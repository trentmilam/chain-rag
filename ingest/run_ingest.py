"""Ingest orchestrator: manifest -> extract/clean/OCR -> section-chunk -> embed -> Qdrant.

Bypasses RAGpack's own ``RAGpack.ingest()`` (which chunks paragraph-only, with no
section awareness) and instead reuses RAGpack's lower-level, proven pieces directly
-- extract_text/is_garbled, the real Embedder, and the Qdrant store helpers -- so
chain-rag controls chunk boundaries exactly (see ingest/chunk.py) while never
re-implementing extraction, embedding, or storage.

Output is the single source of truth chainrag/qdrant_loader.py reads at query time:
every point's payload carries {chain, doc_id, title, section_path, source_url,
doc_type, ocr_engine, ocr_confidence} -- everything ChunkLedger/Plumbline/Legigate
and citation display need.
"""
from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chunk import leaf_chunks, split_into_sections          # noqa: E402
from embed_config import SETTINGS, get_embedder, get_qdrant_client, record_embedder_marker  # noqa: E402
from ocr import needs_ocr, ocr_pdf                            # noqa: E402
from sources import SourceDoc, load_manifest, resolve_local_path, validate_manifest  # noqa: E402

from ragpack.chunk import content_hash, stable_id             # noqa: E402
from ragpack.extract import extract_text                       # noqa: E402
from ragpack.store import ensure_collection, upsert             # noqa: E402


def _clean_html(raw: str) -> tuple[str, bool]:
    """Returns (text, extraction_failed). trafilatura returning empty/None means
    it couldn't find real content (JS-heavy or boilerplate-only page) -- falling
    back to the raw markup is better than losing the document, but the caller
    must be able to tell the difference from a clean extraction."""
    import trafilatura

    extracted = trafilatura.extract(raw, output_format="markdown", include_tables=True)
    if extracted:
        return extracted, False
    return raw, True


def _load_text(doc: SourceDoc) -> tuple[str, str | None, float | None, bool]:
    """Returns (text, ocr_engine_or_None, ocr_confidence_or_None, html_extraction_failed)."""
    path = resolve_local_path(doc.local_path)
    if doc.format == "pdf":
        text = extract_text(path, ocr="never")
        if doc.ocr_required or needs_ocr(path):
            result = ocr_pdf(path)
            return result.text, result.engine, result.mean_confidence, False
        return text, None, None, False
    if doc.format == "html":
        raw = path.read_text(encoding="utf-8", errors="replace")
        text, extraction_failed = _clean_html(raw)
        if extraction_failed:
            print(f"[run_ingest] WARNING: {doc.id}: trafilatura could not extract clean "
                  "content, falling back to raw HTML markup (marked in payload)")
        return text, None, None, extraction_failed
    if doc.format == "mediawiki":
        from mediawiki import mediawiki_to_markdown

        raw = path.read_text(encoding="utf-8", errors="replace")
        return mediawiki_to_markdown(raw), None, None, False
    # md / txt
    return path.read_text(encoding="utf-8", errors="replace"), None, None, False


@dataclass
class DocReport:
    id: str
    chain: str
    n_chunks: int
    ocr_engine: str | None
    ocr_confidence: float | None


@dataclass
class IngestReport:
    collection: str
    model: str
    total_chunks: int
    documents: list = field(default_factory=list)


def _force_clear_local_collection() -> None:
    """qdrant-client's local-mode ``delete_collection`` does
    ``shutil.rmtree(path, ignore_errors=True)`` -- on Windows this can silently fail
    (transient file lock) and leave the old collection directory in place, so a
    *"recreated"* collection actually re-attaches to old data instead of starting
    empty (observed: a stale, buggy chunk surviving a --recreate run alongside its
    corrected replacement). Delete the physical directory ourselves first, with
    errors surfaced instead of swallowed, so --recreate is trustworthy."""
    if not SETTINGS.qdrant or SETTINGS.qdrant == ":memory:" or SETTINGS.qdrant.startswith(("http://", "https://")):
        return
    collection_dir = Path(SETTINGS.qdrant) / "collection" / SETTINGS.collection
    if collection_dir.exists():
        shutil.rmtree(collection_dir)


def run(manifest_path: str, *, recreate: bool = False) -> IngestReport:
    manifest = load_manifest(manifest_path)
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("invalid manifest:\n" + "\n".join(errors))

    if recreate:
        _force_clear_local_collection()

    embedder = get_embedder()
    client = get_qdrant_client()
    ensure_collection(client, SETTINGS.collection, embedder.dim, recreate=recreate)
    record_embedder_marker(recreate=recreate)

    doc_reports: list[DocReport] = []
    total = 0

    for doc in manifest.documents:
        text, ocr_engine, ocr_conf, html_extraction_failed = _load_text(doc)
        sections = split_into_sections(text)
        chunks = leaf_chunks(sections)
        if not chunks:
            raise ValueError(
                f"{doc.id}: extracted zero chunks (empty/garbled text after "
                f"extraction{'+OCR' if ocr_engine else ''}) -- a document silently "
                "missing from the index is worse than a loud failure; fix the source "
                "file or the manifest entry and re-run"
            )

        ids, vectors_input, payloads = [], [], []
        for i, ch in enumerate(chunks):
            chash = content_hash(ch.text)
            pid = stable_id(doc.id, "/".join(ch.section_path), ch.index_in_section, chash)
            ids.append(pid)
            vectors_input.append(ch.text)
            payloads.append({
                "chain": doc.chain,
                "doc_id": doc.id,
                "title": doc.title,
                "section_path": ch.section_path,
                "chunk_index": i,
                "text": ch.text,
                "source_url": doc.source_url,
                "doc_type": doc.doc_type,
                "license_note": doc.license_note,
                "ocr_engine": ocr_engine,
                "ocr_confidence": ocr_conf,
                "html_extraction_failed": html_extraction_failed,
                "content_hash": chash,
            })

        for start in range(0, len(ids), SETTINGS.batch_size):
            batch_ids = ids[start:start + SETTINGS.batch_size]
            batch_texts = vectors_input[start:start + SETTINGS.batch_size]
            batch_payloads = payloads[start:start + SETTINGS.batch_size]
            vectors = embedder.embed(batch_texts)
            upsert(client, SETTINGS.collection, batch_ids, vectors, batch_payloads)

        doc_reports.append(DocReport(
            id=doc.id, chain=doc.chain, n_chunks=len(chunks),
            ocr_engine=ocr_engine, ocr_confidence=ocr_conf,
        ))
        total += len(chunks)

    report = IngestReport(
        collection=SETTINGS.collection, model=SETTINGS.model,
        total_chunks=total, documents=[asdict(d) for d in doc_reports],
    )
    report_path = Path(SETTINGS.qdrant) / f"{SETTINGS.collection}.ingest_report.json" \
        if SETTINGS.qdrant and not SETTINGS.qdrant.startswith("http") else Path("ingest_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()
    r = run(args.manifest, recreate=args.recreate)
    print(json.dumps(asdict(r), indent=2))
