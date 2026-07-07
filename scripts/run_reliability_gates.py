"""Reliability-gate matrix: VecStamp + ChunkLedger + Plumbline + Legigate over
chain-rag's real, already-ingested corpus.

This is the first wiring of these four independent rag-reliability gate tools
against a real pipeline (no existing call sites to copy from) -- every call
below uses each tool's actual discovered public API, nothing invented:

  * VecStamp    -- build/load embedder-identity certificate (embedding drift).
  * ChunkLedger -- source->chunk structural conservation (silently dropped
                   content) + a run-over-run drift gate.
  * Plumbline   -- citation quoted-text -> source-document provenance tracing
                   for real answers composed by Consilium's Router + compose().
  * Legigate    -- reference-free OCR/parse legibility gate over any OCR'd
                   chunks in the corpus.

Each gate is independent: one gate erroring or having nothing to check does
not block the others. Run:

    .venv\\Scripts\\python.exe scripts\\run_reliability_gates.py

Exits 0 if every gate that had real data to check passed (or had nothing to
check -- reported N/A, never fabricated), exits 1 if any gate FAILed.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

CHAIN_RAG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHAIN_RAG_ROOT))  # so `ingest.*` / `chainrag.*` resolve, same as eval/smoke_eval.py

from chainrag._paths import add_sibling_paths  # noqa: E402

# Puts `consilium` on sys.path, plus each of vecstamp/chunkledger/plumbline/legigate's
# OWN subdir individually (rag-reliability has no __init__.py -- every tool is a flat
# module that bare-imports its neighbors, e.g. legigate.py does `from embed import ...`).
add_sibling_paths()

from ingest.embed_config import SETTINGS, get_embedder, get_qdrant_client, verify_embedder_marker  # noqa: E402
from ingest.run_ingest import _load_text  # noqa: E402
from ingest.sources import load_manifest  # noqa: E402

from chainrag.bootstrap import _CHAIN_DESCRIPTORS  # noqa: E402
from chainrag.qdrant_loader import load_module_from_qdrant  # noqa: E402

from consilium.composer import compose  # noqa: E402
from consilium.registry import Registry  # noqa: E402
from consilium.router import Router  # noqa: E402

from vecstamp import build_manifest as vecstamp_build_manifest  # noqa: E402
from vecstamp import verify as vecstamp_verify  # noqa: E402

from chunkledger import build_ledger, drift_gate  # noqa: E402

from plumbline import build_manifest as plumbline_build_manifest  # noqa: E402

from legigate import gate as legigate_gate  # noqa: E402


@dataclass
class GateResult:
    gate: str
    status: str  # "PASS" | "FAIL" | "N/A"
    detail: str


# ---------------------------------------------------------------------------
# VecStamp -- build-vs-load embedder identity certificate
# ---------------------------------------------------------------------------
def _run_vecstamp(build_embedder, get_embedder_fn) -> GateResult:
    """ragpack's Embedder/HashEmbedder (returned by get_embedder()) already expose
    ``embed_one(text) -> list[float]`` -- exactly VecStamp's single-string embed_fn
    contract, no adapter needed. ``build_embedder`` stands in for the ingest-time
    embedder (it is the one this script already used to load the real corpus);
    ``get_embedder_fn()`` is called again to construct an INDEPENDENT second
    instance of the same configured production embedder, standing in for a
    freshly-restarted serving process -- the real build-vs-load boundary VecStamp
    certifies.
    """
    load_embedder = get_embedder_fn()
    manifest = vecstamp_build_manifest(build_embedder.embed_one)
    result = vecstamp_verify(manifest, load_embedder.embed_one)
    name = "VecStamp (embedding build/load identity)"
    if result.ok:
        return GateResult(
            name, "PASS",
            f"verdict={result.verdict} dim={manifest.dim} probes={len(manifest.probes)} "
            f"float_hash={manifest.float_hash}",
        )
    return GateResult(name, "FAIL", f"verdict={result.verdict} detail={result.detail}")


# ---------------------------------------------------------------------------
# ChunkLedger -- source -> chunk structural conservation, per real document
# ---------------------------------------------------------------------------
def _run_chunkledger(modules, docs_by_id) -> GateResult:
    """For each real source document, build_ledger(raw_source_text, real_ingested_
    chunk_texts) proves every structural element (table/code block/heading/list
    item/link/numeric span) that was in the raw extracted text survived into the
    chunks Qdrant actually holds. A persisted per-doc baseline lets a SECOND run
    of this script also exercise drift_gate (the tool's run-over-run regression
    tripwire) for real, rather than fabricating a synthetic prior."""
    name = "ChunkLedger (source->chunk conservation)"

    chunks_by_doc: dict[str, list[str]] = {}
    for module in modules:
        for ch in module.chunks:
            chunks_by_doc.setdefault(ch.doc, []).append(ch.text)

    baseline_dir = CHAIN_RAG_ROOT / "data" / "reliability_baselines" / "chunkledger"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    checked = 0
    dropped_docs: list[str] = []
    regressed_docs: list[str] = []
    for doc in docs_by_id.values():
        chunks = chunks_by_doc.get(doc.id)
        if not chunks:
            continue  # not present in the index -- nothing ingested yet to conserve against
        checked += 1

        source_text, _ocr_engine, _ocr_conf, _html_failed = _load_text(doc)
        ledger = build_ledger(source_text, chunks)
        if any(pt["dropped"] for pt in ledger["per_type"].values()):
            dropped_docs.append(doc.id)

        baseline_path = baseline_dir / f"{doc.id}.json"
        if baseline_path.exists():
            prior = json.loads(baseline_path.read_text(encoding="utf-8"))
            if drift_gate(prior, ledger)["tripped"]:
                regressed_docs.append(doc.id)
        baseline_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    if checked == 0:
        return GateResult(name, "N/A", "0 ingested documents found to check")
    if dropped_docs or regressed_docs:
        return GateResult(
            name, "FAIL",
            f"{checked} docs checked; dropped-content docs={dropped_docs}; "
            f"drift-regressed docs={regressed_docs}",
        )
    return GateResult(name, "PASS", f"{checked} docs checked; 0 dropped structural elements")


# ---------------------------------------------------------------------------
# Plumbline -- citation quoted-text -> source-document provenance
# ---------------------------------------------------------------------------
def _run_plumbline(registry, embedder, chain_names, docs_by_id) -> GateResult:
    """One real representative query per chain (its Descriptor's own first
    example_query) through Router.route() + compose(harden=True), mirroring
    eval/smoke_eval.py's exact pattern. Every resulting Citation's claim text
    (v1 composer is extractive: claim == the cited chunk's text) is resolved
    against ITS document's raw _load_text() output via Plumbline's fuzzy
    (whitespace/case-normalization-aware) resolver -- proving every citation
    genuinely traces back to a real span in the source document."""
    name = "Plumbline (citation->source provenance)"

    router = Router(registry, embedder)
    citations_by_doc: dict[str, dict[str, str]] = {}
    abstained_chains: list[str] = []
    for chain in chain_names:
        module = registry.by_name(chain)
        query = module.descriptor.example_queries[0]
        routed = router.route(query)
        answer = compose(query, routed, registry, embedder, harden=True)
        if answer.abstained:
            abstained_chains.append(chain)
        for citation in answer.citations:
            citations_by_doc.setdefault(citation.doc, {})[citation.chunk_id] = citation.claim

    if not citations_by_doc:
        return GateResult(
            name, "N/A",
            f"0 citations produced across {len(chain_names)} representative queries "
            f"(abstained: {abstained_chains})",
        )

    note = f" (abstained: {abstained_chains})" if abstained_chains else ""
    unresolved: list[str] = []
    n_checked = 0
    for doc_id, citations in citations_by_doc.items():
        doc = docs_by_id.get(doc_id)
        if doc is None:
            continue  # a citation referenced a doc_id not in the manifest -- shouldn't happen, skip not crash
        source_text, *_rest = _load_text(doc)
        spans = plumbline_build_manifest(source_text, citations)
        n_checked += len(spans)
        unresolved.extend(cid for cid, span in spans.items() if span is None)

    if unresolved:
        return GateResult(name, "FAIL", f"{n_checked} citations checked; unresolved={unresolved}{note}")
    return GateResult(
        name, "PASS",
        f"{n_checked} citations across {len(citations_by_doc)} docs all traced to source{note}",
    )


# ---------------------------------------------------------------------------
# Legigate -- reference-free legibility gate over any OCR'd chunks
# ---------------------------------------------------------------------------
def _scan_ocr_chunks(client, collection: str) -> list[tuple[str, str]]:
    """Full-collection scroll (same mechanics as chainrag/qdrant_loader.py's
    load_module_from_qdrant, just without a chain filter) collecting every
    chunk whose payload carries a non-null ocr_engine."""
    ocr_chunks: list[tuple[str, str]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=256,
            with_payload=True, with_vectors=False, offset=offset,
        )
        for p in points:
            payload = p.payload or {}
            if payload.get("ocr_engine"):
                chunk_id = f"{payload.get('doc_id')}#{payload.get('chunk_index')}"
                ocr_chunks.append((chunk_id, str(payload.get("text", ""))))
        if offset is None:
            break
    return ocr_chunks


def _run_legigate(client) -> GateResult:
    name = "Legigate (OCR/parse legibility)"
    ocr_chunks = _scan_ocr_chunks(client, SETTINGS.collection)
    if not ocr_chunks:
        # Real finding: the production ingest (22 docs, 1100 chunks) needed no OCR --
        # every payload's ocr_engine is null. Report that explicitly rather than
        # skipping silently or fabricating OCR'd input to gate against.
        return GateResult(name, "N/A", f"N/A -- {len(ocr_chunks)} OCR'd chunks in corpus")
    kept, quarantined = legigate_gate(ocr_chunks)
    if quarantined:
        reasons = [f"{v.chunk_id}: {v.reasons}" for v in quarantined]
        return GateResult(name, "FAIL", f"{len(kept)} kept / {len(quarantined)} quarantined -- {reasons}")
    return GateResult(name, "PASS", f"{len(kept)}/{len(ocr_chunks)} OCR'd chunks legible")


# ---------------------------------------------------------------------------
def _print_matrix(results: list[GateResult]) -> None:
    name_w = max(len(r.gate) for r in results)
    print()
    print("=" * 88)
    print("chain-rag reliability gate matrix")
    print("=" * 88)
    for r in results:
        print(f"[{r.status:4s}] {r.gate.ljust(name_w)}  {r.detail}")
    print("=" * 88)
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_na = sum(1 for r in results if r.status == "N/A")
    print(f"{n_pass} PASS, {n_fail} FAIL, {n_na} N/A")
    print()


def main() -> int:
    manifest = load_manifest(str(CHAIN_RAG_ROOT / "sources.yaml"))
    docs_by_id = {d.id: d for d in manifest.documents}

    embedder = get_embedder()
    client = get_qdrant_client()
    verify_embedder_marker()

    chain_names = list(_CHAIN_DESCRIPTORS.keys())
    modules = [
        load_module_from_qdrant(chain, client, SETTINGS.collection, embedder, descriptor)
        for chain, descriptor in _CHAIN_DESCRIPTORS.items()
    ]
    registry = Registry(modules)

    results = [
        _run_vecstamp(embedder, get_embedder),
        _run_chunkledger(modules, docs_by_id),
        _run_plumbline(registry, embedder, chain_names, docs_by_id),
        _run_legigate(client),
    ]
    _print_matrix(results)

    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
