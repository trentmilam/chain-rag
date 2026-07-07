"""Smoke eval: proves the ingest -> Qdrant -> Consilium wiring is correct
end-to-end on a tiny 3-document fixture corpus, cheap enough to run before investing
in ingesting the real 15-40 document corpus.

Two independent stages, each its OWN process invocation:

    python eval/smoke_eval.py --stage hash   # zero-cost: deterministic HashEmbedder,
                                              # no GPU/network -- catches wiring bugs cheaply
    python eval/smoke_eval.py --stage real   # the real CUDA embedder + OCR, full round-trip

They must be separate invocations, not one script doing both: ingest/embed_config.SETTINGS
is a module-level object built once at import time from os.environ, so the embedder/qdrant
config has to be set *before* anything under ingest/ or chainrag/ is ever imported in this
process -- there is no supported way to swap it mid-process.

Both stages ingest the 3-document smoke corpus (smoke/sources.smoke.yaml: the Bitcoin
whitepaper PDF, one EIP HTML page, and a synthetic image-only PDF that only has text via
OCR) into their own Qdrant collection, then ask one targeted question per document and
asserts the router selects the right chain and the answer cites the right document --
including the OCR'd one, proving OCR text flows through the identical path as born-digital
text.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKS = [
    # (query, expected chain, expected doc_id substring)
    #
    # Query wording matters more than usual here: the "hash" stage uses a crude
    # deterministic bag-of-words HashEmbedder (lexical, not semantic), and the
    # synthetic ocr-fixture doc is a short, dense paragraph specifically about
    # difficulty adjustment/ten-minute blocks -- so a query using that same
    # phrasing can out-score the much longer, topically-diluted real whitepaper
    # under crude hashing even when the whitepaper is the "right" answer. Each
    # query below targets content unique to its expected document to keep the
    # three checks independent.
    ("What is proof of work and how does SHA-256 mining let nodes reach consensus?",
     "bitcoin", "btc-whitepaper"),
    ("What functions does the ERC-20 token standard interface define?",
     "ethereum", "eip-20"),
    ("How often does the network retarget mining difficulty, every how many blocks?",
     "bitcoin", "ocr-fixture"),
]


def _configure_env(stage: str) -> None:
    if stage == "hash":
        os.environ["CHAINRAG_EMBED_MODEL"] = "hash"
        os.environ["CHAINRAG_QDRANT"] = str(ROOT / "data" / "qdrant_smoke_hash")
        os.environ["CHAINRAG_COLLECTION"] = "chain_rag_smoke_hash"
    else:
        os.environ.setdefault("CHAINRAG_QDRANT", str(ROOT / "data" / "qdrant"))
        os.environ["CHAINRAG_COLLECTION"] = "chain_rag_smoke_real"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["hash", "real"], required=True)
    args = parser.parse_args()

    _configure_env(args.stage)
    sys.path.insert(0, str(ROOT))

    # Imported only now -- after os.environ is set -- so embed_config.SETTINGS picks up
    # this stage's config.
    from chainrag.bootstrap import build_registry
    from consilium.composer import compose
    from consilium.router import Router
    from ingest.embed_config import get_embedder
    from ingest.run_ingest import run as run_ingest

    manifest = str(ROOT / "smoke" / "sources.smoke.yaml")
    print(f"[{args.stage}] ingesting {manifest} ...")
    report = run_ingest(manifest, recreate=True)
    print(f"[{args.stage}] {report.total_chunks} chunks across {len(report.documents)} docs:")
    for d in report.documents:
        print(f"  - {d}")

    embedder = get_embedder()
    # The smoke corpus only covers bitcoin + ethereum -- scope build_registry to
    # just those two so it doesn't fail loudly on the 4 chains with no data yet
    # (that fail-loud behavior is exactly right once the full corpus covers all 6).
    registry = build_registry(embedder, chains=["bitcoin", "ethereum"])
    router = Router(registry, embedder)

    failures = []
    for query, expected_chain, expected_doc in CHECKS:
        routed = router.route(query)
        answer = compose(query, routed, registry, embedder, harden=True)
        cited_docs = {c.doc for c in answer.citations}
        ok_module = expected_chain in answer.modules_used
        ok_doc = any(expected_doc in doc for doc in cited_docs)
        status = "OK" if (not answer.abstained and ok_module and ok_doc) else "FAIL"
        if status == "FAIL":
            failures.append((query, answer.abstained, answer.modules_used, cited_docs))
        print(f"[{status}] {query!r}\n"
              f"    abstained={answer.abstained} modules_used={answer.modules_used} "
              f"cited_docs={cited_docs}")

    if failures:
        print(f"\n[{args.stage}] {len(failures)}/{len(CHECKS)} checks FAILED")
        return 1
    print(f"\n[{args.stage}] all {len(CHECKS)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
