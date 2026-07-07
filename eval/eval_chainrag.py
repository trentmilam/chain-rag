"""Production eval: proves the full ingest -> Qdrant -> Consilium pipeline answers
real technical questions against the real, already-ingested multichain corpus (22
documents, 1100 chunks across all 6 chains). Uses the real embedder
(BAAI/bge-base-en-v1.5) -- the exact one that ingested the corpus, no hash-embedder
shortcut -- so this proves genuine semantic retrieval end-to-end, not wiring alone.

Six per-chain checks (one held-out question per chain: not abstained, the expected
chain contributed a citation), one genuine cross-chain comparison (both expected
chains contributed), and one out-of-scope check that asserts an honest abstain.
Deterministic given the already-ingested corpus; no re-ingestion here (see
eval/smoke_eval.py for the ingest-path smoke test).

MEASURED CALIBRATION NOTE: consilium.router.Router's library defaults (floor=0.11,
anchor_centroid=0.25, anchor_best_chunk=0.25) assume a near-zero baseline cosine
between unrelated text -- true for a bag-of-words HashEmbedder, false for the real
BAAI/bge-base-en-v1.5 dense embedder over this 1100-chunk corpus (best-chunk cosine,
a max over ~1100 chunks, is a saturated order statistic that clears 0.25 for almost
any query, including gibberish). This is measured, not assumed: see
chainrag.bootstrap.ROUTER_KWARGS for the diagnostic numbers and the recalibrated,
per-instance threshold values used below -- consilium's own shared defaults (e.g.
wealth-platform's separate Router instance) are untouched; this is chain-rag's own
Router instantiation exercising the library's documented constructor kwargs.

    python eval/eval_chainrag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chainrag._paths import add_sibling_paths  # noqa: E402

add_sibling_paths()

from chainrag.bootstrap import ROUTER_KWARGS, build_registry  # noqa: E402
from consilium.composer import compose  # noqa: E402
from consilium.router import Router  # noqa: E402
from ingest.embed_config import get_embedder  # noqa: E402

# One held-out, naturally-phrased technical question per chain. Real semantic
# embedder means these don't need to parrot the chain's Descriptor.subjects
# wording -- ordinary technical phrasing routes correctly via cosine similarity.
PER_CHAIN_CHECKS = [
    ("How does Bitcoin's mining difficulty retarget over time so that new blocks "
     "keep arriving roughly every ten minutes?", "bitcoin"),
    ("What's the difference between an externally owned account and a smart "
     "contract account under Ethereum's account model?", "ethereum"),
    ("How does proof of history let a Solana validator order transactions without "
     "waiting on a round of network-wide communication?", "solana"),
    ("How do ring signatures hide which of several possible senders actually "
     "authorized a Monero transaction?", "monero"),
    ("How does Polygon's proof-of-stake sidechain checkpoint its state back to "
     "Ethereum mainnet?", "polygon"),
    ("How does Cardano's Ouroboros protocol choose which stake pool gets to "
     "produce the next block in a given slot?", "cardano"),
]

CROSS_CHAIN_QUESTION = (
    "Compare how Bitcoin's proof-of-work mining and Cardano's Ouroboros "
    "proof-of-stake protocol each decide who is allowed to produce the next block."
)
CROSS_CHAIN_EXPECTED = {"bitcoin", "cardano"}

OOS_QUESTION = "How do you properly season a cast iron skillet before first use?"


def main() -> int:
    embedder = get_embedder()
    registry = build_registry(embedder)
    router = Router(registry, embedder, **ROUTER_KWARGS)
    checks: dict[str, bool] = {}

    for question, expected_chain in PER_CHAIN_CHECKS:
        routed = router.route(question)
        answer = compose(question, routed, registry, embedder, harden=True)
        ok = (not answer.abstained
              and expected_chain in answer.modules_used
              and len(answer.citations) >= 1)
        checks[f"{expected_chain}_answers_correctly"] = ok
        print(f"[{'OK' if ok else 'FAIL'}] ({expected_chain}) {question!r}\n"
              f"    abstained={answer.abstained} modules_used={answer.modules_used} "
              f"citations={len(answer.citations)}")

    routed = router.route(CROSS_CHAIN_QUESTION)
    answer = compose(CROSS_CHAIN_QUESTION, routed, registry, embedder, harden=True)
    ok = not answer.abstained and CROSS_CHAIN_EXPECTED.issubset(set(answer.modules_used))
    checks["cross_chain_comparison"] = ok
    print(f"[{'OK' if ok else 'FAIL'}] (cross-chain) {CROSS_CHAIN_QUESTION!r}\n"
          f"    abstained={answer.abstained} modules_used={answer.modules_used}")

    routed = router.route(OOS_QUESTION)
    answer = compose(OOS_QUESTION, routed, registry, embedder, harden=True)
    ok = answer.abstained
    checks["out_of_scope_abstains"] = ok
    print(f"[{'OK' if ok else 'FAIL'}] (out-of-scope) {OOS_QUESTION!r}\n"
          f"    abstained={answer.abstained} modules_used={answer.modules_used}")

    print("\n=== CHECKS ===")
    for k, v in checks.items():
        print(f"{'OK  ' if v else 'FAIL'} {k}")
    passed = all(checks.values())
    print("\nRESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
