"""chain-rag LIVE DEMO -- a scripted, narrated transcript over the real, already-
ingested multichain corpus (22 documents, 1100 chunks across 6 chains).

    cd projects/chain-rag
    .venv/Scripts/python.exe run_demo.py

Eight real questions, in order: one held-out technical question per chain (6), one
genuine cross-chain comparison, and one question with nothing to do with any of the
6 chains. No mocked output -- every line below is produced live by the real router
+ the real citation-gated composer, reusing the exact queries proven in
eval/eval_chainrag.py. Deterministic + offline beyond the local Qdrant read and
local embedding (no live network calls).

Honesty note on the last question: this system's design is "cited retrieval or
honest abstain -- never a fabricated claim" (see consilium.integrity.gate, which
drops any claim that isn't cosine-bound to a real source chunk). Whether an
out-of-scope QUERY itself gets routed away before that point depends on the
router's topic-relevance thresholds -- recalibrated for this real embedder+corpus
pair, see chainrag.bootstrap.ROUTER_KWARGS. This transcript prints the router/
composer's REAL, live output for that question rather than a scripted "it
abstained" -- whatever actually happens is what gets printed.
"""
from __future__ import annotations

import sys
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_ROOT))

from chainrag._paths import add_sibling_paths  # noqa: E402

add_sibling_paths()

from chainrag.bootstrap import ROUTER_KWARGS, build_registry  # noqa: E402
from consilium.composer import compose  # noqa: E402
from consilium.router import Router  # noqa: E402
from ingest.embed_config import get_embedder  # noqa: E402

BAR = "=" * 78


def ask(router, registry, embedder, n: int, question: str):
    print(f"\n{BAR}\n  Q{n}. {question}\n{BAR}")
    routed = router.route(question)
    answer = compose(question, routed, registry, embedder, harden=True)
    if answer.abstained:
        print("  -> ABSTAINED -- honest refusal (nothing in the corpus supported "
              "an answer to this).")
    else:
        print(f"  -> routed to CITED RETRIEVAL  (chains: {', '.join(answer.modules_used)}, "
              f"{len(answer.citations)} citation(s))")
    return answer


def excerpt(answer, chain: str, chars: int = 220) -> str:
    """The actual retrieved claim text for ``chain``'s top citation, whitespace-
    collapsed (source chunks carry embedded newlines from the original PDF/HTML
    layout) so a reader sees a real sentence, not a layout artifact."""
    for c in answer.citations:
        if c.module == chain:
            flat = " ".join(c.claim.split())
            return flat[:chars] + ("..." if len(flat) > chars else "")
    return ""


def main() -> int:
    print(BAR)
    print("  CHAIN-RAG -- one router across six blockchain knowledge bases,")
    print("  every claim bound to a real source chunk, or the system says so.")
    print(BAR)

    embedder = get_embedder()
    registry = build_registry(embedder)
    router = Router(registry, embedder, **ROUTER_KWARGS)

    # Q1-Q6 -- one held-out technical question per chain.
    a1 = ask(router, registry, embedder, 1,
             "How does Bitcoin's mining difficulty retarget over time so that new "
             "blocks keep arriving roughly every ten minutes?")
    print(f"     \"{excerpt(a1, 'bitcoin')}\"")

    a2 = ask(router, registry, embedder, 2,
             "What's the difference between an externally owned account and a smart "
             "contract account under Ethereum's account model?")
    print(f"     \"{excerpt(a2, 'ethereum')}\"")

    ask(router, registry, embedder, 3,
        "How does proof of history let a Solana validator order transactions "
        "without waiting on a round of network-wide communication?")

    a4 = ask(router, registry, embedder, 4,
             "How do ring signatures hide which of several possible senders "
             "actually authorized a Monero transaction?")
    print(f"     \"{excerpt(a4, 'monero')}\"")

    ask(router, registry, embedder, 5,
        "How does Polygon's proof-of-stake sidechain checkpoint its state back "
        "to Ethereum mainnet?")

    a6 = ask(router, registry, embedder, 6,
             "How does Cardano's Ouroboros protocol choose which stake pool gets "
             "to produce the next block in a given slot?")
    print(f"     \"{excerpt(a6, 'cardano')}\"")

    # Q7 -- a genuine cross-chain comparison: one query, two consensus mechanisms.
    a7 = ask(router, registry, embedder, 7,
             "Compare how Bitcoin's proof-of-work mining and Cardano's Ouroboros "
             "proof-of-stake protocol each decide who is allowed to produce the "
             "next block.")
    print(f"     drew citations from both: {', '.join(a7.modules_used)}")

    # Q8 -- nothing to do with any of the 6 chains. The intended behavior is an
    # honest abstain; ask() above prints whatever the live system actually does.
    ask(router, registry, embedder, 8,
        "How do you properly season a cast iron skillet before first use?")

    print(f"\n{BAR}")
    print("  Every citation above traces to a real chunk of a real source document")
    print("  (see sources.yaml) -- the composer's integrity gate drops any claim")
    print("  that isn't cosine-bound to an actual chunk, and abstains rather than")
    print("  emit an answer with zero supported claims. CITED RETRIEVAL or an")
    print("  HONEST ABSTAIN: never a fabricated guess.")
    print(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
