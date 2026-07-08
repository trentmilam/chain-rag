"""build_registry(embedder) -> consilium.registry.Registry

Assembles chain-rag's own Registry: one Consilium Module per chain, each loaded
from Qdrant (see qdrant_loader.py) rather than from disk -- the ingest pipeline is
the only thing that ever embeds text; this just reads the precomputed vectors back.

This is an entry file: it performs the sibling-path bootstrap (see _paths.py)
before importing anything from consilium.
"""
from __future__ import annotations

from chainrag._paths import add_sibling_paths

add_sibling_paths()

from consilium.module import Descriptor    # noqa: E402
from consilium.registry import Registry    # noqa: E402

from chainrag.qdrant_loader import load_module_from_qdrant  # noqa: E402

_CHAIN_DESCRIPTORS = {
    "bitcoin": Descriptor(
        name="bitcoin",
        subjects=["UTXO model", "proof of work", "difficulty adjustment", "SHA-256 mining",
                  "script opcodes", "SegWit", "Lightning Network", "block reward halving"],
        example_queries=["How does Bitcoin's difficulty adjustment work?",
                          "What is a UTXO?", "How does SegWit change transaction weight?"],
        authority="Bitcoin Core dev docs + BIPs + the Nakamoto whitepaper",
        freshness="static (protocol spec snapshot, not live chain data)",
        trust_tier=0.85,
    ),
    "ethereum": Descriptor(
        name="ethereum",
        subjects=["account model", "EVM", "gas and gas pricing", "EIP standards",
                  "proof of stake consensus", "smart contracts", "Solidity", "ERC token standards"],
        example_queries=["How does EIP-1559 fee pricing work?",
                          "What is the difference between an EOA and a contract account?",
                          "What does EIP-4844 do?"],
        authority="Ethereum Yellow Paper + EIPs + ethereum.org developer docs",
        freshness="static (protocol spec snapshot, not live chain data)",
        trust_tier=0.85,
    ),
    "solana": Descriptor(
        name="solana",
        subjects=["proof of history", "Sealevel parallel execution", "Tower BFT consensus",
                  "accounts model", "program derived addresses", "validators"],
        example_queries=["What is Solana's proof of history?",
                          "How does Sealevel enable parallel transaction execution?"],
        authority="Solana whitepaper + core protocol docs",
        freshness="static (protocol spec snapshot, not live chain data)",
        trust_tier=0.85,
    ),
    "monero": Descriptor(
        name="monero",
        subjects=["CryptoNote protocol", "ring signatures", "RingCT", "stealth addresses",
                  "privacy and fungibility", "dynamic block size"],
        example_queries=["How do ring signatures provide sender ambiguity?",
                          "What does RingCT hide about a transaction?"],
        authority="the CryptoNote whitepaper + Monero's RingCT paper",
        freshness="static (protocol spec snapshot, not live chain data)",
        trust_tier=0.85,
    ),
    "polygon": Descriptor(
        name="polygon",
        subjects=["proof of stake sidechain", "Plasma", "checkpointing to Ethereum",
                  "zkEVM", "validator and heimdall/bor architecture"],
        example_queries=["How does Polygon checkpoint to Ethereum mainnet?",
                          "What is Polygon zkEVM?"],
        authority="Polygon architecture documentation",
        freshness="static (protocol spec snapshot, not live chain data)",
        trust_tier=0.8,
    ),
    "cardano": Descriptor(
        name="cardano",
        subjects=["Ouroboros proof of stake", "eUTXO model", "Plutus smart contracts",
                  "epochs and slots", "stake pools"],
        example_queries=["How does the eUTXO model differ from Ethereum's account model?",
                          "What is Ouroboros consensus?"],
        authority="the Ouroboros paper + Cardano's Plutus documentation",
        freshness="static (protocol spec snapshot, not live chain data)",
        trust_tier=0.8,
    ),
}


# consilium.router.Router's defaults (floor=0.11, anchor_centroid=0.25,
# anchor_best_chunk=0.25) assume a near-zero baseline cosine between unrelated
# text -- true for a bag-of-words HashEmbedder, false for a real dense embedder
# (BAAI/bge-base-en-v1.5) over a ~1100-chunk corpus. Measured directly against
# this corpus (_diag_router_scores.py): every genuine in-scope query's anchor
# module has centroid cosine >=0.626 and subject-token overlap >=2, while 10
# diverse out-of-scope probes (recipes, weather, gibberish) never exceed
# centroid 0.492 and always have 0 subject-token overlap -- but best-chunk
# cosine (a max over ~1100 chunks) is a saturated order statistic that clears
# 0.25 for EVERY probe including gibberish, so it is not usable as an anchor
# signal at this corpus scale. These are per-instance Router/compose kwargs
# (both are the library's own documented calibration knobs, not internal
# consilium state) recalibrated for chain-rag's own embedder+corpus pair only;
# consilium's shared defaults (used by every other Consilium instance)
# are untouched.
ROUTER_KWARGS = {"floor": 0.42, "anchor_centroid": 0.58, "anchor_best_chunk": 0.97}


def build_registry(embedder, client=None, collection: str | None = None,
                    chains: list[str] | None = None) -> Registry:
    """``chains`` restricts which of the 6 declared chains must have data --
    defaults to all 6 (fail loudly if any is missing, the real-corpus/production
    behavior). Pass an explicit subset for a smoke test whose tiny corpus only
    covers some chains; production callers should never need to pass this."""
    from ingest.embed_config import SETTINGS, get_qdrant_client, verify_embedder_marker

    verify_embedder_marker()
    client = client or get_qdrant_client()
    collection = collection or SETTINGS.collection
    wanted = _CHAIN_DESCRIPTORS.items() if chains is None else (
        (chain, _CHAIN_DESCRIPTORS[chain]) for chain in chains
    )
    modules = [
        load_module_from_qdrant(chain, client, collection, embedder, descriptor)
        for chain, descriptor in wanted
    ]
    return Registry(modules)
