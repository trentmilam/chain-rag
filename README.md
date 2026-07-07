# chain-rag

A real, cited retrieval system over primary blockchain protocol documentation —
Bitcoin, Ethereum, Solana, Monero, Polygon, and Cardano — built on [Consilium](../consilium)'s
router/citation-gating spine and [RAGpack](../RAGpack)'s embedder/chunker/vector-store
machinery. Ask a real technical question about any of the six chains (or one that
spans two of them) and get back an answer built entirely out of real cited chunks
from the source documents, or an honest abstain. There is no LLM anywhere in the
answer path — retrieval + a deterministic router + a citation-integrity gate,
nothing else, so every line of every answer traces to an actual document.

This is a second, independent vertical proving the same architecture pattern
[wealth-platform](../wealth-platform) already established generalizes: a standalone
domain, its own Consilium `Registry`/`Router` instance, zero coupling between the
two, sharing only the generic library underneath.

## Honest scope: this repo is NOT standalone

`chain-rag` imports two **sibling directories** that must sit next to it under the
same `projects/` root:

```
projects/
  consilium/          <- the routing spine (Registry, Router, compose/integrity gate)
  rag-reliability/     <- VecStamp / ChunkLedger / Plumbline / Legigate gate tools
  chain-rag/            <- this repo
```

Neither is pip-installed or vendored; each is imported by putting its repo root
(or, for `rag-reliability`, each individual tool subdir — it has no `__init__.py`)
directly on `sys.path`. See `chainrag/_paths.py::add_sibling_paths()`, called by
every entry file before importing anything from a sibling repo. `RAGpack`, unlike
these two, **is** pip-installed editable into chain-rag's own `.venv` — `import
ragpack` needs no sys.path entry.

Move this repo without `consilium` and every import fails immediately and loudly —
there is no silent degraded mode.

## Running it

```
cd projects/chain-rag
app.bat     # chat UI (gr.ChatInterface), own venv, no browser auto-launch
demo.bat    # scripted 8-question transcript, no typing required
```

Both are fully offline at query time beyond a local Qdrant read and local
embedding — no live network calls, no LLM.

## Running the eval

```
cd projects/chain-rag
.venv/Scripts/python.exe eval/eval_chainrag.py
```

Deterministic given the already-ingested corpus (22 documents, ~1,055 chunks
across all 6 chains; see `sources.yaml`). Uses the real embedder
(`BAAI/bge-base-en-v1.5`) — the exact one that ingested the corpus — so this
proves genuine semantic retrieval end-to-end, not wiring alone. Eight checks:

- one held-out, naturally-phrased technical question **per chain** (6) —
  asserts the expected chain contributed a citation and the router didn't abstain;
- one genuine **cross-chain comparison** (Bitcoin PoW vs. Cardano's Ouroboros
  PoS) — asserts both expected chains contributed citations;
- one **out-of-scope** question ("How do you properly season a cast iron
  skillet?") — asserts an honest abstain.

`python scripts/run_all_evals.py` (framework-root aggregator) picks this eval up
alongside the other portfolio evals.

## A measured finding, not a hidden one: real dense embedders need per-corpus calibration

Consilium's `Router` ships library defaults (`floor=0.11`, `anchor_centroid=0.25`,
`anchor_best_chunk=0.25`) that assume a near-zero baseline cosine similarity
between unrelated text — true for a crude bag-of-words `HashEmbedder`, **false**
for a real dense sentence embedder. Measured directly against this corpus: every
genuine in-scope query's anchor module has descriptor-centroid cosine ≥0.626 and
≥2 subject-keyword hits, while 10 diverse out-of-scope probes (recipes, weather,
gibberish) never exceeded centroid 0.492 and always had 0 keyword hits — but
best-chunk cosine (a max over ~1,055 chunks) is a saturated order statistic that
cleared the 0.25 anchor threshold for *every* probe, including gibberish. Left at
the library defaults, the out-of-scope check reliably failed and the answer to
every query, on-topic or not, cited all six chains indiscriminately.

`chainrag/bootstrap.py::ROUTER_KWARGS` recalibrates `floor`/`anchor_centroid`/
`anchor_best_chunk` for this embedder+corpus pair specifically, using Consilium's
own documented per-instance constructor kwargs — Consilium's shared source and its
library-wide defaults (which `wealth-platform`'s own, separate `Router` instance
still uses) are untouched. This is the kind of gap the reliability-gate suite
below is built to surface rather than paper over.

## Reliability gates

```
.venv/Scripts/python.exe scripts/run_reliability_gates.py
```

Four gates from `rag-reliability`, run against the real ingested corpus:

- **VecStamp** — embedding build/load identity (the query-time embedder is
  bit-identical to the one that ingested the corpus).
- **ChunkLedger** — reference-free conservation: no document/chunk silently
  dropped structural content (headings, tables, code blocks, links) during ingest.
- **Plumbline** — every citation resolves back to a real span in its source
  document (fuzzy match survives whitespace/OCR noise, not genuine corruption).
- **Legigate** — OCR legibility gate; correctly reports **N/A** here since the
  real corpus needed zero OCR (all 22 sources are born-digital).

Current reference run: **3 PASS, 0 FAIL, 1 N/A**.

## Layout

```
chainrag/
  _paths.py         sibling-path bootstrap
  bootstrap.py      build_registry(embedder) -> consilium.registry.Registry,
                     the 6 chain Descriptors, ROUTER_KWARGS (see calibration note above)
  qdrant_loader.py  loads a consilium Module's chunks straight from Qdrant
ingest/
  sources.py        SourceDoc manifest schema + loader/validator
  chunk.py          header/section-aware pre-splitter (wraps RAGpack's chunk_text)
  ocr.py            EasyOCR primary / PaddleOCR fallback, confidence-gated
  embed_config.py   single shared Settings (model/qdrant path) for ingest + query time
  run_ingest.py     manifest -> extract/clean/OCR -> chunk -> embed -> Qdrant
eval/
  smoke_eval.py     wiring smoke test (hash + real embedder stages)
  eval_chainrag.py  the 8-check production eval described above
scripts/
  run_reliability_gates.py   VecStamp/ChunkLedger/Plumbline/Legigate over the real corpus
sources.yaml        the 22-document production manifest (chain, source_url, license_note)
app.py               gr.ChatInterface chat UI
run_demo.py          scripted 8-question narrated transcript
```
