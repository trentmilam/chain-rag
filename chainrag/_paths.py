"""Sibling-path bootstrap.

chain-rag is deliberately NOT standalone: it reuses ``consilium`` (the routing/
citation-gating spine, as a library, own Registry instance -- zero coupling to
any other Consilium instance) and 4 of ``rag-reliability``'s gate tools, both living
as SIBLING directories under the same ``projects/`` root, each its own git repo.
None of them are pip-installed; each is imported by putting its root on
``sys.path``. Both siblings are public repos -- clone them next to this one
(see the README).

(RAGpack, unlike these, IS pip-installed editable into chain-rag's own venv --
``import ragpack`` needs no sys.path entry here.)

Every chain-rag entry file calls :func:`add_sibling_paths` first, before importing
anything from a sibling repo. Idempotent -- safe to call repeatedly in one process.
"""
from __future__ import annotations

import os
import sys

CHAIN_RAG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECTS_ROOT = os.path.dirname(CHAIN_RAG_ROOT)

CONSILIUM_ROOT = os.path.join(PROJECTS_ROOT, "consilium")
RAG_RELIABILITY_ROOT = os.path.join(PROJECTS_ROOT, "rag-reliability")

# rag-reliability has NO __init__.py anywhere -- each tool is a flat module that
# imports its neighbors with bare `import <module>` (e.g. legigate.py does
# `from embed import cosine, embed`), so each tool's OWN subdir must be on
# sys.path individually, not just the rag-reliability root.
_RELIABILITY_TOOLS = ["vecstamp", "chunkledger", "plumbline", "legigate"]

_SIBLING_ROOTS = [
    CHAIN_RAG_ROOT,     # so `import chainrag` resolves for entry scripts
    CONSILIUM_ROOT,     # `import consilium`
    *[os.path.join(RAG_RELIABILITY_ROOT, t) for t in _RELIABILITY_TOOLS],
]


def add_sibling_paths() -> None:
    """Insert every sibling repo root at the front of ``sys.path`` (skipping any
    already present). Call before importing anything from a sibling repo."""
    for root in _SIBLING_ROOTS:
        if root not in sys.path:
            sys.path.insert(0, root)
