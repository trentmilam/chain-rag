"""Single source of truth for the embedder + Qdrant connection.

Both the ingest pipeline (writes vectors) and the app/eval (Consilium's Router needs
a live embedder for query-time embed_one() calls) MUST use the exact same model and
device -- a mismatch produces silently meaningless cosine scores with no error.
Import SETTINGS/get_embedder/get_qdrant_client from here, nowhere else.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from ragpack.pipeline import Settings

CHAIN_RAG_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CHAIN_RAG_ROOT / "data"


def _register_cuda_dll_dir() -> None:
    """onnxruntime-gpu's CUDAExecutionProvider depends on cublasLt64_13.dll and
    cudnn64_9.dll, neither on PATH by default on Windows and neither pip-
    installable in reasonable time on a constrained connection (the matching
    nvidia-cudnn-cu12 wheel alone is >700MB). torch's own CUDA wheel already
    ships a matched, working copy of both (installing torch is required anyway
    for EasyOCR) -- point PATH at torch's lib dir instead of a second download.
    No-op if torch isn't installed or on non-Windows."""
    if sys.platform != "win32":
        return
    try:
        import torch
    except ImportError:
        return
    torch_lib = str(Path(torch.__file__).resolve().parent / "lib")
    if os.path.isdir(torch_lib) and torch_lib not in os.environ.get("PATH", ""):
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")


_register_cuda_dll_dir()


def _normalize_qdrant(qdrant: str) -> str:
    """A relative CHAINRAG_QDRANT override is otherwise resolved against whatever
    the *current process's* cwd happens to be -- two processes launched from
    different directories would silently open two different, unrelated on-disk
    stores. Absolute paths and the special :memory:/http(s):// forms pass through
    unchanged."""
    if not qdrant or qdrant == ":memory:" or qdrant.startswith(("http://", "https://")):
        return qdrant
    return str(Path(qdrant).resolve())


SETTINGS = Settings(
    model=os.environ.get("CHAINRAG_EMBED_MODEL", "BAAI/bge-base-en-v1.5"),
    device=os.environ.get("CHAINRAG_DEVICE", "auto"),
    qdrant=_normalize_qdrant(os.environ.get("CHAINRAG_QDRANT", str(DATA_DIR / "qdrant"))),
    collection=os.environ.get("CHAINRAG_COLLECTION", "chain_rag"),
    ocr="never",  # OCR happens upstream in ingest/ocr.py, never inside RAGpack's own path
)


def get_embedder():
    """The one real embedder instance -- construct once, reuse across ingest+query."""
    from ragpack.embed import Embedder, HashEmbedder

    if (SETTINGS.model or "").strip().lower() == "hash":
        return HashEmbedder()
    return Embedder(SETTINGS.model, SETTINGS.device)


def get_qdrant_client():
    from ragpack.store import make_client

    return make_client(SETTINGS.qdrant)


def _marker_path() -> Path | None:
    """Where the "which model owns this collection" marker lives, or None for
    stores with no durable location to persist it (:memory:, a remote server)."""
    q = SETTINGS.qdrant
    if not q or q == ":memory:" or q.startswith(("http://", "https://")):
        return None
    return Path(q) / f"{SETTINGS.collection}.model.json"


def record_embedder_marker(*, recreate: bool = False) -> None:
    """Call once from the ingest path, right after ensure_collection(). Persists
    which model owns this collection so a later query-time process can catch a
    drift instead of silently computing meaningless cosine scores. On
    recreate=True the collection is genuinely fresh, so the marker is always
    (re)written; otherwise it's only written if this is the very first ingest."""
    marker = _marker_path()
    if marker is None:
        return
    if recreate or not marker.exists():
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"model": (SETTINGS.model or "").strip()}), encoding="utf-8")
    else:
        verify_embedder_marker()


def verify_embedder_marker() -> None:
    """Call before querying (e.g. from build_registry()). Raises if this
    process's configured model doesn't match the model that created the
    collection on disk -- the embedder-consistency contract this module's
    docstring promises, actually enforced."""
    marker = _marker_path()
    if marker is None or not marker.exists():
        return  # in-memory/remote store, or no ingest has run yet -- nothing to check
    recorded = json.loads(marker.read_text(encoding="utf-8")).get("model")
    model_name = (SETTINGS.model or "").strip()
    if recorded and recorded != model_name:
        raise ValueError(
            f"embedder mismatch: collection {SETTINGS.collection!r} at {SETTINGS.qdrant!r} "
            f"was ingested with model {recorded!r}, but this process is configured for "
            f"{model_name!r} -- query-time and ingest-time embedders must match exactly, "
            "or cosine scores are silently meaningless."
        )
