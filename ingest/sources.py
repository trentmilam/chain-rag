"""Source-document manifest: schema + load/validate for sources.yaml.

The manifest is the single record of what's actually in the corpus and where it came
from -- every document chain-rag ingests must have an entry here. Populated via a
research pass across each chain's primary documentation; a separate smoke manifest
(smoke/sources.smoke.yaml) uses the same schema for a fast 3-document wiring check.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

_VALID_FORMATS = {"pdf", "html", "md", "txt", "mediawiki"}
_VALID_DOC_TYPES = {"whitepaper", "bip", "eip", "rfc", "spec", "docs"}

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_local_path(local_path: str) -> Path:
    """A manifest's local_path (e.g. "data/raw/bitcoin/x.pdf") is always relative
    to the chain-rag project root, never to the invoking process's cwd -- so
    ingest/eval/app all resolve a document the same way regardless of where
    they're launched from."""
    p = Path(local_path)
    return p if p.is_absolute() else _PROJECT_ROOT / p


@dataclass
class SourceDoc:
    id: str
    chain: str
    title: str
    source_url: str
    local_path: str
    format: str
    doc_type: str
    license_note: str = ""
    ocr_required: Optional[bool] = None
    sha256: Optional[str] = None


@dataclass
class Manifest:
    version: int
    documents: list[SourceDoc] = field(default_factory=list)


def load_manifest(path: str | Path) -> Manifest:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    docs = [SourceDoc(**d) for d in raw.get("documents", [])]
    return Manifest(version=int(raw.get("version", 1)), documents=docs)


def validate_manifest(manifest: Manifest) -> list[str]:
    """Return a list of human-readable error strings; empty means valid."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    for d in manifest.documents:
        if not d.id:
            errors.append(f"document missing id: {d!r}")
            continue
        if d.id in seen_ids:
            errors.append(f"duplicate document id: {d.id!r}")
        seen_ids.add(d.id)
        if not d.chain:
            errors.append(f"{d.id}: missing chain")
        if d.format not in _VALID_FORMATS:
            errors.append(f"{d.id}: format {d.format!r} not one of {sorted(_VALID_FORMATS)}")
        if d.doc_type not in _VALID_DOC_TYPES:
            errors.append(f"{d.id}: doc_type {d.doc_type!r} not one of {sorted(_VALID_DOC_TYPES)}")
        if not d.source_url:
            errors.append(f"{d.id}: missing source_url (provenance is required)")
        if not d.local_path or not resolve_local_path(d.local_path).exists():
            errors.append(f"{d.id}: local_path {d.local_path!r} does not exist")
    return errors
