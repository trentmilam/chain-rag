"""Header/section-aware pre-splitter.

RAGpack's own ``chunk_text`` is paragraph-boundary-only with no notion of document
structure -- a short section can merge into its neighbor, and no chunk records which
section it came from. For citation-gated retrieval over technical specs, a citation
should never straddle a real header boundary. This module splits on markdown ATX
headers first (``split_into_sections``), then leaf-chunks each section's body with
RAGpack's own, already-proven ``chunk_text`` (reused, not reimplemented) so a chunk
never crosses a section it doesn't belong to.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ragpack.chunk import chunk_text, clean_text

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Section:
    title: str
    path: list[str] = field(default_factory=list)
    body: str = ""


def split_into_sections(text: str) -> list[Section]:
    """Split markdown-ish text on ATX headers (#, ##, ...). Text with no headers at
    all (e.g. raw PDF-extracted prose) becomes a single section -- RAGpack's own
    paragraph-aware chunk_text still applies within it, just without section anchors."""
    text = clean_text(text)
    lines = text.split("\n")

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []   # (header level, title) ancestor chain
    current_title = "(document)"
    current_body: list[str] = []

    def flush():
        body = "\n".join(current_body).strip()
        if body:
            path = [t for _, t in stack] or [current_title]
            sections.append(Section(title=current_title, path=path, body=body))

    for line in lines:
        m = _HEADER_RE.match(line)
        if not m:
            current_body.append(line)
            continue
        flush()
        level, title = len(m.group(1)), m.group(2).strip()
        # Truncate by header LEVEL, not stack position: a document that never
        # uses level-1 headers (common in HTML->markdown conversions) must not
        # have its first header become a permanent ancestor of every sibling.
        stack = [(lvl, t) for lvl, t in stack if lvl < level] + [(level, title)]
        current_title = title
        # Keep the header line itself as the start of its own section's body --
        # section_path already carries the title as structured metadata, but no
        # citation/UI in this project actually surfaces section_path, so a header
        # that only lived in metadata was invisible to a reader AND showed up as
        # "dropped content" to any tool checking the chunk text against the source.
        current_body = [line]
    flush()

    if not sections:
        body = text.strip()
        return [Section(title="(document)", path=["(document)"], body=body)] if body else []
    return sections


@dataclass
class LeafChunk:
    section_path: list[str]
    text: str
    index_in_section: int


def leaf_chunks(sections: list[Section], max_chars: int = 1200, overlap: int = 150) -> list[LeafChunk]:
    """Sub-split each section's body into citation-sized pieces via RAGpack's proven
    paragraph-aware chunker, tagging each piece with the header path it came from."""
    out: list[LeafChunk] = []
    for sec in sections:
        for i, piece in enumerate(chunk_text(sec.body, max_chars, overlap)):
            out.append(LeafChunk(section_path=sec.path, text=piece, index_in_section=i))
    return out
