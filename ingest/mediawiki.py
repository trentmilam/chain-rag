"""Light MediaWiki -> markdown cleaner for Bitcoin's BIP source files.

BIPs are published as raw .mediawiki files (github.com/bitcoin/bips) -- not real
markdown, not HTML. Reading them as raw text would dump wiki markup ('''bold''',
[[links]], {{templates}}, == headers ==) straight into chunks as literal noise, the
same "silent garbage in the index" failure mode _clean_html's trafilatura fallback
guards against for HTML. This is NOT a full mediawiki parser (no mwparserfromhell
dependency) -- just enough to (a) convert == headers == to markdown '#' headers so
chunk.py's existing header regex finds them, and (b) strip the common inline markup
that would otherwise read as garbage in a citation.
"""
from __future__ import annotations

import re

_HEADER_RE = re.compile(r"^(=+)\s*(.*?)\s*=+\s*$")
_BOLD_ITALIC_RE = re.compile(r"'''''(.*?)'''''|'''(.*?)'''|''(.*?)''")
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_EXTLINK_RE = re.compile(r"\[(https?://\S+)\s+([^\]]+)\]")
_TEMPLATE_RE = re.compile(r"\{\{.*?\}\}", re.DOTALL)


def mediawiki_to_markdown(raw: str) -> str:
    """Best-effort cleanup, not a full parser -- BIP infobox templates ({{BIP
    Header ...}}) are boilerplate metadata (already captured via the manifest's
    own title/doc_type fields), so dropping them loses no prose content."""
    text = _TEMPLATE_RE.sub("", raw)

    def _header(m: re.Match) -> str:
        return "#" * len(m.group(1)) + " " + m.group(2)

    text = "\n".join(_HEADER_RE.sub(_header, line) for line in text.split("\n"))
    text = _WIKILINK_RE.sub(lambda m: m.group(2) or m.group(1), text)
    text = _EXTLINK_RE.sub(lambda m: m.group(2), text)
    text = _BOLD_ITALIC_RE.sub(lambda m: next(g for g in m.groups() if g is not None), text)
    return text
