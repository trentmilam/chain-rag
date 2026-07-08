"""chain-rag -- multichain blockchain technical-knowledge RAG. Interactive chat UI.

    projects/chain-rag/.venv/Scripts/python.exe app.py

Ask real technical questions about Bitcoin, Ethereum, Solana, Monero, Polygon, and
Cardano protocol internals -- routed and cited against a curated corpus of primary
sources (whitepapers, BIPs, EIPs, protocol docs) via Consilium's router/composer.
Every answer is extractive and citation-gated: the retrieved source chunks ARE the
answer, or the system abstains rather than guess. No LLM anywhere in the answer path
-- retrieval + a deterministic router + a citation-integrity gate, nothing else.

Standalone: no network calls at query time, no dependency on anything outside this
repo + its sibling capability repos (consilium, rag-reliability).
"""
from __future__ import annotations

import html
import os
import sys

CHAIN_RAG_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CHAIN_RAG_ROOT)

from chainrag._paths import add_sibling_paths  # noqa: E402

add_sibling_paths()

import gradio as gr  # noqa: E402

from chainrag.bootstrap import ROUTER_KWARGS, build_registry  # noqa: E402
from consilium.composer import compose  # noqa: E402
from consilium.router import Router  # noqa: E402
from ingest.embed_config import get_embedder  # noqa: E402

EMBEDDER = get_embedder()
REGISTRY = build_registry(EMBEDDER)
ROUTER = Router(REGISTRY, EMBEDDER, **ROUTER_KWARGS)

EXAMPLES = [
    "How does Bitcoin's difficulty adjustment keep block time near ten minutes?",
    "What does EIP-1559 change about how transaction fees are priced?",
    "How does proof of history let Solana validators order transactions?",
    "What does RingCT hide about a Monero transaction?",
    "How does Polygon checkpoint state back to Ethereum mainnet?",
    "How does Ouroboros select the next block-producing stake pool in Cardano?",
    "Compare how Bitcoin and Ethereum each solve double-spending.",
    "What's the weather like in Tokyo tomorrow?",
]

# ---------------------------------------------------------------------------
# Theme -- a quiet SaaS chrome (one neutral surface, one
# accent, plain status pills), a distinct indigo accent for this tool's identity.
# ---------------------------------------------------------------------------
_BG = "#F7F8FA"
_SURFACE = "#FFFFFF"
_BORDER = "#E2E5EA"
_INK = "#14181F"
_INK_MUTED = "#5B6472"
_ACCENT = "#312E81"
_ACCENT_HOVER = "#3F3B9E"
_BLUE_TEXT, _BLUE_BG = "#1E40AF", "#DBEAFE"
_GRAY_TEXT, _GRAY_BG = "#475569", "#F1F5F9"
_SANS = "-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
_MONO = "'SFMono-Regular', Consolas, 'Liberation Mono', ui-monospace, monospace"

THEME = gr.themes.Base(spacing_size="md", radius_size="md", font=[_SANS], font_mono=[_MONO]).set(
    body_background_fill=_BG,
    body_background_fill_dark=_BG,
    body_text_color=_INK,
    body_text_color_dark=_INK,
    body_text_color_subdued=_INK_MUTED,
    background_fill_primary=_SURFACE,
    background_fill_primary_dark=_SURFACE,
    block_background_fill=_SURFACE,
    block_background_fill_dark=_SURFACE,
    block_border_color=_BORDER,
    block_border_color_dark=_BORDER,
    block_label_text_color=_INK_MUTED,
    block_label_text_color_dark=_INK_MUTED,
    block_title_text_color=_INK,
    block_title_text_color_dark=_INK,
    input_background_fill=_SURFACE,
    input_background_fill_dark=_SURFACE,
    border_color_primary=_BORDER,
    border_color_primary_dark=_BORDER,
    button_primary_background_fill=_ACCENT,
    button_primary_background_fill_dark=_ACCENT,
    button_primary_background_fill_hover=_ACCENT_HOVER,
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
)

CUSTOM_CSS = f"""
.gradio-container {{ font-family: {_SANS} !important; max-width: 900px !important; margin: 0 auto !important; }}
#hero h1 {{ font-size: 1.6rem; font-weight: 700; color: {_INK}; margin-bottom: 0.2rem; }}
#hero p {{ color: {_INK_MUTED}; font-size: 0.95rem; max-width: 68ch; }}
.pill {{
    display: inline-flex; align-items: center; gap: 0.35em; font-size: 0.76rem; font-weight: 600;
    padding: 0.18rem 0.6rem; border-radius: 999px; margin: 0 0.3rem 0.4rem 0;
}}
.pill-chain {{ color: {_BLUE_TEXT}; background: {_BLUE_BG}; font-family: {_MONO}; }}
.pill-abstain {{ color: {_GRAY_TEXT}; background: {_GRAY_BG}; }}
.ans-headline {{ font-size: 0.9rem; color: {_INK_MUTED}; margin: 0 0 0.5rem; }}
.citation {{ padding: 0.5rem 0; border-top: 1px solid {_BORDER}; font-size: 0.92rem; line-height: 1.5; color: {_INK}; }}
.citation:first-of-type {{ border-top: none; }}
.citation-src {{
    display: block; font-family: {_MONO}; font-size: 0.76rem; color: {_INK_MUTED}; margin-top: 0.25rem;
}}
details.conflict {{ margin-top: 0.5rem; font-size: 0.82rem; color: {_INK_MUTED}; }}
button:focus-visible, textarea:focus-visible, input:focus-visible {{ outline: 2px solid {_ACCENT} !important; outline-offset: 2px; }}
"""


def _prior_user_turn(history: list) -> str:
    """The immediately-prior user turn only (not the full history) -- a cheap fix
    for pronoun-heavy follow-ups ("why is that?") that's safe on topic-switch,
    since the integrity gate's query-relevance floor still drops stale-topic
    claims even if this turn's query drags in an unrelated prior topic."""
    for msg in reversed(history or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            return content if isinstance(content, str) else ""
    return ""


def _render_answer(answer) -> str:
    if answer.abstained:
        return (
            '<span class="pill pill-abstain">no answer</span>'
            f'<div class="ans-headline">{html.escape(answer.text)}</div>'
        )
    parts = ["".join(f'<span class="pill pill-chain">{html.escape(m)}</span>' for m in answer.modules_used)]
    for c in answer.citations:
        parts.append(
            '<div class="citation">'
            f"{html.escape(c.claim)}"
            f'<span class="citation-src">{html.escape(c.module)}/{html.escape(c.doc)} '
            f"&middot; {html.escape(c.chunk_id)} &middot; support {c.score}</span>"
            "</div>"
        )
    if answer.conflicts:
        parts.append('<details class="conflict"><summary>conflicting sources detected</summary>')
        for cf in answer.conflicts:
            losers = "; ".join(f"{l.module} (trust-resolved out)" for l in cf.losers)
            parts.append(f"<div>resolved to {html.escape(cf.winner.module)}; contested by: {html.escape(losers)}</div>")
        parts.append("</details>")
    return "".join(parts)


def respond(message: str, history: list) -> str:
    prior = _prior_user_turn(history)
    query = f"{prior} {message}".strip() if prior else message
    routed = ROUTER.route(query)
    answer = compose(query, routed, REGISTRY, EMBEDDER, harden=True)
    return _render_answer(answer)


demo = gr.ChatInterface(
    fn=respond,
    title="chain-rag",
    description=(
        "A cited, extractive RAG over primary blockchain protocol documentation -- "
        "Bitcoin, Ethereum, Solana, Monero, Polygon, Cardano. Every claim traces to a "
        "real source chunk, or the system says it doesn't know."
    ),
    examples=EXAMPLES,
    run_examples_on_click=True,
)

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=False, theme=THEME, css=CUSTOM_CSS)
