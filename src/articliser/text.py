"""Shared text utilities: slugs, reading time, and markdown-with-highlights rendering.

The interesting one is `render_body_html`. Spans are character offsets over the
raw markdown, and the naive orders both fail: rendering markdown first loses the
offsets, and wrapping spans in HTML after rendering means walking a DOM. So the
`<mark>` tags are injected into the markdown *before* rendering, and markdown-it
is configured to pass inline HTML through. Because the CRF's spans are snapped to
word boundaries inside single paragraphs, an injected tag never straddles a block
boundary, which is the case that would actually break the renderer.
"""

from __future__ import annotations

import html
import re
import unicodedata

from markdown_it import MarkdownIt

from articliser.highlighting.schema import LABEL_BY_NAME

_WORDS_PER_MINUTE = 220

_md = MarkdownIt("commonmark", {"html": True, "linkify": True, "typographer": True})


def slugify(value: str, max_length: int = 80) -> str:
    """Lowercase hyphenated slug, truncated at a word boundary.

    The boundary matters because the slug is the article's URL: a hard character
    cut produced `...without-specialized-hard`, which reads as a typo rather than
    an abbreviation. Falls back to the hard cut only when the first word alone
    already exceeds the limit.
    """
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    value = re.sub(r"[-\s]+", "-", value).strip("-")
    if len(value) <= max_length:
        return value or "untitled"

    cut = value[: max_length + 1]
    if "-" in cut[1:]:
        cut = cut[: cut.rindex("-")]
    return cut[:max_length].strip("-") or "untitled"


def unique_slug(base: str, taken: set[str]) -> str:
    """Append -2, -3, ... until the slug is free. Slugs are the article URL, so
    a collision would otherwise make one article unreachable."""
    slug = slugify(base)
    if slug not in taken:
        return slug
    for n in range(2, 1000):
        candidate = f"{slug}-{n}"
        if candidate not in taken:
            return candidate
    raise ValueError(f"could not find a free slug for {base!r}")


def reading_minutes(text: str) -> int:
    return max(1, round(len(text.split()) / _WORDS_PER_MINUTE))


def _dedupe_spans(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Drop overlaps, keeping the earliest-starting span. Post-CRF overlap
    shouldn't happen, but injecting nested <mark> tags from bad offsets would
    produce broken HTML rather than a visible error, so it's guarded here."""
    ordered = sorted(spans, key=lambda s: (s[0], -s[1]))
    kept: list[tuple[int, int, str]] = []
    cursor = 0
    for start, end, label in ordered:
        if start < cursor or end <= start:
            continue
        kept.append((start, end, label))
        cursor = end
    return kept


def render_body_html(body_md: str, spans: list[tuple[int, int, str]] | None = None) -> str:
    """Render article markdown to HTML, wrapping any tagged spans in <mark>."""
    if not spans:
        return _md.render(body_md)

    parts: list[str] = []
    cursor = 0
    for start, end, label in _dedupe_spans(spans):
        if start >= len(body_md):
            break
        end = min(end, len(body_md))
        parts.append(body_md[cursor:start])
        text = body_md[start:end]
        known = label if label in LABEL_BY_NAME else "Unknown"
        parts.append(
            f'<mark class="hl" data-label="{html.escape(known)}" '
            f'title="{html.escape(known)}">{text}</mark>'
        )
        cursor = end
    parts.append(body_md[cursor:])
    return _md.render("".join(parts))


def strip_markdown(body_md: str) -> str:
    """Plain text for embeddings and excerpts -- not a full parser, just enough to
    stop markdown punctuation dominating a MiniLM vector."""
    text = re.sub(r"```.*?```", " ", body_md, flags=re.S)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_`>#]+", "", text)
    return re.sub(r"\s+", " ", text).strip()
