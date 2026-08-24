"""Ask the model what to read next, given what the corpus already covers.

The other two strategies look backwards -- references are what these papers
cited, tags are what they were about. This one looks sideways: it shows the model
the corpus's own summaries and asks for directions adjacent to them, including
ones none of the papers took.

**It asks for concepts, not queries.** An earlier version asked for arXiv field
syntax directly and got back things like `(abs:` and `(all:"robot control" OR
all:` -- truncated, unbalanced, or parenthesised in ways arXiv answers with a
400. Sanitising that was a losing game: each fix revealed another malformation,
and a query that survives sanitising but means the wrong thing fails silently by
returning nothing. Asking for two or three plain search terms plays to what the
model is good at, and this module composes the syntax, which removes the entire
class of bug rather than filtering it.
"""

from __future__ import annotations

import json
import logging
import re

import httpx

from articliser.config import settings
from articliser.worker.gpu import ollama_base_url

log = logging.getLogger(__name__)

MIN_TERMS = 2
MAX_TERMS = 3

_SCHEMA = {
    "type": "object",
    "properties": {
        "directions": {
            "type": "array",
            "minItems": 3,
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "terms": {
                        "type": "array",
                        "minItems": MIN_TERMS,
                        "maxItems": MAX_TERMS,
                        "items": {"type": "string"},
                    },
                    "reason": {"type": "string"},
                },
                "required": ["terms", "reason"],
            },
        }
    },
    "required": ["directions"],
}

_PROMPT = """These are summaries of papers someone has been reading:

{summaries}

Propose {n} research directions they would want to read next. Favour adjacent
work and open questions over more of the same.

For each direction give:
- "terms": {min_terms}-{max_terms} short search terms that together pin down the topic.
  Use the plain words a paper's abstract would use. No boolean operators, no
  field prefixes, no punctuation. Good: ["tactile feedback", "teleoperation"].
  Bad: ["all:tactile", "robotics OR control"].
- "reason": one sentence naming the gap it fills.
"""

# The model is asked for plain words, but still occasionally reaches for syntax.
_SYNTAX_RE = re.compile(r"\b(all|ti|abs|cat|au|co|jr|rn):|\b(AND|OR|ANDNOT)\b|[()\"\\]")


def _clean_term(term: str) -> str | None:
    """Reduce a proposed term to plain words, or reject it."""
    cleaned = _SYNTAX_RE.sub(" ", str(term or ""))
    cleaned = re.sub(r"[^A-Za-z0-9 \-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # One-character fragments and bare numbers identify nothing.
    if len(cleaned) < 3 or not re.search(r"[A-Za-z]{3,}", cleaned):
        return None
    return cleaned


def build_query(terms: list[str]) -> str | None:
    """Compose an arXiv query from plain terms. This module owns the syntax."""
    cleaned = [t for t in (_clean_term(t) for t in terms) if t]
    # Deduplicate while preserving order: "robot" and "robot" AND-ed together
    # narrows nothing and looks like a bug in the logs.
    seen: dict[str, None] = {}
    for term in cleaned:
        seen.setdefault(term.lower(), None)
    unique = list(seen)
    if len(unique) < MIN_TERMS:
        return None
    return " AND ".join(
        f'all:"{t}"' if " " in t else f"all:{t}" for t in unique[:MAX_TERMS]
    )


def propose_queries(summaries: list[str], n: int = 5) -> list[tuple[str, str]]:
    """Return (query, reason) pairs, or an empty list if the model is unreachable."""
    if not summaries:
        return []

    joined = "\n".join(f"- {s.strip()[:300]}" for s in summaries[:12] if s.strip())
    if not joined:
        return []

    prompt = _PROMPT.format(
        summaries=joined, n=n, min_terms=MIN_TERMS, max_terms=MAX_TERMS
    )
    try:
        response = httpx.post(
            f"{ollama_base_url()}/api/chat",
            timeout=settings.ollama_timeout_s,
            json={
                "model": settings.ollama_model,
                "messages": [{"role": "user", "content": prompt}],
                "format": _SCHEMA,
                "think": False,
                "options": {
                    "temperature": 0.6,
                    "num_ctx": settings.ollama_num_ctx,
                    # Without an explicit budget the earlier version's output
                    # came back truncated mid-token.
                    "num_predict": 800,
                },
                "stream": False,
                "keep_alive": settings.ollama_keep_alive,
            },
        )
        response.raise_for_status()
        payload = json.loads(response.json()["message"]["content"])
    except Exception as exc:  # noqa: BLE001 - discovery degrades, it doesn't fail
        log.warning("prompted query generation failed: %s", exc)
        return []

    out: list[tuple[str, str]] = []
    for item in payload.get("directions", []):
        query = build_query(item.get("terms") or [])
        if query is None:
            log.debug("dropped a direction with unusable terms: %r", item.get("terms"))
            continue
        out.append((query, str(item.get("reason", "")).strip()[:160]))
    return out[:n]
