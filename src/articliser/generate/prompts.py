"""Prompt construction and response parsing for article generation.

The parsing half is deliberately generous even though the Ollama backend now
constrains decoding against this module's own schema, so responses arrive as
valid JSON by construction. Two reasons to keep it. Constrained decoding is a
property of the backend rather than of the format, so a backend without it would
silently start returning prose. And a rigid parser that rejects a fenced code
block or a trailing comma throws away a whole generation over punctuation, which
is a bad trade at any speed.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)

_H1_RE = re.compile(r"^\s{0,3}#\s")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")

CATEGORIES: tuple[str, ...] = (
    "Machine Learning",
    "Systems",
    "Robotics",
    "Human-Computer Interaction",
    "Theory",
    "Applications",
    "Safety",
)


# Abbreviations models reach for constantly. Substring matching alone can't help
# here -- "ml" is not a substring of "machine learning" -- and dropping these to
# "Applications" would put most articles in one bucket.
_CATEGORY_ALIASES: dict[str, str] = {
    "ml": "Machine Learning",
    "ai": "Machine Learning",
    "nlp": "Machine Learning",
    "deep learning": "Machine Learning",
    "rl": "Machine Learning",
    "reinforcement learning": "Machine Learning",
    "cv": "Machine Learning",
    "computer vision": "Machine Learning",
    "hci": "Human-Computer Interaction",
    "systems": "Systems",
    "os": "Systems",
    "distributed systems": "Systems",
    "security": "Safety",
    "alignment": "Safety",
    "ethics": "Safety",
    "math": "Theory",
    "mathematics": "Theory",
    "algorithms": "Theory",
}


class ArticleDraft(BaseModel):
    """The single structured object a generation call must produce."""

    title: str
    standfirst: str = ""
    category: str = "Applications"
    tags: list[str] = Field(default_factory=list)
    body_md: str
    image_prompt: str = ""

    @field_validator("category")
    @classmethod
    def _known_category(cls, value: str) -> str:
        """Snap to the closest known category rather than rejecting the draft.

        Models reliably invent near-misses ("ML", "machine learning systems").
        Losing an entire generation over a label that only drives a filter chip
        would be a poor trade.
        """
        cleaned = (value or "").strip().lower()
        if not cleaned:
            return "Applications"
        for known in CATEGORIES:
            if cleaned == known.lower():
                return known
        if cleaned in _CATEGORY_ALIASES:
            return _CATEGORY_ALIASES[cleaned]
        for known in CATEGORIES:
            if cleaned in known.lower() or known.lower() in cleaned:
                return known
        # Logged rather than silent: "Applications" is both a real category and
        # the fallback, so without this there is no way to tell a deliberate
        # classification from a rejected one when reviewing a skewed feed.
        log.info("category %r not recognised; filed under Applications", value)
        return "Applications"

    @field_validator("body_md", mode="before")
    @classmethod
    def _normalise_headings(cls, value) -> str:
        """Demote any level-1 headings in the body to level 2.

        The template already renders the article title as the page's only <h1>,
        so an <h1> inside the body is both a duplicate landmark for screen
        readers and a visual break in the type scale. Models emit one anyway --
        the prompt asks for "## subheadings, no title heading" and it is ignored
        often enough to be worth fixing rather than re-prompting.

        A first heading that merely restates the title is dropped outright.
        """
        body = str(value or "").strip()
        if not body:
            return body

        lines = body.splitlines()
        out: list[str] = []
        in_fence = False
        for line in lines:
            if _FENCE_RE.match(line):
                in_fence = not in_fence
                out.append(line)
                continue
            if not in_fence and _H1_RE.match(line):
                line = "#" + line.lstrip()  # "# Foo" -> "## Foo"
            out.append(line)
        return "\n".join(out).strip()

    @field_validator("tags", mode="before")
    @classmethod
    def _coerce_tags(cls, value) -> list[str]:
        if isinstance(value, str):
            value = [part for part in re.split(r"[,;]", value)]
        if not isinstance(value, list):
            return []
        return [str(tag).strip().lstrip("#") for tag in value if str(tag).strip()][:6]


SYSTEM_INSTRUCTION = """You are a science writer. You turn research papers into \
articles that a curious non-specialist can finish in a few minutes and come away \
genuinely understanding.

Rules:
- Explain what the researchers did, why it matters, and what the limits are.
- Lead with the finding, not with background.
- Define jargon in-line the first time it appears, or avoid it.
- Never invent results, numbers, or citations. If the source does not say it, \
leave it out.
- Do not describe the paper's structure ("this paper is organised as...").
- Write in plain British English. No hype, no "revolutionary", no "game-changing"."""

_RESPONSE_SHAPE = """Respond with a single JSON object and nothing else:

{
  "title": "a specific, concrete headline; not the paper's own title",
  "standfirst": "one sentence, under 30 words, saying what the reader will learn",
  "category": "one of: %s",
  "tags": ["three", "to", "five", "lowercase", "topics"],
  "body_md": "600-900 words of Markdown. Use ## subheadings. No title heading.",
  "image_prompt": "one sentence describing an abstract editorial illustration; no text, no people"
}""" % ", ".join(CATEGORIES)


def build_prompt(title: str, evidence: str) -> str:
    """Assemble the single generation call.

    One call, not a map-reduce over sections: with layer-streamed inference the
    dominant cost is the number of forward sweeps, so five cheap calls cost far
    more than one expensive one.
    """
    source_title = title.strip() or "Untitled"
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"{_RESPONSE_SHAPE}\n\n"
        f"---\n"
        f"Source paper: {source_title}\n\n"
        f"{evidence.strip()}\n"
        f"---\n\n"
        f"JSON:"
    )


BOOK_INSTRUCTION = """You are a science writer. You turn sections of a technical \
book into standalone articles that a curious non-specialist can finish in a few \
minutes and come away genuinely understanding.

Rules:
- Explain the idea itself: what it is, why it exists, what it lets you do.
- Lead with the intuition, then the mechanism. Never open with a definition.
- Define jargon in-line the first time it appears, or avoid it.
- Where the source gives a formula, explain what it means rather than restating it. \
Do not reproduce equations, figure numbers, or exercise references.
- Never invent results or numbers. If the source does not say it, leave it out.
- This is one part of a series, so do not write an introduction to the whole book \
and do not close by summarising what the reader has learned.
- Write in plain British English. No hype."""


def build_book_prompt(
    book: str,
    chapter: str,
    section: str,
    evidence: str,
    part: int,
    total: int,
) -> str:
    """Assemble the generation call for one section of a book.

    The paper prompt's framing -- "what the researchers did, why it matters, what
    the limits are" -- is wrong here. A textbook section is not reporting a
    result, it is teaching an idea, and prompting for the former produces
    articles that invent findings the source never claimed.
    """
    where = f"part {part} of {total}"
    if chapter and chapter != section:
        where += f', from the chapter "{chapter}"'
    return (
        f"{BOOK_INSTRUCTION}\n\n"
        f"{_RESPONSE_SHAPE}\n\n"
        f"---\n"
        f'Book: {book}\n'
        f'Section: {section} ({where})\n\n'
        f"{evidence.strip()}\n"
        f"---\n\n"
        f"JSON:"
    )


def build_repair_prompt(broken: str) -> str:
    """Second and final attempt, when the first response wouldn't parse."""
    return (
        "The following was meant to be a single JSON object but could not be "
        "parsed. Return the same content as valid JSON, with all six keys "
        "(title, standfirst, category, tags, body_md, image_prompt) and nothing "
        "else. Escape newlines inside strings.\n\n"
        f"{broken[:4000]}\n\nJSON:"
    )


def _extract_json_object(text: str) -> str | None:
    """Find the outermost {...} in a response, ignoring braces inside strings.

    A plain regex for the first "{" to the last "}" breaks on any body_md that
    contains a brace, which happens often enough (code snippets, set notation)
    to matter. This walks the text tracking string state instead.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _repair_common_json_faults(raw: str) -> str:
    """Fix the two faults small models produce most: trailing commas, and raw
    newlines inside string literals."""
    without_trailing = re.sub(r",(\s*[}\]])", r"\1", raw)

    out: list[str] = []
    in_string = False
    escaped = False
    for char in without_trailing:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            elif char == "\n":
                out.append("\\n")
                continue
            elif char == "\t":
                out.append("\\t")
                continue
        elif char == '"':
            in_string = True
        out.append(char)
    return "".join(out)


def parse_draft(response: str) -> ArticleDraft | None:
    """Parse a generation response into an ArticleDraft, or None if unusable."""
    candidate = _extract_json_object(response)
    if candidate is None:
        return None

    for attempt in (candidate, _repair_common_json_faults(candidate)):
        try:
            payload = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        # A draft with no body is a failed generation, not a short article --
        # returning None here lets the caller spend its one retry usefully.
        if not str(payload.get("body_md", "")).strip():
            return None
        try:
            return ArticleDraft.model_validate(payload)
        except Exception:  # noqa: BLE001 - pydantic ValidationError and friends
            return None
    return None
