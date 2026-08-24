# VENDORED from semantic-highlighting-slm @ 5d4235c (2026-07-28).
#
# Originally copied because airllm pinned transformers<5.13 while that project
# required >=5.14.1, so a path dependency was unresolvable. airllm has since been
# removed, so that constraint is gone and a path dependency would now resolve --
# it is still vendored only because that would pull gradio, datasets and
# scikit-learn in for seven files. Only the import paths were flattened to
# articliser.highlighting.*; the logic is unchanged, so upstream fixes can be
# re-copied. Weights still load from the Hub, not from here.
"""Post-processing over raw predicted spans: merge fragments, clean up
boundaries, and cap total highlight coverage.

Five independent passes, each fixing a distinct problem found by actually
using the deployed model (see docs/findings.md for the concrete examples
that prompted each one):

- `merge_adjacent_spans`: stitches same-label fragments the model failed to
  continue with an I- tag back into one span.
- `snap_to_word_boundaries`: fixes label boundaries landing mid-word (a BPE
  subword-token artifact), so a rendered highlight never visibly cuts a
  word in half.
- `fix_dangling_edges`: expands (or drops) spans that start/end on a bare
  comma or function word -- e.g. ", 18, and 21 minutes of" is missing the
  "12" before it and whatever noun follows "of", and reads as broken even
  though every individual word is correct.
- `drop_short_spans`: removes single-word/fragment spans that aren't useful
  highlights regardless of how the model scored them.
- `apply_highlight_budget`: caps total highlighted coverage to a fraction
  of the document, ranked by confidence -- without this, the model
  highlights everything it has any opinion about, defeating the point.

All of these are deliberately separate from `labels.bio_ids_to_spans`
(which is the *correct*, literal decoding of what the model predicted) so
raw and post-processed output can still be compared directly -- these are
heuristic cleanup passes, not part of the model's own decoding.
"""

from __future__ import annotations

import re

from articliser.highlighting.spans import Span

# Sentence-ending punctuation in the gap between two spans means they're
# almost certainly not one continuous thought -- don't bridge over it even
# if the character gap is small.
_SENTENCE_END = {".", "!", "?"}


def merge_adjacent_spans(spans: list[Span], text: str, max_gap: int = 10) -> list[Span]:
    """Merge same-label spans separated by a short gap (e.g. "it", "for", "the").

    max_gap is a character count, not a token count -- 10 characters covers
    most short connector words/phrases without being so large it bridges
    genuinely separate spans.
    """
    if not spans:
        return []

    ordered = sorted(spans, key=lambda s: s.start)
    merged: list[Span] = [ordered[0]]

    for span in ordered[1:]:
        prev = merged[-1]
        gap = text[prev.end : span.start]
        bridges_sentence_end = any(ch in _SENTENCE_END for ch in gap)

        if span.label == prev.label and len(gap) <= max_gap and not bridges_sentence_end:
            merged_score = (prev.score + span.score) / 2
            merged[-1] = Span(prev.start, span.end, prev.label, text[prev.start : span.end], merged_score)
        else:
            merged.append(span)

    return merged


def snap_to_word_boundaries(spans: list[Span], text: str) -> list[Span]:
    """Expand each span's start/end outward to the nearest whitespace, so a
    span never ends mid-word.

    Needed because BPE token boundaries don't always line up with word
    boundaries -- e.g. "copilot" can split into "cop"/"ilot" as separate
    subword tokens, and if a label boundary lands on that internal split,
    the rendered highlight visibly breaks the word in half.
    """
    snapped: list[Span] = []
    for s in spans:
        start, end = s.start, s.end
        while start > 0 and text[start - 1].isalnum() and text[start].isalnum():
            start -= 1
        while end < len(text) and text[end].isalnum() and text[end - 1].isalnum():
            end += 1
        snapped.append(Span(start, end, s.label, text[start:end], s.score))
    return snapped


_DANGLING_WORDS = {
    "and", "or", "but", "nor", "as",
    "of", "in", "on", "at", "by", "with", "from", "to", "into", "onto", "via",
    "the", "a", "an",
    "which", "that", "using",
}
_WORD_RE = re.compile(r"\w+")
_LEADING_PUNCT_RE = re.compile(r"^\s*[,;:)\]]")


def _first_word(span_text: str) -> str:
    m = _WORD_RE.search(span_text)
    return m.group(0).lower() if m else ""


def _last_word(span_text: str) -> str:
    last = None
    for last in _WORD_RE.finditer(span_text):
        pass
    return last.group(0).lower() if last else ""


def _is_dangling_start(span_text: str) -> bool:
    return bool(_LEADING_PUNCT_RE.match(span_text)) or _first_word(span_text) in _DANGLING_WORDS


def _is_dangling_end(span_text: str) -> bool:
    return _last_word(span_text) in _DANGLING_WORDS


def _word_before(text: str, pos: int) -> tuple[int, int] | None:
    match = None
    for match in _WORD_RE.finditer(text[:pos]):
        pass
    return (match.start(), match.end()) if match else None


def _word_after(text: str, pos: int) -> tuple[int, int] | None:
    match = _WORD_RE.search(text, pos)
    return (match.start(), match.end()) if match else None


def fix_dangling_edges(spans: list[Span], text: str, max_expand: int = 4) -> list[Span]:
    """Expand spans whose edges land on a dangling comma/conjunction/
    preposition/article into the adjacent word, so a highlight doesn't
    render as an ungrammatical fragment.

    Concrete case this fixes: a span reading ", 18, and 21 minutes of" is
    missing the "12" before the leading comma (part of the same "12, 18,
    and 21" list) and whatever noun follows the trailing "of" -- every
    word in it is individually correct, but it doesn't read as a complete
    thought on its own.

    Bounded to max_expand words per edge so a stubborn case can't run away
    consuming a whole sentence. If a span is *still* dangling on either
    edge after that many attempts, it's dropped rather than shown broken --
    downstream, `apply_highlight_budget` ranks by confidence and fills the
    coverage cap from what's left, so dropping a broken span here is
    exactly "replace it with the next-best span" with no special-case
    logic needed for that part.
    """
    fixed: list[Span] = []
    for s in spans:
        start, end = s.start, s.end

        for _ in range(max_expand):
            if not _is_dangling_start(text[start:end]):
                break
            prev = _word_before(text, start)
            if prev is None:
                break
            start = prev[0]

        for _ in range(max_expand):
            if not _is_dangling_end(text[start:end]):
                break
            nxt = _word_after(text, end)
            if nxt is None:
                break
            end = nxt[1]

        if _is_dangling_start(text[start:end]) or _is_dangling_end(text[start:end]):
            continue  # unfixable within the budget -- drop rather than show broken

        fixed.append(Span(start, end, s.label, text[start:end], s.score))
    return fixed


def drop_short_spans(spans: list[Span], min_words: int = 2) -> list[Span]:
    """Drop spans with fewer than min_words whitespace-separated words.

    A single word like "We" isn't a useful highlight and violates the
    annotation guideline's own "prefer clause- or sentence-length spans"
    rule -- these are almost always fragments the model shouldn't have
    split off on their own.
    """
    return [s for s in spans if len(s.text.split()) >= min_words]


def apply_highlight_budget(
    spans: list[Span], text_len: int, budget_fraction: float = 0.2
) -> list[Span]:
    """Keep only the highest-confidence spans until their combined coverage
    reaches budget_fraction of the document.

    This is RESEARCH.MD's own design principle -- "if someone could only
    read 20% of this page, which phrases would give them the best
    understanding" -- made real: without it, the model highlights
    everything it has any opinion about, which defeats the point of
    highlighting at all. Requires spans to carry a real `.score` (see
    predict.predict_spans_crf); spans with no score all rank equally and
    the budget cut becomes arbitrary.
    """
    if not spans:
        return []
    budget_chars = text_len * budget_fraction
    ranked = sorted(spans, key=lambda s: s.score, reverse=True)
    kept: list[Span] = []
    covered = 0
    for s in ranked:
        span_len = s.end - s.start
        if kept and covered + span_len > budget_chars:
            continue
        kept.append(s)
        covered += span_len
    return sorted(kept, key=lambda s: s.start)
