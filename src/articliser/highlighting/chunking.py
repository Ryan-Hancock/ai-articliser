# VENDORED from semantic-highlighting-slm @ 5d4235c (2026-07-28).
#
# Originally copied because airllm pinned transformers<5.13 while that project
# required >=5.14.1, so a path dependency was unresolvable. airllm has since been
# removed, so that constraint is gone and a path dependency would now resolve --
# it is still vendored only because that would pull gradio, datasets and
# scikit-learn in for seven files. Only the import paths were flattened to
# articliser.highlighting.*; the logic is unchanged, so upstream fixes can be
# re-copied. Weights still load from the Hub, not from here.
"""Split a long document into model-safe chunks, run inference per chunk,
and stitch predictions back into full-document character offsets.

Needed because the classifier was trained (and its 512-token max_length
applies) on single abstracts, not full documents -- a real "highlight this
paper" tool has to handle text well beyond that. Chunking happens at
sentence boundaries (same spaCy sentencizer approach as the M1 rule
tagger) so a chunk boundary never lands mid-sentence, which would risk
splitting a label-worthy span across two chunks.

max_chars=1800 is a conservative proxy for the 512-token budget: measured
~5.4 chars/token on our own abstract text, so 1800 chars is ~330 tokens,
leaving headroom for special tokens and text denser than our measurement.
"""

from __future__ import annotations

from functools import lru_cache

import spacy

from articliser.highlighting.postprocess import (
    apply_highlight_budget,
    drop_short_spans,
    fix_dangling_edges,
    merge_adjacent_spans,
    snap_to_word_boundaries,
)
from articliser.highlighting.predict import predict_spans_crf
from articliser.highlighting.spans import Span

DEFAULT_MAX_CHARS = 1800


@lru_cache(maxsize=1)
def _sentencizer():
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def split_into_chunks(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[tuple[int, int]]:
    """Return (start, end) character ranges, each a run of whole sentences
    that fits under max_chars. A single sentence longer than max_chars
    still becomes its own (oversized) chunk rather than being cut mid-word
    -- the tokenizer's own truncation is the fallback for that rare case."""
    doc = _sentencizer()(text)
    sentences = [(s.start_char, s.end_char) for s in doc.sents]
    if not sentences:
        return []

    chunks: list[tuple[int, int]] = []
    chunk_start, chunk_end = sentences[0]
    for sent_start, sent_end in sentences[1:]:
        if sent_end - chunk_start <= max_chars:
            chunk_end = sent_end
        else:
            chunks.append((chunk_start, chunk_end))
            chunk_start, chunk_end = sent_start, sent_end
    chunks.append((chunk_start, chunk_end))
    return chunks


def predict_document(
    text: str,
    model,
    tokenizer,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_words: int = 2,
    budget_fraction: float | None = 0.2,
) -> list[Span]:
    """Chunk `text`, run the CRF classifier on each chunk, translate spans
    back into full-document offsets, and clean up the result for display:
    merge fragments, snap boundaries to whole words, expand/drop spans
    that dangle on a bare comma or function word, drop leftover
    single-word spans, then (if budget_fraction is set) keep only the
    highest-confidence spans up to that fraction of document coverage --
    see classifier/postprocess.py for why each step exists.

    Pass budget_fraction=None to get every cleaned-up span with no coverage
    cap (used by eval code that wants to compare against gold's full
    recall, not the deployment-facing "only show the top slice" behavior).
    """
    spans: list[Span] = []
    for chunk_start, chunk_end in split_into_chunks(text, max_chars):
        chunk_text = text[chunk_start:chunk_end]
        for s in predict_spans_crf(chunk_text, model, tokenizer):
            spans.append(Span(s.start + chunk_start, s.end + chunk_start, s.label, s.text, s.score))

    spans = merge_adjacent_spans(spans, text)
    spans = snap_to_word_boundaries(spans, text)
    spans = fix_dangling_edges(spans, text)
    spans = drop_short_spans(spans, min_words=min_words)
    if budget_fraction is not None:
        spans = apply_highlight_budget(spans, text_len=len(text), budget_fraction=budget_fraction)
    return spans
