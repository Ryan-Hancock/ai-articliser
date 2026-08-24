# VENDORED from semantic-highlighting-slm @ 5d4235c (2026-07-28).
#
# Originally copied because airllm pinned transformers<5.13 while that project
# required >=5.14.1, so a path dependency was unresolvable. airllm has since been
# removed, so that constraint is gone and a path dependency would now resolve --
# it is still vendored only because that would pull gradio, datasets and
# scikit-learn in for seven files. Only the import paths were flattened to
# articliser.highlighting.*; the logic is unchanged, so upstream fixes can be
# re-copied. Weights still load from the Hub, not from here.
"""BIO tagging scheme: convert between character-offset Spans and per-token labels.

BIO = Begin/Inside/Outside, the standard scheme for treating span
classification as token classification (originally from NER). Every role in
schema.LABELS gets two tags (B-<role> starts a span, I-<role> continues one);
"O" covers everything else. This is the most bug-prone part of the whole
classifier pipeline -- see tests/test_bio.py for the cases it has to get
right (multi-token spans, adjacent spans of the same label, special tokens).
"""

from __future__ import annotations

from articliser.highlighting.spans import Span
from articliser.highlighting.schema import LABEL_NAMES

BIO_LABELS: list[str] = ["O"] + [
    f"{prefix}-{name}" for name in LABEL_NAMES for prefix in ("B", "I")
]
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(BIO_LABELS)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}


def _effective_start(text: str, tok_start: int, tok_end: int) -> int:
    """Skip leading whitespace some tokenizers fold into a token's offset.

    Byte-level BPE tokenizers (GPT-2/RoBERTa/ModernBERT-style) attach the
    space before a word to that word's token: "propose" preceded by a space
    reports as a single token with offset starting at the space, not at "p".
    Without correcting for that, a span whose quote starts cleanly at
    "propose" would fail strict containment against a token offset that
    starts one character earlier.
    """
    start = tok_start
    while start < tok_end and text[start].isspace():
        start += 1
    return start


def _find_containing_span(
    text: str, spans: list[Span], tok_start: int, tok_end: int
) -> Span | None:
    eff_start = _effective_start(text, tok_start, tok_end)
    for span in spans:
        if eff_start >= span.start and tok_end <= span.end:
            return span
    return None


def spans_to_bio_ids(
    text: str, spans: list[Span], offset_mapping: list[tuple[int, int]]
) -> list[int]:
    """Assign a BIO label id to each token, given its (start, end) char offsets.

    `offset_mapping` comes from a fast tokenizer with return_offsets_mapping=True.
    Special tokens ([CLS], [SEP], padding) have offset (0, 0) and get -100,
    the PyTorch/HF convention for "ignore this position when computing loss."

    A token counts as inside a span only if it's fully contained by it
    (strict containment, not partial overlap, modulo the leading-whitespace
    correction above) -- if our label boundaries ever land mid-token, that
    token is left as O rather than guessed at.
    """
    ids: list[int] = []
    prev_span_key: tuple[int, int, str] | None = None
    spans_sorted = sorted(spans, key=lambda s: s.start)

    for tok_start, tok_end in offset_mapping:
        if tok_start == tok_end:
            ids.append(-100)
            prev_span_key = None
            continue

        current = _find_containing_span(text, spans_sorted, tok_start, tok_end)
        if current is None:
            ids.append(LABEL2ID["O"])
            prev_span_key = None
        else:
            key = (current.start, current.end, current.label)
            prefix = "I" if key == prev_span_key else "B"
            ids.append(LABEL2ID[f"{prefix}-{current.label}"])
            prev_span_key = key

    return ids


def bio_ids_to_spans(
    label_ids: list[int],
    offset_mapping: list[tuple[int, int]],
    text: str,
    token_scores: list[float] | None = None,
) -> list[Span]:
    """Inverse of spans_to_bio_ids: collapse per-token predictions back into Spans.

    Tolerant of malformed sequences a model might actually predict (e.g. an
    I-Method with no preceding B-Method) by treating a "stray" I- tag as
    starting a new span rather than raising.

    If `token_scores` is given (one confidence value per token, e.g. the
    softmax probability of the predicted label), each returned span's
    `.score` is the mean of its tokens' scores -- used to rank spans by
    confidence for highlight-budget filtering (see classifier/postprocess.py).
    Omitting it leaves every span's score at the Span default (0.0).
    """
    spans: list[Span] = []
    current_label: str | None = None
    current_start: int | None = None
    current_end: int | None = None
    current_scores: list[float] = []

    def flush() -> None:
        nonlocal current_label, current_start, current_end, current_scores
        if current_label is not None:
            assert current_start is not None and current_end is not None
            mean_score = sum(current_scores) / len(current_scores) if current_scores else 0.0
            spans.append(
                Span(current_start, current_end, current_label, text[current_start:current_end], mean_score)
            )
        current_label = None
        current_start = None
        current_end = None
        current_scores = []

    for i, (label_id, (tok_start, tok_end)) in enumerate(zip(label_ids, offset_mapping)):
        if tok_start == tok_end:
            continue  # special token
        label = ID2LABEL[label_id]
        tok_score = token_scores[i] if token_scores is not None else 0.0
        if label == "O":
            flush()
            continue

        prefix, role = label.split("-", 1)
        if prefix == "B" or role != current_label:
            flush()
            current_label = role
            # Same leading-whitespace correction as spans_to_bio_ids -- a
            # new span's first token may report an offset that folds in the
            # preceding space, which would otherwise leak into the
            # recovered span's text and break exact-match comparisons.
            current_start = _effective_start(text, tok_start, tok_end)
            current_end = tok_end
            current_scores = [tok_score]
        else:  # "I" continuing the same role
            current_end = tok_end
            current_scores.append(tok_score)

    flush()
    return spans
