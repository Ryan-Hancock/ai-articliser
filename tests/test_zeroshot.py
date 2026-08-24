"""Zero-shot tagger tests.

Only the parts that don't need the model: markdown masking, sentence
boundaries, and the coverage budget. Those are where the bugs live -- the
classifier itself is someone else's problem, but injecting a <mark> into a code
fence or spending the whole budget on one label are ours.
"""

from __future__ import annotations

from articliser.highlighting.schema import LABEL_NAMES
from articliser.highlighting.zeroshot import (
    HYPOTHESES,
    ZeroShotTagger,
    prose_regions,
    sentence_spans,
)


def _texts(body: str, spans):
    return [body[start:end] for start, end in spans]


def test_hypotheses_cover_exactly_the_schema_labels():
    # The legend, the CSS and the model all read from schema.py; drift here would
    # render highlights with no matching colour and no legend entry.
    assert set(HYPOTHESES) == set(LABEL_NAMES)


def test_headings_are_never_tagged():
    body = "## A Heading Here\n\nA real sentence of prose follows it.\n"
    assert "## A Heading Here" not in "".join(_texts(body, prose_regions(body)))
    assert _texts(body, sentence_spans(body)) == ["A real sentence of prose follows it."]


def test_fenced_code_is_excluded():
    body = "Some prose here.\n\n```python\nx = {1: 2}\n```\n\nMore prose here.\n"
    regions = "".join(_texts(body, prose_regions(body)))
    assert "x = {1: 2}" not in regions
    assert "Some prose here." in regions and "More prose here." in regions


def test_list_markers_are_stripped_from_the_region():
    body = "- first bullet point here\n2. second bullet point here\n"
    texts = _texts(body, prose_regions(body))
    assert texts == ["first bullet point here", "second bullet point here"]


def test_sentence_spans_exclude_trailing_whitespace():
    body = "One sentence here.   Another sentence here.\n"
    for start, end in sentence_spans(body):
        assert not body[start:end].endswith((" ", "\n"))


def test_sentence_offsets_index_the_original_string():
    body = "## Heading\n\nAlpha runs first here. Beta runs second here.\n"
    for start, end in sentence_spans(body):
        assert body[start:end] in body
    assert _texts(body, sentence_spans(body)) == [
        "Alpha runs first here.",
        "Beta runs second here.",
    ]


def test_budget_none_keeps_everything_in_document_order():
    scored = [(0.9, 50, 60, "Result"), (0.7, 10, 20, "Method")]
    kept = ZeroShotTagger._apply_budget(scored, document_length=100, budget_fraction=None)
    assert [span[0] for span in kept] == [10, 50]


def test_budget_caps_total_coverage():
    scored = [(0.9 - i / 100, i * 10, i * 10 + 10, "Result") for i in range(10)]
    kept = ZeroShotTagger._apply_budget(scored, document_length=100, budget_fraction=0.2)
    assert sum(end - start for start, end, _ in kept) <= 20


def test_budget_spreads_across_labels_rather_than_taking_the_top_scores():
    # Evidence scores highest because numbers are unambiguous. Taking the top N by
    # score alone would return four Evidence spans and nothing else.
    scored = [
        (0.99, 0, 10, "Evidence"),
        (0.98, 10, 20, "Evidence"),
        (0.97, 20, 30, "Evidence"),
        (0.80, 30, 40, "Limitation"),
        (0.75, 40, 50, "FutureWork"),
    ]
    kept = ZeroShotTagger._apply_budget(scored, document_length=100, budget_fraction=0.3)
    labels = {label for _, _, label in kept}
    assert len(labels) >= 2, f"budget collapsed to a single label: {labels}"


def test_budget_always_keeps_at_least_one_span():
    # A single sentence longer than the whole budget should still be highlighted
    # rather than returning nothing at all.
    kept = ZeroShotTagger._apply_budget([(0.9, 0, 90, "Result")], 100, 0.2)
    assert len(kept) == 1


def test_spans_are_returned_in_document_order():
    scored = [(0.9, 80, 90, "Result"), (0.95, 10, 20, "Method"), (0.7, 40, 50, "Safety")]
    kept = ZeroShotTagger._apply_budget(scored, document_length=200, budget_fraction=0.9)
    assert [start for start, _, _ in kept] == sorted(start for start, _, _ in kept)
