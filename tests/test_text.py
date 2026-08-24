"""Slug, reading-time and highlight-rendering tests.

render_body_html is the one with real failure modes: it injects HTML into
markdown before rendering, so bad offsets would silently produce broken pages
rather than raising.
"""

from __future__ import annotations

from articliser.text import (
    reading_minutes,
    render_body_html,
    slugify,
    strip_markdown,
    unique_slug,
)


def test_slugify_handles_punctuation_and_case():
    assert slugify("Layer Streaming: A Disk Problem, Not Memory!") == (
        "layer-streaming-a-disk-problem-not-memory"
    )


def test_unique_slug_avoids_collisions():
    taken = {"a-title", "a-title-2"}
    assert unique_slug("A Title", taken) == "a-title-3"


def test_reading_minutes_is_at_least_one():
    assert reading_minutes("") == 1
    assert reading_minutes("word " * 440) == 2


def test_render_without_spans_is_plain_markdown():
    html = render_body_html("# Heading\n\nSome *text*.")
    assert "<h1>Heading</h1>" in html
    assert "<em>text</em>" in html
    assert "<mark" not in html


def test_render_wraps_spans_and_keeps_them_inside_the_paragraph():
    body = "The model achieves 91% accuracy on the benchmark."
    start = body.index("achieves 91% accuracy")
    html = render_body_html(body, [(start, start + len("achieves 91% accuracy"), "Result")])
    assert '<mark class="hl" data-label="Result"' in html
    assert "achieves 91% accuracy</mark>" in html
    assert html.strip().startswith("<p>")


def test_render_spans_survive_a_line_break_inside_a_paragraph():
    body = "We use a CRF\nlayer for decoding."
    start = body.index("a CRF\nlayer")
    html = render_body_html(body, [(start, start + len("a CRF\nlayer"), "Method")])
    assert html.count("<mark") == 1
    assert html.count("<p>") == 1  # still one paragraph, not split by the tag


def test_render_drops_overlapping_spans_rather_than_nesting_marks():
    body = "alpha beta gamma"
    html = render_body_html(body, [(0, 10, "Method"), (5, 16, "Result")])
    assert html.count("<mark") == 1


def test_render_clamps_out_of_range_offsets():
    body = "short body"
    html = render_body_html(body, [(4, 999, "Evidence")])
    assert "</mark>" in html
    # Clamped to len(body): the tag closes at the end of the text, not past it.
    assert html.strip().endswith("</mark></p>")
    assert "999" not in html


def test_unknown_label_does_not_leak_into_the_dom():
    html = render_body_html("text here", [(0, 4, "NotALabel")])
    assert 'data-label="Unknown"' in html
    assert "NotALabel" not in html


def test_strip_markdown_removes_syntax_and_code_blocks():
    plain = strip_markdown("# Title\n\n`code` and [link](http://x) and ```\nblock\n```")
    assert "Title" in plain and "link" in plain
    assert "```" not in plain and "block" not in plain and "http" not in plain


def test_slug_truncates_at_a_word_boundary():
    # A hard character cut produced "...without-specialized-hard", which reads as
    # a typo in the URL rather than an abbreviation.
    title = "Turning Standard Computer Memory into an AI Accelerator without Specialized Hardware"
    slug = slugify(title)
    assert len(slug) <= 80
    assert not slug.endswith("-")
    assert slug.split("-")[-1] == "specialized"


def test_slug_falls_back_to_a_hard_cut_for_one_long_word():
    slug = slugify("x" * 200)
    assert len(slug) == 80


def test_short_slugs_are_untouched():
    assert slugify("A Short One") == "a-short-one"
