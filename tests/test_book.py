"""Book splitting tests.

The outline is synthesised rather than read from a real PDF: the logic under
test is how outline entries become publishable sections -- boundary arithmetic,
adaptive granularity, and front/back-matter filtering -- none of which needs a
644-page file to exercise. Text extraction and running-head removal are checked
against constructed page lists for the same reason.
"""

from __future__ import annotations

import pytest

from articliser.ingest import book as book_module
from articliser.ingest.book import BookSection, _strip_running_heads

# (level, title, page) -- the shape pymupdf's get_toc() returns.
OUTLINE = [
    (1, "Foreword", 5),
    (1, "Preface", 8),
    (1, "Configuration Space", 31),
    (2, "Degrees of Freedom of a Rigid Body", 32),
    (3, "Robot Joints", 36),
    (2, "Task Space and Workspace", 52),
    (2, "Summary", 56),
    (2, "Exercises", 58),
    (1, "Rigid-Body Motions", 77),
    (2, "Rotations and Angular Velocities", 86),
    (1, "Appendix A", 600),
    (1, "Index", 620),
]
PAGE_COUNT = 640


@pytest.fixture()
def sections(monkeypatch):
    monkeypatch.setattr(book_module, "_raw_entries", lambda _p: (OUTLINE, PAGE_COUNT))
    return book_module.outline_sections("fake.pdf")


def test_front_and_back_matter_are_dropped(sections):
    titles = {s.title for s in sections}
    for unwanted in ("Foreword", "Preface", "Index", "Appendix A", "Summary", "Exercises"):
        assert unwanted not in titles


def test_subchapters_are_preferred_over_the_chapter(sections):
    # A chapter with usable children should not itself become a section.
    assert "Configuration Space" not in {s.title for s in sections}
    assert "Degrees of Freedom of a Rigid Body" in {s.title for s in sections}


def test_grandchildren_are_not_used(sections):
    # Level 3 entries average a page or two: too granular to be an article.
    assert "Robot Joints" not in {s.title for s in sections}


def test_each_section_records_its_chapter(sections):
    by_title = {s.title: s for s in sections}
    assert by_title["Task Space and Workspace"].chapter == "Configuration Space"


def test_section_ends_before_the_next_entry(sections):
    by_title = {s.title: s for s in sections}
    # Ends at 51: "Task Space and Workspace" starts at 52.
    assert by_title["Degrees of Freedom of a Rigid Body"].end_page == 51


def test_last_section_runs_to_the_end_of_the_book(sections):
    # "Rotations and Angular Velocities" is followed only by skipped back matter,
    # so its end must come from the following level-1 entry, not the page count.
    by_title = {s.title: s for s in sections}
    assert by_title["Rotations and Angular Velocities"].end_page == 599


def test_sections_are_numbered_contiguously_in_reading_order(sections):
    assert [s.index for s in sections] == list(range(1, len(sections) + 1))
    assert [s.start_page for s in sections] == sorted(s.start_page for s in sections)


def test_a_chapter_without_usable_children_stands_alone(monkeypatch):
    # The child must be genuinely short: it runs from page 11 until the next
    # entry at page 12, so one page. (A child followed by a distant next chapter
    # is not a stub, however few outline entries sit between them.)
    outline = [(1, "Short Chapter", 10), (2, "A Stub", 11), (1, "Next", 12)]
    monkeypatch.setattr(book_module, "_raw_entries", lambda _p: (outline, 60))
    titles = [s.title for s in book_module.outline_sections("fake.pdf")]
    assert titles == ["Short Chapter", "Next"]


def test_a_child_is_usable_when_it_spans_real_pages(monkeypatch):
    # The mirror of the case above: one child, but it covers most of the book,
    # so it is the better unit and the chapter should not be emitted.
    outline = [(1, "Chapter", 10), (2, "A Real Section", 11), (1, "Next", 40)]
    monkeypatch.setattr(book_module, "_raw_entries", lambda _p: (outline, 60))
    titles = [s.title for s in book_module.outline_sections("fake.pdf")]
    assert titles == ["A Real Section", "Next"]


def test_empty_outline_yields_nothing(monkeypatch):
    monkeypatch.setattr(book_module, "_raw_entries", lambda _p: ([], 100))
    assert book_module.outline_sections("fake.pdf") == []


def test_page_count_is_inclusive():
    assert BookSection("t", 2, 10, 12, "c", 1).page_count == 3
    assert BookSection("t", 2, 10, 10, "c", 1).page_count == 1


# --- running heads ----------------------------------------------------------


def test_repeated_headers_and_page_numbers_are_removed():
    pages = [f"3.2. Rotations\nChapter 3.\n{60 + i}\nreal prose on page {i}." for i in range(6)]
    cleaned = _strip_running_heads(pages)
    assert "3.2. Rotations" not in cleaned
    assert "Chapter 3." not in cleaned
    assert "62" not in cleaned
    for i in range(6):
        assert f"real prose on page {i}." in cleaned


def test_body_text_that_merely_repeats_twice_survives():
    # The threshold is half the pages: a phrase on two of eight is prose.
    pages = ["the torque is applied" if i < 2 else f"other prose {i}" for i in range(8)]
    assert "the torque is applied" in _strip_running_heads(pages)


def test_short_sections_only_lose_page_numbers():
    # Under three pages, frequency means nothing, so nothing but numbers goes.
    pages = ["Heading\n1\nprose here", "Heading\n2\nmore prose"]
    cleaned = _strip_running_heads(pages)
    assert cleaned.count("Heading") == 2
    assert "\n1\n" not in cleaned and "\n2\n" not in cleaned
