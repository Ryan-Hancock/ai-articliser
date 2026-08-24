"""Section detection and evidence-bundle tests.

These run against string fixtures rather than real PDFs on purpose: the
heuristics here are the part that gets tuned, and tying them to a binary fixture
would make every tweak require a new PDF.
"""

from __future__ import annotations

from articliser.ingest.pdf import (
    PREFERRED_SECTIONS,
    build_evidence_bundle,
    guess_title,
    normalise,
    split_sections,
)

PAPER = """Streaming Transformer Layers From Disk

Jane Doe, John Roe

Abstract
We show that layer-streaming inference turns a memory problem into a bandwidth
problem.

1. Introduction
Large models do not fit on consumer cards.

2. Related Work
Others have quantised.

3. Method
We load one layer at a time.

4. Results
Throughput tracks read bandwidth.

5. Conclusion
Batch use only.

References
[1] Someone.
"""


def test_split_sections_finds_canonical_names():
    names = [section.name for section in split_sections(PAPER)]
    for expected in ("abstract", "introduction", "related work", "method", "results", "conclusion"):
        assert expected in names


def test_numbered_and_unnumbered_headings_both_match():
    numbered = split_sections("1. Introduction\nbody\n")
    unnumbered = split_sections("Introduction\nbody\n")
    assert numbered[-1].name == "introduction"
    assert unnumbered[-1].name == "introduction"


def test_unknown_headings_do_not_start_a_section():
    # A figure caption must not be mistaken for a heading and shred the body.
    sections = split_sections("Abstract\nclaim one.\nFigure 1\nclaim two.\n")
    abstract = [s for s in sections if s.name == "abstract"]
    assert len(abstract) == 1
    assert "claim two." in abstract[0].text


def test_evidence_bundle_prefers_claim_bearing_sections():
    bundle = build_evidence_bundle(PAPER, char_budget=4000)
    assert "bandwidth" in bundle  # abstract
    assert "Batch use only." in bundle  # conclusion
    assert "Others have quantised." not in bundle  # related work is dropped
    assert "[1] Someone." not in bundle  # references are dropped


def test_evidence_bundle_respects_its_budget():
    bundle = build_evidence_bundle(PAPER, char_budget=120)
    assert len(bundle) <= 120


def test_evidence_bundle_falls_back_when_no_headings_found():
    # Scanned or heading-less PDFs must degrade to head-truncation, not to empty.
    plain = "just one long run of text without any recognisable headings at all. " * 20
    bundle = build_evidence_bundle(plain, char_budget=200)
    assert bundle == plain[:200]


def test_preferred_sections_are_all_reachable_names():
    known = {section.name for section in split_sections(PAPER)}
    assert known & set(PREFERRED_SECTIONS)


def test_normalise_rejoins_hyphen_broken_words_and_ligatures():
    assert "bandwidth" in normalise("band-\nwidth")
    assert "efficient" in normalise("eﬃcient")


def test_guess_title_skips_headings_and_identifiers():
    assert guess_title(PAPER) == "Streaming Transformer Layers From Disk"
    assert guess_title("arXiv:2401.00001\nA Real Title Here\n") == "A Real Title Here"


# --- real-world publisher layouts -------------------------------------------
# Every case below comes from a PDF that failed before the fix. On a seven-paper
# corpus, none of the abstracts were being detected at all, which silently
# dropped the single most useful section from every generation prompt.


def test_letter_spaced_heading_is_recognised():
    # Elsevier sets section headings letter-spaced: "A B S T R A C T".
    paper = "A B S T R A C T\nWe present a review of offshore wind robotics.\n"
    assert [s.name for s in split_sections(paper)] == ["abstract"]


def test_letter_spacing_does_not_swallow_ordinary_headings():
    assert [s.name for s in split_sections("Related Work\nothers did things.\n")] == ["related work"]


def test_inline_abstract_marker_keeps_its_own_first_sentence():
    # IEEE runs the abstract into its first sentence: "Abstract-Teleoperation..."
    sections = split_sections("Abstract—Teleoperation can be very difficult.\nMore text.\n")
    assert sections[0].name == "abstract"
    assert "Teleoperation can be very difficult." in sections[0].text
    assert "More text." in sections[0].text


def test_inline_marker_accepts_colon_and_dash_forms():
    for sep in ("—", "–", "-", ":"):
        sections = split_sections(f"Abstract{sep} the claim goes here.\n")
        assert sections[0].name == "abstract", sep
        assert "the claim goes here." in sections[0].text


def test_guess_title_skips_publisher_front_matter():
    text = (
        "Energy and AI 8 (2022) 100146\n"
        "Available online 15 February 2022\n"
        "Contents lists available at ScienceDirect\n"
        "A review: Challenges and opportunities for artificial intelligence\n"
    )
    assert guess_title(text).startswith("A review")


def test_guess_title_skips_all_caps_conference_banners():
    text = (
        "THIS VERSION HAS BEEN ACCEPTED AS A PAPER IN THE 2024 IEEE CONFERENCE\n"
        "Intelligent Mode-switching Framework for Teleoperation\n"
    )
    assert guess_title(text) == "Intelligent Mode-switching Framework for Teleoperation"


def test_guess_title_skips_short_journal_names():
    # "Energy and AI" passes every other filter; the word-count floor is what
    # separates a journal name from a title.
    assert guess_title("Energy and AI\nCyber-physical-human systems for mobile robots\n") == (
        "Cyber-physical-human systems for mobile robots"
    )


def test_guess_title_keeps_a_four_word_title():
    # The floor must not be raised: real titles in the corpus go down to four.
    assert guess_title("Task-Oriented Prediction and Communication\n") == (
        "Task-Oriented Prediction and Communication"
    )


def test_guess_title_stops_at_the_abstract():
    assert guess_title("A B S T R A C T\nthis is body text and not a title at all\n") == "Untitled"
