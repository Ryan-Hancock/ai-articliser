"""Discovery tests.

No network and no model: what's covered is reference parsing, the ranking and
merging that turn 500 entries into a shortlist, query composition, and the
dedupe/interleave that decides what actually gets queued. Every case below is
drawn from a real failure on the eight-paper corpus.
"""

from __future__ import annotations

from articliser.discovery import Candidate, _dedupe, _interleave, broaden
from articliser.discovery.keywords import aggregate_tags, build_queries
from articliser.discovery.prompted import build_query
from articliser.discovery.references import (
    Citation,
    merge_citations,
    signature,
    split_entries,
)
from articliser.ingest.arxiv import ArxivPaper, best_title_match, title_similarity

REFS_BRACKET = """[1] Cheah W, Groves K, Martin H. MIRRAX: A reconfigurable robot for
limited access environments. IEEE Trans Robot 2023;39(2):1341-52.
[2] Placed JA, Strader J. A survey on active simultaneous localization
and mapping. IEEE Trans Robot 2023;39(3):1686-705.
"""

REFS_NUMBERED = """1. Sahil Anchal, Bodhibrata Mukhopadhyay. Person identification and
imposter detection using footstep generated seismic signals. IEEE
Transactions on Instrumentation and Measurement, 70:1-11, 2020.
2. Another Author. A second work entirely. Some Journal, 2021.
"""


# --- reference parsing ------------------------------------------------------


def test_split_entries_handles_bracketed_style():
    entries = split_entries(REFS_BRACKET)
    assert len(entries) == 2
    assert entries[0].startswith("Cheah W")
    # Wrapped lines must be rejoined, or the title is cut in half.
    assert "limited access environments" in entries[0]


def test_split_entries_handles_numbered_style():
    entries = split_entries(REFS_NUMBERED)
    assert len(entries) == 2
    assert "footstep generated seismic signals" in entries[0]


def test_split_entries_on_empty_input():
    assert split_entries("") == []
    assert split_entries("no markers here at all\n") == []


def test_signature_drops_venue_and_publisher_noise():
    sig = signature("Cheah W. MIRRAX robot. IEEE Trans Robot 2023. doi:10.1109/TRO.1")
    assert "mirrax" in sig
    assert "ieee" not in sig and "trans" not in sig
    assert not any(w.startswith("10.") for w in sig)


def test_merge_collapses_the_same_work_cited_by_two_papers():
    # The real case: the same reference formatted differently by two publishers.
    a = Citation("Blanche J, Mitchell D. Asset integrity monitoring of wind turbine blades",
                 signature("Blanche J, Mitchell D. Asset integrity monitoring of wind turbine blades"),
                 cited_by={"paper-a"})
    b = Citation("J. Blanche and D. Mitchell, Asset integrity monitoring of wind turbine blades, 2021",
                 signature("J. Blanche and D. Mitchell, Asset integrity monitoring of wind turbine blades, 2021"),
                 cited_by={"paper-b"})
    merged = merge_citations([a, b])
    assert len(merged) == 1
    assert merged[0].count == 2


def test_merge_uses_identifiers_before_similarity():
    a = Citation("one rendering of the work", signature("one rendering of the work"),
                 doi="10.1/x", cited_by={"a"})
    b = Citation("a completely different set of descriptive words entirely",
                 signature("a completely different set of descriptive words entirely"),
                 doi="10.1/x", cited_by={"b"})
    merged = merge_citations([a, b])
    assert len(merged) == 1 and merged[0].count == 2


def test_merge_keeps_genuinely_different_works_apart():
    a = Citation("robot navigation in cluttered indoor environments",
                 signature("robot navigation in cluttered indoor environments"), cited_by={"a"})
    b = Citation("transformer language models for protein folding prediction",
                 signature("transformer language models for protein folding prediction"), cited_by={"b"})
    assert len(merge_citations([a, b])) == 2


# --- title matching ---------------------------------------------------------


def test_title_similarity_survives_a_pdf_merged_hyphen():
    # "multiple-antenna" wrapped across lines becomes "multipleantenna", which
    # shares no *word* with the real title. Character-level comparison catches it.
    wanted = "Quasi-static multipleantenna fading channels at finite blocklength"
    assert title_similarity(wanted, "Quasi-Static Multiple-Antenna Fading Channels at Finite Blocklength") > 0.9


def test_best_match_prefers_the_exact_paper_over_a_similar_one():
    # arXiv ranked the SIMO paper above the cited Multiple-Antenna one.
    wanted = "Quasi-static multipleantenna fading channels at finite blocklength"
    papers = [
        ArxivPaper("1302.1302", "Quasi-Static SIMO Fading Channels at Finite Blocklength", "a"),
        ArxivPaper("1311.2012", "Quasi-Static Multiple-Antenna Fading Channels at Finite Blocklength", "a"),
    ]
    assert best_title_match(wanted, papers).arxiv_id == "1311.2012"


def test_best_match_rejects_an_unrelated_paper():
    papers = [ArxivPaper("1234.5678", "Attention Is All You Need", "a")]
    assert best_title_match("Sustainable decommissioning of an offshore wind farm", papers) is None


# --- keyword queries --------------------------------------------------------


def test_tags_are_counted_once_per_article():
    ranked = dict(aggregate_tags([["robotics", "robotics"], ["robotics"], ["vision"]]))
    assert ranked["robotics"] == 2


def test_generic_tags_are_dropped():
    ranked = dict(aggregate_tags([["machine learning", "teleoperation"], ["ai", "teleoperation"]]))
    assert "teleoperation" in ranked
    assert "machine learning" not in ranked and "ai" not in ranked


def test_keyword_queries_pair_the_top_tags():
    queries = [q for q, _ in build_queries([["teleoperation", "robotics"]] * 3, limit=4)]
    assert any(" AND " in q for q in queries)
    assert all(q.startswith(("all:", "abs:")) for q in queries)


def test_slugified_tags_lose_their_hyphens():
    # arXiv reads a hyphen as an operator, so "digital-twins" literally matches nothing.
    queries = [q for q, _ in build_queries([["digital-twins", "robotics"]] * 2, limit=2)]
    assert not any("-" in q for q in queries)


# --- prompted query composition ---------------------------------------------


def test_build_query_composes_syntax_from_plain_terms():
    assert build_query(["tactile feedback", "teleoperation"]) == (
        'all:"tactile feedback" AND all:teleoperation'
    )


def test_build_query_strips_syntax_the_model_should_not_have_sent():
    # The model was asked for plain words but still reaches for field prefixes.
    assert build_query(["all:tactile", "robot control"]) == (
        'all:tactile AND all:"robot control"'
    )


def test_build_query_rejects_fragments_and_duplicates():
    assert build_query(["(abs:"]) is None
    assert build_query(["a", "b"]) is None
    assert build_query(["robot", "robot"]) is None  # AND-ing a term with itself


def test_broaden_drops_the_last_term_only():
    assert broaden('all:"a b" AND all:c AND all:d') == 'all:"a b" AND all:c'
    assert broaden('all:"a b" AND all:c') == 'all:"a b"'
    assert broaden("all:x") is None


# --- shortlisting -----------------------------------------------------------


def _c(arxiv_id, strategy, score):
    return Candidate(ArxivPaper(arxiv_id, f"T{arxiv_id}", "a"), strategy, "r", score)


def test_dedupe_credits_a_paper_found_by_two_strategies():
    merged = _dedupe([_c("1", "references", 11.0), _c("1", "keywords", 6.0)])
    assert len(merged) == 1
    assert merged[0].also_found_by == ["keywords"]
    assert merged[0].score > 11.0  # corroboration is a stronger signal


def test_interleave_stops_one_strategy_filling_the_batch():
    # References always outscore the others, so a plain sort would return only
    # references and the feed would never see anything new.
    candidates = [_c(str(i), "references", 12.0 - i * 0.1) for i in range(5)]
    candidates += [_c("k1", "keywords", 6.0), _c("p1", "prompted", 4.0)]
    strategies = {c.strategy for c in _interleave(_dedupe(candidates), 3)}
    assert len(strategies) == 3


def test_interleave_respects_the_limit():
    candidates = [_c(str(i), "keywords", 6.0) for i in range(10)]
    assert len(_interleave(candidates, 4)) == 4
