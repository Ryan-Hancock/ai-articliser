"""Response-parsing tests.

The Ollama backend constrains decoding against this schema, so in practice these
faults no longer arrive. They are kept because that guarantee belongs to the
backend rather than to the format: each case here is a real failure mode from an
unconstrained generate loop, and the parser is what stands between a backend
without `format` support and a lost generation.
"""

from __future__ import annotations

from articliser.generate.prompts import (
    ArticleDraft,
    build_prompt,
    build_repair_prompt,
    parse_draft,
)

MINIMAL = '{"title": "T", "standfirst": "S", "category": "Systems", ' \
          '"tags": ["a"], "body_md": "## H\\nbody", "image_prompt": "p"}'


def test_parses_a_clean_object():
    draft = parse_draft(MINIMAL)
    assert draft is not None
    assert draft.title == "T"
    assert draft.category == "Systems"


def test_parses_through_surrounding_prose():
    assert parse_draft(f"Certainly! Here is the article:\n\n{MINIMAL}\n\nLet me know.") is not None


def test_parses_a_fenced_block():
    assert parse_draft(f"```json\n{MINIMAL}\n```") is not None


def test_body_containing_braces_does_not_truncate_the_object():
    # A regex from first "{" to last "}" would work here but breaks the moment
    # the body's braces are unbalanced; the parser tracks string state instead.
    # Replace the value only -- "body_md" contains "body", so a blind replace
    # would rename the key and test nothing.
    raw = MINIMAL.replace('"## H\\nbody"', '"## H\\nbody {braces} and a set {1, 2}"')
    draft = parse_draft(raw)
    assert draft is not None
    assert "{braces}" in draft.body_md


def test_trailing_comma_is_repaired():
    assert parse_draft(MINIMAL.replace('"p"}', '"p",}')) is not None


def test_raw_newline_inside_a_string_is_repaired():
    broken = '{"title": "T", "body_md": "line one\nline two", "category": "Theory"}'
    draft = parse_draft(broken)
    assert draft is not None
    assert "line one" in draft.body_md and "line two" in draft.body_md


def test_missing_body_is_treated_as_a_failed_generation():
    assert parse_draft('{"title": "T", "body_md": "   "}') is None


def test_no_json_at_all_returns_none():
    assert parse_draft("I'm sorry, I can't help with that.") is None
    assert parse_draft("") is None


def test_optional_fields_default_rather_than_failing():
    draft = parse_draft('{"title": "T", "body_md": "b"}')
    assert draft is not None
    assert draft.standfirst == "" and draft.tags == [] and draft.category == "Applications"


def test_category_aliases_and_fallback():
    assert ArticleDraft(title="t", body_md="b", category="ML").category == "Machine Learning"
    assert ArticleDraft(title="t", body_md="b", category="HCI").category == (
        "Human-Computer Interaction"
    )
    assert ArticleDraft(title="t", body_md="b", category="???").category == "Applications"


def test_tags_are_coerced_capped_and_cleaned():
    draft = ArticleDraft(title="t", body_md="b", tags="#one, two; three,,")
    assert draft.tags == ["one", "two", "three"]
    assert len(ArticleDraft(title="t", body_md="b", tags=[str(n) for n in range(20)]).tags) == 6


def test_prompt_contains_the_evidence_and_asks_for_json():
    prompt = build_prompt("A Paper", "## Abstract\nWe did a thing.")
    assert "We did a thing." in prompt
    assert "A Paper" in prompt
    assert "body_md" in prompt


def test_repair_prompt_carries_the_broken_response():
    assert "not json" in build_repair_prompt("not json at all")


def test_unrecognised_category_is_logged_not_just_swallowed(caplog):
    # "Applications" doubles as a real category and the fallback, so a skewed
    # feed is otherwise indistinguishable from a broken taxonomy.
    with caplog.at_level("INFO", logger="articliser.generate.prompts"):
        assert ArticleDraft(title="t", body_md="b", category="Astrology").category == "Applications"
    assert "Astrology" in caplog.text


def test_a_deliberate_applications_choice_is_not_logged(caplog):
    with caplog.at_level("INFO", logger="articliser.generate.prompts"):
        assert ArticleDraft(title="t", body_md="b", category="Applications").category == (
            "Applications"
        )
    assert "not recognised" not in caplog.text
