"""Ollama backend tests.

No network: what's under test is the schema we hand to Ollama for constrained
decoding, and the heading normalisation applied to whatever comes back. Those
are the two places where a defect produces a subtly wrong article rather than a
loud failure.
"""

from __future__ import annotations

from articliser.generate.backend import Summariser
from articliser.generate.ollama import OllamaSummariser, _draft_schema
from articliser.generate.prompts import ArticleDraft


def test_ollama_backend_satisfies_the_protocol():
    assert isinstance(OllamaSummariser(), Summariser)


def test_ollama_declares_that_it_owns_its_vram():
    # The pipeline reads this to decide whether the preflight may unload the
    # host's models -- which for this backend would unload the generator itself.
    assert OllamaSummariser().manages_own_vram is True


def test_schema_requires_every_field():
    # Pydantic marks a field optional as soon as it has a default, which lets a
    # constrained decoder skip it. Observed as articles arriving with no tags.
    schema = _draft_schema()
    assert set(schema["required"]) == set(schema["properties"])
    assert "tags" in schema["required"]


def test_schema_constrains_tag_count():
    tags = _draft_schema()["properties"]["tags"]
    assert tags["minItems"] == 3 and tags["maxItems"] == 5


def test_schema_build_does_not_mutate_the_model():
    _draft_schema()
    fresh = ArticleDraft.model_json_schema()
    assert "minItems" not in fresh["properties"]["tags"]


def test_h1_in_the_body_is_demoted():
    # The template renders the title as the page's only <h1>; another one in the
    # body is a duplicate landmark and a break in the type scale.
    draft = ArticleDraft(title="T", body_md="# Section One\n\ntext\n\n## Section Two\n")
    assert "\n## Section One" in "\n" + draft.body_md
    assert not draft.body_md.startswith("# ")
    assert "## Section Two" in draft.body_md


def test_hash_inside_a_code_fence_is_left_alone():
    body = "intro\n\n```sh\n# a shell comment\n```\n"
    assert "# a shell comment" in ArticleDraft(title="T", body_md=body).body_md


def test_body_normalisation_handles_empty_input():
    assert ArticleDraft(title="T", body_md="   ").body_md == ""
