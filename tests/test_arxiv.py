"""arXiv id parsing. Network calls are not exercised here on purpose."""

from __future__ import annotations

import pytest

from articliser.ingest.arxiv import ArxivPaper, parse_arxiv_id


@pytest.mark.parametrize(
    "value,expected",
    [
        ("https://arxiv.org/abs/2401.00001", "2401.00001"),
        ("https://arxiv.org/abs/2401.00001v3", "2401.00001"),
        ("https://arxiv.org/pdf/2401.00001v2.pdf", "2401.00001"),
        ("http://arxiv.org/abs/2401.12345", "2401.12345"),
        ("2401.00001", "2401.00001"),
        ("2401.00001v2", "2401.00001"),
        ("https://example.com/paper.pdf", None),
        ("not a url", None),
        ("", None),
    ],
)
def test_parse_arxiv_id(value, expected):
    assert parse_arxiv_id(value) == expected


def test_paper_urls_are_derived_from_the_id():
    paper = ArxivPaper("2401.00001", "T", "A")
    assert paper.abs_url == "https://arxiv.org/abs/2401.00001"
    assert paper.pdf_url == "https://arxiv.org/pdf/2401.00001"
