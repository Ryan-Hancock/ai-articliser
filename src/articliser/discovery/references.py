"""Mine the reference lists of ingested PDFs for papers worth reading next.

The premise is that what your corpus cites repeatedly is what your corpus is
about. Measured on a real eight-paper corpus: 528 reference entries, of which
only 23% carry a DOI and 1% an arXiv id, so identifier matching cannot carry
this -- titles have to. Co-citation is real but sparse at this corpus size (five
cross-paper matches in 493 entries, because the papers span offshore wind, DRAM
acceleration and 6G teleoperation alike), so citation count *ranks* candidates
rather than gating them. It gets stronger as the corpus grows.

Entry splitting handles the three styles observed in the wild: `[1] ` brackets
(IEEE and Elsevier), `1. ` numbering, and entries wrapped across several lines
by the PDF extractor.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from articliser.ingest.pdf import split_sections

# A new entry begins at "[12] " or "12. " at the start of a line. Everything
# until the next such marker belongs to the entry, because PDF extraction wraps
# long references across many lines.
_ENTRY_START_RE = re.compile(r"^\s*(?:\[(\d{1,3})\]|(\d{1,3})\.)\s+")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.I)
_ARXIV_RE = re.compile(r"arxiv[:\s/]*(\d{4}\.\d{4,5})", re.I)
_URL_RE = re.compile(r"https?://\S+")

# Dropped before building an entry's signature: they say nothing about which
# paper is being cited.
_STOPWORDS = frozenset(
    """the a an of and for in on with to by using via from at is are as its their our we
    this that these those be been was were will can could may might such which then than
    proc proceedings conf conference trans transactions journal vol pp eds ed press
    ieee acm springer elsevier available online accessed http https doi arxiv preprint
    """.split()
)

MIN_SIGNATURE_WORDS = 6
# Two entries this similar are treated as the same cited work. Chosen from the
# real corpus: genuine matches scored 0.63-1.00, and the highest scoring
# non-match sat well below.
SAME_WORK_THRESHOLD = 0.55


@dataclass
class Citation:
    """One cited work, possibly seen in several of the corpus's papers."""

    text: str
    signature: frozenset[str]
    doi: str | None = None
    arxiv_id: str | None = None
    cited_by: set[str] = field(default_factory=set)

    @property
    def count(self) -> int:
        return len(self.cited_by)


def split_entries(references_text: str) -> list[str]:
    """Split a references section into one string per entry."""
    entries: list[str] = []
    current: list[str] = []
    for line in references_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _ENTRY_START_RE.match(line):
            if current:
                entries.append(" ".join(current))
            current = [_ENTRY_START_RE.sub("", stripped)]
        elif current:
            current.append(stripped)
    if current:
        entries.append(" ".join(current))
    return [re.sub(r"\s+", " ", e).strip() for e in entries if e.strip()]


def signature(entry: str) -> frozenset[str]:
    """Distinctive words of an entry, for matching the same work across styles.

    A bag of words rather than a parsed title, because the same reference is
    formatted differently by every publisher -- author initials move, the venue
    abbreviates, the year relocates -- while the title's distinctive words
    survive all of it.
    """
    cleaned = _URL_RE.sub(" ", entry)
    cleaned = _DOI_RE.sub(" ", cleaned)
    words = re.findall(r"[a-z]{4,}", cleaned.lower())
    return frozenset(w for w in words if w not in _STOPWORDS)


def _similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def extract_citations(raw_text: str, cited_by: str) -> list[Citation]:
    """Every reference entry in one paper."""
    sections = {s.name: s.text for s in split_sections(raw_text)}
    references = sections.get("references", "")
    if not references:
        return []

    citations: list[Citation] = []
    for entry in split_entries(references):
        sig = signature(entry)
        if len(sig) < MIN_SIGNATURE_WORDS:
            continue  # too short to identify -- usually a bare URL or a fragment
        doi = _DOI_RE.search(entry)
        arxiv = _ARXIV_RE.search(entry)
        citations.append(
            Citation(
                text=entry,
                signature=sig,
                doi=doi.group(0).rstrip(".") if doi else None,
                arxiv_id=arxiv.group(1) if arxiv else None,
                cited_by={cited_by},
            )
        )
    return citations


def merge_citations(citations: list[Citation]) -> list[Citation]:
    """Collapse entries that refer to the same work, across papers and styles.

    Identifier matching first because it is exact; signature similarity after,
    for the ~76% of entries that carry no identifier at all.
    """
    by_identifier: dict[str, Citation] = {}
    unidentified: list[Citation] = []

    for citation in citations:
        key = citation.arxiv_id or citation.doi
        if key is None:
            unidentified.append(citation)
            continue
        existing = by_identifier.get(key)
        if existing is None:
            by_identifier[key] = citation
        else:
            existing.cited_by |= citation.cited_by

    merged: list[Citation] = list(by_identifier.values())
    for citation in unidentified:
        for existing in merged:
            if _similarity(citation.signature, existing.signature) >= SAME_WORK_THRESHOLD:
                existing.cited_by |= citation.cited_by
                # Keep the longer rendering: it usually has the fuller title.
                if len(citation.text) > len(existing.text):
                    existing.text = citation.text
                break
        else:
            merged.append(citation)

    merged.sort(key=lambda c: (-c.count, -len(c.signature)))
    return merged


def rank_citations(sources: dict[str, str], limit: int = 40) -> list[Citation]:
    """Every cited work across the corpus, most co-cited first.

    `sources` maps a label (the citing paper's title) to its raw text.
    """
    everything: list[Citation] = []
    for label, raw_text in sources.items():
        everything.extend(extract_citations(raw_text, cited_by=label))
    return merge_citations(everything)[:limit]


def citations_by_paper(sources: dict[str, str]) -> dict[str, int]:
    """Entry counts per paper, for logging what the parser actually saw."""
    counts: dict[str, int] = defaultdict(int)
    for label, raw_text in sources.items():
        counts[label] = len(extract_citations(raw_text, cited_by=label))
    return dict(counts)
