"""Corpus-driven discovery: decide what to read next from what has been read.

Three strategies, deliberately different in what they look at, because each one
alone has a characteristic blind spot:

- **references** -- what the corpus cites, ranked by how many of its papers cite
  it. The most precise signal available (these are works the authors themselves
  judged relevant) and the most backward-looking: it can only ever surface prior
  work.
- **keywords** -- what the corpus is about, taken from the tags the generator
  already writes. Finds current work, but only in vocabulary the corpus already
  uses, so it never escapes the existing topic.
- **prompted** -- what the model thinks is adjacent, given the summaries. The
  only strategy that can propose a direction none of the papers took, and the
  only one whose output is unverifiable, so its queries are sanitised hard and
  its candidates ranked below the other two.

Results are interleaved rather than concatenated. Reference candidates score
highest, and a plain sort would let one prolific bibliography fill the entire
batch -- so each strategy contributes in turn until the limit is reached.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from articliser.discovery.keywords import build_queries
from articliser.discovery.prompted import propose_queries
from articliser.discovery.references import rank_citations
from articliser.discovery.titles import extract_titles
from articliser.ingest.arxiv import (
    ArxivPaper,
    best_title_match,
    fetch_papers,
    search_by_title,
)

log = logging.getLogger(__name__)

STRATEGIES = ("references", "keywords", "prompted")

# How many top-ranked citations get a title extracted and an arXiv lookup. Each
# costs up to two throttled requests (a phrase search, then a word-AND fallback),
# so at a 3s gate this is already ~90s of the run. The corpus has ~500 entries,
# almost all cited once, so looking every one up would spend an hour to find the
# same handful.
MAX_CITATIONS_RESOLVED = 12


@dataclass
class Candidate:
    paper: ArxivPaper
    strategy: str
    reason: str
    score: float
    also_found_by: list[str] = field(default_factory=list)

    @property
    def identifier(self) -> str:
        return self.paper.abs_url


def _from_references(sources: dict[str, str], limit: int) -> list[Candidate]:
    """Cited works that turn out to be on arXiv, most co-cited first."""
    citations = rank_citations(sources, limit=MAX_CITATIONS_RESOLVED)
    if not citations:
        return []

    # Anything that already carries an arXiv id needs no title lookup at all.
    direct = [c for c in citations if c.arxiv_id]
    needs_title = [c for c in citations if not c.arxiv_id]

    candidates: list[Candidate] = []
    for citation in direct:
        paper = next(iter(search_by_title(citation.text[:120], max_results=1)), None)
        if paper is None or paper.arxiv_id != citation.arxiv_id:
            from articliser.ingest.arxiv import fetch_by_id

            paper = fetch_by_id(citation.arxiv_id)
        if paper:
            candidates.append(
                Candidate(
                    paper,
                    "references",
                    f"cited by {citation.count} paper(s) in the corpus (arXiv id in reference)",
                    score=10.0 + citation.count,
                )
            )

    titles = extract_titles([c.text for c in needs_title])
    for citation, title in zip(needs_title, titles):
        if not title or len(title.split()) < 3:
            continue
        # Best match, not first: arXiv orders by its own relevance, which can put
        # a topically similar paper above the one actually cited.
        paper = best_title_match(title, search_by_title(title, max_results=5))
        if paper is not None:
            candidates.append(
                Candidate(
                    paper,
                    "references",
                    f"cited by {citation.count} paper(s) in the corpus",
                    score=10.0 + citation.count,
                )
            )
        if len(candidates) >= limit:
            break

    return candidates


def broaden(query: str) -> str | None:
    """Drop the last AND-ed term, or None if there is nothing left to drop.

    Two exact phrases AND-ed together are often too narrow for arXiv: three of
    four generated queries returned nothing at all. Dropping the trailing term
    keeps precision on the primary concept, which is more useful than OR-ing the
    two and matching every paper that mentions either.
    """
    parts = re.split(r"\s+AND\s+", query)
    if len(parts) < 2:
        return None
    return " AND ".join(parts[:-1])


def _search(query: str, strategy: str, per_query: int) -> list[ArxivPaper]:
    """Run a query, broadening it while it returns nothing."""
    attempt: str | None = query
    while attempt:
        try:
            papers = fetch_papers(query=attempt, max_results=per_query, sort_by="submittedDate")
        except Exception as exc:  # noqa: BLE001 - one bad query shouldn't end discovery
            log.warning("%s query failed (%s): %s", strategy, attempt, exc)
            return []
        if papers:
            if attempt != query:
                log.info("%s: broadened to %r -> %d paper(s)", strategy, attempt, len(papers))
            else:
                log.info("%s: %r -> %d paper(s)", strategy, attempt, len(papers))
            return papers
        attempt = broaden(attempt)
    log.info("%s: %r -> nothing, even broadened", strategy, query)
    return []


def _from_queries(
    queries: list[tuple[str, str]], strategy: str, per_query: int, base_score: float
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for query, reason in queries:
        for rank, paper in enumerate(_search(query, strategy, per_query)):
            candidates.append(
                Candidate(paper, strategy, reason or query, base_score - rank * 0.1)
            )
    return candidates


def _dedupe(candidates: list[Candidate]) -> list[Candidate]:
    """One entry per arXiv id, keeping the best-scoring find and noting the rest.

    A paper surfaced by two strategies is a stronger signal than one surfaced by
    either, so the corroborating strategies are recorded on the survivor.
    """
    best: dict[str, Candidate] = {}
    for candidate in sorted(candidates, key=lambda c: -c.score):
        existing = best.get(candidate.paper.arxiv_id)
        if existing is None:
            best[candidate.paper.arxiv_id] = candidate
        elif candidate.strategy not in existing.also_found_by + [existing.strategy]:
            existing.also_found_by.append(candidate.strategy)
            existing.score += 1.0
    return list(best.values())


def _interleave(candidates: list[Candidate], limit: int) -> list[Candidate]:
    """Round-robin across strategies so no single one fills the batch."""
    by_strategy: dict[str, list[Candidate]] = {}
    for candidate in sorted(candidates, key=lambda c: -c.score):
        by_strategy.setdefault(candidate.strategy, []).append(candidate)

    out: list[Candidate] = []
    while by_strategy and len(out) < limit:
        for strategy in list(by_strategy):
            if len(out) >= limit:
                break
            out.append(by_strategy[strategy].pop(0))
            if not by_strategy[strategy]:
                del by_strategy[strategy]
    return out


def discover_candidates(
    sources: dict[str, str],
    tag_lists: list[list[str]],
    summaries: list[str],
    limit: int = 6,
    strategies: tuple[str, ...] = STRATEGIES,
    known_arxiv_ids: set[str] | None = None,
) -> list[Candidate]:
    """Run the enabled strategies and return a mixed, deduped, ranked shortlist."""
    known = known_arxiv_ids or set()
    found: list[Candidate] = []

    if "references" in strategies and sources:
        refs = _from_references(sources, limit=limit * 2)
        log.info("references: %d candidate(s)", len(refs))
        found += refs

    if "keywords" in strategies and tag_lists:
        found += _from_queries(build_queries(tag_lists, limit=4), "keywords", 3, base_score=6.0)

    if "prompted" in strategies and summaries:
        found += _from_queries(propose_queries(summaries, n=4), "prompted", 3, base_score=4.0)

    fresh = [c for c in _dedupe(found) if c.paper.arxiv_id not in known]
    log.info(
        "discovery: %d raw, %d after dedupe/known-filter", len(found), len(fresh)
    )
    return _interleave(fresh, limit)
