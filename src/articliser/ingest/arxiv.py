"""arXiv fetching, adapted from semantic-highlighting-slm's data/arxiv.py @ 5d4235c.

Kept stdlib-only for the same reason the original was: pulling id/title/summary
out of an Atom feed does not justify a feed-parsing dependency. The polite delay,
the Retry-After handling and the exponential backoff are carried over unchanged
-- they were tuned against arXiv's real rate limits during an unattended
multi-hour collection run, and rediscovering those numbers by getting 429'd
would be a waste.

Added here: PDF download, and category metadata (the original only needed
abstracts, whereas this pipeline summarises full papers).
"""

from __future__ import annotations

import logging
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_API_URL = "http://export.arxiv.org/api/query"
_PAGE_SIZE = 100  # arXiv's own client examples page in chunks this size
_POLITE_DELAY_S = 3  # arXiv asks for a delay between consecutive requests
_RETRY_STATUS = {429, 503}
_MAX_RETRIES = 5
_RETRY_BASE_DELAY_S = 10  # doubles each retry: 10s, 20s, 40s, 80s, 160s

# Every request goes through one throttle, not just successive pages of the same
# query. Discovery issues tens of separate lookups in a loop -- a title search per
# citation, plus a query per keyword and per prompted suggestion -- and without a
# global gate those arrive back to back and earn an HTTP 429. Enforcing the delay
# here rather than at each call site means no caller can forget it.
_last_request_at = 0.0
_throttle_lock = threading.Lock()


def _throttle() -> None:
    global _last_request_at
    with _throttle_lock:
        wait = _POLITE_DELAY_S - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()

_ABS_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^/?#]+?)(?:v\d+)?(?:\.pdf)?$")


@dataclass(frozen=True)
class ArxivPaper:
    arxiv_id: str
    title: str
    abstract: str
    categories: tuple[str, ...] = field(default=())

    @property
    def abs_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"

    @property
    def pdf_url(self) -> str:
        return f"https://arxiv.org/pdf/{self.arxiv_id}"


def parse_arxiv_id(url_or_id: str) -> str | None:
    """Pull a bare arXiv id out of an abs/pdf URL, or pass through an id."""
    candidate = url_or_id.strip()
    match = _ABS_URL_RE.search(candidate)
    if match:
        return match.group(1)
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", candidate):
        return re.sub(r"v\d+$", "", candidate)
    return None


def _fetch_page(query: str, start: int, count: int, sort_by: str) -> list[ArxivPaper]:
    params = {
        "search_query": query,
        "sortBy": sort_by,
        "sortOrder": "descending",
        "start": str(start),
        "max_results": str(count),
    }
    url = f"{_API_URL}?{urllib.parse.urlencode(params)}"

    raw = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            _throttle()
            with urllib.request.urlopen(url, timeout=30) as resp:
                raw = resp.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRY_STATUS or attempt == _MAX_RETRIES:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            delay = float(retry_after) if retry_after else _RETRY_BASE_DELAY_S * (2**attempt)
            log.warning(
                "arXiv HTTP %s, retrying in %.0fs (attempt %d/%d)",
                exc.code,
                delay,
                attempt + 1,
                _MAX_RETRIES,
            )
            time.sleep(delay)

    assert raw is not None
    root = ET.fromstring(raw)
    papers: list[ArxivPaper] = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        raw_id = entry.findtext(f"{_ATOM_NS}id", default="")
        arxiv_id = re.sub(r"v\d+$", "", raw_id.rsplit("/", 1)[-1])
        title = " ".join(entry.findtext(f"{_ATOM_NS}title", default="").split())
        abstract = " ".join(entry.findtext(f"{_ATOM_NS}summary", default="").split())
        categories = tuple(
            term
            for element in entry.findall(f"{_ATOM_NS}category")
            if (term := element.get("term"))
        )
        if arxiv_id and abstract:
            papers.append(ArxivPaper(arxiv_id, title, abstract, categories))
    return papers


def fetch_papers(
    query: str = "cat:cs.LG OR cat:cs.CL",
    max_results: int = 20,
    start: int = 0,
    sort_by: str = "submittedDate",
) -> list[ArxivPaper]:
    """Fetch up to `max_results` papers starting at result offset `start`.

    `start` is what makes repeated runs add *new* papers instead of re-fetching
    the same top results; the scheduler persists the offset between runs for
    exactly this reason.
    """
    papers: list[ArxivPaper] = []
    offset = start
    remaining = max_results

    while remaining > 0:
        # No per-page sleep here: _throttle() gates every request globally.
        count = min(_PAGE_SIZE, remaining)
        page = _fetch_page(query, offset, count, sort_by)
        if not page:
            break  # ran out of results for this query
        papers.extend(page)
        offset += len(page)
        remaining -= len(page)

    return papers


def _escape_query_phrase(phrase: str) -> str:
    """Strip characters the arXiv query parser treats as syntax.

    Titles routinely contain colons and hyphens, which the API reads as field
    separators and operators; leaving them in silently returns nothing.
    """
    cleaned = re.sub(r'["\\:()\[\]{}]', " ", phrase)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def search_by_title(title: str, max_results: int = 3) -> list[ArxivPaper]:
    """Look a cited work up on arXiv by title.

    Used by reference-driven discovery: a reference list gives a title but rarely
    an arXiv id (measured: 1% of entries), so the title is the only way to find
    the free full text.
    """
    phrase = _escape_query_phrase(title)
    words = phrase.split()
    if len(words) < 3:
        return []  # too generic to identify a paper

    hits = _fetch_page(f'ti:"{phrase}"', start=0, count=max_results, sort_by="relevance")
    if hits:
        return hits

    # A phrase search fails outright on one corrupted token, and reference text
    # supplies them: PDF extraction joins hyphen-broken words without the hyphen,
    # turning "multiple-antenna" into "multipleantenna", which matches nothing.
    # AND-ing the individual words tolerates that, and dropping implausibly long
    # tokens removes the artifact itself.
    usable = [w for w in words if 3 <= len(w) <= 14]
    if len(usable) < 3:
        return []
    return _fetch_page(
        " AND ".join(f"ti:{w}" for w in usable[:8]),
        start=0,
        count=max_results,
        sort_by="relevance",
    )


def _title_words(title: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", title.lower()))


def title_similarity(wanted: str, candidate: str) -> float:
    """How alike two titles are, 0.0-1.0, by the better of two measures.

    Word overlap (Jaccard, not containment -- containment scores a short generic
    title highly against a long specific one) handles the ordinary case. But
    reference text carries tokens that PDF extraction has merged: "multiple-
    antenna" wrapped across lines becomes "multipleantenna", which shares *no*
    word with the real title and drags the correct paper below a wrong one. On
    that exact case, word overlap ranked "Quasi-Static SIMO Fading Channels"
    (0.75) above the cited "Quasi-Static Multiple-Antenna Fading Channels"
    (0.67).

    Comparing the letters alone, with spacing and punctuation removed, is immune
    to that -- a merged hyphen changes nothing at character level. Taking the max
    of the two lets each cover the other's weakness.
    """
    a, b = _title_words(wanted), _title_words(candidate)
    if not a or not b:
        return 0.0
    word_overlap = len(a & b) / len(a | b)

    letters = lambda t: re.sub(r"[^a-z]", "", t.lower())  # noqa: E731
    char_ratio = SequenceMatcher(None, letters(wanted), letters(candidate)).ratio()
    return max(word_overlap, char_ratio)


def title_matches(wanted: str, candidate: str, threshold: float = 0.6) -> bool:
    """Whether an arXiv hit is plausibly the work that was cited."""
    return title_similarity(wanted, candidate) >= threshold


def best_title_match(
    wanted: str, papers: list[ArxivPaper], threshold: float = 0.6
) -> ArxivPaper | None:
    """The closest of several arXiv hits, or None if none is close enough.

    Taking the first hit is wrong: arXiv orders by its own relevance, which put
    "Quasi-Static SIMO Fading Channels" above the exact "Quasi-Static
    Multiple-Antenna Fading Channels" for that title. Both look plausible; only
    one is the cited work.
    """
    scored = [(title_similarity(wanted, p.title), p) for p in papers]
    scored.sort(key=lambda pair: -pair[0])
    if scored and scored[0][0] >= threshold:
        return scored[0][1]
    return None


def fetch_by_id(arxiv_id: str) -> ArxivPaper | None:
    papers = _fetch_page(f"id:{arxiv_id}", start=0, count=1, sort_by="submittedDate")
    return papers[0] if papers else None


def download_pdf(paper: ArxivPaper, destination_dir: Path) -> Path:
    """Download a paper's PDF, skipping the request if it's already on disk."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / f"arxiv-{paper.arxiv_id.replace('/', '-')}.pdf"
    if target.exists() and target.stat().st_size > 0:
        return target

    log.info("downloading %s", paper.pdf_url)
    request = urllib.request.Request(
        paper.pdf_url,
        # arXiv returns 403 to the default urllib agent.
        headers={"User-Agent": "articliser/0.1 (local research summariser)"},
    )
    _throttle()
    with urllib.request.urlopen(request, timeout=60) as response:
        target.write_bytes(response.read())
    return target
