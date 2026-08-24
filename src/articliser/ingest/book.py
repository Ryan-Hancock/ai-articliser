"""Split a book-length PDF into per-section units using its embedded outline.

A 644-page textbook cannot go through the paper pipeline: the evidence bundle
holds 12,000 characters and a single chapter runs to 46 pages. It has to be cut
first, and the PDF's own outline is far better than any heuristic -- it gives
exact titles, exact page boundaries and the real nesting, all authored by the
publisher rather than inferred from font sizes.

**Granularity is chosen per chapter, not globally.** Measured on Modern Robotics
(644 pages, 231 outline entries): chapters average 46 pages, which overflows the
budget several times over, while subchapters average 4 and fit comfortably. But
not every chapter has subchapters, so each one uses its own children when it has
usable ones and stands alone when it does not.

Front and back matter is dropped. A foreword, an exercise set and an index are
not articles, and generating them would cost a minute each to publish something
nobody wants to read.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from articliser.ingest.pdf import normalise

log = logging.getLogger(__name__)

# Sections that exist to serve the book, not the reader of an article.
_SKIP_TITLE_RE = re.compile(
    r"(?i)^\s*(foreword|preface|preview|summary|exercises?|notes and references|"
    r"software|bibliography|index|appendix|acknowledg|contents|about the|"
    r"list of (figures|tables)|glossary|errata|colophon|dedication)\b"
)

# Below this a section is a stub -- a heading with a sentence under it -- and
# produces an article with nothing in it.
MIN_SECTION_CHARS = 1500
# A chapter's children are only used if most of them are substantial; otherwise
# the chapter itself is the better unit.
MIN_USABLE_CHILD_FRACTION = 0.5


@dataclass(frozen=True)
class BookSection:
    """One publishable unit of a book."""

    title: str
    level: int
    start_page: int  # 1-based, inclusive, as the outline reports it
    end_page: int  # 1-based, inclusive
    chapter: str  # the enclosing level-1 title, for series ordering
    index: int  # position within the book, 1-based

    @property
    def page_count(self) -> int:
        return max(0, self.end_page - self.start_page + 1)


def has_outline(pdf_path: Path) -> bool:
    """Whether this PDF carries an embedded outline worth splitting on."""
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        return len(doc.get_toc()) > 1


def looks_like_a_book(pdf_path: Path, min_pages: int = 80) -> bool:
    """A long PDF with a real outline. Papers have neither."""
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        return doc.page_count >= min_pages and len(doc.get_toc()) > 1


def _raw_entries(pdf_path: Path) -> tuple[list[tuple[int, str, int]], int]:
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        return doc.get_toc(), doc.page_count


def _end_page(entries: list[tuple[int, str, int]], position: int, page_count: int) -> int:
    """Where the entry at `position` stops: just before the next entry at its own
    level or shallower."""
    level = entries[position][0]
    for other_level, _title, page in entries[position + 1 :]:
        if other_level <= level:
            return max(page - 1, entries[position][2])
    return page_count


def outline_sections(pdf_path: Path) -> list[BookSection]:
    """Publishable sections, one per chapter or subchapter as appropriate."""
    entries, page_count = _raw_entries(pdf_path)
    if not entries:
        return []

    chapters = [i for i, (level, _t, _p) in enumerate(entries) if level == 1]
    sections: list[BookSection] = []

    for position in chapters:
        _level, title, start = entries[position]
        if _SKIP_TITLE_RE.match(title):
            continue
        end = _end_page(entries, position, page_count)

        # Direct children only: grandchildren are usually a paragraph each.
        children = []
        for other in range(position + 1, len(entries)):
            child_level, child_title, child_page = entries[other]
            if child_level <= 1:
                break
            if child_level != 2 or _SKIP_TITLE_RE.match(child_title):
                continue
            children.append((other, child_title, child_page))

        usable = [
            (o, t, p)
            for o, t, p in children
            if _end_page(entries, o, page_count) - p + 1 >= 2
        ]
        if children and len(usable) >= max(1, int(len(children) * MIN_USABLE_CHILD_FRACTION)):
            for other, child_title, child_page in usable:
                sections.append(
                    BookSection(
                        title=child_title.strip(),
                        level=2,
                        start_page=child_page,
                        end_page=_end_page(entries, other, page_count),
                        chapter=title.strip(),
                        index=len(sections) + 1,
                    )
                )
        else:
            sections.append(
                BookSection(
                    title=title.strip(),
                    level=1,
                    start_page=start,
                    end_page=end,
                    chapter=title.strip(),
                    index=len(sections) + 1,
                )
            )

    return sections


def _strip_running_heads(pages: list[str]) -> str:
    """Remove the headers, footers and page numbers repeated across a section.

    Textbooks repeat the chapter and section title on every page, and PDF text
    extraction interleaves them with the prose. Left in, they are the most
    frequent phrases in the extracted text and the model treats them as emphasis.
    A line short enough to be furniture and present on at least half the pages is
    furniture.
    """
    if len(pages) < 3:
        # Too few pages for frequency to mean anything; only page numbers go.
        return "\n\n".join(
            "\n".join(l for l in p.splitlines() if not re.fullmatch(r"\s*\d{1,4}\s*", l))
            for p in pages
        )

    counts: Counter[str] = Counter()
    for page in pages:
        for line in {l.strip() for l in page.splitlines() if 3 < len(l.strip()) < 80}:
            counts[line] += 1

    threshold = max(2, len(pages) // 2)
    furniture = {line for line, n in counts.items() if n >= threshold}

    kept: list[str] = []
    for page in pages:
        lines = [
            line
            for line in page.splitlines()
            if line.strip() not in furniture
            and not re.fullmatch(r"\s*\d{1,4}\s*", line)
        ]
        kept.append("\n".join(lines))
    return "\n\n".join(kept)


def extract_section_text(pdf_path: Path, section: BookSection) -> str:
    """The text of one section, with running heads and page numbers removed."""
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        first = max(0, section.start_page - 1)
        last = min(doc.page_count, section.end_page)
        pages = [doc[p].get_text("text") for p in range(first, last)]
    return normalise(_strip_running_heads(pages))


def publishable_sections(pdf_path: Path) -> list[BookSection]:
    """Sections with enough text to be worth an article."""
    out: list[BookSection] = []
    for section in outline_sections(pdf_path):
        text = extract_section_text(pdf_path, section)
        if len(text) < MIN_SECTION_CHARS:
            log.info("skipping %r: only %d chars", section.title, len(text))
            continue
        out.append(section)
    # Renumber so the series has no gaps after the short ones are dropped.
    return [
        BookSection(s.title, s.level, s.start_page, s.end_page, s.chapter, i + 1)
        for i, s in enumerate(out)
    ]


def book_title(pdf_path: Path) -> str:
    """The book's title: metadata, then the title page, then the filename.

    Metadata is authoritative when present but very often empty -- it was on the
    first book tried, whose filename was "MR-v2", giving a series called "MR v2".
    The title page carries the real thing, typically as the first line or two
    set in capitals.
    """
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        metadata_title = ((doc.metadata or {}).get("title") or "").strip()
        if metadata_title:
            return metadata_title
        first_page = doc[0].get_text("text") if doc.page_count else ""

    lines = [line.strip() for line in first_page.splitlines() if line.strip()]
    caps: list[str] = []
    for line in lines[:6]:
        letters = [c for c in line if c.isalpha()]
        # A title page sets the title in capitals; the author line and the date
        # that follow are not, which is what ends the run.
        if len(line) < 4 or not letters:
            continue
        if sum(c.isupper() for c in letters) / len(letters) < 0.85:
            break
        caps.append(line)

    if caps:
        # "MODERN ROBOTICS" + "MECHANICS, PLANNING, AND CONTROL" -> title: subtitle
        return ": ".join(part.title() for part in caps[:2])
    return Path(pdf_path).stem.replace("-", " ").replace("_", " ").strip()
