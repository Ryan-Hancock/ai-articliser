"""PDF text extraction and section detection.

Split deliberately into an I/O half (`extract_text`, needs PyMuPDF and a file)
and a pure-text half (`split_sections`, `build_evidence_bundle`), so the section
heuristics -- the part that actually gets tuned -- are testable against string
fixtures without a PDF or a GPU anywhere in the loop.

The evidence bundle is a quality decision as much as a cost one. Sending
abstract + introduction + results + conclusion keeps the parts that carry the
paper's claims and drops the parts (related work, appendices, references) that
mostly carry other people's. It also has to fit the model's context window --
`ollama_num_ctx` -- and overflow there is silently truncated from the front,
which would drop the instructions rather than the paper.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# Ordered by how much a summary-writer needs them, not by how they appear in the
# paper -- build_evidence_bundle fills its budget in this order.
PREFERRED_SECTIONS: tuple[str, ...] = (
    "abstract",
    "conclusion",
    "introduction",
    "results",
    "discussion",
    "method",
)

_CANONICAL_SECTIONS: tuple[tuple[str, str], ...] = (
    (r"abstract", "abstract"),
    (r"introduction|overview|background", "introduction"),
    (r"related\s+work|prior\s+work|literature", "related work"),
    (r"method|approach|model|architecture|materials", "method"),
    (r"experiment|evaluation|setup|implementation", "experiments"),
    (r"result|finding", "results"),
    (r"discussion|analysis", "discussion"),
    (r"conclusion|summary|closing|future\s+work", "conclusion"),
    (r"reference|bibliography", "references"),
    (r"acknowledg", "acknowledgements"),
    (r"appendix|supplement", "appendix"),
)

# A heading line: optionally numbered ("3.", "IV."), short, and not ending in a
# sentence-final period. Papers vary wildly, so this stays deliberately loose and
# is validated by matching against the canonical names above rather than by
# trusting the layout.
_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.)?\s*([A-Za-z][A-Za-z\s\-]{2,48})\s*$"
)

# Publishers set section headings in ways a plain heading regex misses. Elsevier
# letter-spaces them ("A B S T R A C T"); IEEE runs the abstract inline with its
# first sentence ("Abstract-Teleoperation can be..."). Both were dropping the
# abstract out of the evidence bundle entirely on real papers.
_INLINE_SECTION_RE = re.compile(
    r"^\s*(abstract|keywords?|index terms)\s*[\u2014\u2013:-]\s*(.+)$", re.I
)

_LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"})


@dataclass(frozen=True)
class Section:
    name: str  # canonical name, or "preamble" for text before any heading
    heading: str  # the heading line as it appeared
    text: str


def extract_text(pdf_path: Path) -> str:
    """Extract a PDF's text, page by page, normalised for downstream text work."""
    import pymupdf  # imported lazily: the pure-text helpers below shouldn't need it

    with pymupdf.open(pdf_path) as doc:
        pages = [page.get_text("text") for page in doc]
    return normalise("\n\n".join(pages))


def normalise(text: str) -> str:
    """Undo the three things PDF extraction reliably gets wrong.

    Ligatures come out as single codepoints, words are hyphen-broken across line
    ends, and hard-wrapped lines leave newlines mid-sentence. All three confuse
    both the sentence splitter and the model, and none of them survive a round
    trip through the article body.
    """
    text = unicodedata.normalize("NFKC", text.translate(_LIGATURES))
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join hyphen-broken words
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _despace(text: str) -> str:
    """Collapse a letter-spaced heading: "A B S T R A C T" -> "ABSTRACT".

    Only fires when every token is a single character, so ordinary headings
    ("Related Work") and initialisms in prose are left alone.
    """
    tokens = text.split()
    if len(tokens) >= 3 and all(len(t) == 1 and t.isalpha() for t in tokens):
        return "".join(tokens)
    return text


def _canonical_section(heading: str) -> str | None:
    """Map a heading line to a canonical section name, or None if unrecognised.

    The trailing `[a-z]*` matters more than it looks: an anchored `\b` after a
    stem cannot match its own plural, because the character following "result"
    in "Results" is a word character. That silently folded Results into Method
    and References into Conclusion -- the section was never missing from the
    output, just attributed to the wrong heading, which is the kind of bug that
    only shows up as a slightly wrong evidence bundle.
    """
    lowered = _despace(heading.strip()).lower()
    for pattern, canonical in _CANONICAL_SECTIONS:
        if re.match(rf"^(?:{pattern})[a-z]*(?:\b|$)", lowered):
            return canonical
    return None


def split_sections(text: str) -> list[Section]:
    """Split a paper into canonical sections by scanning for heading lines.

    Unrecognised headings do not start a new section -- their text is folded into
    the section above. That's the safe direction to err: a missed heading costs
    some precision in the bundle, whereas treating every short line as a heading
    (figure captions, author names, table cells) would shred the body.
    """
    sections: list[Section] = []
    current_name = "preamble"
    current_heading = ""
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body or current_heading:
            sections.append(Section(current_name, current_heading, body))

    for line in text.splitlines():
        # An inline marker carries its own first sentence, so the remainder has to
        # survive into the new section rather than being discarded with the heading.
        inline = _INLINE_SECTION_RE.match(line)
        canonical = _canonical_section(inline.group(1)) if inline else None
        if canonical is not None:
            flush()
            current_name, current_heading = canonical, inline.group(1).strip()
            buffer = [inline.group(2).strip()]
            continue

        match = _HEADING_RE.match(line)
        canonical = _canonical_section(match.group(1)) if match else None
        if canonical is not None:
            flush()
            current_name, current_heading, buffer = canonical, line.strip(), []
        else:
            buffer.append(line)

    flush()
    return sections


def build_evidence_bundle(text: str, char_budget: int) -> str:
    """Assemble the prompt's source material, best sections first.

    Falls back to a plain head-truncation when section detection finds nothing
    usable, which happens with scanned or single-column-no-headings PDFs -- the
    first N characters of a paper are still mostly abstract and introduction, so
    this degrades rather than fails.
    """
    sections = split_sections(text)
    by_name: dict[str, str] = {}
    for section in sections:
        if section.text:
            by_name.setdefault(section.name, section.text)

    separator = "\n\n"
    parts: list[str] = []
    remaining = char_budget
    for name in PREFERRED_SECTIONS:
        body = by_name.get(name)
        if not body:
            continue
        header = f"## {name.title()}\n"
        # The separator this part will need once joined is charged up front,
        # otherwise the joined result overshoots char_budget by 2 per section.
        overhead = len(header) + (len(separator) if parts else 0)
        chunk = body[: max(0, remaining - overhead)]
        if not chunk.strip():
            break
        parts.append(header + chunk)
        remaining -= overhead + len(chunk)
        if remaining <= 0:
            break

    if not parts:
        return text[:char_budget]
    return separator.join(parts)


# Front-matter that sits above the real title on published PDFs. Getting one of
# these as the "source paper" makes the generation prompt actively misleading --
# observed titles included "Energy and AI 8 (2022) 100146" and "THIS VERSION HAS
# BEEN ACCEPTED AS A PAPER IN THE 2024 IEEE...".
_BOILERPLATE_RE = re.compile(
    r"(?i)(available online|contents lists|sciencedirect|elsevier|springer|"
    r"journal homepage|this version has been accepted|\bdoi\b|\bissn\b|"
    r"creativecommons|all rights reserved|©|preprint|accepted as a paper|"
    r"^\s*\d+\s*$|copyright)"
)


# "Energy and AI 8 (2022) 100146" -- a journal name with volume/year/article id.
# Passes every other filter, so it needs its own shape.
_JOURNAL_LINE_RE = re.compile(r"\(\s*(19|20)\d{2}\s*\)|\b\d{4,}\s*$|^\s*vol\.?\s*\d+", re.I)


def guess_title(text: str) -> str:
    """Best-effort title: the first substantial line that isn't front matter.

    Only a fallback -- the LLM writes the real article title, and an arXiv source
    carries an authoritative one. It still matters because it is passed to the
    model as "Source paper:", so publisher boilerplate here is worse than nothing.
    """
    for line in text.splitlines()[:60]:
        stripped = line.strip()
        # Front matter ends at the abstract; anything past it is body text.
        if _canonical_section(stripped) or _INLINE_SECTION_RE.match(stripped):
            break
        if not 12 <= len(stripped) <= 200:
            continue
        if stripped.lower().startswith(("arxiv:", "doi:", "http")):
            continue
        if _BOILERPLATE_RE.search(stripped) or _JOURNAL_LINE_RE.search(stripped):
            continue
        # A line with almost no lowercase is usually a running header or a
        # conference banner rather than a title.
        letters = [c for c in stripped if c.isalpha()]
        if letters and sum(c.islower() for c in letters) / len(letters) < 0.3:
            continue
        # A four-word floor separates titles from journal names ("Energy and AI").
        # Deliberately not higher: real titles in the corpus go down to four
        # words ("Task-Oriented Prediction and Communication"). Taking the first
        # survivor rather than the longest matters too -- the longest front-matter
        # line is usually an author list or an affiliation.
        if len(stripped.split()) < 4:
            continue
        return stripped[:200]
    return "Untitled"
