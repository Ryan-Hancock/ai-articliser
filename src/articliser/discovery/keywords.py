"""Turn the corpus's own vocabulary into arXiv queries.

The tags the generator already writes for every article are the cheapest usable
keyword source in the system -- they were produced by a model that had just read
the paper, they are normalised to three to five lowercase topics, and they cost
nothing extra to collect. Frequency across the corpus is the signal: a tag that
recurs is a standing interest, a tag that appears once is an accident of one
paper.
"""

from __future__ import annotations

import re
from collections import Counter

# Too broad to be worth a query -- they would match most of arXiv's CS listing.
# Written as an explicit tuple, not a `.split()` of a sentence: the multi-word
# entries are the whole point, and splitting on whitespace silently turned
# "machine learning" into "machine" and "learning", so it never matched a tag.
_TOO_GENERIC = frozenset(
    (
        "ai",
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "neural networks",
        "research",
        "paper",
        "study",
        "survey",
        "review",
        "technology",
        "system",
        "systems",
        "data",
        "model",
        "models",
        "algorithm",
        "algorithms",
        "computing",
        "software",
        "hardware",
        "network",
        "networks",
        "engineering",
        "science",
    )
)

MIN_TAG_LENGTH = 3


def aggregate_tags(tag_lists: list[list[str]]) -> list[tuple[str, int]]:
    """Corpus-wide tag counts, most frequent first, generic terms removed."""
    counter: Counter[str] = Counter()
    for tags in tag_lists:
        # A tag repeated inside one article still only counts once for that
        # article, so a single verbose piece can't dominate the corpus.
        for tag in {t.strip().lower() for t in tags if t and t.strip()}:
            if len(tag) < MIN_TAG_LENGTH or tag in _TOO_GENERIC:
                continue
            counter[tag] += 1
    return counter.most_common()


def _quote(term: str) -> str:
    """Render a tag as an arXiv field query.

    Hyphens become spaces: tags arrive slugified ("digital-twins") but arXiv
    treats a hyphen as an operator, so the literal form matches nothing.
    """
    cleaned = re.sub(r'["\\:()\[\]{}-]', " ", term).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return f'all:"{cleaned}"' if " " in cleaned else f"all:{cleaned}"


def build_queries(
    tag_lists: list[list[str]], limit: int = 6, pair_top: int = 3
) -> list[tuple[str, str]]:
    """Return (query, reason) pairs for the most characteristic tags.

    Pairs the very top tags with each other as well as querying them alone: a
    single term like "teleoperation" returns arXiv's whole teleoperation
    listing, whereas "teleoperation AND latency" returns the corner of it this
    corpus actually occupies.
    """
    ranked = aggregate_tags(tag_lists)
    if not ranked:
        return []

    queries: list[tuple[str, str]] = []
    top = [tag for tag, _ in ranked[:pair_top]]
    for i, first in enumerate(top):
        for second in top[i + 1 :]:
            queries.append(
                (f"{_quote(first)} AND {_quote(second)}", f"recurring tags: {first} + {second}")
            )

    for tag, count in ranked:
        if len(queries) >= limit:
            break
        queries.append((_quote(tag), f"tag appears in {count} article(s)"))

    return queries[:limit]
