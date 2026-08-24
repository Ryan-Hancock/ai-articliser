"""Rhetorical-role tagging.

Two implementations behind one interface. `ZeroShotTagger` is the default and
`RhetoricalTagger` (the fine-tuned ModernBERT+CRF) is kept because it is roughly
30x faster and may be the better choice if the label schema is ever retrained on
article prose rather than abstracts.
"""

from __future__ import annotations

from articliser.config import settings


def build_tagger(kind: str | None = None):
    """Return the configured tagger. `kind` overrides ARTICLISER_TAGGER."""
    choice = (kind or settings.tagger).lower()
    if choice == "crf":
        from articliser.highlighting.tagger import RhetoricalTagger

        return RhetoricalTagger()
    if choice == "zeroshot":
        from articliser.highlighting.zeroshot import ZeroShotTagger

        return ZeroShotTagger()
    raise ValueError(f"unknown tagger {choice!r}; expected 'zeroshot' or 'crf'")
