"""Zero-shot sentence tagging, the default replacement for the CRF.

Why this rather than the fine-tuned ModernBERT+CRF it replaces (kept alongside,
see `tagger.py`): the CRF was trained on paper abstracts, and this pipeline feeds
it generated article prose -- plain-language explanatory writing that
deliberately avoids the "we propose" phrasing the tagger keys on. The measured
result was not wrong labels so much as wrong *boundaries*: it emitted spans like
'383 held' and 'should extend', fragments that read as noise when highlighted.

A zero-shot NLI classifier sidesteps both problems. Its labels are natural
language, so there is no training domain to shift away from, and it classifies
whole sentences, so a boundary can only ever fall at a sentence edge. See
docs/findings.md for the head-to-head.

Two details do real work here:

- **A competing "background" hypothesis.** Without one, every sentence gets a
  role, because the model will always find *something* -- it labelled
  "Researchers have studied bearing faults for decades" as Limitation at 0.99
  confidence. The background option gives that sentence somewhere to go.
- **A coverage budget.** Even with the background option, ~80% of sentences come
  back tagged, and highlighting 80% of an article is the same as highlighting
  none of it. Sentences are ranked by confidence and kept until the budget is
  spent, which is the same approach the CRF's own postprocessing took.
"""

from __future__ import annotations

import logging
import re
import time
from functools import lru_cache

from articliser.config import settings
from articliser.highlighting.schema import LABEL_NAMES

log = logging.getLogger(__name__)

# One natural-language hypothesis per label in schema.py. These are the model's
# entire "training", so they are worth editing carefully: phrasing them as claims
# about the text ("This text reports...") rather than as topic labels ("Result")
# is what makes an entailment model usable as a classifier at all.
HYPOTHESES: dict[str, str] = {
    "Contribution": "This text states what the authors did, built, or propose.",
    "Method": "This text describes how something was done: a technique, model, or dataset used.",
    "Result": "This text reports an outcome or finding.",
    "Evidence": "This text gives specific numbers, measurements, or benchmark figures.",
    "Limitation": "This text describes a weakness, constraint, or failure case.",
    "FutureWork": (
        "This text describes work that has not been done yet but is planned or suggested."
    ),
    "Safety": "This text describes a safety property, guarantee, or risk.",
}

# Not a label -- a decoy that competes with the real ones so that unremarkable
# sentences have somewhere to land.
BACKGROUND_HYPOTHESIS = (
    "This text is background, context, or narrative filler that makes no specific claim."
)
_BACKGROUND_KEY = "\x00background"

DEFAULT_MIN_SCORE = 0.5
DEFAULT_BUDGET_FRACTION = 0.22
MIN_SENTENCE_WORDS = 5

# Markdown constructs that must never be tagged: a highlighted heading looks like
# a rendering bug, and a <mark> injected inside a fenced block corrupts the code.
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_LIST_MARKER_RE = re.compile(r"^\s*([-*+]|\d+\.)\s+")


@lru_cache(maxsize=1)
def _sentencizer():
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def prose_regions(body_md: str) -> list[tuple[int, int]]:
    """Character ranges of `body_md` that are ordinary prose.

    Excludes fenced code blocks and heading lines. Returned as offsets into the
    original string so spans found inside them need no later re-alignment.
    """
    regions: list[tuple[int, int]] = []
    offset = 0
    in_fence = False
    for line in body_md.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if _FENCE_RE.match(stripped):
            in_fence = not in_fence
        elif not in_fence and not _HEADING_RE.match(stripped) and stripped.strip():
            # Drop a leading list marker so a bullet's punctuation can't end up
            # inside the highlight.
            marker = _LIST_MARKER_RE.match(stripped)
            start = offset + (marker.end() if marker else 0)
            regions.append((start, offset + len(stripped)))
        offset += len(line)
    return regions


def sentence_spans(body_md: str) -> list[tuple[int, int]]:
    """Sentence boundaries over the prose regions of `body_md`."""
    nlp = _sentencizer()
    spans: list[tuple[int, int]] = []
    for region_start, region_end in prose_regions(body_md):
        chunk = body_md[region_start:region_end]
        for sent in nlp(chunk).sents:
            start = region_start + sent.start_char
            end = region_start + sent.end_char
            # Trim trailing whitespace so the <mark> doesn't swallow the gap
            # between sentences.
            while end > start and body_md[end - 1].isspace():
                end -= 1
            if end > start:
                spans.append((start, end))
    return spans


class ZeroShotTagger:
    """Sentence-level rhetorical tagging via a zero-shot NLI classifier."""

    name = "zeroshot-nli"

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or settings.zeroshot_model_id
        self._pipe = None

    def load(self) -> None:
        if self._pipe is not None:
            return

        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        log.info("loading zero-shot tagger %s (device=%s)", self.model_id, device)
        self._pipe = pipeline(
            "zero-shot-classification",
            model=self.model_id,
            device=device,
            # fp16 halves an already-small model and costs nothing measurable in
            # agreement; the classifier is not the pipeline's precision-critical
            # component.
            dtype=torch.float16 if device == 0 else torch.float32,
        )

    def tag(
        self,
        body_md: str,
        budget_fraction: float | None = DEFAULT_BUDGET_FRACTION,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> list[tuple[int, int, str]]:
        """Return (start, end, label) tuples over `body_md`."""
        if not body_md.strip():
            return []

        candidates = [
            (start, end)
            for start, end in sentence_spans(body_md)
            if len(body_md[start:end].split()) >= MIN_SENTENCE_WORDS
        ]
        if not candidates:
            return []

        self.load()
        assert self._pipe is not None

        keys = [*HYPOTHESES, _BACKGROUND_KEY]
        hypotheses = [*HYPOTHESES.values(), BACKGROUND_HYPOTHESIS]
        by_hypothesis = dict(zip(hypotheses, keys))

        started = time.perf_counter()
        results = self._pipe(
            [body_md[start:end] for start, end in candidates],
            candidate_labels=hypotheses,
            # The hypotheses are already complete sentences; the default template
            # ("This example is {}.") would wrap them into nonsense.
            hypothesis_template="{}",
            # Independent per-label scores. With multi_label=False the scores are
            # softmaxed across labels, so the background option would suppress
            # every real label instead of competing with them.
            multi_label=True,
        )

        scored: list[tuple[float, int, int, str]] = []
        for (start, end), result in zip(candidates, results):
            hypothesis, score = max(
                zip(result["labels"], result["scores"]), key=lambda pair: pair[1]
            )
            key = by_hypothesis[hypothesis]
            if key == _BACKGROUND_KEY or score < min_score:
                continue
            scored.append((score, start, end, key))

        kept = self._apply_budget(scored, len(body_md), budget_fraction)
        elapsed = time.perf_counter() - started
        log.info(
            "tagged %d/%d sentences over %d words in %.2fs (%.1fs/1000w)",
            len(kept),
            len(candidates),
            len(body_md.split()),
            elapsed,
            elapsed / max(len(body_md.split()), 1) * 1000,
        )
        return kept

    @staticmethod
    def _apply_budget(
        scored: list[tuple[float, int, int, str]],
        document_length: int,
        budget_fraction: float | None,
    ) -> list[tuple[int, int, str]]:
        """Keep the best sentences up to a coverage budget, one label at a time.

        Two things this has to get right. Without any budget the tagger marks
        roughly four sentences in five, which carries no information -- the point
        of highlighting is what it leaves out. But spending the budget on raw
        confidence alone produces the opposite failure: Evidence sentences score
        near 0.99 because numbers are unambiguous, so they crowd out every other
        label and the article comes back highlighted entirely one colour.

        So the budget is spent in rounds -- the best remaining sentence for each
        distinct label, then the next best for each, and so on. A varied set of
        roles is more useful to a reader than the seven most confident sentences,
        which in practice are all the same role.
        """
        if budget_fraction is None:
            ordered = sorted(scored, key=lambda item: item[1])
            return [(start, end, label) for _score, start, end, label in ordered]

        by_label: dict[str, list[tuple[float, int, int, str]]] = {}
        for item in sorted(scored, key=lambda item: -item[0]):
            by_label.setdefault(item[3], []).append(item)

        budget = document_length * budget_fraction
        used = 0
        kept: list[tuple[float, int, int, str]] = []
        # Round-robin over labels, best-first within each, strongest label first.
        while by_label:
            for label in sorted(by_label, key=lambda name: -by_label[name][0][0]):
                item = by_label[label].pop(0)
                if not by_label[label]:
                    del by_label[label]
                length = item[2] - item[1]
                if used + length > budget and kept:
                    continue
                kept.append(item)
                used += length
            if used >= budget:
                break

        return [
            (start, end, label)
            for _score, start, end, label in sorted(kept, key=lambda item: item[1])
        ]

    def unload(self) -> None:
        if self._pipe is None:
            return
        self._pipe = None
        from articliser.worker.gpu import empty_cache

        empty_cache()


def _assert_labels_match_schema() -> None:
    """The legend, the CSS and the model must agree on the label set."""
    missing = set(LABEL_NAMES) - set(HYPOTHESES)
    extra = set(HYPOTHESES) - set(LABEL_NAMES)
    if missing or extra:
        raise RuntimeError(
            f"zero-shot hypotheses drifted from schema.py "
            f"(missing={sorted(missing)}, extra={sorted(extra)})"
        )


_assert_labels_match_schema()
