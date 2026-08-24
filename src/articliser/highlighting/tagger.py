"""Loads the published CRF tagger and turns an article body into stored spans.

The wrapper exists so the pipeline never touches the vendored modules directly:
everything above this line deals in plain `(start, end, label)` tuples over the
article's markdown, which is what the database stores and the renderer consumes.

Offsets are computed over the raw markdown, not over stripped prose, so the
renderer can inject <mark> tags without a second alignment pass. The cost is
that the model sees a little markdown punctuation; the postprocessing snaps span
edges to word boundaries, so that shows up as slightly conservative boundaries
rather than as broken tags.
"""

from __future__ import annotations

import logging
import time

from articliser.config import settings

log = logging.getLogger(__name__)


class RhetoricalTagger:
    """ModernBERT+CRF span tagger, loaded from the Hub."""

    name = "modernbert-crf"

    def __init__(self, model_repo: str | None = None) -> None:
        self.model_repo = model_repo or settings.crf_model_repo
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoTokenizer

        from articliser.highlighting.crf_model import ModernBertCrfForTokenClassification

        log.info("loading CRF tagger from %s", self.model_repo)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_repo)
        self._model = ModernBertCrfForTokenClassification.from_pretrained(self.model_repo)
        self._model.eval()

    def tag(self, body_md: str, budget_fraction: float | None = 0.2) -> list[tuple[int, int, str]]:
        """Return (start, end, label) tuples over `body_md`.

        `budget_fraction` caps highlighted coverage at a fraction of the document,
        keeping the highest-confidence spans. Highlighting everything is the same
        as highlighting nothing, so the default keeps roughly a fifth.
        """
        if not body_md.strip():
            return []

        self.load()
        from articliser.highlighting.chunking import predict_document

        started = time.perf_counter()
        spans = predict_document(
            body_md, self._model, self._tokenizer, budget_fraction=budget_fraction
        )
        words = len(body_md.split())
        elapsed = time.perf_counter() - started
        log.info(
            "tagged %d spans over %d words in %.2fs (%.1fs/1000w)",
            len(spans),
            words,
            elapsed,
            elapsed / max(words, 1) * 1000,
        )
        return [(span.start, span.end, span.label) for span in spans]

    def unload(self) -> None:
        if self._model is None:
            return
        self._model = None
        self._tokenizer = None
        from articliser.worker.gpu import empty_cache

        empty_cache()
