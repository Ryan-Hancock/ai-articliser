"""MiniLM sentence embeddings, used only for "related articles".

all-MiniLM-L6-v2 is already in the local HF cache and is small enough (~88MB)
that it never competes with the pipeline's real memory consumers, so it is the
one model here without a residency ceremony around it.
"""

from __future__ import annotations

import logging

from articliser.config import settings

log = logging.getLogger(__name__)

_model = None


def embed(text: str) -> list[float] | None:
    """Return a normalised embedding, or None if the model can't be loaded.

    Failure is non-fatal by design: without an embedding the article still
    publishes and simply falls back to same-category related links.
    """
    global _model

    if not text.strip():
        return None
    try:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading %s", settings.embedding_model_id)
            _model = SentenceTransformer(settings.embedding_model_id, device="cpu")

        # normalize_embeddings makes cosine similarity a plain dot product, which
        # is what web/app.py's _related computes.
        vector = _model.encode(text[:4000], normalize_embeddings=True)
        return [float(value) for value in vector]
    except Exception as exc:  # noqa: BLE001
        log.warning("embedding failed: %s", exc)
        return None
