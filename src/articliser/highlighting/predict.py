# VENDORED from semantic-highlighting-slm @ 5d4235c (2026-07-28).
#
# Originally copied because airllm pinned transformers<5.13 while that project
# required >=5.14.1, so a path dependency was unresolvable. airllm has since been
# removed, so that constraint is gone and a path dependency would now resolve --
# it is still vendored only because that would pull gradio, datasets and
# scikit-learn in for seven files. Only the import paths were flattened to
# articliser.highlighting.*; the logic is unchanged, so upstream fixes can be
# re-copied. Weights still load from the Hub, not from here.
"""Run a trained token classifier on raw text and get back Spans.

Shared by scripts/eval_classifier.py now and the eventual M5 deployment
code -- "text in, Spans out" is the same operation in both places.
"""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from articliser.highlighting.labels import bio_ids_to_spans
from articliser.highlighting.spans import Span


@torch.no_grad()
def predict_spans(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
) -> list[Span]:
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = encoding.pop("offset_mapping")[0].tolist()
    encoding = {k: v.to(model.device) for k, v in encoding.items()}

    logits = model(**encoding).logits[0]
    pred_ids = logits.argmax(dim=-1).tolist()

    return bio_ids_to_spans(pred_ids, offset_mapping, text)


@torch.no_grad()
def predict_spans_crf(
    text: str,
    model,  # ModernBertCrfForTokenClassification -- avoids a circular import for the type hint
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
) -> list[Span]:
    """Same as predict_spans, but for the CRF variant: uses Viterbi decoding
    (model.decode) instead of per-token argmax, since that's the CRF's
    actual prediction rule.

    Each returned span's `.score` is the softmax confidence of its
    predicted label, averaged over its tokens (from the raw emissions, not
    a true CRF marginal probability -- pytorch-crf doesn't expose
    forward-backward marginals, so this is a practical proxy). Used to rank
    spans for highlight-budget filtering; see classifier/postprocess.py.
    """
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = encoding.pop("offset_mapping")[0].tolist()
    device = next(model.parameters()).device
    encoding = {k: v.to(device) for k, v in encoding.items()}

    emissions = model.emissions(**encoding)[0]
    probs = torch.softmax(emissions, dim=-1)
    pred_ids = model.decode(**encoding)[0]
    token_scores = [probs[i, label_id].item() for i, label_id in enumerate(pred_ids)]

    return bio_ids_to_spans(pred_ids, offset_mapping, text, token_scores)
