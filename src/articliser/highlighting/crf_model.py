# VENDORED from semantic-highlighting-slm @ 5d4235c (2026-07-28).
#
# Originally copied because airllm pinned transformers<5.13 while that project
# required >=5.14.1, so a path dependency was unresolvable. airllm has since been
# removed, so that constraint is gone and a path dependency would now resolve --
# it is still vendored only because that would pull gradio, datasets and
# scikit-learn in for seven files. Only the import paths were flattened to
# articliser.highlighting.*; the logic is unchanged, so upstream fixes can be
# re-copied. Weights still load from the Hub, not from here.
"""ModernBERT encoder + linear head + CRF, for structured BIO decoding.

Plain token classification (AutoModelForTokenClassification, the v1/v2
approach) picks each token's label independently via per-token softmax --
nothing in the model penalizes the label changing every token, which is
exactly the failure mode found in v2's predictions (label flip-flopping
mid-word). A CRF layer adds learned transition scores between adjacent
labels and decodes the *whole sequence* jointly (Viterbi), so "Method
directly followed by Contribution" can be learned as rare/unlikely instead
of being invisible to the loss.

Not a transformers.PreTrainedModel subclass (a CRF's Viterbi decode doesn't
fit that class's generate/forward conventions cleanly), so local
checkpoints during training/eval still use plain torch.save/load. For
publishing to the HF Hub (M7), PyTorchModelHubMixin is a lighter-weight fit
than forcing this into PreTrainedModel: it adds save_pretrained/push_to_hub/
from_pretrained to any nn.Module, auto-generating config.json from the
constructor's own keyword arguments (all plain str/int/float here, so no
custom config class needed) without requiring transformers' architecture
conventions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from torchcrf import CRF
from transformers import AutoModel
from transformers.modeling_outputs import TokenClassifierOutput

from articliser.highlighting.labels import LABEL2ID


class ModernBertCrfForTokenClassification(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/Ryan-Hancock/semantic-highlighting-improve-reading-accessibility",
    tags=["token-classification", "crf", "modernbert"],
):
    def __init__(
        self,
        base_model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        o_weight: float = 0.1,
        aux_loss_weight: float = 0.5,
    ):
        """o_weight/aux_loss_weight implement class-weighted training: the
        vast majority of tokens in any abstract are "O" (not part of any
        labeled span), so a model that just predicts O everywhere gets most
        tokens "right" for free. down-weighting O's contribution to the
        loss pushes training to care more about getting the actual labeled
        spans right. See the auxiliary CE loss in forward() for why this
        can't just be a `weight=` kwarg on the CRF's own loss.
        """
        super().__init__()
        self.num_labels = num_labels
        self.aux_loss_weight = aux_loss_weight
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        class_weights = torch.ones(num_labels)
        class_weights[LABEL2ID["O"]] = o_weight
        self.register_buffer("class_weights", class_weights)

    def emissions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Raw per-token, per-label scores before CRF decoding -- exposed
        publicly so callers can derive a confidence score per predicted
        span (softmax over these), not just the single best label sequence
        that `decode` returns."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.classifier(self.dropout(hidden))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> TokenClassifierOutput:
        emissions = self.emissions(input_ids, attention_mask)
        mask = attention_mask.bool()

        loss = None
        if labels is not None:
            # CRF requires a valid (non-negative) tag at every position where
            # mask=True, but our labels use -100 (the HF ignore-index
            # convention) for special tokens (CLS/SEP), which the attention
            # mask marks as valid (mask=1, they're real tokens, just not
            # content). Swap -100 for a dummy valid id ("O") so CRF doesn't
            # index-error; attention_mask still correctly excludes actual
            # padding from the loss regardless.
            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = 0

            # reduction="sum" (not "mean") divided by valid-token-count,
            # not batch size -- pytorch-crf's "mean" averages per SEQUENCE,
            # so a 300-token abstract's loss is ~300x a per-token loss
            # (the partition function accumulates over every position).
            # Dividing by token count instead puts this on the same scale
            # as the per-token cross-entropy term below, so aux_loss_weight
            # is a meaningful knob rather than a number that only matters
            # because it happens to cancel out an unrelated scale mismatch.
            n_valid = mask.sum().clamp(min=1)
            crf_loss = -self.crf(emissions, safe_labels, mask=mask, reduction="sum") / n_valid

            # The CRF's own loss has no per-class weighting hook (it's a
            # sequence-level score via the forward algorithm, not a simple
            # per-token sum you can reweight token-by-token). Class
            # weighting happens through this auxiliary independent-softmax
            # loss on the same emissions instead, added alongside the CRF
            # loss -- the CRF still enforces sequence consistency (why it
            # was added in the first place), while this term pushes the
            # model to care more about the rare, real labels than the
            # majority "O" class.
            ce_loss = F.cross_entropy(
                emissions.view(-1, self.num_labels),
                labels.view(-1),
                weight=self.class_weights,
                ignore_index=-100,
            )

            loss = crf_loss + self.aux_loss_weight * ce_loss

        return TokenClassifierOutput(loss=loss, logits=emissions)

    @torch.no_grad()
    def decode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[list[int]]:
        """Viterbi-decode the single best label sequence per example (not
        argmax-per-token) -- this is the real prediction path, used at
        inference/eval time instead of `forward().logits.argmax(-1)`."""
        emissions = self.emissions(input_ids, attention_mask)
        mask = attention_mask.bool()
        return self.crf.decode(emissions, mask=mask)
