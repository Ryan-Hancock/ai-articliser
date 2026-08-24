# VENDORED from semantic-highlighting-slm @ 5d4235c (2026-07-28).
#
# Originally copied because airllm pinned transformers<5.13 while that project
# required >=5.14.1, so a path dependency was unresolvable. airllm has since been
# removed, so that constraint is gone and a path dependency would now resolve --
# it is still vendored only because that would pull gradio, datasets and
# scikit-learn in for seven files. Only the import paths were flattened to
# articliser.highlighting.*; the logic is unchanged, so upstream fixes can be
# re-copied. Weights still load from the Hub, not from here.
"""Span-level evaluation harness.

This is the scaffolding every model in the project (rule baseline now,
transformer classifier and fine-tuned SLM later) gets scored against, so it's
built once here and reused rather than re-implemented per model.

Two scoring modes are provided because the baselines answer different
questions:

- `score_spans`: exact (start, end, label) match, CoNLL/NER-style. Use this
  for anything that claims to assign a *role* (the rule tagger, and later
  the trained classifiers).
- `score_unlabeled_overlap`: character-level overlap, ignoring label and
  exact boundaries. Use this for keyword/importance baselines (YAKE,
  TextRank) that only rank spans, they don't claim a role, so exact-match
  scoring would be unfairly strict.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Span:
    start: int  # char offset, inclusive
    end: int  # char offset, exclusive
    label: str
    text: str = ""
    score: float = 0.0  # ranking confidence, used by keyword baselines (label is "" for those)

    def key(self) -> tuple[int, int, str]:
        return (self.start, self.end, self.label)


@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def support(self) -> int:
        return self.tp + self.fn


@dataclass
class SpanMetrics:
    per_label: dict[str, PRF] = field(default_factory=dict)
    micro: PRF = field(default_factory=PRF)

    @property
    def macro_f1(self) -> float:
        if not self.per_label:
            return 0.0
        return sum(m.f1 for m in self.per_label.values()) / len(self.per_label)

    def __str__(self) -> str:
        lines = [f"{'label':<14}{'P':>7}{'R':>7}{'F1':>7}{'support':>9}"]
        for label, m in sorted(self.per_label.items()):
            lines.append(
                f"{label:<14}{m.precision:>7.2f}{m.recall:>7.2f}{m.f1:>7.2f}{m.support:>9}"
            )
        lines.append(
            f"{'micro avg':<14}{self.micro.precision:>7.2f}"
            f"{self.micro.recall:>7.2f}{self.micro.f1:>7.2f}{self.micro.support:>9}"
        )
        lines.append(f"{'macro F1':<14}{self.macro_f1:>7.2f}")
        return "\n".join(lines)


def score_spans(pred: list[Span], gold: list[Span]) -> SpanMetrics:
    """Exact (start, end, label) match, CoNLL/NER-style."""
    labels = {s.label for s in pred} | {s.label for s in gold}
    metrics = SpanMetrics(per_label={label: PRF() for label in labels})

    gold_keys = {s.key() for s in gold}
    pred_keys = {s.key() for s in pred}

    for label in labels:
        prf = metrics.per_label[label]
        gold_l = {k for k in gold_keys if k[2] == label}
        pred_l = {k for k in pred_keys if k[2] == label}
        prf.tp = len(pred_l & gold_l)
        prf.fp = len(pred_l - gold_l)
        prf.fn = len(gold_l - pred_l)
        metrics.micro.tp += prf.tp
        metrics.micro.fp += prf.fp
        metrics.micro.fn += prf.fn

    return metrics


def _to_char_set(spans: list[Span]) -> set[int]:
    chars: set[int] = set()
    for s in spans:
        chars.update(range(s.start, s.end))
    return chars


def score_unlabeled_overlap(pred: list[Span], gold: list[Span]) -> PRF:
    """Character-level precision/recall/F1, ignoring label and exact boundaries."""
    pred_chars = _to_char_set(pred)
    gold_chars = _to_char_set(gold)
    prf = PRF()
    prf.tp = len(pred_chars & gold_chars)
    prf.fp = len(pred_chars - gold_chars)
    prf.fn = len(gold_chars - pred_chars)
    return prf
