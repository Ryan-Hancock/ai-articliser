# VENDORED from semantic-highlighting-slm @ 5d4235c (2026-07-28).
#
# Originally copied because airllm pinned transformers<5.13 while that project
# required >=5.14.1, so a path dependency was unresolvable. airllm has since been
# removed, so that constraint is gone and a path dependency would now resolve --
# it is still vendored only because that would pull gradio, datasets and
# scikit-learn in for seven files. Only the import paths were flattened to
# articliser.highlighting.*; the logic is unchanged, so upstream fixes can be
# re-copied. Weights still load from the Hub, not from here.
"""Label schema for rhetorical-role tagging, shared by every baseline and model.

Kept in one place so the rule baseline (M1), the labeling guideline (M2), and
the trained classifiers (M3/M4) all agree on the same label set and colors.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Label:
    name: str
    color: str  # hex, for terminal/HTML rendering
    description: str


LABELS: tuple[Label, ...] = (
    Label(
        "Contribution",
        "#4A90D9",
        "States what the authors did or built (e.g. 'we propose', 'we introduce').",
    ),
    Label(
        "Method",
        "#D9B44A",
        "Describes how it was done (techniques, models, data used).",
    ),
    Label(
        "Result",
        "#4AD98B",
        "Reports an outcome or finding (e.g. 'achieves', 'outperforms').",
    ),
    Label(
        "Evidence",
        "#4AD98B",
        "Quantitative support for a result (numbers, sample sizes, benchmarks).",
    ),
    Label(
        "Limitation",
        "#D94A4A",
        "Flags a weakness, constraint, or failure case.",
    ),
    Label(
        "FutureWork",
        "#A34AD9",
        "Points to work the authors plan to do or leave open.",
    ),
    Label(
        "Safety",
        "#4AD9C7",
        "Notes a safety property, guarantee, or constraint that was respected.",
    ),
)

LABEL_NAMES: tuple[str, ...] = tuple(l.name for l in LABELS)
LABEL_BY_NAME: dict[str, Label] = {l.name: l for l in LABELS}
