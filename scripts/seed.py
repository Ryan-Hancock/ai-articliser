"""Insert sample articles so the reading site is reviewable before any model runs.

M1 is deliberately verifiable end-to-end with no GPU, no weights and no network:
these fixtures exercise the same code paths a generated article will (markdown
body, stored spans, embeddings, related-article scoring), so layout and
highlighting bugs surface now rather than after a multi-hour model download.

Usage:
    uv run python scripts/seed.py [--reset]
"""

from __future__ import annotations

import argparse
import math
import random

from sqlmodel import col, select

from articliser.db.models import Article, Source, SourceKind, Span
from articliser.db.session import init_db, session_scope
from articliser.text import reading_minutes, strip_markdown, unique_slug

FIXTURES: list[dict] = [
    {
        "category": "Machine Learning",
        "title": "Layer streaming makes big models a disk problem, not a memory problem",
        "standfirst": "Running a 32B model on a 12GB card stops being about VRAM and starts being about how fast you can read from an SSD.",
        "identifier": "fixture:layer-streaming",
        "tags": ["inference", "quantisation", "systems"],
        "body": """The usual framing for running a large language model locally is a memory
question: does the model fit in VRAM? Layer-streaming inference reframes it. Instead of
holding every parameter resident, the runtime loads one transformer layer, runs the forward
pass through it, discards it, and loads the next.

## What actually changes

The peak memory requirement collapses to roughly the size of a single layer plus activations.
A 32B model that would need 65GB in bf16 becomes tractable on a 12GB card. The cost moves
somewhere else entirely: every token now pays for a full sweep of reads across the whole
model.

We measure a 4-bit compressed checkpoint against its uncompressed equivalent and find the
throughput gap tracks read bandwidth almost exactly, which suggests the bottleneck is
genuinely storage rather than compute.

## Where it stops making sense

This is a batch technique. For anything interactive the latency is disqualifying, and no
amount of quantisation closes that gap, because the cost is structural rather than
arithmetic. The honest use case is unattended overnight work where wall time is cheap.

Future work should look at overlapping the load of layer *n+1* with the compute of layer
*n*, which the current implementation does not attempt.""",
        "spans": [("Contribution", "reframes"), ("Method", "We measure a 4-bit compressed checkpoint"), ("Limitation", "the latency is disqualifying"), ("FutureWork", "overlapping the load of layer")],
    },
    {
        "category": "Information Design",
        "title": "Highlighting is a summarisation problem wearing a different hat",
        "standfirst": "Tagging which sentences carry a paper's claims turns out to need most of the machinery that writing a summary does — and one thing it doesn't.",
        "identifier": "fixture:highlighting-summarisation",
        "tags": ["nlp", "reading", "interfaces"],
        "body": """Ask a model to summarise a paper and it must decide what matters, then
write. Ask it to highlight the paper instead and it must still decide what matters, but the
writing step disappears. That missing step is worth more than it sounds.

## The structural argument

A generative summariser produces text that did not exist before, which means it can be
wrong in ways that are invisible: a plausible sentence that no source supports. A tagger
can only ever point at text that is already there. It can point at the wrong span, but it
cannot invent one.

The classifier reads every token's label in a single parallel forward pass. Sequence
labelling with a CRF layer decodes the whole sequence jointly, so implausible label
transitions get penalised rather than being invisible to the loss.

## The cost

Highlighting cannot compress. A reader who wants a paper reduced to three sentences is not
served by a version of the paper with three sentences underlined — the other ten pages are
still there. The two techniques answer different questions, and a reading tool probably
wants both.""",
        "spans": [("Contribution", "That missing step is worth more than it sounds"), ("Evidence", "single parallel forward pass"), ("Method", "Sequence\nlabelling with a CRF layer"), ("Limitation", "Highlighting cannot compress")],
    },
    {
        "category": "Systems",
        "title": "The GPU you think you have and the GPU you actually have",
        "standfirst": "On a WSL2 guest sharing a card with its Windows host, the memory numbers you trust are the ones that lie to you.",
        "identifier": "fixture:shared-gpu",
        "tags": ["wsl2", "cuda", "operations"],
        "body": """A pipeline that loads models in sequence has one hard invariant: only one
model is resident at a time. Enforcing it turns out to be harder than writing it down.

## The failure

Training aborted with a CUDA out-of-memory error while allocating a model small enough to
fit several times over. The Linux-side process list showed nothing holding memory. The
total-used figure disagreed.

The cause was a model server running on the Windows host, still holding weights loaded from
a previous step. Under WSL2 the physical card is shared, so host allocations never appear
in the guest's process list — only in the aggregate.

## The fix

Preflight every stage: read total free memory rather than summing per-process usage, and
explicitly request an unload from anything on the host side before allocating. Waiting for
an idle timeout is not a fix, because the timeout is longer than the gap between pipeline
stages.

This is a small operational detail that costs an entire overnight run when it goes wrong.""",
        "spans": [("Result", "Training aborted with a CUDA out-of-memory error"), ("Evidence", "The Linux-side process list showed nothing holding memory"), ("Contribution", "Preflight every stage"), ("Safety", "explicitly request an unload")],
    },
]


def _fake_embedding(seed_text: str, dim: int = 384) -> list[float]:
    """Deterministic unit vector standing in for MiniLM, so related-article
    scoring is exercised without loading sentence-transformers in M1."""
    rng = random.Random(seed_text)
    vector = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def seed(reset: bool = False) -> int:
    init_db()
    created = 0

    with session_scope() as session:
        if reset:
            for article in session.exec(select(Article)).all():
                session.delete(article)
            for source in session.exec(
                select(Source).where(col(Source.identifier).like("fixture:%"))
            ).all():
                session.delete(source)
            session.commit()

        taken = {slug for slug in session.exec(select(col(Article.slug))).all()}

        for fixture in FIXTURES:
            if session.exec(
                select(Source).where(Source.identifier == fixture["identifier"])
            ).first():
                continue

            source = Source(
                kind=SourceKind.PDF,
                identifier=fixture["identifier"],
                title=fixture["title"],
                raw_text=fixture["body"],
            )
            session.add(source)
            session.commit()
            session.refresh(source)

            body = fixture["body"]
            slug = unique_slug(fixture["title"], taken)
            taken.add(slug)

            article = Article(
                source_id=source.id,
                slug=slug,
                title=fixture["title"],
                standfirst=fixture["standfirst"],
                category=fixture["category"],
                tags=fixture["tags"],
                body_md=body,
                reading_minutes=reading_minutes(body),
                embedding=_fake_embedding(strip_markdown(body)),
            )
            session.add(article)
            session.commit()
            session.refresh(article)

            # Spans are located by searching the body for the quoted phrase rather
            # than hard-coding offsets, so editing a fixture can't silently
            # misplace a highlight.
            for label, phrase in fixture["spans"]:
                start = body.find(phrase)
                if start == -1:
                    raise SystemExit(f"fixture phrase not found in body: {phrase!r}")
                session.add(
                    Span(article_id=article.id, start=start, end=start + len(phrase), label=label)
                )
            session.commit()
            created += 1

    return created


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="delete existing fixture rows first")
    args = parser.parse_args()
    count = seed(reset=args.reset)
    print(f"seeded {count} article(s)")
