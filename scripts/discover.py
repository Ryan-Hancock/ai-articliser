"""Find papers worth reading next, based on the corpus already ingested.

Three strategies, which can be run together or individually:

  references  what your PDFs cite, ranked by how many of them cite it
  keywords    arXiv searches built from the tags your articles were given
  prompted    arXiv searches the model proposes from your article summaries

With no corpus yet, falls back to a fixed subject query.

Usage:
    uv run python scripts/discover.py                      # all three, queue only
    uv run python scripts/discover.py --run                # queue, then generate
    uv run python scripts/discover.py --strategy references # just one
    uv run python scripts/discover.py --dry-run            # show, don't queue
    uv run python scripts/discover.py --query "cat:cs.RO"  # ignore the corpus
"""

from __future__ import annotations

import argparse
import logging

from sqlmodel import select

from articliser.db.models import Article, Job, JobStatus, Source
from articliser.db.session import init_db, session_scope
from articliser.discovery import STRATEGIES, discover_candidates
from articliser.ingest.arxiv import parse_arxiv_id
from articliser.worker.runner import Worker


def _corpus() -> tuple[dict[str, str], list[list[str]], list[str], set[str]]:
    with session_scope() as session:
        sources = session.exec(select(Source)).all()
        known = {parse_arxiv_id(s.identifier) for s in sources if parse_arxiv_id(s.identifier)}
        corpus = {
            s.title: s.raw_text
            for s in sources
            if s.raw_text and not s.identifier.startswith("fixture:")
        }
        articles = [
            a
            for a in session.exec(select(Article)).all()
            if a.source and not a.source.identifier.startswith("fixture:")
        ]
        return (
            corpus,
            [a.tags for a in articles if a.tags],
            [f"{a.title}. {a.standfirst}" for a in articles],
            known,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        action="append",
        choices=STRATEGIES,
        help="restrict to one strategy; repeatable. Defaults to all three.",
    )
    parser.add_argument("--limit", type=int, default=6, help="papers to queue")
    parser.add_argument("--query", default=None, help="ignore the corpus, use a fixed arXiv query")
    parser.add_argument("--start", type=int, default=0, help="arXiv offset, fixed-query mode only")
    parser.add_argument("--dry-run", action="store_true", help="show candidates, queue nothing")
    parser.add_argument("--run", action="store_true", help="generate the queued papers now")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    init_db()
    strategies = tuple(args.strategy) if args.strategy else STRATEGIES

    if args.dry_run:
        corpus, tags, summaries, known = _corpus()
        print(f"corpus: {len(corpus)} paper(s), {len(summaries)} article(s)\n")
        candidates = discover_candidates(
            corpus, tags, summaries, limit=args.limit, strategies=strategies, known_arxiv_ids=known
        )
        for c in candidates:
            also = f" (+{', '.join(c.also_found_by)})" if c.also_found_by else ""
            print(f"\n  [{c.strategy}{also}] {c.paper.arxiv_id}  score={c.score:.1f}")
            print(f"    {c.paper.title[:88]}")
            print(f"    why: {c.reason[:100]}")
        print(f"\n{len(candidates)} candidate(s); nothing queued (--dry-run)")
        return 0

    worker = Worker()
    queued = worker.discover(
        limit=args.limit, strategies=strategies, query=args.query, start=args.start
    )
    print(f"\n{queued} paper(s) queued")

    if args.run and queued:
        print("\ngenerating -- roughly a minute each\n")
        print(f"processed {worker.drain()} job(s)")
        with session_scope() as session:
            for job in session.exec(select(Job).where(Job.status == JobStatus.FAILED)).all():
                print(f"  FAILED {job.payload}: {job.error}")
    elif queued:
        print("run `make worker` or re-run with --run to generate them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
