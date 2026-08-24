"""Queue every PDF in the data directory that hasn't been turned into an article.

The intended workflow for a folder of papers: drop them in `data/pdfs/`, run
this, then let the worker drain the queue. Sources are keyed by resolved path,
so re-running is safe -- it queues what is new and skips what already has an
article.

Usage:
    uv run python scripts/ingest.py             # queue everything new
    uv run python scripts/ingest.py --run       # queue, then generate now
    uv run python scripts/ingest.py --dir path  # somewhere other than data/pdfs
    uv run python scripts/ingest.py --dry-run   # just show what would be queued
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sqlmodel import select

from articliser.config import settings
from articliser.db.models import Article, Job, JobKind, JobStatus, Source
from articliser.db.session import init_db, session_scope
from articliser.generate.pipeline import source_for_pdf
from articliser.worker.runner import Worker

log = logging.getLogger(__name__)


def find_pdfs(directory: Path) -> list[Path]:
    """Every readable PDF in `directory`, sorted.

    `*.pdf` rather than a case-insensitive walk of everything, and an explicit
    guard against the `file.pdf:Zone.Identifier` companions WSL surfaces for
    files downloaded on Windows -- they are alternate data streams, not PDFs,
    and PyMuPDF fails on them with an unhelpful error.
    """
    if not directory.is_dir():
        raise SystemExit(f"not a directory: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".pdf"
        and ":" not in path.name
        and path.stat().st_size > 0
    )


def queue(directory: Path, dry_run: bool = False) -> int:
    init_db()
    pdfs = find_pdfs(directory)
    if not pdfs:
        print(f"no PDFs found in {directory}")
        return 0

    queued = 0
    with session_scope() as session:
        for pdf in pdfs:
            if dry_run:
                # source_for_pdf commits, so a dry run must not call it -- the
                # enclosing rollback cannot undo a commit that already happened.
                identifier = f"file:{pdf.resolve()}"
                existing = session.exec(
                    select(Source).where(Source.identifier == identifier)
                ).first()
                state = "known" if existing else "new  "
                print(f"  would   [{state}] {pdf.name[:60]}")
                queued += 1
                continue

            source = source_for_pdf(session, pdf)

            if session.exec(select(Article).where(Article.source_id == source.id)).first():
                print(f"  skip    {pdf.name[:66]}  (already published)")
                continue

            pending = session.exec(
                select(Job).where(
                    Job.status.in_([JobStatus.PENDING, JobStatus.RUNNING]),  # type: ignore[attr-defined]
                )
            ).all()
            if any(job.payload.get("source_id") == source.id for job in pending):
                print(f"  skip    {pdf.name[:66]}  (already queued)")
                continue

            session.add(Job(kind=JobKind.INGEST, payload={"source_id": source.id}))
            print(f"  queued  {pdf.name[:66]}")
            queued += 1

    return queued


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=None, help="defaults to data/pdfs")
    parser.add_argument("--run", action="store_true", help="drain the queue after queueing")
    parser.add_argument("--dry-run", action="store_true", help="show what would be queued")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    settings.ensure_dirs()

    directory = args.dir or settings.pdf_dir
    print(f"scanning {directory}\n")
    n = queue(directory, dry_run=args.dry_run)
    print(f"\n{n} PDF(s) {'would be ' if args.dry_run else ''}queued")

    # Drain on --run even when nothing new was queued: a previous run may have
    # queued these already, and refusing to generate them would be surprising.
    if args.run and not args.dry_run:
        print("\ndraining the queue -- roughly a minute per article\n")
        done = Worker().drain()
        print(f"\nprocessed {done} job(s)")
        with session_scope() as session:
            failed = session.exec(select(Job).where(Job.status == JobStatus.FAILED)).all()
            for job in failed:
                print(f"  FAILED {job.payload}: {job.error}")
    elif n and not args.dry_run:
        print("run `make ingest RUN=1` (or `make worker`) to generate them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
