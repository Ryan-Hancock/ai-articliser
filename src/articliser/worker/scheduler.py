"""The long-running worker: periodic discovery, periodic generation.

Two cadences rather than one, because the two costs are wildly different.
Discovery is a handful of HTTP requests and can run hourly without noticing;
generation is a layer-streamed 32B model and is measured in minutes per article,
so it runs on a slow tick and processes whatever the queue holds.

`max_instances=1` and `coalesce=True` matter here: if a generation run overruns
its interval -- which it will -- APScheduler must skip the missed ticks rather
than stack up a backlog of concurrent runs fighting over one GPU.
"""

from __future__ import annotations

import logging
import os
import signal
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from sqlmodel import select

from articliser.db.models import Job, JobKind, JobStatus, Source
from articliser.db.session import init_db, session_scope
from articliser.worker.runner import DEFAULT_QUERY, Worker

log = logging.getLogger(__name__)

DISCOVER_INTERVAL_MIN = int(os.environ.get("ARTICLISER_DISCOVER_MINUTES", 180))
GENERATE_INTERVAL_MIN = int(os.environ.get("ARTICLISER_GENERATE_MINUTES", 15))
DISCOVER_BATCH = int(os.environ.get("ARTICLISER_DISCOVER_BATCH", 5))
ARXIV_QUERY = os.environ.get("ARTICLISER_ARXIV_QUERY", DEFAULT_QUERY)


def _recover_orphaned_jobs() -> int:
    """Return jobs left RUNNING by a previous crash to the queue.

    Safe because the worker is serial: anything still marked RUNNING at startup
    is by definition not running.
    """
    with session_scope() as session:
        orphans = session.exec(select(Job).where(Job.status == JobStatus.RUNNING)).all()
        for job in orphans:
            job.status = JobStatus.PENDING
            job.started_at = None
            session.add(job)
    if orphans:
        log.warning("requeued %d job(s) left running by a previous crash", len(orphans))
    return len(orphans)


def _next_discovery_offset() -> int:
    """Where a fixed-query run should resume paging.

    Only used on a cold start, when there is no corpus to learn from and
    discovery falls back to a fixed subject query. Without it every such run
    re-fetches the same newest N papers, sees them all as known, and queues
    nothing. Corpus-driven runs need no offset: what they return changes as the
    corpus changes.
    """
    with session_scope() as session:
        done = session.exec(
            select(Job).where(Job.kind == JobKind.DISCOVER, Job.status == JobStatus.DONE)
        ).all()
    return sum(int(job.payload.get("limit", DISCOVER_BATCH)) for job in done)


def _has_corpus() -> bool:
    with session_scope() as session:
        return any(
            s.raw_text and not s.identifier.startswith("fixture:")
            for s in session.exec(select(Source)).all()
        )


def queue_discovery() -> None:
    with session_scope() as session:
        pending = session.exec(
            select(Job).where(Job.kind == JobKind.DISCOVER, Job.status == JobStatus.PENDING)
        ).first()
        if pending is not None:
            log.info("discovery already queued; skipping this tick")
            return

        # A corpus-driven run takes no query and no offset -- it reads what has
        # been ingested. The fixed query is the cold-start path only.
        payload: dict = {"limit": DISCOVER_BATCH}
        if not _has_corpus():
            payload |= {"query": ARXIV_QUERY, "start": _next_discovery_offset()}
            log.info("no corpus yet; queueing a fixed-query discovery run")
        session.add(Job(kind=JobKind.DISCOVER, payload=payload))
    log.info("queued a discovery run")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    init_db()
    _recover_orphaned_jobs()

    worker = Worker()
    scheduler = BlockingScheduler()

    # An interval trigger's first fire is one interval from now, which is the
    # wanted behaviour here. Passing next_run_time=None to try to express that
    # does the opposite: APScheduler reads None as "paused" and the job never
    # runs at all.
    scheduler.add_job(
        queue_discovery,
        "interval",
        minutes=DISCOVER_INTERVAL_MIN,
        id="discover",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        worker.drain,
        "interval",
        minutes=GENERATE_INTERVAL_MIN,
        id="drain",
        max_instances=1,
        coalesce=True,
    )

    for received in (signal.SIGINT, signal.SIGTERM):
        signal.signal(received, lambda *_: scheduler.shutdown(wait=False))

    log.info(
        "worker up: discovery every %dmin (%s), queue drain every %dmin",
        DISCOVER_INTERVAL_MIN,
        ARXIV_QUERY,
        GENERATE_INTERVAL_MIN,
    )
    # Drain anything already waiting before settling into the schedule.
    worker.drain()
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
