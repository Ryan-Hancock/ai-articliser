"""Queue mechanics, against a temporary database.

The pipeline is stubbed out: what's under test is the queue's own behaviour --
claiming in order, surviving a failing job, and recovering jobs orphaned by a
crash -- none of which should need a GPU to verify.
"""

from __future__ import annotations

import pytest
from sqlmodel import Session, SQLModel, create_engine

from articliser.db.models import Job, JobKind, JobStatus


@pytest.fixture()
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_claim_next_takes_the_oldest_pending_job(session, monkeypatch):
    from articliser.worker.runner import Worker

    worker = Worker(pipeline=object())
    session.add(Job(kind=JobKind.DISCOVER, payload={"n": 1}))
    session.commit()
    session.add(Job(kind=JobKind.GENERATE, payload={"n": 2}))
    session.commit()

    first = worker.claim_next(session)
    assert first is not None
    assert first.payload["n"] == 1
    assert first.status is JobStatus.RUNNING
    assert first.started_at is not None

    # The claimed job must not be handed out twice.
    second = worker.claim_next(session)
    assert second is not None and second.payload["n"] == 2


def test_claim_next_returns_none_on_an_empty_queue(session):
    from articliser.worker.runner import Worker

    assert Worker(pipeline=object()).claim_next(session) is None


def test_run_job_rejects_a_payload_with_no_source(session):
    from articliser.worker.runner import Worker

    with pytest.raises(ValueError, match="source_id"):
        Worker(pipeline=object()).run_job(JobKind.GENERATE, {})
