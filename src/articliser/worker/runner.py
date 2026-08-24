"""Drains the job queue, one job at a time.

Strictly serial, and that is the design rather than a simplification: the three
GPU stages already have to take the card in turn, so a second concurrent job
could only ever wait. Serial execution also means a crash leaves exactly one job
marked RUNNING, which is recoverable, instead of an indeterminate set.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import Session, col, select

from articliser.config import settings
from articliser.db.models import Article, Job, JobKind, JobStatus, Source, SourceKind
from articliser.db.session import session_scope
from articliser.generate.pipeline import ArticlePipeline
from articliser.discovery import STRATEGIES, Candidate, discover_candidates
from articliser.ingest.arxiv import download_pdf, fetch_by_id, fetch_papers, parse_arxiv_id

log = logging.getLogger(__name__)

DEFAULT_QUERY = "cat:cs.LG OR cat:cs.CL OR cat:cs.RO"


class Worker:
    def __init__(self, pipeline: ArticlePipeline | None = None) -> None:
        self.pipeline = pipeline or ArticlePipeline()

    # --- queue plumbing -----------------------------------------------------

    def claim_next(self, session: Session) -> Job | None:
        job = session.exec(
            select(Job)
            .where(Job.status == JobStatus.PENDING)
            .order_by(col(Job.created_at))
            .limit(1)
        ).first()
        if job is None:
            return None
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        session.add(job)
        session.commit()
        session.refresh(job)
        return job

    def drain(self, limit: int | None = None) -> int:
        """Run pending jobs until the queue empties. Returns how many ran."""
        processed = 0
        while limit is None or processed < limit:
            with session_scope() as session:
                job = self.claim_next(session)
                if job is None:
                    break
                job_id, kind, payload = job.id, job.kind, dict(job.payload)

            log.info("job %s: %s %s", job_id, kind.value, payload)
            try:
                self.run_job(kind, payload)
                status, error = JobStatus.DONE, None
            except Exception as exc:  # noqa: BLE001 - one bad job must not stop the queue
                log.exception("job %s failed", job_id)
                status, error = JobStatus.FAILED, f"{type(exc).__name__}: {exc}"

            with session_scope() as session:
                job = session.get(Job, job_id)
                if job is not None:
                    job.status = status
                    job.error = error
                    job.finished_at = datetime.now(timezone.utc)
                    session.add(job)
            processed += 1

        return processed

    # --- job kinds ----------------------------------------------------------

    def run_job(self, kind: JobKind, payload: dict) -> None:
        if kind is JobKind.DISCOVER:
            self.discover(
                limit=int(payload.get("limit", 6)),
                strategies=tuple(payload.get("strategies", STRATEGIES)),
                query=payload.get("query"),
                start=int(payload.get("start", 0)),
            )
        elif kind in (JobKind.INGEST, JobKind.GENERATE):
            source_id = payload.get("source_id")
            if source_id is None:
                raise ValueError("job payload has no source_id")
            self.generate(int(source_id))
        elif kind is JobKind.HIGHLIGHT:
            self.rehighlight(int(payload["article_id"]))
        elif kind is JobKind.ILLUSTRATE:
            self.reillustrate(int(payload["article_id"]))
        else:
            raise ValueError(f"unknown job kind: {kind}")

    def generate(self, source_id: int) -> None:
        with session_scope() as session:
            source = session.get(Source, source_id)
            if source is None:
                raise ValueError(f"no source {source_id}")

            # A URL source has no PDF yet; resolve it before the pipeline runs.
            if source.pdf_path is None:
                source.pdf_path = str(self._resolve_pdf(source))
                session.add(source)
                session.commit()
                session.refresh(source)

            existing = session.exec(
                select(Article).where(Article.source_id == source_id)
            ).first()
            if existing is not None:
                log.info("source %s already has article %s; skipping", source_id, existing.slug)
                return

            result = self.pipeline.run(session, source)
            log.info(
                "generated %s (%d spans, illustrated=%s)",
                result.slug,
                result.span_count,
                result.illustrated,
            )

    def _resolve_pdf(self, source: Source) -> Path:
        """Turn a URL/arXiv source into a PDF on disk."""
        settings.ensure_dirs()
        arxiv_id = parse_arxiv_id(source.identifier)
        if arxiv_id:
            paper = fetch_by_id(arxiv_id)
            if paper is None:
                raise ValueError(f"arXiv has no record of {arxiv_id}")
            if not source.title or source.title == source.identifier:
                source.title = paper.title
            return download_pdf(paper, settings.pdf_dir)

        raise ValueError(
            f"don't know how to fetch a PDF for {source.identifier!r}; "
            f"only arXiv links and uploaded PDFs are supported"
        )

    def discover(
        self,
        limit: int = 6,
        strategies: tuple[str, ...] = STRATEGIES,
        query: str | None = None,
        start: int = 0,
    ) -> int:
        """Find papers worth reading next and queue them. Returns how many.

        Driven by the corpus rather than a fixed subject query: what it cites,
        what it is about, and what the model thinks is adjacent to it. `query`
        forces the old fixed-subject behaviour, which is also the automatic
        fallback on a cold start when there is no corpus to learn from yet.
        """
        with session_scope() as session:
            known = {
                parse_arxiv_id(s.identifier)
                for s in session.exec(select(Source)).all()
                if parse_arxiv_id(s.identifier)
            }
            corpus = {
                s.title: s.raw_text
                for s in session.exec(select(Source)).all()
                if s.raw_text and not s.identifier.startswith("fixture:")
            }
            articles = [
                a
                for a in session.exec(select(Article)).all()
                if a.source and not a.source.identifier.startswith("fixture:")
            ]
            tag_lists = [a.tags for a in articles if a.tags]
            summaries = [f"{a.title}. {a.standfirst}" for a in articles]

        if query or not corpus:
            fixed = query or DEFAULT_QUERY
            log.info("no corpus to learn from; falling back to a fixed query: %s", fixed)
            papers = fetch_papers(query=fixed, max_results=limit, start=start)
            candidates = [
                Candidate(paper, "fallback", f"fixed query: {fixed}", 1.0) for paper in papers
            ]
            candidates = [c for c in candidates if c.paper.arxiv_id not in known]
        else:
            candidates = discover_candidates(
                sources=corpus,
                tag_lists=tag_lists,
                summaries=summaries,
                limit=limit,
                strategies=strategies,
                known_arxiv_ids=known,
            )

        queued = 0
        with session_scope() as session:
            for candidate in candidates:
                if session.exec(
                    select(Source).where(Source.identifier == candidate.identifier)
                ).first():
                    continue
                source = Source(
                    kind=SourceKind.ARXIV,
                    identifier=candidate.identifier,
                    title=candidate.paper.title,
                )
                session.add(source)
                session.commit()
                session.refresh(source)
                # The reason rides on the job payload rather than a new Source
                # column: it is visible in the queue UI, and it avoids a schema
                # migration on a database that has no migration story yet.
                session.add(
                    Job(
                        kind=JobKind.GENERATE,
                        payload={
                            "source_id": source.id,
                            "via": candidate.strategy,
                            "why": candidate.reason[:200],
                        },
                    )
                )
                log.info(
                    "  queued [%s] %s -- %s",
                    candidate.strategy,
                    candidate.paper.arxiv_id,
                    candidate.paper.title[:64],
                )
                queued += 1
            session.commit()

        log.info("discovery: %d candidate(s), %d newly queued", len(candidates), queued)
        return queued

    def rehighlight(self, article_id: int) -> None:
        with session_scope() as session:
            article = session.get(Article, article_id)
            if article is None:
                raise ValueError(f"no article {article_id}")
            self.pipeline.highlight(session, article)

    def reillustrate(self, article_id: int) -> None:
        with session_scope() as session:
            article = session.get(Article, article_id)
            if article is None:
                raise ValueError(f"no article {article_id}")
            self.pipeline.illustrate(session, article)
