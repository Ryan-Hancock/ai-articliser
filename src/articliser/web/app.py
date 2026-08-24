"""The reading site.

Deliberately model-free: this process imports no torch, loads no weights, and
does no generation. It reads articles the worker has already written and, for
submissions, appends a Job row for the worker to pick up. That separation is
what keeps page loads instant while a 32B model streams layers from disk in the
other process.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, col, func, select

from articliser.config import settings
from articliser.db.models import (
    Article,
    Job,
    JobKind,
    JobStatus,
    Series,
    Source,
    SourceKind,
    Span,
)
from articliser.db.session import get_session, init_db
from articliser.highlighting.schema import LABELS
from articliser.text import render_body_html, slugify

BASE_DIR = Path(__file__).resolve().parent

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # The worker normally creates the schema first, but the site has to come up
    # cleanly on a fresh checkout where nothing has run yet.
    init_db()
    yield


app = FastAPI(title="Articliser", lifespan=lifespan)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Directories must exist before StaticFiles is constructed, and mounts have to be
# registered at import time -- adding one inside a startup hook leaves it out of
# the router for requests that arrive first.
settings.ensure_dirs()
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
# Generated hero images live under data/, outside the package, so they are
# mounted separately from the site's own CSS.
app.mount("/media", StaticFiles(directory=str(settings.image_dir)), name="media")


def _template(request: Request, name: str, **context) -> HTMLResponse:
    return templates.TemplateResponse(request, name, {"labels": LABELS, **context})


@app.get("/", response_class=HTMLResponse)
def feed(
    request: Request,
    session: Session = Depends(get_session),
    category: str | None = None,
    q: str | None = None,
) -> HTMLResponse:
    statement = select(Article).order_by(col(Article.created_at).desc())
    if category:
        statement = statement.where(Article.category == category)
    if q:
        pattern = f"%{q}%"
        statement = statement.where(
            col(Article.title).ilike(pattern) | col(Article.standfirst).ilike(pattern)
        )
    articles = session.exec(statement.limit(60)).all()

    categories = session.exec(
        select(Article.category, func.count(col(Article.id)))
        .group_by(col(Article.category))
        .order_by(func.count(col(Article.id)).desc())
    ).all()

    return _template(
        request,
        "feed.html",
        articles=articles,
        categories=categories,
        active_category=category,
        query=q or "",
    )


@app.get("/article/{slug}", response_class=HTMLResponse)
def article_page(
    request: Request, slug: str, session: Session = Depends(get_session)
) -> HTMLResponse:
    article = session.exec(select(Article).where(Article.slug == slug)).first()
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found")

    spans = session.exec(select(Span).where(Span.article_id == article.id)).all()
    body_html = render_body_html(article.body_md, [(s.start, s.end, s.label) for s in spans])

    return _template(
        request,
        "article.html",
        article=article,
        body_html=body_html,
        has_spans=bool(spans),
        related=_related(session, article),
        **_series_context(session, article),
    )


def _series_context(session: Session, article: Article) -> dict:
    """Neighbouring parts, for the in-article series navigation.

    Previous and next are found by position rather than by id: parts can be
    generated out of order, or regenerated, and an id-ordered series would then
    read in the wrong sequence.
    """
    if article.series_id is None:
        return {"series": None, "series_total": 0, "prev_part": None, "next_part": None}

    parts = session.exec(
        select(Article)
        .where(Article.series_id == article.series_id)
        .order_by(col(Article.series_index))
    ).all()
    position = next(
        (i for i, part in enumerate(parts) if part.id == article.id), None
    )
    series = session.get(Series, article.series_id)
    return {
        "series": series,
        # The planned total, not the published count -- a series is built over
        # several runs and "part 3 of 3" with 54 still to come is wrong.
        "series_total": (series.total_parts if series and series.total_parts else len(parts)),
        "series_published": len(parts),
        "prev_part": parts[position - 1] if position else None,
        "next_part": parts[position + 1] if position is not None and position + 1 < len(parts) else None,
    }


@app.get("/series", response_class=HTMLResponse)
def series_index(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    rows = []
    for series in session.exec(select(Series).order_by(col(Series.created_at).desc())).all():
        parts = session.exec(
            select(Article)
            .where(Article.series_id == series.id)
            .order_by(col(Article.series_index))
        ).all()
        if parts:
            rows.append((series, parts))
    return _template(request, "series_index.html", rows=rows)


@app.get("/series/{slug}", response_class=HTMLResponse)
def series_page(
    request: Request, slug: str, session: Session = Depends(get_session)
) -> HTMLResponse:
    series = session.exec(select(Series).where(Series.slug == slug)).first()
    if series is None:
        raise HTTPException(status_code=404, detail="Series not found")

    parts = session.exec(
        select(Article)
        .where(Article.series_id == series.id)
        .order_by(col(Article.series_index))
    ).all()

    # Grouped by the book's own chapters, preserving reading order rather than
    # sorting: a dict keyed by chapter would reorder them.
    chapters: list[tuple[str, list[Article]]] = []
    for part in parts:
        name = part.series_chapter or ""
        if not chapters or chapters[-1][0] != name:
            chapters.append((name, []))
        chapters[-1][1].append(part)

    return _template(
        request, "series.html", series=series, parts=parts, chapters=chapters
    )


def _related(session: Session, article: Article, limit: int = 3) -> list[Article]:
    """Nearest neighbours by MiniLM cosine similarity, falling back to same-category.

    No vector store: at this corpus size a dot product over every row is
    microseconds, and adding one would be infrastructure without a problem.
    """
    others = session.exec(select(Article).where(col(Article.id) != article.id)).all()
    if not others:
        return []

    if article.embedding:
        scored = [
            (sum(a * b for a, b in zip(article.embedding, other.embedding)), other)
            for other in others
            if other.embedding and len(other.embedding) == len(article.embedding)
        ]
        if scored:
            scored.sort(key=lambda pair: pair[0], reverse=True)
            return [other for _, other in scored[:limit]]

    return [other for other in others if other.category == article.category][:limit]


@app.get("/submit", response_class=HTMLResponse)
def submit_form(request: Request) -> HTMLResponse:
    return _template(request, "submit.html")


@app.post("/submit/pdf")
async def submit_pdf(
    file: UploadFile, session: Session = Depends(get_session)
) -> RedirectResponse:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Expected a .pdf file")

    settings.ensure_dirs()
    identifier = f"upload:{slugify(Path(file.filename).stem)}"
    destination = settings.pdf_dir / f"{slugify(Path(file.filename).stem)}.pdf"
    destination.write_bytes(await file.read())

    _enqueue_source(session, SourceKind.PDF, identifier, str(destination), Path(file.filename).stem)
    return RedirectResponse(url="/jobs", status_code=303)


@app.post("/submit/url")
def submit_url(url: str = Form(...), session: Session = Depends(get_session)) -> RedirectResponse:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Expected an http(s) URL")

    # arXiv links are worth recognising rather than treating as generic URLs: the
    # id gives a stable identifier for dedupe and a direct PDF route.
    kind = SourceKind.ARXIV if "arxiv.org" in httpx.URL(url).host else SourceKind.URL
    _enqueue_source(session, kind, url, None, url)
    return RedirectResponse(url="/jobs", status_code=303)


def _enqueue_source(
    session: Session, kind: SourceKind, identifier: str, pdf_path: str | None, title: str
) -> None:
    source = session.exec(select(Source).where(Source.identifier == identifier)).first()
    if source is None:
        source = Source(kind=kind, identifier=identifier, pdf_path=pdf_path, title=title)
        session.add(source)
        session.commit()
        session.refresh(source)

    session.add(Job(kind=JobKind.INGEST, payload={"source_id": source.id}))
    session.commit()


@app.get("/jobs", response_class=HTMLResponse)
def jobs_page(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    return _template(request, "jobs.html", jobs=_recent_jobs(session))


@app.get("/jobs/rows", response_class=HTMLResponse)
def jobs_rows(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    """HTMX polling target -- returns just the table rows, not the page."""
    return _template(request, "_job_rows.html", jobs=_recent_jobs(session))


def _recent_jobs(session: Session, limit: int = 25) -> list[Job]:
    return list(
        session.exec(select(Job).order_by(col(Job.created_at).desc()).limit(limit)).all()
    )


@app.get("/healthz")
def healthz(session: Session = Depends(get_session)) -> dict:
    return {
        "status": "ok",
        "articles": len(session.exec(select(Article)).all()),
        "pending_jobs": len(
            session.exec(select(Job).where(Job.status == JobStatus.PENDING)).all()
        ),
    }
