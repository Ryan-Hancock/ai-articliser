"""Source -> published article.

The stage order is a VRAM schedule, not a logical one. Each model is loaded,
used, and explicitly unloaded before the next is touched, because none of the
three fit alongside each other on a 12GB card. Cheapest-first would be the
natural reading order; instead the LLM goes first because it is the only stage
whose failure means there is no article at all, and the two decorating stages
(highlighting, illustration) follow so that their failure downgrades the result
rather than losing it.

Every stage after generation is best-effort for that reason: a published article
with no picture and no highlights is a worse article, but a lost generation is
hours of wall time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from sqlmodel import Session, col, select

from articliser.config import settings
from articliser.db.models import Article, Series, Source, SourceKind, Span
from articliser.generate.ollama import OllamaSummariser
from articliser.generate.backend import GenerationError, Summariser
from articliser.generate.embeddings import embed
from articliser.generate.prompts import (
    ArticleDraft,
    build_book_prompt,
    build_prompt,
    build_repair_prompt,
    parse_draft,
)
from articliser.highlighting import build_tagger
from articliser.images.sdxl import SDXLTurboIllustrator
from articliser.ingest.book import (
    BookSection,
    book_title,
    extract_section_text,
    publishable_sections,
)
from articliser.ingest.pdf import build_evidence_bundle, extract_text, guess_title
from articliser.text import reading_minutes, strip_markdown, unique_slug
from articliser.worker.gpu import gpu_stage

log = logging.getLogger(__name__)

# Rough residency ceilings used for the VRAM preflight. Deliberately generous:
# refusing to start is cheap, and an OOM three hours into a generation is not.
LLM_VRAM_MB = 4000
TAGGER_VRAM_MB = 2500
# SDXL-Turbo is ~7GB fp16 and lives on the card for the whole stage.
IMAGE_VRAM_MB = 8000


@dataclass
class SeriesResult:
    series_id: int
    slug: str
    published: list[str]
    skipped: list[str]


@dataclass
class PipelineResult:
    article_id: int
    slug: str
    illustrated: bool
    span_count: int


class ArticlePipeline:
    def __init__(
        self,
        summariser: Summariser | None = None,
        tagger=None,
        illustrator=None,
    ) -> None:
        self.summariser = summariser or OllamaSummariser()
        self.tagger = tagger or build_tagger()
        self.illustrator = illustrator or SDXLTurboIllustrator()

    # --- ingest -------------------------------------------------------------

    def ingest(self, session: Session, source: Source) -> Source:
        """Populate `source.raw_text` from its PDF. Idempotent."""
        if source.raw_text:
            return source
        if not source.pdf_path:
            raise ValueError(f"source {source.id} has no PDF to ingest")

        path = Path(source.pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"missing PDF: {path}")

        log.info("extracting text from %s", path.name)
        source.raw_text = extract_text(path)
        if not source.title or source.title == path.stem:
            source.title = guess_title(source.raw_text)
        session.add(source)
        session.commit()
        session.refresh(source)
        return source

    # --- generation ---------------------------------------------------------

    def _generate(self, prompt: str) -> ArticleDraft:
        """Run the one expensive call, with a single repair retry."""
        owns_vram = getattr(self.summariser, "manages_own_vram", False)
        with gpu_stage("llm", 0 if owns_vram else LLM_VRAM_MB, release_host=not owns_vram):
            self.summariser.load()
            try:
                response = self.summariser.generate(prompt, settings.max_new_tokens)
                draft = parse_draft(response)
                if draft is None:
                    log.warning("response did not parse; attempting one repair pass")
                    response = self.summariser.generate(
                        build_repair_prompt(response), settings.max_new_tokens
                    )
                    draft = parse_draft(response)
            finally:
                self.summariser.unload()

        if draft is None:
            raise GenerationError(
                "model did not return usable JSON after a repair attempt; "
                "see the logged response for what it produced instead"
            )
        return draft

    def draft(self, source: Source) -> ArticleDraft:
        """Run the one expensive generation call, with a single repair retry."""
        if not source.raw_text:
            raise ValueError("cannot draft from a source with no extracted text")

        evidence = build_evidence_bundle(source.raw_text, settings.evidence_char_budget)
        return self._generate(build_prompt(source.title, evidence))

    # --- persistence --------------------------------------------------------

    def publish(self, session: Session, source: Source, draft: ArticleDraft) -> Article:
        taken = set(session.exec(select(col(Article.slug))).all())
        body = draft.body_md.strip()

        article = Article(
            source_id=source.id,
            slug=unique_slug(draft.title, taken),
            title=draft.title.strip(),
            standfirst=draft.standfirst.strip(),
            category=draft.category,
            tags=draft.tags,
            body_md=body,
            image_prompt=draft.image_prompt.strip() or None,
            reading_minutes=reading_minutes(body),
            embedding=embed(f"{draft.title}\n\n{strip_markdown(body)}"),
        )
        session.add(article)
        session.commit()
        session.refresh(article)
        return article

    # --- decoration (best-effort) -------------------------------------------

    def highlight(self, session: Session, article: Article) -> int:
        """Tag the body and replace any existing spans. Returns the span count."""
        try:
            with gpu_stage(self.tagger.name, TAGGER_VRAM_MB):
                try:
                    spans = self.tagger.tag(article.body_md)
                finally:
                    self.tagger.unload()
        except Exception as exc:  # noqa: BLE001
            log.warning("highlighting failed for %s: %s", article.slug, exc)
            return 0

        for existing in session.exec(select(Span).where(Span.article_id == article.id)).all():
            session.delete(existing)
        for start, end, label in spans:
            session.add(Span(article_id=article.id, start=start, end=end, label=label))
        session.commit()
        return len(spans)

    def illustrate(self, session: Session, article: Article) -> bool:
        if not article.image_prompt:
            return False
        try:
            with gpu_stage(self.illustrator.name, IMAGE_VRAM_MB):
                try:
                    filename = self.illustrator.illustrate(article.image_prompt, article.slug)
                finally:
                    self.illustrator.unload()
        except Exception as exc:  # noqa: BLE001
            log.warning("illustration stage failed for %s: %s", article.slug, exc)
            return False

        if filename is None:
            return False
        article.image_path = filename
        session.add(article)
        session.commit()
        return True

    # --- the whole thing ----------------------------------------------------

    def run(
        self,
        session: Session,
        source: Source,
        *,
        illustrate: bool = True,
        highlight: bool = True,
    ) -> PipelineResult:
        source = self.ingest(session, source)
        draft = self.draft(source)
        article = self.publish(session, source, draft)
        log.info("published %s", article.slug)

        span_count = self.highlight(session, article) if highlight else 0
        illustrated = self.illustrate(session, article) if illustrate else False

        return PipelineResult(
            article_id=article.id or 0,
            slug=article.slug,
            illustrated=illustrated,
            span_count=span_count,
        )


    # --- books --------------------------------------------------------------

    def run_book(
        self,
        session: Session,
        source: Source,
        *,
        illustrate: bool = True,
        highlight: bool = True,
        limit: int | None = None,
        skip_existing: bool = True,
    ) -> SeriesResult:
        """Turn one long PDF into a series, one article per outline section.

        Resumable by design: a 57-part book is an hour of generation, and a run
        that has to restart from zero after a failure at part 50 is not usable.
        Parts already published are skipped, so re-running finishes the job.
        """
        pdf_path = Path(source.pdf_path or "")
        if not pdf_path.exists():
            raise FileNotFoundError(f"missing PDF: {pdf_path}")

        sections = publishable_sections(pdf_path)
        if not sections:
            raise ValueError(f"{pdf_path.name} has no usable outline sections")

        series = self._series_for(session, source, pdf_path, len(sections))
        published: list[str] = []
        skipped: list[str] = []

        for section in sections:
            if limit is not None and len(published) >= limit:
                break

            existing = session.exec(
                select(Article).where(
                    Article.series_id == series.id, Article.series_index == section.index
                )
            ).first()
            if existing is not None and skip_existing:
                skipped.append(section.title)
                continue

            log.info(
                "part %d/%d: %s (p%d-%d)",
                section.index,
                len(sections),
                section.title,
                section.start_page,
                section.end_page,
            )
            try:
                article = self._publish_part(
                    session, source, series, section, pdf_path, len(sections)
                )
            except Exception as exc:  # noqa: BLE001 - one bad part must not end the book
                log.warning("part %d (%s) failed: %s", section.index, section.title, exc)
                skipped.append(section.title)
                continue

            published.append(article.slug)
            if highlight:
                self.highlight(session, article)
            if illustrate:
                self.illustrate(session, article)

        return SeriesResult(
            series_id=series.id or 0,
            slug=series.slug,
            published=published,
            skipped=skipped,
        )

    def _series_for(
        self, session: Session, source: Source, pdf_path: Path, total: int
    ) -> Series:
        series = session.exec(select(Series).where(Series.source_id == source.id)).first()
        if series is not None:
            # The outline can yield a different count than last time if the
            # section filters changed; the planned total should follow it.
            if series.total_parts != total:
                series.total_parts = total
                session.add(series)
                session.commit()
                session.refresh(series)
            return series

        title = book_title(pdf_path)
        taken = set(session.exec(select(col(Series.slug))).all())
        series = Series(
            source_id=source.id,
            slug=unique_slug(title, taken),
            title=title,
            total_parts=total,
            description=f"Generated from {pdf_path.name}",
        )
        session.add(series)
        session.commit()
        session.refresh(series)
        return series

    def _publish_part(
        self,
        session: Session,
        source: Source,
        series: Series,
        section: BookSection,
        pdf_path: Path,
        total: int,
    ) -> Article:
        text = extract_section_text(pdf_path, section)
        evidence = build_evidence_bundle(text, settings.evidence_char_budget)
        draft = self._generate(
            build_book_prompt(
                book=series.title,
                chapter=section.chapter,
                section=section.title,
                evidence=evidence,
                part=section.index,
                total=total,
            )
        )

        taken = set(session.exec(select(col(Article.slug))).all())
        body = draft.body_md.strip()
        article = Article(
            source_id=source.id,
            series_id=series.id,
            series_index=section.index,
            series_part=section.title,
            series_chapter=section.chapter,
            slug=unique_slug(draft.title, taken),
            title=draft.title.strip(),
            standfirst=draft.standfirst.strip(),
            category=draft.category,
            tags=draft.tags,
            body_md=body,
            image_prompt=draft.image_prompt.strip() or None,
            reading_minutes=reading_minutes(body),
            embedding=embed(f"{draft.title}\n\n{strip_markdown(body)}"),
        )
        session.add(article)
        session.commit()
        session.refresh(article)
        return article


def source_for_pdf(session: Session, pdf_path: Path) -> Source:
    """Find or create the Source row for a PDF already on disk."""
    identifier = f"file:{pdf_path.resolve()}"
    source = session.exec(select(Source).where(Source.identifier == identifier)).first()
    if source is None:
        source = Source(
            kind=SourceKind.PDF,
            identifier=identifier,
            pdf_path=str(pdf_path.resolve()),
            title=pdf_path.stem,
        )
        session.add(source)
        session.commit()
        session.refresh(source)
    return source
