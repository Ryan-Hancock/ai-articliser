"""Persistence schema.

Four tables, matching the four stages a paper passes through: a Source is the
raw material (a PDF on disk, an arXiv entry, a submitted link), an Article is
what the LLM made of it, Spans are the CRF's rhetorical tags over that article's
body, and a Job is a unit of pending work for the offline worker.

Spans are stored rather than computed at render time on purpose: the CRF is
fast enough to run live, but keeping it out of the web process means the server
holds no models at all and the request path stays free of torch entirely.
"""

# No `from __future__ import annotations` here on purpose: it turns the
# Relationship annotations into plain strings, and SQLAlchemy then rejects
# list["Article"] as "a generic class as the argument to relationship()".
import enum
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, Text
from sqlmodel import Field, Relationship, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SourceKind(str, enum.Enum):
    PDF = "pdf"
    ARXIV = "arxiv"
    URL = "url"


class JobKind(str, enum.Enum):
    INGEST = "ingest"
    GENERATE = "generate"
    ILLUSTRATE = "illustrate"
    HIGHLIGHT = "highlight"
    DISCOVER = "discover"


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Source(SQLModel, table=True):
    __tablename__ = "source"

    id: int | None = Field(default=None, primary_key=True)
    kind: SourceKind
    title: str = ""
    # arXiv id, canonical URL, or original filename -- whatever identifies this
    # source upstream. Unique so re-discovery doesn't create duplicates.
    identifier: str = Field(index=True, unique=True)
    pdf_path: str | None = None
    raw_text: str | None = Field(default=None, sa_column=Column(Text))
    created_at: datetime = Field(default_factory=utcnow)

    articles: list["Article"] = Relationship(back_populates="source")


class Series(SQLModel, table=True):
    """An ordered run of articles from one long source, usually a book.

    A book is one Source but many Articles, which is the only place the schema
    departs from one-source-one-article. The series carries what they share --
    the book's title, its slug, the PDF it came from -- so an article only needs
    its position in it.
    """

    __tablename__ = "series"

    id: int | None = Field(default=None, primary_key=True)
    source_id: int | None = Field(default=None, foreign_key="source.id", index=True)
    slug: str = Field(index=True, unique=True)
    title: str
    description: str = ""
    # Sections the outline yielded, which is not the same as parts published: a
    # series is generated over multiple resumable runs, and "part 3 of 3" while
    # 54 remain is worse than no count at all.
    total_parts: int = 0
    created_at: datetime = Field(default_factory=utcnow)

    articles: list["Article"] = Relationship(back_populates="series")


class Article(SQLModel, table=True):
    __tablename__ = "article"

    id: int | None = Field(default=None, primary_key=True)
    source_id: int | None = Field(default=None, foreign_key="source.id", index=True)

    # Series membership. Null for the ordinary one-paper-one-article case.
    series_id: int | None = Field(default=None, foreign_key="series.id", index=True)
    series_index: int | None = None  # 1-based position, for ordering and "part N of M"
    series_part: str | None = None  # the section's own heading, e.g. "Rotation Matrices"
    series_chapter: str | None = None  # the enclosing chapter, for grouping the index

    slug: str = Field(index=True, unique=True)
    title: str
    standfirst: str = ""
    category: str = Field(default="Uncategorised", index=True)
    tags: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    body_md: str = Field(default="", sa_column=Column(Text))

    image_path: str | None = None
    image_prompt: str | None = None

    # MiniLM sentence embedding, stored as a plain float list so "related
    # articles" needs no vector store -- at this corpus size a numpy dot product
    # over every row is microseconds.
    embedding: list[float] | None = Field(default=None, sa_column=Column(JSON))

    reading_minutes: int = 1
    created_at: datetime = Field(default_factory=utcnow, index=True)

    source: Source | None = Relationship(back_populates="articles")
    series: Series | None = Relationship(back_populates="articles")
    spans: list["Span"] = Relationship(
        back_populates="article",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class Span(SQLModel, table=True):
    """A CRF-tagged character range over Article.body_md."""

    __tablename__ = "span"

    id: int | None = Field(default=None, primary_key=True)
    article_id: int = Field(foreign_key="article.id", index=True)
    start: int
    end: int
    label: str

    article: Article | None = Relationship(back_populates="spans")


class Job(SQLModel, table=True):
    __tablename__ = "job"

    id: int | None = Field(default=None, primary_key=True)
    kind: JobKind
    status: JobStatus = Field(default=JobStatus.PENDING, index=True)
    payload: dict = Field(default_factory=dict, sa_column=Column(JSON))
    error: str | None = Field(default=None, sa_column=Column(Text))
    created_at: datetime = Field(default_factory=utcnow, index=True)
    started_at: datetime | None = None
    finished_at: datetime | None = None
