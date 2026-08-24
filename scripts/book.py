"""Turn a book-length PDF into a series of articles, one per outline section.

Splits on the PDF's embedded outline: subchapters where a chapter has them,
the chapter itself where it does not. Front and back matter (forewords,
exercises, indexes) is dropped.

Resumable -- parts already published are skipped, so a run interrupted at part
40 of 57 is finished by running it again.

Usage:
    uv run python scripts/book.py MR-v2.pdf --dry-run   # show the split
    uv run python scripts/book.py MR-v2.pdf --limit 3   # first 3 parts only
    uv run python scripts/book.py MR-v2.pdf             # the whole book
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from articliser.db.session import init_db, session_scope
from articliser.generate.pipeline import ArticlePipeline, source_for_pdf
from articliser.ingest.book import looks_like_a_book, publishable_sections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="stop after N new parts")
    parser.add_argument("--dry-run", action="store_true", help="show the split, generate nothing")
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--no-highlight", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.pdf.exists():
        print(f"no such file: {args.pdf}")
        return 2
    if not looks_like_a_book(args.pdf):
        print(
            f"{args.pdf.name} has no embedded outline, or is too short to split. "
            f"Use `make generate SOURCE={args.pdf}` to treat it as a single paper."
        )
        return 2

    sections = publishable_sections(args.pdf)
    if args.dry_run:
        chapter = None
        for section in sections:
            if section.chapter != chapter:
                chapter = section.chapter
                print(f"\n  {chapter}")
            same = section.title == section.chapter
            print(
                f"    {section.index:>3}. p{section.start_page:>4}-{section.end_page:<4} "
                f"({section.page_count:>2}p) {'[whole chapter] ' if same else ''}{section.title[:56]}"
            )
        print(f"\n{len(sections)} part(s); roughly {len(sections) * 65 / 60:.0f} min to generate")
        return 0

    init_db()
    started = time.perf_counter()
    with session_scope() as session:
        source = source_for_pdf(session, args.pdf)
        result = ArticlePipeline().run_book(
            session,
            source,
            illustrate=not args.no_image,
            highlight=not args.no_highlight,
            limit=args.limit,
        )

    print(
        f"\nseries /series/{result.slug}\n"
        f"  published: {len(result.published)}\n"
        f"  skipped:   {len(result.skipped)}\n"
        f"  wall time: {time.perf_counter() - started:.0f}s"
    )
    if result.skipped:
        print(f"  skipped parts: {', '.join(result.skipped[:6])}"
              f"{' ...' if len(result.skipped) > 6 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
