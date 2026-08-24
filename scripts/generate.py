"""Run the full pipeline against one PDF, bypassing the job queue.

The queue is how articles normally get made; this is the version you run while
developing, when you want the traceback in your terminal rather than in a Job
row's error column.

Usage:
    uv run python scripts/generate.py path/to/paper.pdf [--no-image] [--no-highlight]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from articliser.db.session import init_db, session_scope
from articliser.generate.pipeline import ArticlePipeline, source_for_pdf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--no-image", action="store_true", help="skip the image stage")
    parser.add_argument("--no-highlight", action="store_true", help="skip the CRF stage")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.pdf.exists():
        print(f"no such file: {args.pdf}", file=sys.stderr)
        return 2

    init_db()
    started = time.perf_counter()
    with session_scope() as session:
        source = source_for_pdf(session, args.pdf)
        result = ArticlePipeline().run(
            session,
            source,
            illustrate=not args.no_image,
            highlight=not args.no_highlight,
        )

    print(
        f"\npublished /article/{result.slug}\n"
        f"  spans:       {result.span_count}\n"
        f"  illustrated: {result.illustrated}\n"
        f"  wall time:   {time.perf_counter() - started:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
