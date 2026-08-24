"""Pull the cited work's title out of a raw reference entry.

Done with the LLM rather than a regex because the three citation styles in the
corpus put the title in three different places and mark none of them: IEEE wraps
it in smart quotes, Elsevier leaves it bare between the author list and an
abbreviated venue, and numbered styles run it into the journal name. A parser
that handles all three is a parser that mostly handles none.

Batched so the cost stays bounded -- one call per twenty entries, and only for
the top-ranked citations rather than all 500-odd in a corpus.
"""

from __future__ import annotations

import json
import logging

import httpx

from articliser.config import settings
from articliser.worker.gpu import ollama_base_url

log = logging.getLogger(__name__)

BATCH_SIZE = 20

_SCHEMA = {
    "type": "object",
    "properties": {
        "titles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"n": {"type": "integer"}, "title": {"type": "string"}},
                "required": ["n", "title"],
            },
        }
    },
    "required": ["titles"],
}

_PROMPT = """Extract the title of the work cited in each reference below.

Rules:
- Return the title only: no authors, no venue, no year, no page numbers.
- If an entry is a web page or has no identifiable paper title, return "".
- Keep the numbering.

{entries}"""


def extract_titles(entries: list[str]) -> list[str]:
    """Return one title per entry, "" where none could be identified."""
    if not entries:
        return []

    titles: list[str] = []
    for start in range(0, len(entries), BATCH_SIZE):
        batch = entries[start : start + BATCH_SIZE]
        numbered = "\n".join(f"{i + 1}. {e[:400]}" for i, e in enumerate(batch))
        try:
            response = httpx.post(
                f"{ollama_base_url()}/api/chat",
                timeout=settings.ollama_timeout_s,
                json={
                    "model": settings.ollama_model,
                    "messages": [{"role": "user", "content": _PROMPT.format(entries=numbered)}],
                    "format": _SCHEMA,
                    "think": False,
                    "options": {"temperature": 0, "num_ctx": settings.ollama_num_ctx},
                    "stream": False,
                    "keep_alive": settings.ollama_keep_alive,
                },
            )
            response.raise_for_status()
            payload = json.loads(response.json()["message"]["content"])
            by_index = {int(item["n"]): str(item["title"]).strip() for item in payload["titles"]}
        except Exception as exc:  # noqa: BLE001 - a failed batch costs titles, not the run
            log.warning("title extraction failed for a batch of %d: %s", len(batch), exc)
            by_index = {}
        titles.extend(by_index.get(i + 1, "") for i in range(len(batch)))

    return titles
