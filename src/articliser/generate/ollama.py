"""Ollama backend: generation on the host GPU, out of this process entirely.

Chosen as the default because of a constraint the other options can't meet --
generating articles without tying up the machine. Three properties do that work:

- **It isn't our process.** The weights live in Ollama's address space on the
  Windows host, so the worker never holds gigabytes of VRAM or system RAM, and
  killing the worker can't strand any.
- **VRAM comes back on demand.** `unload()` posts `keep_alive: 0`, which frees
  the card immediately rather than after Ollama's idle timeout. That is what lets
  the illustration stage have the whole GPU a second later.
- **It finishes.** A 799-word article takes ~23s. The approach this replaced --
  streaming a 32B model's layers from disk -- took 6.8 hours for the same article
  (docs/findings.md, Finding 7).

The model choice matters more than it looks: a model that overflows VRAM is both
slow *and* disruptive, because the spill contends with everything else on the
card. Measured on this 12GB machine, `qwen3.5:latest` keeps all 7.6GB resident at
64 tok/s, while the task-specific Paper-Summarizer-14B needs 20.3GB, spills, and
drops to 5.4 tok/s. The smaller model wins on both counts.
"""

from __future__ import annotations

import json
import logging
import time

import httpx

from articliser.config import settings
from articliser.generate.backend import GenerationError
from articliser.generate.prompts import ArticleDraft
from articliser.worker.gpu import ollama_base_url

log = logging.getLogger(__name__)


class OllamaSummariser:
    """Summariser backed by an Ollama server, usually on the WSL2 host."""

    # Tells the pipeline not to ask the host to drop its models before this stage
    # -- for this backend, the host's loaded model *is* the generator.
    manages_own_vram = True

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or settings.ollama_model
        self.base_url = (base_url or ollama_base_url()).rstrip("/")
        self.name = f"ollama:{self.model}"

    # --- lifecycle ----------------------------------------------------------

    def load(self) -> None:
        """Warm the model so the first real call isn't paying for a cold load.

        An empty prompt makes Ollama load the weights and return immediately.
        """
        try:
            httpx.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": "", "keep_alive": settings.ollama_keep_alive},
                timeout=settings.ollama_timeout_s,
            ).raise_for_status()
        except httpx.HTTPError as exc:
            raise GenerationError(
                f"could not reach Ollama at {self.base_url} ({exc}). "
                f"Is it running, and is {self.model!r} pulled?"
            ) from exc

    def unload(self) -> None:
        """Free the host's VRAM now rather than at Ollama's idle timeout.

        Deliberately best-effort: failing to unload should not fail an article
        that has already been generated. The next GPU stage's preflight will
        catch it if the memory really didn't come back.
        """
        try:
            httpx.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": "", "keep_alive": 0},
                timeout=settings.ollama_timeout_s,
            )
            time.sleep(1)  # the unload is asynchronous on Ollama's side
        except httpx.HTTPError as exc:
            log.debug("could not unload %s (%s)", self.model, exc)

    # --- generation ---------------------------------------------------------

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """Generate an article as JSON, constrained by the ArticleDraft schema.

        Ollama enforces the schema during decoding, so the response is valid JSON
        with the right keys by construction. `parse_draft`'s repair logic still
        runs behind it -- it costs nothing, and it is what would catch a server
        too old to honour `format`.
        """
        budget = max_new_tokens or settings.max_new_tokens
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "format": _draft_schema(),
            # Qwen3-family models emit a reasoning trace by default. It would be
            # discarded anyway, and it is charged at the same tokens/sec as the
            # article.
            "think": False,
            "options": {
                "temperature": settings.generation_temperature,
                "num_predict": budget,
                # Explicit, because the default context is smaller than the
                # evidence bundle and the overflow would be silently truncated
                # from the *front* -- dropping the instructions, not the paper.
                "num_ctx": settings.ollama_num_ctx,
            },
            "stream": False,
            "keep_alive": settings.ollama_keep_alive,
        }

        log.info("generating via %s (%d prompt chars, budget %d)", self.name, len(prompt), budget)
        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat", json=payload, timeout=settings.ollama_timeout_s
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            raise GenerationError(f"Ollama request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise GenerationError(f"Ollama returned a non-JSON envelope: {exc}") from exc

        elapsed = time.perf_counter() - started
        generated = body.get("eval_count") or 0
        log.info(
            "generated %d tokens in %.1fs (%.1f tok/s)",
            generated,
            elapsed,
            generated / elapsed if elapsed else 0.0,
        )

        content = (body.get("message") or {}).get("content", "")
        if not content.strip():
            raise GenerationError("Ollama returned an empty message")
        return content


def _draft_schema() -> dict:
    """The ArticleDraft schema, tightened for constrained decoding.

    Pydantic marks a field optional whenever it has a default, which lets the
    model skip it -- observed in practice as articles arriving with an empty tag
    list. Everything is required here so the decoder has to produce all six.
    """
    schema = ArticleDraft.model_json_schema()
    schema["required"] = sorted(schema.get("properties", {}))
    tags = schema.get("properties", {}).get("tags")
    if isinstance(tags, dict):
        tags["minItems"] = 3
        tags["maxItems"] = 5
    return schema
