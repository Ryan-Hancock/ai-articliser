"""GPU arbitration for the sequential pipeline.

Two models pass through this machine's 12GB card in a run -- the sentence tagger
and the image model -- while the generator runs on the host GPU inside Ollama.
None may be resident at the same time. That would be a simple lock, except for
one thing this WSL2 guest does that a normal Linux box does not: it shares the
physical card with the Windows host, where Ollama's weights appear in *no*
Linux-side process list. The semantic-highlighting-slm project lost an overnight
run to exactly this (docs/findings.md), so the check here reads the driver's
aggregate free memory rather than summing per-process usage, and asks the host to
unload before allocating instead of waiting out an idle timeout.

The generation stage is the exception and passes `release_host=False`: there, the
host's loaded model is the thing doing the work.
"""

from __future__ import annotations

import gc
import logging
import os
import subprocess
import time
from contextlib import contextmanager

import httpx

log = logging.getLogger(__name__)

_OLLAMA_TIMEOUT_S = 10


def _detect_wsl_gateway() -> str:
    """The Windows host's IP, which is the WSL2 default gateway under NAT."""
    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return result.stdout.split()[2]
    except Exception:
        return "127.0.0.1"


def ollama_base_url() -> str:
    host = os.environ.get("OLLAMA_HOST")
    if host:
        return host if host.startswith("http") else f"http://{host}"
    return f"http://{_detect_wsl_gateway()}:11434"


def release_host_models() -> list[str]:
    """Ask Ollama on the host to drop anything it currently holds in VRAM.

    Returns the models it was holding. A missing or unreachable Ollama is not an
    error -- most machines running this won't have one -- so failures here are
    logged and swallowed rather than aborting the pipeline.
    """
    base = ollama_base_url()
    released: list[str] = []
    try:
        with httpx.Client(timeout=_OLLAMA_TIMEOUT_S) as client:
            loaded = client.get(f"{base}/api/ps").json().get("models", [])
            for entry in loaded:
                name = entry.get("name") or entry.get("model")
                if not name:
                    continue
                # keep_alive=0 unloads immediately; an empty prompt means this
                # costs nothing but the unload itself.
                client.post(
                    f"{base}/api/generate",
                    json={"model": name, "prompt": "", "keep_alive": 0},
                )
                released.append(name)
    except Exception as exc:  # noqa: BLE001 - an absent Ollama is the common case
        log.debug("could not reach Ollama at %s (%s)", base, exc)
        return []

    if released:
        log.info("asked host Ollama to unload: %s", ", ".join(released))
        time.sleep(2)  # the unload is asynchronous on Ollama's side
    return released


def free_vram_mb() -> int | None:
    """Free VRAM as the driver reports it, or None without CUDA.

    `torch.cuda.mem_get_info` is the right call rather than `memory_allocated`:
    it reports the device total, which includes allocations made by the Windows
    host that this guest cannot otherwise see.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, _total = torch.cuda.mem_get_info()
        return free_bytes // (1024 * 1024)
    except Exception as exc:  # noqa: BLE001
        log.debug("could not read VRAM (%s)", exc)
        return None


def empty_cache() -> None:
    """Return this process's cached blocks to the driver.

    gc first: torch only frees a tensor's memory once Python has dropped the last
    reference, so emptying the cache before collecting reclaims noticeably less.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        log.debug("could not empty CUDA cache (%s)", exc)


def ensure_vram(required_mb: int, *, release_host: bool = True) -> None:
    """Raise unless `required_mb` is free, after trying to reclaim.

    Deliberately raises rather than proceeding: an OOM part-way through a
    multi-hour generation costs the whole run, and failing before allocating
    leaves the job queue in a state the worker can retry.
    """
    empty_cache()
    available = free_vram_mb()
    if available is None:
        log.warning("no CUDA device visible; running on CPU")
        return

    if available < required_mb and release_host:
        release_host_models()
        empty_cache()
        available = free_vram_mb() or 0

    if available < required_mb:
        raise RuntimeError(
            f"need {required_mb}MB free VRAM, only {available}MB available. "
            f"On this WSL2 setup the usual cause is the Windows host holding "
            f"weights that don't appear in the Linux-side nvidia-smi process "
            f"list -- check Ollama at {ollama_base_url()}."
        )


@contextmanager
def gpu_stage(name: str, required_mb: int, *, release_host: bool = True):
    """Bracket one model's residency: preflight, run, then hand the card back.

    `release_host=False` for a stage whose work *is* the host's loaded model --
    asking Ollama to unload immediately before generating through it would be
    self-defeating.
    """
    log.info("gpu stage %s: preflight (need %dMB)", name, required_mb)
    ensure_vram(required_mb, release_host=release_host)
    started = time.perf_counter()
    try:
        yield
    finally:
        empty_cache()
        log.info(
            "gpu stage %s: released after %.1fs (%sMB free)",
            name,
            time.perf_counter() - started,
            free_vram_mb(),
        )
