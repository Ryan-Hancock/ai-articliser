"""Check the pipeline's dependencies before a run rather than during one.

Every check here corresponds to something that has actually gone wrong on this
machine: Ollama unreachable across the WSL2 boundary, a model that overflows VRAM
and silently drops to a twelfth of its throughput, or the host holding weights
that a Linux-side `nvidia-smi` process list will not show.

Usage:
    uv run python scripts/doctor.py
"""

from __future__ import annotations

import sys

import httpx

from articliser.config import settings
from articliser.worker.gpu import free_vram_mb, ollama_base_url

OK, WARN, FAIL = "  ok  ", " warn ", " FAIL "


def _line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))


def check_gpu() -> bool:
    free = free_vram_mb()
    if free is None:
        _line(WARN, "GPU", "no CUDA device visible; highlighting and imagery will use the CPU")
        return True
    # SDXL needs ~5GB with component offload; the tagger ~1GB. Below this the
    # illustration stage will fail rather than run slowly.
    if free < 6000:
        _line(FAIL, "GPU", f"only {free}MB free -- something else is holding the card")
        return False
    _line(OK, "GPU", f"{free}MB free")
    return True


def check_ollama() -> bool:
    base = ollama_base_url()
    try:
        tags = httpx.get(f"{base}/api/tags", timeout=10).json().get("models", [])
    except Exception as exc:  # noqa: BLE001
        _line(FAIL, "Ollama", f"unreachable at {base} ({type(exc).__name__})")
        print("         On WSL2 the server usually runs on the Windows host; override "
              "with OLLAMA_HOST if the gateway guess is wrong.")
        return False

    names = {m.get("name", "") for m in tags}
    _line(OK, "Ollama", f"{base}, {len(names)} model(s)")

    if settings.ollama_model not in names:
        _line(FAIL, "model", f"{settings.ollama_model!r} not pulled")
        print(f"         run: ollama pull {settings.ollama_model}")
        return False

    entry = next(m for m in tags if m.get("name") == settings.ollama_model)
    size_gb = entry.get("size", 0) / 1e9
    free = free_vram_mb()
    _line(OK, "model", f"{settings.ollama_model} ({size_gb:.1f}GB)")
    # A model larger than free VRAM does not fail, it spills to system memory and
    # runs ~12x slower while disrupting everything else on the card.
    if free is not None and size_gb * 1024 > free:
        _line(
            WARN,
            "model fit",
            f"{size_gb:.1f}GB against {free/1024:.1f}GB free VRAM -- expect it to spill and "
            f"run several times slower",
        )
    return True


def check_taggers() -> bool:
    from huggingface_hub import model_info

    ok = True
    repos = (
        ("tagger", settings.zeroshot_model_id),
        ("embeddings", settings.embedding_model_id),
        ("images", settings.image_model_id),
    )
    for label, repo in repos:
        try:
            model_info(repo)
            _line(OK, label, repo)
        except Exception as exc:  # noqa: BLE001
            _line(WARN, label, f"{repo} not reachable ({type(exc).__name__}); cached copy may still work")
            ok = ok and True
    return ok


def main() -> int:
    print(f"generation=ollama  tagger={settings.tagger}  images={settings.image_model_id}\n")
    results = [check_gpu(), check_ollama(), check_taggers()]
    print()
    if all(results):
        print("ready: try `make generate SOURCE=path/to/paper.pdf`")
        return 0
    print("not ready -- see the FAIL lines above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
