"""Runtime settings, resolved once from the environment.

Everything that differs between "developing on this laptop" and "running the
overnight batch" lives here rather than being threaded through call sites --
in particular the model ids and the generation budget, which are the two knobs
most likely to be retuned once M2 produces real wall-time numbers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    # --- storage ---
    data_dir: Path = field(default_factory=lambda: _env_path("ARTICLISER_DATA", PROJECT_ROOT / "data"))

    # --- generation ---
    # Measured on a 12GB card: this keeps all 7.6GB resident at ~64 tok/s. Larger
    # models here are a false economy -- Paper-Summarizer-14B needs 20.3GB, spills
    # to system memory, and lands at 5.4 tok/s while disrupting everything else.
    ollama_model: str = field(
        default_factory=lambda: os.environ.get("ARTICLISER_OLLAMA_MODEL", "qwen3.5:latest")
    )
    # How long Ollama holds the weights between calls. Long enough to generate a
    # run of articles back to back, short enough that a crashed worker doesn't
    # leave the card occupied indefinitely.
    ollama_keep_alive: str = field(
        default_factory=lambda: os.environ.get("ARTICLISER_OLLAMA_KEEP_ALIVE", "5m")
    )
    # Must exceed the evidence bundle plus the instructions, or Ollama silently
    # truncates from the front and the model never sees its own instructions.
    ollama_num_ctx: int = field(default_factory=lambda: _env_int("ARTICLISER_OLLAMA_NUM_CTX", 8192))
    ollama_timeout_s: int = field(default_factory=lambda: _env_int("ARTICLISER_OLLAMA_TIMEOUT", 600))
    generation_temperature: float = field(
        default_factory=lambda: float(os.environ.get("ARTICLISER_TEMPERATURE", "0.4"))
    )

    # Char budget for the evidence bundle handed to the model. Must leave room
    # inside ollama_num_ctx for the instructions, or the overflow is truncated
    # from the front and the model loses them rather than losing the paper.
    evidence_char_budget: int = field(
        default_factory=lambda: _env_int("ARTICLISER_EVIDENCE_CHARS", 12000)
    )
    max_new_tokens: int = field(default_factory=lambda: _env_int("ARTICLISER_MAX_NEW_TOKENS", 2000))

    # --- highlighting ---
    # "zeroshot" (default) or "crf". See docs/findings.md for the comparison --
    # the CRF is kept because it is 30x faster, not because it is better here.
    tagger: str = field(default_factory=lambda: os.environ.get("ARTICLISER_TAGGER", "zeroshot"))
    zeroshot_model_id: str = field(
        default_factory=lambda: os.environ.get(
            "ARTICLISER_ZEROSHOT_MODEL", "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
        )
    )
    crf_model_repo: str = field(
        default_factory=lambda: os.environ.get(
            "ARTICLISER_CRF_REPO", "Rychanfox/semantic-highlighting-modernbert-crf"
        )
    )
    embedding_model_id: str = field(
        default_factory=lambda: os.environ.get(
            "ARTICLISER_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # --- imagery ---
    # Must fit in VRAM. FLUX.1-dev was the original choice and needs ~33GB of
    # weights held in system RAM under offload, against 13GB usable here.
    image_model_id: str = field(
        default_factory=lambda: os.environ.get("ARTICLISER_IMAGE_MODEL", "stabilityai/sdxl-turbo")
    )

    # --- GPU arbitration ---
    # This WSL2 guest shares the physical GPU with the Windows host, where Ollama
    # holds VRAM that never appears in the Linux-side nvidia-smi process list.
    # The worker refuses to start a GPU stage with less than this free.
    min_free_vram_mb: int = field(default_factory=lambda: _env_int("ARTICLISER_MIN_FREE_VRAM_MB", 9000))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "articliser.db"

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def image_dir(self) -> Path:
        return self.data_dir / "images"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.pdf_dir, self.image_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
