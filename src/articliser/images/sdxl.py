"""SDXL-Turbo illustrator: the whole model stays on the GPU.

Chosen over FLUX.1-dev, the original plan, which cannot run on this machine
without thrashing. Its weights are ~33GB in bf16 (an 11.9B transformer
plus a 4.7B T5 encoder), and `enable_sequential_cpu_offload` holds those in
*system* RAM while paging submodules to the GPU. Against 13GB of usable RAM that
means swapping, and a single 1024x768 image did not finish in ten minutes. See
docs/findings.md, Finding 9.

SDXL-Turbo is ~6.6GB in fp16, which fits in this card's VRAM outright, and its
distilled schedule needs 1-4 steps instead of 28. The trade is real -- FLUX
renders better images -- but an illustration that costs a minute of thrashing is
not worth having on a machine someone is trying to use.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from articliser.config import settings

log = logging.getLogger(__name__)

# Appended to whatever the LLM proposed, so artwork stays visually consistent
# across articles even as the generation prompt changes.
STYLE_SUFFIX = (
    "abstract editorial illustration, muted earth tones with a single accent colour, "
    "soft geometric shapes, generous negative space, matte texture, "
    "no text, no letters, no people, no logos"
)


class SDXLTurboIllustrator:
    name = "sdxl-turbo"

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or settings.image_model_id
        self._pipe = None

    def load(self) -> None:
        if self._pipe is not None:
            return

        import torch
        from diffusers import AutoPipelineForText2Image

        log.info("loading %s", self.model_id)
        started = time.perf_counter()
        pipe = AutoPipelineForText2Image.from_pretrained(
            # diffusers still takes torch_dtype here; `dtype` is silently ignored,
            # which loads fp32 weights and then fails the first matmul as
            # "mat1 and mat2 must have the same dtype".
            self.model_id, torch_dtype=torch.float16, variant="fp16"
        )
        # Straight onto the card. The entire point of this model over FLUX is that
        # it fits, so any offload here would give back the reason for choosing it.
        # Component-level offload: only the active component (text encoders, then
        # UNet, then VAE) sits on the card. Measured against keeping everything
        # resident, this is both lighter *and* faster -- 5.03GB peak vs 9.55GB,
        # 3.6s vs 5.2s per image -- because holding it all resident reserved
        # 12.3GB on a 12GB card and the allocator spent the difference on
        # fragmentation pressure.
        #
        # Worth being explicit that this is the same call that made FLUX unusable
        # (Finding 9). The difference is arithmetic, not technique: SDXL's ~7GB of
        # weights fit in system RAM with room to spare, where FLUX's ~33GB did not.
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        log.info("sdxl-turbo ready in %.1fs", time.perf_counter() - started)

    def illustrate(
        self,
        prompt: str,
        slug: str,
        *,
        steps: int = 4,
        width: int = 1024,
        height: int = 768,
        seed: int | None = None,
    ) -> str | None:
        """Generate a hero image; return its filename relative to the image dir.

        Returns None rather than raising: a missing illustration should cost the
        article its picture, not its publication.
        """
        if not prompt.strip():
            return None

        try:
            import torch

            self.load()
            assert self._pipe is not None
            settings.ensure_dirs()

            full_prompt = f"{prompt.strip().rstrip('.')}. {STYLE_SUFFIX}"
            generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None

            log.info("illustrating %s (%d steps)", slug, steps)
            started = time.perf_counter()
            image = self._pipe(
                prompt=full_prompt,
                num_inference_steps=steps,
                # Turbo is a distilled model trained without classifier-free
                # guidance; any value above 0 degrades it rather than sharpening it.
                guidance_scale=0.0,
                width=width,
                height=height,
                generator=generator,
            ).images[0]

            filename = f"{slug}.png"
            image.save(Path(settings.image_dir) / filename)
            log.info("illustrated %s in %.1fs", slug, time.perf_counter() - started)
            return filename
        except Exception as exc:  # noqa: BLE001
            log.warning("illustration failed for %s: %s", slug, exc)
            return None

    def unload(self) -> None:
        if self._pipe is None:
            return
        self._pipe = None
        from articliser.worker.gpu import empty_cache

        empty_cache()
