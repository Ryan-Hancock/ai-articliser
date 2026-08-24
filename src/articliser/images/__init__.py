"""Hero image generation.

One implementation: `SDXLTurboIllustrator`. FLUX.1-dev was the original choice
and was removed rather than left selectable -- it cannot run on this machine
(docs/findings.md, Finding 9), and keeping the option would have meant a config
flag whose only effect was to re-download 38GB and then swap.
"""

from __future__ import annotations

from articliser.images.sdxl import SDXLTurboIllustrator

__all__ = ["SDXLTurboIllustrator"]
