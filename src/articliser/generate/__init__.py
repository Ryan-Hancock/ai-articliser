"""Article generation.

One backend: `OllamaSummariser`, which runs the model out-of-process on the host
GPU. The `Summariser` protocol it satisfies is still worth having -- it pins down
the load/generate/unload lifecycle the pipeline relies on for VRAM arbitration --
but there is no longer anything to choose between, so there is no factory.
"""

from __future__ import annotations

from articliser.generate.ollama import OllamaSummariser

__all__ = ["OllamaSummariser"]
