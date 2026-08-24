"""The generation interface the pipeline codes against.

Narrow on purpose. The pipeline needs exactly three things from a text
generator, and the third -- `unload` -- is why this is a protocol with an
explicit lifecycle rather than a plain function: on a 12GB card shared with the
Windows host, *when* the weights leave VRAM is part of the contract, not an
implementation detail the garbage collector can be trusted with.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Summariser(Protocol):
    """A loadable text generator with a caller-controlled residency window."""

    name: str

    def load(self) -> None:
        """Make the model ready to generate. Idempotent."""

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        """Return only the continuation, with the prompt stripped."""

    def unload(self) -> None:
        """Release VRAM. Idempotent, and safe to call without a prior load."""


class GenerationError(RuntimeError):
    """Raised when a backend cannot produce usable output."""
