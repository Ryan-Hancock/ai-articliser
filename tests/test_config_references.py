"""Every `settings.X` in the codebase must exist on Settings.

Added after removing the AirLLM and FLUX backends broke `scripts/doctor.py`:
it still read `settings.backend`, and nothing caught it. Attribute access on a
dataclass is invisible to pyflakes, and doctor.py has no unit test of its own,
so the failure surfaced only when the script was run by hand.

Static, so it needs no imports of the modules it checks -- a script that fails at
import time for an unrelated reason still gets scanned.
"""

from __future__ import annotations

import ast
from pathlib import Path

from articliser.config import Settings, settings

ROOTS = ("src", "scripts")


def _settings_attributes() -> set[str]:
    return {name for name in dir(settings) if not name.startswith("_")}


def _referenced() -> list[tuple[Path, int, str]]:
    known_names = {"settings"}
    found: list[tuple[Path, int, str]] = []
    for root in ROOTS:
        for path in Path(root).rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in known_names
                ):
                    found.append((path, node.lineno, node.attr))
    return found


def test_every_settings_reference_resolves():
    known = _settings_attributes()
    dangling = [
        f"{path}:{line} -> settings.{attr}"
        for path, line, attr in _referenced()
        if attr not in known
    ]
    assert not dangling, "settings attributes referenced but not defined:\n" + "\n".join(dangling)


def test_the_scan_actually_finds_references():
    # A guard on the guard: if the AST walk silently stopped matching, the test
    # above would pass vacuously.
    assert len(_referenced()) > 10


def test_settings_is_constructible_without_environment():
    # Every field needs a default; a required one would break `make serve` on a
    # fresh checkout rather than at the point of use.
    assert Settings().data_dir is not None
