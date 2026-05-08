"""Filter out minigrid's 'Sampling rejected' chatter from stdout.

minigrid's BabyAI levels print messages like
    Sampling rejected: unreachable object at (1, 2)
to stdout every time the level-generator retries a random object placement
that ends up unreachable. These show up many times per episode and bury
our actual training/eval logs.

Calling `install_minigrid_noise_filter()` once at script start wraps
sys.stdout with a line-buffered writer that drops any line containing
'Sampling rejected'. Everything else passes through unchanged.
"""

from __future__ import annotations

import sys
from typing import TextIO


class _MinigridNoiseFilter:
    """File-like wrapper that drops lines matching the noisy minigrid prints.

    Buffers partial writes until a newline arrives, then either emits or
    drops the completed line based on the substring match. Any pre-newline
    flush() also flushes whatever's buffered, EXCEPT when the partial line
    looks like the start of a noise line (rare; prevents the noise from
    sneaking through on a forced flush).
    """

    _PATTERN = "Sampling rejected"

    def __init__(self, target: TextIO):
        self._target = target
        self._buffer = ""

    def write(self, s: str) -> int:
        self._buffer += s
        emitted = 0
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if self._PATTERN not in line:
                emitted += self._target.write(line + "\n")
        return emitted + len(self._buffer)  # report bytes consumed, not emitted

    def flush(self) -> None:
        if self._buffer and self._PATTERN not in self._buffer:
            self._target.write(self._buffer)
            self._buffer = ""
        self._target.flush()

    # Pass through attributes like .isatty / .fileno so libraries that
    # introspect stdout still see the underlying terminal.
    def __getattr__(self, name):
        return getattr(self._target, name)


def install_minigrid_noise_filter() -> None:
    """Wrap BOTH sys.stdout and sys.stderr to drop minigrid's
    'Sampling rejected' lines. Some minigrid versions print to stderr
    (depending on level class), so we cover both. Idempotent."""
    if not isinstance(sys.stdout, _MinigridNoiseFilter):
        sys.stdout = _MinigridNoiseFilter(sys.stdout)
    if not isinstance(sys.stderr, _MinigridNoiseFilter):
        sys.stderr = _MinigridNoiseFilter(sys.stderr)
