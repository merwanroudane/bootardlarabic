"""Lightweight progress reporting. شريط التقدم."""
from __future__ import annotations

import sys


class ProgressBar:
    """Minimal text progress bar (no hard dependency on tqdm)."""

    def __init__(self, total: int, label: str = "البوتستراب", enabled: bool = True, width: int = 40):
        self.total = max(int(total), 1)
        self.label = label
        self.enabled = enabled
        self.width = width
        self._last = -1

    def update(self, i: int) -> None:
        if not self.enabled:
            return
        frac = (i + 1) / self.total
        filled = int(frac * self.width)
        if filled == self._last and (i + 1) != self.total:
            return
        self._last = filled
        bar = "#" * filled + " " * (self.width - filled)
        sys.stdout.write(f"\r{self.label}: [{bar}] {int(frac * 100)}%")
        sys.stdout.flush()
        if (i + 1) == self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()
