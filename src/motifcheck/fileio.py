"""File helpers shared by acquisition and dataset construction.

Acquisition renames every asset to ``<role>__<id>``, which drops the original
file extension. Anything that dispatches on ``.gz`` therefore breaks silently
and reads compressed bytes as text, so gzip is detected by magic byte here and
nowhere else.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import IO, Iterator

GZIP_MAGIC = b"\x1f\x8b"


def is_gzip(path: str | Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(2) == GZIP_MAGIC


def open_text(path: str | Path) -> IO[str]:
    """Open a text stream, transparently decompressing gzip."""
    path = Path(path)
    if is_gzip(path):
        return gzip.open(path, "rt")  # type: ignore[return-value]
    return open(path, "r")


def iter_lines(path: str | Path) -> Iterator[str]:
    with open_text(path) as fh:
        for line in fh:
            yield line.rstrip("\n")
