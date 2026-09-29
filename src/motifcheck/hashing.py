"""Canonical serialisation and hashing.

Every artifact that can influence a result is hashed through this module so
that protocol, code, data and checkpoints can be tied together at evaluation
time, as required by protocol sections 10 and 11.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

_CHUNK = 1 << 20


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def hash_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_config(config: dict) -> str:
    return hash_obj(config)


def stable_hash_int(*parts: object, seed: int = 0, nbits: int = 64) -> int:
    """Deterministic integer hash.

    Python's built-in ``hash`` is salted per process and must never appear in
    anything that determines a split. This uses BLAKE2b, which is stable
    across processes, machines and interpreter versions.
    """
    key = "\x1f".join(str(p) for p in parts).encode()
    digest = hashlib.blake2b(key, digest_size=nbits // 8, salt=seed.to_bytes(8, "big")[:8] if seed else b"\x00" * 8)
    return int.from_bytes(digest.digest(), "big")


def stable_unit_interval(*parts: object, seed: int = 0) -> float:
    """Deterministic value in [0, 1) used for split and sampling decisions."""
    return stable_hash_int(*parts, seed=seed, nbits=64) / float(1 << 64)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> str:
    """Write JSON and return the file's SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, sort_keys=True, default=str) + "\n")
    return hash_file(path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_yaml(path: str | Path, obj: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False, default_flow_style=False))
    return hash_file(path)


def append_jsonl(path: str | Path, record: dict) -> str:
    """Append-only log. Existing content is never rewritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(canonical_json(record) + "\n")
    return hash_file(path)


def iter_lines(path: str | Path) -> Iterable[str]:
    with open(path) as fh:
        for line in fh:
            yield line.rstrip("\n")
