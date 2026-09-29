"""Protocol configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .hashing import hash_config


@dataclass(frozen=True)
class Protocol:
    raw: dict
    path: Path
    sha256: str

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # Convenience accessors used throughout the pipeline -------------------
    @property
    def window_length(self) -> int:
        return int(self.raw["task"]["window_length"])

    @property
    def seeds(self) -> list[int]:
        return [int(s) for s in self.raw["optimization"]["seeds"]]

    @property
    def amendments(self) -> list[dict]:
        return self.raw.get("amendments", [])

    def section(self, name: str) -> dict:
        return self.raw[name]


_REQUIRED_TOP_LEVEL = [
    "protocol_version",
    "amendments",
    "task",
    "sources",
    "candidates",
    "splits",
    "models",
    "tinycnn",
    "optimization",
    "intervention",
    "evaluation",
    "motif_audit",
    "attribution",
    "statistics",
    "primary_comparison",
]


def load_protocol(path: str | Path = "configs/protocol.yaml") -> Protocol:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())

    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in raw]
    if missing:
        raise ValueError(f"protocol {path} is missing required sections: {missing}")

    # The plan forbids silently combining assemblies (section 4.1).
    assemblies = {src.get("assembly") for src in raw["sources"].values() if src.get("assembly")}
    if assemblies != {"GRCh38"}:
        raise ValueError(
            f"all sources must share one assembly; found {sorted(a for a in assemblies if a)}"
        )

    # A declared parameter count that disagrees with the architecture is a
    # protocol error, not something to discover after training.
    cfg = raw["tinycnn"]
    n_expected = _predicted_param_count(cfg)
    if int(cfg["expected_params"]) != n_expected:
        raise ValueError(
            f"tinycnn.expected_params={cfg['expected_params']} but architecture implies "
            f"{n_expected}; architecture and declared count must agree before modeling"
        )

    return Protocol(raw=raw, path=path, sha256=hash_config(raw))


def _predicted_param_count(cfg: dict) -> int:
    """Analytic parameter count for the section 6.1 architecture."""
    cin = int(cfg["in_channels"])
    ch = int(cfg["channels"])
    stack = cfg["conv_stack"]
    total = 0
    prev = cin
    for layer in stack:
        k = int(layer["kernel"])
        total += ch * prev * k + ch          # conv weights + bias
        prev = ch
    pooled = 2 * ch                          # mean ++ max
    total += int(cfg["hidden"]) * pooled + int(cfg["hidden"])   # Linear + bias
    total += 1 * int(cfg["hidden"]) + 1                          # Linear 32->1 + bias
    return total


def predicted_receptive_field(cfg: dict) -> int:
    """Receptive field of the dilated stack, in nucleotides."""
    rf = 1
    prev = int(cfg["in_channels"])
    for layer in cfg["conv_stack"]:
        k = int(layer["kernel"])
        d = int(layer["dilation"])
        rf += (k - 1) * d
        prev = int(cfg["channels"])
    return rf
