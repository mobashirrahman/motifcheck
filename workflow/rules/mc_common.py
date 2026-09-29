"""Shared paths, seeds and command helpers for every workflow rule."""

from __future__ import annotations

import os
from pathlib import Path

CONFIG = "configs/protocol.yaml"

# Interpreter used for every rule. Point PYTHON at the locked environment, e.g.
#   PYTHON=/path/to/env/bin/python snakemake --cores 8
PY = os.environ.get("PYTHON", "python")

DATA = Path("data")
EXTERNAL = DATA / "external"
PROCESSED = DATA / "processed"
MANIFEST = DATA / "source_manifest.json"
DATASET = PROCESSED / "dataset.parquet"
EXCLUSIONS = PROCESSED / "exclusions.parquet"
SPLIT_DATA = PROCESSED / "dataset_split.parquet"
TRAIN_BINS = PROCESSED / "train_bins.json"

RESULTS = Path("results")
TRAINING = RESULTS / "training"
BASELINES = RESULTS / "baselines"
AUDIT = RESULTS / "audit"
TEST = RESULTS / "test"

REPORTS = Path("reports")
GATES = REPORTS / "gates"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

SEEDS = [17, 29, 43, 71, 101]
CONDITIONS = ["C0", "C1"]

def checkpoint_paths() -> list[str]:
    """All ten checkpoints: two conditions x five paired seeds."""
    return [str(TRAINING / f"{c}_seed{s}.pt") for c in CONDITIONS for s in SEEDS]


def cli(cmd: str, *args, config: str | None = None) -> str:
    """Build one command from the frozen CLI contract.

    The rule supplies --config explicitly so the executed protocol file is
    visible in the job log; nothing is inferred here.

    Gate stages exit non-zero when a gate fails, and `run` enables pipefail, so
    a failed gate becomes a failed rule rather than a warning buried in a log.
    """
    parts = [PY, "-m", "motifcheck.cli", cmd]
    if config is not None:
        parts += ["--config", config]
    parts += [str(a) for a in args]
    return " ".join(parts)


def run(cmd: str) -> str:
    return f"set -euo pipefail; {cmd}"
