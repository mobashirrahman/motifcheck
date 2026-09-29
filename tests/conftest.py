"""Shared test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SPLIT_DATA = ROOT / "data" / "processed" / "dataset_split.parquet"


@pytest.fixture(scope="session")
def protocol():
    from motifcheck.config import load_protocol
    return load_protocol(ROOT / "configs" / "protocol.yaml")


@pytest.fixture(scope="session")
def split_data():
    """The built dataset with splits, or skip if the pipeline has not run yet."""
    import pandas as pd
    if not SPLIT_DATA.exists():
        pytest.skip("dataset_split.parquet not built; run `motifcheck check-splits` first")
    return pd.read_parquet(SPLIT_DATA)
