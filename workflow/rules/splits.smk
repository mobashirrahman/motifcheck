"""Leakage components, immutable splits, and the split audit (G2)."""

from mc_common import CONFIG, GATES, REPORTS, SPLIT_DATA, TRAIN_BINS, cli, run


rule check_splits:
    """Build leakage components, assign splits by component hash, audit support.

    Exits non-zero if any forbidden cross-split overlap survives or minimum
    support is not met, so training cannot start on an unsound split.
    """
    input:
        config=CONFIG,
        dataset="data/processed/dataset.parquet",
        gate="reports/gates/G1.json",
    output:
        dataset=SPLIT_DATA,
        audit=REPORTS / "split_audit.json",
        bins=TRAIN_BINS,
        record=GATES / "G2.json",
    log:
        "logs/check_splits.log",
    shell:
        run(cli("check-splits", "--config", input.config,
                "--dataset", input.dataset,
                "--out", output.dataset,
                "--audit", output.audit,
                "--bins", output.bins,
                "--record", output.record)) + " 2>&1 | tee {log}"
