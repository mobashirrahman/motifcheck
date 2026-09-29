"""Deterministic unit tests and synthetic learning fixtures (G3)."""

from mc_common import CONFIG, GATES, REPORTS, cli, run


rule self_test:
    """Mandatory deterministic tests plus the fixed synthetic fixtures."""
    input:
        config=CONFIG,
        dataset="data/processed/dataset_split.parquet",
        gate="reports/gates/G2.json",
    output:
        report=REPORTS / "self_test.json",
        record=GATES / "G3.json",
    log:
        "logs/self_test.log",
    shell:
        run(cli("self-test", "--config", input.config, "--suite", "core",
                "--report", output.report, "--record", output.record)
            + " 2>&1 | tee {log}")
