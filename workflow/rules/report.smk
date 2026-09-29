"""Frozen-result reporting: tables, figures, cards and the final report (G7)."""

from mc_common import (FIGURES, GATES, REPORTS, TABLES, TEST, cli, run)


rule report:
    """Generate every deliverable from frozen results only.

    Reads results/test/evaluation.json and never recomputes a result, so the
    report cannot disagree with the artifacts it cites.
    """
    input:
        config="configs/protocol.yaml",
        evaluation=TEST / "evaluation.json",
        ism=TEST / "ism_examples.parquet",
        gate="reports/gates/G6.json",
    output:
        final=REPORTS / "final_report.md",
        data_card=REPORTS / "data_card.md",
        model_card=REPORTS / "model_card.md",
        tables=TABLES / "primary_comparison.csv",
        predictive=TABLES / "predictive_results.csv",
        seeds=TABLES / "seed_results.csv",
        limitations=REPORTS / "limitations.json",
        replay=REPORTS / "replay_check.json",
        summary=REPORTS / "gate_summary.json",
        record=GATES / "G7.json",
    log:
        "logs/report.log",
    shell:
        run(cli("report", "--config", input.config,
                "--from-frozen-results",
                "--final", output.final,
                "--data-card", output.data_card,
                "--model-card", output.model_card,
                "--tables", str(TABLES),
                "--limitations", output.limitations,
                "--summary", output.summary,
                "--record", output.record)) + " 2>&1 | tee {log}"
