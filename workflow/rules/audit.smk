"""Validate the motif-order audit and attribution machinery on validation only."""

from mc_common import AUDIT, CONFIG, REPORTS, RESULTS, cli, run


rule audit_validation:
    """Build validation audit panels, controls and the ISM panel.

    Nothing here touches the test split: the audit machinery must be shown to
    work before the analysis is locked.
    """
    input:
        config=CONFIG,
        manifest="data/source_manifest.json",
        dataset="data/processed/dataset_split.parquet",
        gate="reports/gates/G5.json",
        thresholds=RESULTS / "motif_threshold.json",
    output:
        report=AUDIT / "audit_validation.json",
        controls=AUDIT / "control_panel_validation.parquet",
    log:
        "logs/audit_validation.log",
    shell:
        run(cli("audit", "--config", input.config,
                "--manifest", input.manifest,
                "--dataset", input.dataset,
                "--split", "validation",
                "--out", str(AUDIT),
                "--report", output.report,
                "--controls", output.controls)) + " 2>&1 | tee {log}"
