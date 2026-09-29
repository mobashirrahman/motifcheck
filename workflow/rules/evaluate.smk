"""Lock the analysis, then run the single frozen test evaluation (G6, G7)."""

from mc_common import CONFIG, GATES, REPORTS, RESULTS, TEST, cli, run


rule lock_analysis:
    """Freeze code, config, data, checkpoints and endpoints before test access.

    Exits non-zero unless every technical gate through G5 has passed.
    """
    input:
        config=CONFIG,
        dataset="data/processed/dataset_split.parquet",
        baselines=RESULTS / "baselines" / "baselines.json",
        audit=RESULTS / "audit" / "audit_validation.json",
        qc="reports/intervention_qc.json",
        gate="reports/gates/G5.json",
    output:
        lock=REPORTS / "analysis_lock.json",
        record=GATES / "G6.json",
    log:
        "logs/lock_analysis.log",
    shell:
        run(cli("lock-analysis", "--config", input.config,
                "--dataset", input.dataset,
                "--lock", output.lock,
                "--record", output.record)) + " 2>&1 | tee {log}"


rule evaluate_test:
    """The one frozen test evaluation.

    The CLI refuses to run without a matching analysis-lock hash and passing
    technical gate records, and appends every evaluation to an append-only log.
    """
    input:
        config=CONFIG,
        manifest="data/source_manifest.json",
        dataset="data/processed/dataset_split.parquet",
        lock=REPORTS / "analysis_lock.json",
        gate="reports/gates/G6.json",
    output:
        evaluation=TEST / "evaluation.json",
        controls=TEST / "control_panel_test.parquet",
        log=RESULTS / "test_evaluation_log.jsonl",
    log:
        "logs/evaluate_test.log",
    shell:
        run(cli("evaluate", "--config", input.config,
                "--manifest", input.manifest,
                "--split", "test",
                "--dataset", input.dataset,
                "--lock", input.lock,
                "--evaluation", output.evaluation,
                "--controls", output.controls,
                "--log", output.log)) + " 2>&1 | tee {log}"
