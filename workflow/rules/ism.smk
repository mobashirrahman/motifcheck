"""Exact-ISM examples for the report, selected by a fixed rule."""

from mc_common import CONFIG, REPORTS, RESULTS, TEST, cli, run


rule ism_examples:
    """Compute exact substitution effects for the frozen ISM panel and keep a
    fixed-rule selection of representative examples for the report figure."""
    input:
        config=CONFIG,
        manifest="data/source_manifest.json",
        dataset="data/processed/dataset_split.parquet",
        evaluation=TEST / "evaluation.json",
    output:
        examples=TEST / "ism_examples.parquet",
        metrics=TEST / "ism_metrics.json",
    log:
        "logs/ism_examples.log",
    shell:
        run(cli("ism-examples", "--config", input.config,
                "--manifest", input.manifest,
                "--dataset", input.dataset,
                "--split", "test",
                "--out", output.examples,
                "--metrics-out", output.metrics)) + " 2>&1 | tee {log}"
