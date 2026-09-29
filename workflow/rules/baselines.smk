"""CPU baselines B0-B4 with validation-only regularisation selection."""

from mc_common import BASELINES, CONFIG, REPORTS, RESULTS, cli, run


rule fit_baselines:
    """Fit B0-B4 on train; select regularisation by validation average precision."""
    input:
        config=CONFIG,
        manifest="data/source_manifest.json",
        dataset="data/processed/dataset_split.parquet",
        gate="reports/gates/G3.json",
    output:
        results=BASELINES / "baselines.json",
        motif=RESULTS / "motif_threshold.json",
        scores=RESULTS / "motif_scores.parquet",
    log:
        "logs/fit_baselines.log",
    shell:
        run(cli("fit-baselines", "--config", input.config,
                "--manifest", input.manifest,
                "--dataset", input.dataset,
                "--split", "validation",
                "--out", str(BASELINES),
                "--motif-threshold", output.motif,
                "--motif-scores", output.scores)) + " 2>&1 | tee {log}"
