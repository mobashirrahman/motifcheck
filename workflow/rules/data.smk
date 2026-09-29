"""Build the candidate universe and the 201-nt window dataset."""

from mc_common import CONFIG, DATASET, EXCLUSIONS, REPORTS, cli, run


rule build_data:
    """Candidate universe, sequence extraction, labels, exclusions, nuisance variables."""
    input:
        config=CONFIG,
        manifest="data/source_manifest.json",
        gate="reports/gates/G1.json",
    output:
        dataset=DATASET,
        exclusions=EXCLUSIONS,
        report=REPORTS / "dataset_build.json",
    log:
        "logs/build_data.log",
    threads: 1
    resources:
        mem_mb=8000,
    shell:
        run(cli("build-data", "--config", input.config,
                "--manifest", input.manifest,
                "--dataset", output.dataset,
                "--exclusions", output.exclusions,
                "--report", output.report)) + " 2>&1 | tee {log}"
