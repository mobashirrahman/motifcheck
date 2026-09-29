"""Paired CNN training C0/C1 across five seeds (G4, G5)."""

from mc_common import (CONFIG, GATES, RESULTS, SEEDS, TRAINING, checkpoint_paths, cli, run)


rule train_pairs:
    """Ten full runs: two loss-weighting conditions x five paired seeds.

    The intervention is instantiated and QC'd here (G5); fit sanity and
    checkpoint reload parity are checked here (G4).
    """
    input:
        config=CONFIG,
        dataset="data/processed/dataset_split.parquet",
        gate="reports/gates/G3.json",
        baselines=RESULTS / "baselines" / "baselines.json",
    output:
        checkpoints=checkpoint_paths(),
        pairs=RESULTS / "training" / "pairs.jsonl",
        weights=RESULTS / "training" / "weights.npz",
        qc="reports/intervention_qc.json",
        parity="reports/checkpoint_reload_parity.json",
        record_g4=GATES / "G4.json",
        record_g5=GATES / "G5.json",
    log:
        "logs/train_pairs.log",
    threads: 1
    resources:
        gpus=1,
        mem_mb=8000,
    shell:
        run(cli("train-pairs", "--config", input.config,
                "--models", "cnn",
                "--seeds", *SEEDS,
                "--dataset", input.dataset,
                "--out", str(TRAINING),
                "--pairs", output.pairs,
                "--weights", output.weights,
                "--qc", output.qc,
                "--parity", output.parity)) + " 2>&1 | tee {log}"
