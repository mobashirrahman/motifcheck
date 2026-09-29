# MotifCheck — Snakemake workflow
#
# Implements the command contract of MotifCheck_Implementation_Plan.md section 11.
# Every rule calls the frozen `motifcheck` CLI; no rule computes a result itself.
#
# Run:
#   PYTHON=/path/to/env/bin/python snakemake --cores 8
#
# Gate stages exit non-zero on failure, so a failed technical gate stops the
# workflow rather than allowing downstream stages to run on unvalidated inputs.

import os
import sys
from pathlib import Path

configfile: "configs/protocol.yaml"

# Make the shared rule helpers importable from every rules/*.smk module.
sys.path.insert(0, os.path.join(str(workflow.basedir), "workflow", "rules"))

from mc_common import (  # noqa: E402
    AUDIT, BASELINES, CONFIG, DATA, DATASET, EXCLUSIONS,
    EXTERNAL, FIGURES, GATES, MANIFEST, PROCESSED, PY, REPORTS, RESULTS,
    SEEDS, SPLIT_DATA, TABLES, TEST, TRAIN_BINS, TRAINING, checkpoint_paths, cli, run,
)

RULES = Path(workflow.snakefile).parent / "workflow" / "rules"

include: "workflow/rules/sources.smk"
include: "workflow/rules/data.smk"
include: "workflow/rules/splits.smk"
include: "workflow/rules/selftest.smk"
include: "workflow/rules/baselines.smk"
include: "workflow/rules/train.smk"
include: "workflow/rules/audit.smk"
include: "workflow/rules/evaluate.smk"
include: "workflow/rules/ism.smk"
include: "workflow/rules/report.smk"


rule all:
    input:
        REPORTS / "final_report.md",
        REPORTS / "data_card.md",
        REPORTS / "model_card.md",
        REPORTS / "gate_summary.json",


rule clean_intermediates:
    """Remove derived artifacts but keep downloaded sources and checkpoints."""
    run:
        for p in (SPLIT_DATA, DATASET, EXCLUSIONS, TRAIN_BINS):
            Path(p).unlink(missing_ok=True)
        print("removed derived dataset files")


rule clean_results:
    run:
        import shutil
        for d in (RESULTS,):
            shutil.rmtree(d, ignore_errors=True)
        print("removed results/")


rule clean_gates:
    run:
        import shutil
        shutil.rmtree(GATES, ignore_errors=True)
        print("removed gate records")
