# MotifCheck

A small, leakage-controlled benchmark asking a narrow question:

> Does balancing training examples for broad sequence composition and transcript
> context improve evidence of motif-specific recognition, while preserving
> prediction of experimentally observed binding peaks?

Task: classify 201-nt, strand-oriented genomic windows as *observed reproducible
PTBP1 peak* vs *sampled eligible background*, using public ENCODE eCLIP data
(**ENCSR981WKN**, K562, GRCh38). The main experiment compares the **same
13,857-parameter CNN** trained normally (C0) or with a prespecified
nuisance-balancing training-loss intervention (C1), across five paired seeds.

Full design, gates, thresholds and claim rules:
[`MotifCheck_Implementation_Plan.md`](MotifCheck_Implementation_Plan.md).
Amendments to that plan are recorded in
[`configs/protocol.yaml`](configs/protocol.yaml) under `amendments`.

---

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[workflow,dev]'
cp /path/to/locked/env/bin/python .venv/bin/python   # optional

# Point the workflow at your interpreter and run it end to end
PYTHON=$(which python) snakemake --cores 8
```

`snakemake` is the only entry point; every rule calls the frozen `motifcheck`
CLI, so nothing is computed by an ad hoc shell command.

```bash
PYTHON=$(which python) snakemake -n            # dry run, show the DAG
PYTHON=$(which python) snakemake --cores 8 -k   # keep going after a rule fails
PYTHON=$(which python) snakemake --cores 8 \
    --config configs/protocol.yaml             # the protocol is explicit
```

## The workflow

```
protocol-freeze     G0   freeze configs/protocol.yaml and its amendment log
validate-sources    G1   download + SHA-256 + schema-validate every asset
build-data               candidate universe, 201-nt windows, labels, exclusions
check-splits        G2   leakage components, 70/15/15 split, support audit
self-test           G3   57 deterministic tests + 5 synthetic fixtures
fit-baselines            B0-B4, regularisation chosen on validation AP only
train-pairs         G4   2 conditions x 5 paired seeds, reload parity
                    G5   intervention instantiated and QC'd
audit                    validate motif-order panels and ISM on validation only
lock-analysis       G6   hash code, config, data, checkpoints, endpoints
evaluate                 the single frozen test evaluation
report               G7   tables, figures, data card, model card, final report
```

Each gate rule exits non-zero when its gate fails, so a failed gate stops the
workflow rather than letting downstream stages consume unvalidated inputs.

## Test isolation

The test split is scored exactly once. `motifcheck evaluate` refuses to run
unless all three hold:

1. `reports/analysis_lock.json` exists,
2. its `protocol_sha256` matches the current protocol,
3. every technical gate G0-G6 has a `pass` record, and the dataset hash is
   unchanged since the lock.

Every evaluation is appended to `results/test_evaluation_log.jsonl`, which is
never rewritten. A test-aware code fix preserves the old results, is described as
an amendment, and any rerun is marked exploratory rather than confirmatory.

## Repository layout

```
configs/protocol.yaml        frozen protocol: seeds, thresholds, margins, budgets, amendments
workflow/rules/*.smk         Snakemake rules, one per pipeline stage
src/motifcheck/
  config.py                  protocol loading + architecture/parameter-count assertions
  acquire.py                 download, checksum, schema validation, provenance manifest
  data.py                    candidate universe, window extraction, labels, exclusions
  splits.py                  leakage components, hashed splits, split audit
  features.py                B0-B4 feature matrices
  weights.py                 the intervention: strata, weights, ESS, balance QC
  models.py                  TinyCNN (asserted to 13,857 params) + optional transformer
  train.py                   paired seeded training, checkpoints, reload parity
  panels.py                  P1, P2, motif-order audit panel, ISM panel
  motif.py                   PWM loading, log-odds scanning, training-null thresholds
  audit.py                   order-disrupted controls, decoys, exact and Taylor ISM
  stats.py                   metrics, paired component bootstrap, claim classification
  gates.py                   G0-G7 machine-readable gate records
  report.py                  frozen-result tables, figures, cards, final report
  synthetic.py               the five synthetic learning fixtures
  cli.py                     the command contract
tests/                       deterministic unit and integration tests
reports/                     protocol, data card, model card, figures, tables, final report
data/source_manifest.json    every source asset with SHA-256, size, license and validation
```

## Outputs

- [`reports/final_report.md`](reports/final_report.md) — results, claim level, limitations
- [`reports/data_card.md`](reports/data_card.md) — target population, exclusions, prevalence
- [`reports/model_card.md`](reports/model_card.md) — capacity, optimisation, per-seed record
- `reports/tables/*.csv` — predictive results, seed results, primary comparison
- `reports/figures/*.png` — predictive comparison, motif preference, ISM examples
- `reports/gates/G*.json` — one machine-readable record per technical gate

## Findings

**Claim level: `valid_demo`.** The prespecified intervention claim is **not
supported**. This is a valid, reportable negative result, not a failure of the
study.

### 1. Peak prediction in K562 is overwhelmingly a composition problem

Test set, panel P1 (n = 3,878, benchmark prevalence 0.194):

| model | inputs | params | AUROC | AP |
|---|---|---|---|---|
| B0 | none (train prevalence) | 0 | 0.500 | 0.194 |
| **B1** | **4 mononucleotide frequencies** | **4** | **0.966** | **0.926** |
| B1d | + dinucleotide frequencies | 20 | 0.974 | 0.945 |
| B2 | composition + context + log1p TPM | 24 | 0.982 | 0.953 |
| B3 | normalised 3/4/5-mer counts | 1,345 | 0.980 | 0.954 |
| C0 / C1 | TinyCNN, one-hot sequence only | 13,857 | 0.983–0.986 | 0.959–0.966 |
| B4 | reference motif score | 2 | 0.866 | 0.684 |

**Four composition features reach AUROC 0.966 — within 0.02 of the 13,857-parameter
CNN.** Essentially all of the CNN's discriminative power is reproducible from bulk
base composition. B0 landing exactly on prevalence confirms the metric harness is
wired correctly.

### 2. Composition balancing bought nothing

The single intervention (C1: nuisance-balanced training loss) versus the identical
standard-training control (C0), five paired seeds, test set:

| endpoint | C1 − C0 | 95% CI | prespecified rule | result |
|---|---|---|---|---|
| AP on P1 | −0.0003 | [−0.0062, +0.0058] | lower CI > −0.02 | **pass** |
| MPA (motif preference) | +0.0142 | [−0.0086, +0.0355] | ≥ +0.10, lower CI > 0, ≥ 4/5 seeds | **fail** (3/5) |
| Specificity guard ΔS | +0.0210 | [−0.0021, +0.0444] | lower CI > 0 | **fail** |

Balancing cost essentially nothing in accuracy and produced no detectable gain in
motif-order evidence. Intervals are paired component bootstraps (2,000 replicates)
conditional on the trained seed pairs.

### 3. The models are only weakly order-sensitive

MPA — the fraction of mononucleotide-count- and length-preserving order-disrupted
controls scored below the unedited window, averaged within leakage component over
148 components:

| seed | C0 | C1 |
|---|---|---|
| 17 | 0.553 | 0.547 |
| 29 | 0.555 | 0.578 |
| 43 | 0.535 | 0.560 |
| 71 | 0.522 | 0.558 |
| 101 | 0.570 | 0.564 |
| **mean** | **0.547** | **0.561** |

Chance is 0.5. Both conditions sit only 5–6 points above it, so the models predict
PTBP1 peaks well while barely preferring the motif's *correct base order*. Note
that B4 — the reference motif score itself — is the *weakest* sequence-based model
in the table (AUROC 0.866).

### 4. The intervention itself could not be instantiated as planned

Gate **G5 failed**: residual expression imbalance reached SMD +0.138 against a
0.10 limit, after all three permitted training-only attempts. Effective sample
size (0.552) and GC balance (+0.053) passed; no stratum lacked common support.

The handling rule for this was written into the frozen protocol as **amendment
A5 before any model was trained**, on training data only. It caps the claim at
`valid_demo` and forbids describing C1 as a successful balancing experiment. So
the defensible statement is *"this much balancing did not help"*, **not**
*"balancing does not work"* — C1 is an under-balanced arm.

### What the numbers point at

The CNN outperforms the reference motif by a wide margin while outperforming a
composition model by almost nothing. Whatever the CNN does beyond composition is
not motif-order recognition. That is the question worth carrying forward.

## Gates

| gate | status | |
|---|---|---|
| G0 protocol | pass | frozen with 5 documented amendments |
| G1 provenance | pass | 9/9 sources, SHA-256 verified |
| G2 splits | pass | zero detected cross-split overlap |
| G3 implementation | pass | 57 tests + 5/5 synthetic fixtures |
| G4 fit sanity | pass | finite losses, paired inputs, reload Δlogit = 0.00e+00 |
| **G5 intervention** | **fail** | SMD(log1p TPM) +0.138 > 0.10 (amendment A5) |
| G6 final lock | pass | code, config, data, checkpoints hashed |
| G7 release | pass | every planned result reported |

## What this project does not claim

The claim level is decided by prespecified rules in the protocol and is never
chosen after seeing the result. "Valid demo" is a successful outcome; an
intervention that fails its co-primary requirements is reported as a negative
result, not re-tuned. Causal biological recognition, universal debiasing and
mechanism are explicitly **not** available from this computational demo.
