# MotifCheck: final report

*Small-model study of composition shortcuts in PTBP1 binding prediction (ENCODE eCLIP ENCSR981WKN, K562).*

*Generated 2026-09-29T19:58:06+00:00 · motifcheck 1.0.0 · protocol sha256 `d60baba1cfb141ac` · dataset `motifcheck-ptbp1-k562-grch38-v1`*

## Question

> Does balancing training examples for broad sequence composition and transcript context improve evidence of motif-specific recognition while preserving prediction of experimentally observed binding peaks?

## Headline

**Claim level: `valid_demo`.** A reproducible audit of a small PTBP1 peak predictor. The prespecified intervention claim is not supported (improved motif preference failed). This is a valid, reportable negative or inconclusive result.

> A reproducible audit of a small PTBP1 peak predictor. The prespecified intervention claim is not supported (improved motif preference failed). This is a valid, reportable negative or inconclusive result.

## Dataset as built

- 27,288 windows: 5,352 positive, 21,936 background (4.099:1)
- split by leakage component: test 3,878 rows / 754 positives / 1,011 components, train 19,232 rows / 3,786 positives / 4,822 components, validation 4,178 rows / 812 positives / 1,015 components
- zero detected cross-split gene, locus, exact-sequence or near-duplicate overlap

## Predictive results (test set, single frozen evaluation)

| model | seed | panel | n | prevalence | AP | AUROC | log loss | Brier |
|---|---|---|---|---|---|---|---|---|
| B0 | — | P1 | 3,878 | 0.194 | 0.194 | 0.500 | 0.493 | 0.157 |
| B1 | — | P1 | 3,878 | 0.194 | 0.926 | 0.966 | 0.166 | 0.044 |
| B1d | — | P1 | 3,878 | 0.194 | 0.945 | 0.974 | 0.135 | 0.034 |
| B2 | — | P1 | 3,878 | 0.194 | 0.953 | 0.982 | 0.121 | 0.032 |
| B3 | — | P1 | 3,878 | 0.194 | 0.954 | 0.980 | 0.119 | 0.031 |
| B4 | — | P1 | 3,878 | 0.194 | 0.684 | 0.866 | 0.326 | 0.098 |
| C0 | 17 | P1 | 3,878 | 0.194 | 0.959 | 0.984 | 0.987 | 0.034 |
| C0 | 29 | P1 | 3,878 | 0.194 | 0.965 | 0.985 | 1.238 | 0.041 |
| C0 | 43 | P1 | 3,878 | 0.194 | 0.962 | 0.983 | 1.560 | 0.051 |
| C0 | 71 | P1 | 3,878 | 0.194 | 0.965 | 0.985 | 0.927 | 0.030 |
| C0 | 101 | P1 | 3,878 | 0.194 | 0.966 | 0.986 | 1.175 | 0.039 |
| C1 | 17 | P1 | 3,878 | 0.194 | 0.960 | 0.984 | 0.982 | 0.032 |
| C1 | 29 | P1 | 3,878 | 0.194 | 0.965 | 0.985 | 1.146 | 0.038 |
| C1 | 43 | P1 | 3,878 | 0.194 | 0.961 | 0.983 | 0.871 | 0.030 |
| C1 | 71 | P1 | 3,878 | 0.194 | 0.963 | 0.986 | 0.912 | 0.030 |
| C1 | 101 | P1 | 3,878 | 0.194 | 0.965 | 0.986 | 1.118 | 0.036 |
| B0 | — | P2 | 1,504 | 0.500 | 0.500 | 0.500 | 0.922 | 0.342 |
| B1 | — | P2 | 1,504 | 0.500 | 0.969 | 0.963 | 0.306 | 0.084 |
| B1d | — | P2 | 1,504 | 0.500 | 0.976 | 0.970 | 0.249 | 0.065 |
| B2 | — | P2 | 1,504 | 0.500 | 0.978 | 0.975 | 0.227 | 0.063 |
| B3 | — | P2 | 1,504 | 0.500 | 0.982 | 0.977 | 0.224 | 0.059 |
| B4 | — | P2 | 1,504 | 0.500 | 0.873 | 0.865 | 0.600 | 0.195 |
| C0 | 17 | P2 | 1,504 | 0.500 | 0.985 | 0.982 | 1.809 | 0.058 |
| C0 | 29 | P2 | 1,504 | 0.500 | 0.988 | 0.984 | 1.418 | 0.047 |
| C0 | 43 | P2 | 1,504 | 0.500 | 0.986 | 0.983 | 1.605 | 0.053 |
| C0 | 71 | P2 | 1,504 | 0.500 | 0.988 | 0.985 | 1.895 | 0.059 |
| C0 | 101 | P2 | 1,504 | 0.500 | 0.988 | 0.985 | 1.283 | 0.043 |
| C1 | 17 | P2 | 1,504 | 0.500 | 0.987 | 0.984 | 1.950 | 0.063 |
| C1 | 29 | P2 | 1,504 | 0.500 | 0.987 | 0.984 | 1.488 | 0.049 |
| C1 | 43 | P2 | 1,504 | 0.500 | 0.986 | 0.983 | 1.656 | 0.054 |
| C1 | 71 | P2 | 1,504 | 0.500 | 0.987 | 0.985 | 1.606 | 0.054 |
| C1 | 101 | P2 | 1,504 | 0.500 | 0.988 | 0.985 | 1.364 | 0.042 |

AP is average precision in the sklearn sense. P1 prevalence is the artificial benchmark prevalence; P2 is 1:1 matched and therefore has a different prevalence, so P1→P2 differences are **not** transfer degradation.

## Primary comparison: C1 minus C0

| endpoint | estimate | 95% CI | prespecified rule | passes |
|---|---|---|---|---|
| AP on P1 | -0.0003 | [-0.0062, +0.0058] | lower CI > -0.02 | **yes** |
| MPA (motif preference) | +0.0142 | [-0.0086, +0.0355] | >= 0.1, lower CI > 0, >= 4/5 seeds agree (3/5) | **no** |
| S = MPA_target − MPA_outside | +0.0210 | [-0.0021, +0.0444] | lower CI > 0 | **no** |

All requirements and the specificity guard form a joint rule; the favourable condition was not selected on its own.

## Motif-order audit

- motif: `M00249_2.00` (unknown, unknown, unknown), length 7, consensus CUUUUCU
- scanning threshold 5.139, set from 3,750,240 training-null scan positions at a 1% window-level false-positive rate (a scanning cutoff, **not** a binding probability)
- audit panel: 500 windows from 154 components
- controls: 8120 target, 8120 outside-motif; 94 windows marked uninformative and excluded rather than relaxed
- all controls preserve mononucleotide counts and window length: True; mean |Δ dinucleotide| = 4.508866995073892

## Attribution

- reference method: exact in silico mutagenesis, 603 substitutions per 201-nt sequence
- TISM (gradient) approximation: **accepted** (median Spearman 0.871965, median sign agreement 0.9875). The model was not changed to make the approximation pass.

## Technical gates

| gate | status |
|---|---|
| G0 | pass |
| G1 | pass |
| G2 | pass |
| G3 | pass |
| G4 | pass |
| G5 | fail |
| G6 | pass |
| G7 | pass |

## What this does and does not support

**Not claimable from this experiment:** causal biological recognition; universal debiasing; disease prediction; resolution of the lab's reported problem; mechanism establishment (not available from this computational demo alone).

## Limitations and deviations

- **No processed control signal.** ENCSR981WKN releases no size-matched input, so B2's input-coverage feature was removed rather than zero-filled (amendment A2). Without a control, observed peaks cannot be separated from assay background the way RBPNet does.
- **Positive ceiling below plan.** The plan asked for up to 10,000 positive windows; only 7,797 reproducible peaks exist and 5,352 survived filtering (amendment A1). Statistical power for the MPA endpoint is correspondingly limited.
- **Background is assay-negative, not unbound.** Any false negative in the eCLIP dataset is learned as a genuine negative.
- **One RBP, one cell type.** Results do not generalise to other RBPs or to HepG2. No transfer cohort was run.
- **Five seeds are not five experiments.** Intervals are conditional on the trained seed pairs and describe test-component sampling only.
- **Synthetic edits are out of distribution.** Order-disrupted controls change RNA structure and other motifs. The audit shows sensitivity to sequence order beyond preserved composition; it does not establish biological causality.
- **The optional transformer (H3) was not run.** It is excluded by the frozen protocol and was deferred until after the core report, as the plan requires.
- **Track A (SPIDRnet audit) was not performed.** The plan's two-day audit of the public SPIDRnet repository was out of scope for this run, so nothing here is a reproduction of, or a comparison against, its reported motif-correlation result.

## Figures

1. `reports/figures/predictive.png` — predictive comparison, all models and seeds.
2. `reports/figures/motif_preference.png` — paired motif-preference effects with target and outside-motif controls.
3. `reports/figures/ism_examples.png` — representative exact-ISM examples selected by a fixed rule.

## Reproducing

```bash
snakemake --cores 1 --use-conda   # or point PYTHON at the locked environment
```

The test evaluation refuses to run without a matching analysis lock and passing technical gate records; every test evaluation is appended to `results/test_evaluation_log.jsonl`.
