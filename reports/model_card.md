# Model card — MotifCheck

*Generated 2026-09-29T19:58:06+00:00 · motifcheck 1.0.0 · dataset `motifcheck-ptbp1-k562-grch38-v1`*

## C0 / C1 — TinyCNN

| property | value |
|---|---|
| Input | one-hot (batch, 4, 201) RNA only — A,C,G,U |
| Conv stack | 4→32 k11 d1 → 32→32 k5 d2 → 32→32 k5 d4 |
| Pooling | mean + max → 64 features |
| Head | Linear 64→32, GELU, dropout 0.1, Linear 32→1 |
| **Parameters** | **13,857** (asserted at construction) |
| Receptive field | 35 nt (asserted at construction) |
| Normalisation layers | none |
| Output | single logit, no sigmoid (BCEWithLogitsLoss) |

### Optimisation

- AdamW, lr 0.001, weight decay 0.0001
- batch size 128, 30 epochs, grad clip norm 1.0
- checkpoint selection: best_validation_ap (tie-break: lower validation log loss, then earlier epoch)
- seeds: [17, 29, 43, 71, 101]
- fallback learning rate 0.0003: **not used**
- precision: float32; fallback LR was not required

### The only difference between C0 and C1

Per-example training-loss weights. C0 uses `w0(y) = N / (2·n_y)`; C1 uses `w1(y,s) = n_s / (2·n_ys)` over context × training-GC quintile × training-expression tertile strata, capped then normalised to mean one. Labels, test examples, CNN inputs and capacity are identical, and within each seed pair both conditions start from identical parameters and see identical batches in identical order.

### Per-seed training record

| seed | condition | best epoch | val AP | val log loss | params | seconds | reload Δlogit |
|---|---|---|---|---|---|---|---|
| 17 | C0 | 1 | 0.951 | 0.130 | 13,857 | 312.04 | 0.00e+00 |
| 17 | C1 | 1 | 0.950 | 0.121 | 13,857 | 305.72 | 0.00e+00 |
| 29 | C0 | 1 | 0.952 | 0.188 | 13,857 | 307.04 | 0.00e+00 |
| 29 | C1 | 1 | 0.956 | 0.155 | 13,857 | 298.08 | 0.00e+00 |
| 43 | C0 | 1 | 0.952 | 0.208 | 13,857 | 294.9 | 0.00e+00 |
| 43 | C1 | 1 | 0.955 | 0.118 | 13,857 | 297.04 | 0.00e+00 |
| 71 | C0 | 1 | 0.952 | 0.128 | 13,857 | 296.81 | 0.00e+00 |
| 71 | C1 | 1 | 0.955 | 0.123 | 13,857 | 294.42 | 0.00e+00 |
| 101 | C0 | 1 | 0.957 | 0.180 | 13,857 | 294.32 | 0.00e+00 |
| 101 | C1 | 1 | 0.957 | 0.162 | 13,857 | 296.19 | 0.00e+00 |

## Baselines B0–B4

| id | model | features | params | selected C | validation AP |
|---|---|---|---|---|---|
| B0 | constant_prevalence | 0 | 0 | None | 0.194 |
| B1 | composition_mono_only | 4 | 5 | 0.01 | 0.918 |
| B1d | composition_mono_plus_dinucleotide | 20 | 21 | 0.01 | 0.935 |
| B2 | composition_plus_context | 23 | 24 | 0.01 | 0.945 |
| B3 | kmer_3_4_5_normalized | 1344 | 1345 | 0.01 | 0.939 |
| B4 | motif_reference | 2 | 3 | 0.1 | 0.706 |

B2 uses additional context and expression information and is **not** a sequence-only peer. B4 uses the motif score that defines it and therefore cannot validate itself; it is reported as a known motif-sensitive reference for label prediction only.

## Known limits

- The receptive field is 35 nt against a 201-nt window, so this model *cannot* represent long-range regulatory structure. That is intentional and bounds the claims.
- Five seeds on one dataset are five optimisation replicates, **not** five independent biological experiments. Confidence intervals describe test-component sampling uncertainty conditional on those seed pairs.
- Trained on background that is only *not detected by this assay*; systematic false negatives would be learned as negatives.
- The CNN is not expected to beat the k-mer baseline. That outcome, if it occurs, is reported rather than tuned away.

## Runtime and footprint

- Total CNN training wall time across all 10 runs: **2996.6 s** on one RTX 2060 SUPER (8 GB).
- Baselines and all inference run on CPU; no custom CUDA kernels are used.
