# MotifCheck: a small-model study of shortcuts in RNA-binding prediction

Implementation and validation plan • Version 1.0 • 29 September 2026


## 1. Decision and scope

Build an interpretable, reproducible benchmark around **human PTBP1 binding in K562**, using public ENCODE eCLIP data. The main experiment compares the **same approximately 14,000-parameter CNN**, trained normally or with a prespecified nuisance-balancing intervention. CPU baselines establish how much performance simple sequence statistics explain. A tiny transformer is optional and must wait until the core study is complete.

The central question is:

> Does balancing training examples for broad sequence composition and transcript context improve evidence of motif-specific recognition while preserving prediction of experimentally observed binding peaks?

The project succeeds as a research demonstration if it answers this question rigorously, including a negative or inconclusive answer. It does not have to beat a large model.

**Core limits:** one RBP, one cell type, one primary task, two CNN training conditions, five paired seeds, one primary intervention, a fixed test set, and one small report. Use processed public data. No wet-lab work, foundation-model pretraining, genome-wide model training, or raw-read reprocessing is required.

**Status:** this is an implementation specification, not a completed experiment. Candidate accessions are leads, not a verified downloadable manifest. All numerical acceptance margins below are proposed project rules, not field standards or guarantees of statistical power. Freeze or amend them using training/validation information before accessing test performance.

## 2. Evidence, motivation, and the reproduction boundary

The public SPIDRnet teaching repository reports reasonable predictive performance for PTBP1 but motif tests suggesting dependence on sequence composition. This is a hypothesis-generating project report, not proof that a particular mechanism caused the behavior. [S1]

Run two explicitly separate tracks:

| Track | Purpose | Allowed claim |
|---|---|---|
| A: SPIDRnet audit, time-boxed to two working days | Inspect its preprocessing, model, saved artifacts, and motif test; reproduce only what resources permit | Exact reproduction only if data, target, split, model, and checkpoint are sufficiently matched; otherwise partial reproduction or code audit |
| B: MotifCheck core | Test the shortcut hypothesis independently on an ENCODE peak-classification task | Independent conceptual follow-up; not a reproduction of the reported SPIDRnet correlation |

Do not compare SPIDRnet regression correlations with classification average precision. Do not call a smaller, independently trained CNN “SPIDRnet reproduced.” If checkpoints or exact inputs are unavailable, record that fact and continue Track B.

RBPNet establishes that separating experimental background from binding-related signal is important; its base-resolution task is more ambitious than this demo. MotifCheck is not a replacement for RBPNet and should not claim base-resolution binding prediction. [S2]

## 3. Hypotheses and conclusions that are permitted

**H1 — shortcut evidence:** broad sequence/context information explains substantial predictive behavior, while controlled motif-order tests show limited specificity. This requires converging evidence. A strong GC baseline alone is insufficient because composition can be biologically relevant.

**H2 — intervention:** balancing training nuisance variables increases motif-preference performance relative to the identical standard CNN, without an unacceptable decrease in held-out predictive performance.

**H3 — optional architecture comparison:** a tiny transformer changes these outcomes under the same data and evaluation rules. This tests an architecture, not the benefits of language-model pretraining.

Possible conclusions:

- Improved motif evidence with preserved prediction: supports the narrow intervention hypothesis.
- Improved motif evidence with lower prediction: a trade-off; potentially real contextual information was removed.
- No improvement: the intervention is ineffective in this setting, or the remaining uncertainty is too large.
- Strong motif behavior already present: the suspected failure did not generalize to this dataset/model.
- Insufficient valid controls or assay support: inconclusive; report the limitation.

Avoid claiming causal biological recognition, universal debiasing, disease prediction, or resolution of the lab's reported problem from this experiment alone.

## 4. Dataset contract

### 4.1 Inputs and provenance

The starting candidate is PTBP1 K562 eCLIP **ENCSR981WKN**. PTBP1 HepG2 **ENCSR384KAN** is an optional later transfer cohort. These identifiers appear in the ENCODE RBP inventory; revalidate status, assembly, controls, and file relationships at acquisition. [S3]

Required core assets:

1. Released reproducible PTBP1 K562 peak files, preferably the portal's documented reproducibility/IDR output.
2. The associated size-matched input/control metadata and usable processed control signal when available.
3. Matching genome FASTA and gene annotation on the same assembly.
4. Public K562 RNA-seq gene quantification for defining an expressed, observable candidate universe.
5. An experimentally determined human PTBP1 motif from CisBP-RNA, with motif ID, assay provenance, version, and matrix orientation recorded. Inferred motifs must be labeled separately. [S4]

Use a single assembly, preferably GRCh38. Never silently combine hg19 coordinates with GRCh38 sequence. Do not perform implicit liftover. If a change of assembly is unavoidable, make it a documented dataset-version change.

For each source, persist URL, accession, filename, assembly, biological replicate, processing pipeline, control relationship, release status, retrieval date, license/use terms, byte size, and SHA-256. The manifest must distinguish failed downloads from legitimate empty files.

Do not guess BED score meanings or assume a universal peak threshold. Read the selected file's schema and portal pipeline documentation. Prefer released reproducible peaks over inventing a new caller. If such files are unavailable, stop the confirmatory dataset build and register a different peak-selection protocol before training.

### 4.2 Prediction unit and label

- Input: a **201-nucleotide, strand-oriented contiguous genomic window** around a retained peak midpoint. Midpoint is a window anchor, not an asserted crosslink nucleotide.
- Output: one logit for **observed reproducible PTBP1 peak versus sampled eligible background**.
- Positive: retained reproducible peak centered in the window.
- Background: an eligible window with no overlap with the excluded peak neighborhood.
- Biological limitation: background means “not detected by this assay under these rules,” not proven unbound RNA.

Use A/C/G/U in the logical sequence representation, with a single tested T-to-U conversion boundary. On the negative genomic strand, reverse-complement the DNA before converting T to U. Reject noncanonical or out-of-bounds windows in the core dataset and report their counts. No silent padding.

PTBP1 analysis must retain intronic sequence when appropriate. Do not reconstruct only mature transcripts and accidentally discard pre-mRNA binding contexts. Do not automatically reverse-complement augment RNA inputs: RNA binding is directional.

Assign one unambiguous gene and a conservative context category: intronic, exonic, or ambiguous. Exclude ambiguous assignments in the core experiment and report the resulting target population. More detailed UTR/CDS categories are optional only if annotation consistency across isoforms is explicitly handled.

### 4.3 Eligible background and filtering

Define the candidate universe before comparing models:

- Expressed, unambiguously annotated genes; proposed starting threshold is gene TPM >= 1 in a compatible K562 quantification.
- Valid sequence and matching assembly/strand.
- Exclude windows overlapping a retained peak or any available replicate-level PTBP1 peak expanded by 100 nt on each side. This is a conservative background exclusion buffer, not a biological boundary.
- Apply the same blacklist/mappability and observability rules to positives and background.
- If using processed input coverage to restrict observability, document its units, distinguish missing from zero coverage, and freeze a training-derived rule. Do not equate absence of an input file with zero background signal.

Prefer up to **10,000 positive windows and 40,000 background windows** initially, with deterministic sampling. Use fewer if supported data are limited. Collapse duplicate/strongly overlapping positives into one deterministic representative per locus rather than multiplying nearly identical examples. Preserve all exclusion counts.

The proposed 1:4 positive/background ratio defines an artificial benchmark prevalence, not transcriptome-wide prevalence. Report it on every precision-recall figure. Positive selection and background sampling must not use motif scores or model predictions.

Keep a frozen master candidate list. Diagnostic test subsets are selected from this list using prespecified metadata rules, never by searching for a favorable model result.

### 4.4 Minimal row schema

| Field | Requirement |
|---|---|
| window_id | Stable hash of coordinates, strand, assembly, and dataset version |
| chrom, start, end, strand | Zero-based, half-open coordinates; explicit assembly |
| gene_id, context | Unambiguous assignment and annotation version |
| sequence | 201 canonical RNA bases; sequence hash retained |
| label | 1 = observed reproducible peak; 0 = eligible background |
| peak_accession, source_accessions | Traceable provenance; nullable only when appropriate |
| gc, mono_counts, log1p_tpm, input_signal | QC/nuisance variables; no missing-as-zero substitution |
| component_id, split | Leakage grouping and immutable split assignment |
| exclusion_reason | Stored in a separate exclusion table for rejected candidates |

Motif scores and audit-edit metadata live in a separate audit table. They must not enter CNN inputs or the primary balancing intervention.

## 5. Leakage prevention and test isolation

Before fitting any model, build connected components joining windows that share a gene, overlap at a genomic locus, have identical sequence, or meet a prespecified near-duplicate sequence criterion. Initial near-duplicate rule: >=90% identity over >=80% of the shorter window, accounting for reverse-complement duplicates for leakage detection only.

Pin a nucleotide-similarity implementation, record its settings, and validate its behavior on known exact, near-duplicate, and unrelated fixtures. This screen reduces homology leakage; it does not prove that all distant evolutionary relationships have been removed.

Assign complete components by stable seeded hashing to approximately **70% train, 15% validation, 15% test**. The component's canonical identifier, not filesystem order, determines the split. Define a giant component as one containing more than 5% of candidate windows; report and remove such components before modeling rather than breaking them across splits. If this removes more than 20% of candidates, stop and revise the candidate-universe/similarity protocol before training. Record this selection bias in the data card.

Minimum proposed confirmatory support:

- At least 2,000 positives and 200 independent components in training.
- At least 500 positives and 100 components in each validation and test partition.
- No cross-split gene, overlapping-locus, exact-sequence, or detected near-duplicate overlap.

If support is smaller, revise scope before training or label the experiment exploratory. Never search many split seeds for a favorable score.

Fit scalers, feature vocabularies, matching-bin boundaries, weights, calibrators, and feature selection on training data only, except explicitly validation-fitted calibration. Test data cannot select any of these.

Permit automated test QC to inspect metadata, labels, and sample counts. Developers use training/validation examples for debugging. Test scores, attribution figures, and handpicked sequences stay hidden until the final protocol is locked. Record all test evaluations in an append-only log.

## 6. Small model set

| ID | Model | Inputs | Size/budget | Purpose |
|---|---|---|---|---|
| B0 | Constant prevalence | None | 1 parameter | Sanity baseline; expected random average precision equals prevalence |
| B1 | Composition logistic regression | Mononucleotide and dinucleotide frequencies | About 21 coefficients | Quantify simple sequence information |
| B2 | Context logistic regression | B1 plus context, log1p TPM, and eligible input-coverage features | Usually <50 coefficients | Diagnostic nuisance baseline; not a sequence-only peer |
| B3 | k-mer logistic regression | Normalized 3-, 4-, and 5-mer counts | About 1,345 coefficients | Strong inexpensive sequence baseline |
| B4 | Motif logistic regression | Frozen motif maximum score and hit count | A few coefficients | Biological reference, evaluated with its own limitations |
| C0 | TinyCNN, standard training | One-hot sequence only | 13,857 parameters for architecture below | Primary control |
| C1 | TinyCNN, nuisance-balanced training | Same sequence input and architecture | Exactly same count as C0 | Primary intervention |
| T0/T1 | Optional tiny transformer, standard/balanced | Nucleotide tokens only | Hard cap 300,000 parameters | Architecture sensitivity, only after core report |

B2 uses additional information and must be labeled accordingly. B4 should not be “validated” only by the same motif score that defines its input. Use B4 primarily for experimental-label prediction and as a known motif-sensitive reference.

### 6.1 Exact CNN starting architecture

1. One-hot input of shape batch x 4 x 201.
2. Conv1d: 4 -> 32 channels, kernel 11, dilation 1, padding 5; GELU.
3. Conv1d: 32 -> 32, kernel 5, dilation 2, padding 4; GELU.
4. Conv1d: 32 -> 32, kernel 5, dilation 4, padding 8; GELU.
5. Concatenate global mean and global maximum pooling: 64 features.
6. Linear 64 -> 32; GELU; dropout 0.1.
7. Linear 32 -> 1 logit.

Use convolution biases, no normalization layers in this starting architecture, and no sigmoid before BCEWithLogitsLoss. Expected parameter count: 13,857. The local convolutional receptive field is 35 nt; pooling aggregates across the full window. This model intentionally cannot capture every long-range regulatory interaction.

Assert shape and parameter count programmatically. If architecture changes, version the model configuration and recalculate the count.

The optional transformer uses embedding dimension 64, two encoder layers, four attention heads, feed-forward dimension 128, positional encoding, pooled output, and the same classification target. Record the actual parameter count and enforce the cap. It is trained from scratch; do not call it a foundation model.

### 6.2 Optimization protocol

- AdamW; initial learning rate 0.001; weight decay 0.0001.
- Batch size 128, reduced only for resource limits and identically across paired CNN conditions.
- Thirty epochs; gradient clipping norm 1.0.
- Complete all 30 epochs for a fixed update budget; save the checkpoint with best validation average precision. Tie-break by lower validation log loss, then earlier epoch.
- Paired random seeds: 17, 29, 43, 71, 101.
- Within each seed, C0/C1 start from identical parameters and receive identical batches in identical order. Only training loss weights differ.
- No motif-based auxiliary loss or model selection in the primary experiment.
- Start in float32. Enable mixed precision only after numerical parity checks; explanations use float32.
- At most one prespecified fallback learning rate, 0.0003, if validation/training diagnostics show instability. Apply the same rule to both conditions and record an amendment before final evaluation.

For logistic baselines, select regularization from a small declared grid using validation AP. Do not compare dozens of CNN architectures against a weak untuned baseline.

## 7. One intervention: training loss weighting

Change training weights, not labels, test examples, CNN inputs, or model capacity.

Construct nuisance strata using **context category x training-GC quintile x training-expression tertile**. The initial intervention targets GC/context/expression, not every nucleotide statistic. Monitor C/U content separately because matching all composition features could remove part of PTBP1's real preference.

For sparse strata, merge adjacent numeric bins using a deterministic training-only rule until each retained stratum has at least ten positives and ten background windows. Never merge intronic and exonic categories merely to force a pass. If common support is absent, exclude unsupported strata from both training conditions, document the fraction lost, and define the resulting estimand narrowly.

Let N be training size, n_y the number of examples with label y, n_s the size of stratum s, and n_ys the count for label y in stratum s.

- Standard C0 weight: w0(y) = N / (2 n_y).
- Balanced C1 weight: w1(y,s) = n_s / (2 n_ys).

These target globally balanced classes for C0, and equal class mass within each retained stratum for C1. Normalize each weight vector to training mean one. Cap C1 weights at five before final renormalization, and report the balance that remains after clipping. Never assert that the intervention worked solely because weights were computed.

Use the mean of weighted per-example BCE losses with fixed global normalization. Do not independently renormalize weights by each batch's label composition. Both conditions see the same examples and update count.

Intervention QC, checked on weighted training distributions:

- Effective sample size (sum w)^2 / sum(w^2) >= 50% of retained training rows.
- Absolute standardized mean difference for GC and log1p TPM <= 0.10 between classes.
- Absolute difference in each context-category proportion <= 0.05.
- Report residual balance of all mononucleotides, dinucleotides, input signal, and gene length; these are diagnostic, not additional hidden matching requirements.

If thresholds fail, revise strata/weights on training data only, with at most two documented attempts. Otherwise conclude the balancing intervention could not be instantiated as planned. Continue an observational audit if useful, but do not label it a successful balancing experiment.

## 8. Evaluation panels

### 8.1 Predictive panels

**P1: fixed sampled assay panel.** All eligible held-out test windows after the predefined sampling policy. This is the primary predictive test population, with actual prevalence reported.

**P2: matched diagnostic panel.** A prespecified 1:1 subset of P1 matching positive and background windows on context, GC, and expression using training-derived bins. Freeze selection before test scoring. Report retained counts and unmatched positives. This panel asks whether sequence models discriminate examples after simple nuisance information is reduced; it is not representative of the transcriptome.

Report average precision (primary), AUROC, log loss, and Brier score separately for each panel. Use the precise term **average precision (AP)** for sklearn-style average precision, rather than treating every numerical AUPRC implementation as interchangeable.

Do not interpret an AP change from P1 to P2 as transfer degradation: their prevalences differ. Compare C0 versus C1 within each fixed panel. If calibrated probabilities are shown, fit a separate prespecified calibration step on validation examples drawn under the same panel policy, and report raw as well as calibrated metrics. Weighted training logits are not automatically calibrated to assay prevalence.

### 8.2 Motif-order audit: primary mechanistic evidence

Select up to 500 held-out windows from at least 100 components with a supported PTBP1 motif occurrence. Selection is based on label, annotation, motif score, and fixed hashes—not model confidence. Prefer natural windows containing one clear audit target. Predeclare how competing motif occurrences are handled.

Choose the primary experimentally measured motif before model evaluation. Freeze motif-scanning background frequencies and score threshold using training-derived null sequences; proposed threshold corresponds to a 1% window-level false-positive rate in that training null, not a calibrated biological binding probability. Record matrix conventions and pseudocounts. Do not select whichever motif best agrees with the final model.

For each selected motif instance, construct a small fixed panel of up to 20 local order-disrupted controls preserving the edited segment's mononucleotide counts and overall window length. Require a decreased target motif score and check the surrounding context for newly introduced high-scoring occurrences. Retain all valid controls under a deterministic rule; if a low-complexity segment has no valid control, mark it uninformative and exclude it with a count.

For each natural window and model, define motif preference as the fraction of valid controls scored below the unedited window, with numerical ties receiving 0.5. Define ties by an absolute logit difference <=1e-6 in float32 and freeze this tolerance before scoring. Average within each independent component, then across components. This **motif preference accuracy (MPA)** is the primary motif endpoint. It is bounded and avoids directly equating different models' arbitrary logit scales.

Composition-only B1 should be invariant to mononucleotide-preserving edits only if its dinucleotide features are also unchanged; therefore include a separate mononucleotide-only analytic predictor as the exact-invariance control. B1's possible response to changed dinucleotides is expected and must not be called an implementation bug.

Add the following controls:

- Equal-length order changes outside the motif in the same natural windows.
- Composition-matched unrelated/decoy motifs, with their selection fixed before test scoring.
- Original versus edited sequence composition checks, and actual changes in dinucleotide content documented.
- A secondary dinucleotide-preserving local-control analysis when enough valid alternatives exist; low-complexity sites may make this impossible.
- Natural matched-window motif-score analysis, which does not depend on synthetic edits.

This audit supports sensitivity to sequence order beyond preserved composition. Edits can still change RNA structure or other motifs, and synthetic controls can be out of distribution. Failure or success on this panel alone does not establish biological causality. Report coverage and exclusions, not only the favorable examples.

If fewer than 100 independent components or 200 valid natural windows remain, report motif effects as exploratory with uncertainty; do not relax control validity rules to manufacture significance.

### 8.3 Exact attribution and optional approximation

Use explicit in silico single-base substitution scoring (ISM) as the reference for **model behavior**, not experimental mutation effects. For length 201 there are 603 alternative substitutions per sequence.

Select 64 held-out windows in advance: 16 from each positive/background x motif-present/absent stratum, using fixed hashes. If a stratum is too small, report that and use all available examples. Use this same panel for C0/C1 and all seeds. Batch the 603 substitutions; do not build millions of Python objects at once.

Compute effects on logits. For a one-hot input x with observed base r at position i and alternative base a:

    exact_delta(i,a) = f(x with base i replaced by a) - f(x)
    taylor_delta(i,a) = df/dx[i,a] - df/dx[i,r]

The observed-base effect is zero by definition. TISM is optional and should be checked against exact ISM before being used to summarize examples. Sasse's public TISM implementation is a useful reference. [S5]

On validation sequences, require median per-sequence Spearman agreement >=0.8 and sign agreement >=0.8 among the top 10% largest absolute exact effects before using the approximation in headline figures. These are project utility thresholds. If they fail, retain exact ISM; do not change the model just to make TISM pass. Mark constant/near-zero sequences as undefined rather than converting their correlations to zero or one.

Integrated gradients and motif-logo clustering are optional. They must not delay the core result. Attention weights alone are not an explanation validation method.

## 9. Statistical analysis and scientific decision rules

### 9.1 Independent units

Windows from a gene/sequence component are correlated. Bootstrap complete held-out components, not individual bases, edited variants, or repeated windows. Keep C0/C1 predictions paired on exactly the same resampled components.

For AP, compute the metric on all windows in each bootstrap sample and average the resulting paired differences across the five matched seeds. For MPA, average within component before the component bootstrap. Use 2,000 fixed-seed bootstrap replicates and two-sided 95% percentile intervals. If a bootstrap sample has only one label, handle it explicitly and report the fraction of invalid draws.

These intervals describe test-component sampling uncertainty conditional on the five trained seed pairs. Report seed-wise results, median/range, and the fraction of seeds agreeing with the effect direction separately. Do not treat five seeds on the same data as five independent biological experiments or pool them into a falsely enlarged sample size.

### 9.2 Prespecified primary comparison

The sole primary comparison is **C1 minus C0**. Two co-primary requirements define the desired claim:

1. **Predictive non-inferiority on P1:** the lower 95% CI for delta AP must exceed -0.02.
2. **Improved motif preference:** delta MPA must be at least +0.10 in the point estimate, its lower 95% CI must exceed zero, and at least four of five seed pairs must agree in direction.

The margins 0.02 and 0.10 are planning choices. Check feasibility and approximate precision using validation components before final lock. If precision is insufficient, increase independent samples if available or declare the study exploratory. Never reinterpret a wide CI or a nonsignificant difference as equivalence.

Add a **specificity guard** before claiming improved motif recognition: on the same component set with valid target and outside-motif controls, compute S = MPA_target - MPA_outside. Require the lower 95% CI for S(C1) - S(C0) to exceed zero. Outside-motif edits must follow the same length/count-preservation rules, and their sites must be selected without model scores. This guards against a model merely preferring all unedited natural sequences. If paired support is below 100 components, the specificity conclusion remains exploratory regardless of the target-only result.

P2 AP, calibration, attribution behavior, decoy controls, and other RBPs are secondary/supporting analyses. Report all prespecified outcomes. For families of additional hypothesis tests, use a declared multiple-testing correction; do not attach uncorrected significance stars to an exploratory heatmap. The two co-primary requirements and specificity guard form a joint claim rule: all must pass; do not choose whichever one is favorable.

### 9.3 Claim levels

| Level | Required evidence | Permitted wording |
|---|---|---|
| Valid demo | Technical gates passed; frozen results and limitations reported | “A reproducible audit of a small PTBP1 peak predictor” |
| Intervention supported | Both primary requirements and the specificity guard pass | “Balancing improved motif-order evidence while preserving performance in this benchmark” |
| Generalization supported | Above plus a prespecified independent RBP or external cohort replication with new provenance audit | “The finding replicated in the tested additional setting” |
| Mechanism established | Not available from this computational demo alone | Do not claim |

An experiment that does not reach the second level is still deliverable. The validity of the work must never depend on obtaining the hoped-for result.

## 10. Tests and gates

Technical gates stop downstream execution when correctness is uncertain. Scientific decision rules classify results; they do not authorize repeated test-set tuning.

| Gate | Required checks | Pass rule | Failure action |
|---|---|---|---|
| G0: protocol | Scope, hypotheses, file schema, metrics, split algorithm, seeds, budgets, margins | Protocol/config saved with hashes before modeling | Resolve contradictions; no model search |
| G1: provenance | Accessions, checksums, assembly, controls, annotation, peak semantics | All required assets validated; no unresolved assembly/control mismatch | Stop dataset build; document alternative protocol |
| G2: split | Gene/locus/sequence/near-duplicate audit; sample support | Zero detected forbidden overlap; minimum support or explicit exploratory designation | Rebuild data before training; never ignore overlap |
| G3: implementation | Encoding, gradients, metrics, synthetic fixtures, overfit check | Mandatory deterministic tests pass; synthetic learning checks pass | Fix code using synthetic/train data |
| G4: fit sanity | Finite losses, valid checkpoints, label permutation, reproducibility, capacity cap | No invalid numerics; permutation suspiciousness investigated; deterministic reload parity | Diagnose train pipeline before full runs |
| G5: intervention | Weighted balance, effective sample size, identical paired inputs | Balance and ESS thresholds pass; only intended loss weights differ | Revise training-only intervention or label it infeasible |
| G6: final lock | Validation report, code/config/data hashes, checkpoint list, audit panel, analysis script | All technical gates satisfied; final analysis dry run passes on validation | Repair before unlocking test |
| G7: release | One frozen test evaluation, paired CIs, controls, limitations, artifact replay | Every planned result accounted for; clean-environment inference/report reproduction passes | Fix reporting/technical errors transparently; mark test-aware changes exploratory |

### 10.1 Mandatory unit and integration tests

**Coordinates and sequence**

- Hand-checked toy FASTA with positive/negative-strand loci, boundary cases, and a known annotation join.
- Exact 201-base extraction, half-open indexing, reverse-complement round trip, and T/U conversion.
- Reject out-of-bounds/ambiguous bases as configured; no silent truncation.
- Motif scan matches a hand-computed score and does not accidentally scan the wrong RNA orientation.
- The same locus represented in different input files gets one stable window identity.

**Dataset and leakage**

- Deliberately inject same-gene windows, overlaps, exact duplicates, and a detectable near-duplicate into different partitions: the checker must fail.
- Change row order: split IDs and selected panels must not change.
- Add a test-only extreme value: training scalers/bin edges/weights must remain identical.
- Ensure background does not overlap excluded neighborhoods and all source relationships are traceable.
- Prove CNN input tensors contain sequence only; accession, genomic coordinate, gene ID, label, motif score, and coverage cannot become hidden features.

**Models and training**

- Output shape batch x 1; finite forward/backward; parameter count exactly matches the declared architecture.
- In evaluation mode, prediction is invariant to batch order and batch size within float32 tolerance.
- Save/reload reproduces logits with max absolute difference <=1e-6 on the same device/configuration.
- Overfit a fixed 128-example nonconflicting synthetic batch to >=99% training accuracy with dropout disabled. This checks wiring, not generalization.
- C0/C1 initial parameters, batch indices, epoch count, and examples match for each paired seed.
- Weight formulas match a hand-calculated tiny table; effective sample size and clipping logic have fixtures.

**Metrics and explanations**

- AP, AUROC, tie handling, and one-class errors match known examples and the pinned reference implementation.
- ISM leaves input tensors unchanged and gives exactly zero for the reference base.
- On an analytic linear model, exact and Taylor substitution effects agree within 1e-6.
- A nonlinear toy model demonstrates that the approximation can disagree; tests must not assume universal equality.
- Edit validity checks preserve declared counts/length and detect accidental new motif instances.
- Component bootstrap never separates a natural window from its derived edits and uses paired model predictions.

### 10.2 Synthetic learning fixtures

Use artificial nucleotide patterns to test the pipeline independently of biological claims:

1. **True-order fixture:** labels depend on a planted sequence-order pattern, with broad composition matched. TinyCNN should reach AUROC >=0.95 and recover the relevant positions; a mononucleotide-only predictor should remain near chance.
2. **Composition-only fixture:** labels depend only on composition. The simple composition model should succeed. The audit must not incorrectly attribute this to a specific planted order pattern.
3. **Shortcut-shift fixture:** a composition cue is correlated with the true pattern during training but removed or reversed in a test fixture. The reporting pipeline must reveal the resulting performance/behavior change. No requirement that balancing always solves this toy problem.
4. **Random-label fixture:** labels are independent of sequence. Held-out discrimination should be compatible with chance under the fixture's uncertainty.

Use fixed generated datasets and broad tolerances established on development fixtures. Do not make a flaky exact neural-training metric a per-commit gate. Fast deterministic tests run in every CI job; fixed-seed synthetic training runs before release/model changes.

### 10.3 Permutation and randomization controls

Run three small CNN training controls with labels independently permuted in training and validation; evaluate on untouched held-out validation labels. Use a smaller fixed subset for cost control. A consistent AUROC >0.60 or a 95% component CI wholly above 0.5 is an investigation trigger, not automatic proof of leakage. Check chance fluctuations, label handling, duplicate contamination, and preprocessing before proceeding.

Randomize trained network weights and repeat a small explanation panel. Inspect whether explanations lose task-specific behavior. High visual similarity is a warning to investigate, not a universal hard cutoff. Report methods whose explanations are dominated by input composition.

## 11. Implementation layout and interfaces

Proposed modules:

| Path | Responsibility |
|---|---|
| configs/protocol.yaml | All defaults, seeds, thresholds, endpoints, budgets |
| data/source_manifest.json | Verified immutable source metadata and hashes |
| src/motifcheck/data.py | Acquisition validation, interval processing, extraction |
| src/motifcheck/splits.py | Components, hashing, leakage checks |
| src/motifcheck/features.py | Composition, k-mer, context, and motif baselines |
| src/motifcheck/weights.py | Training-only balancing and QC |
| src/motifcheck/models.py | TinyCNN and optional transformer |
| src/motifcheck/train.py | Seeded training, checkpoints, paired run records |
| src/motifcheck/audit.py | Motif controls, ISM, optional TISM |
| src/motifcheck/stats.py | Metrics, paired component bootstrap, claim classification |
| src/motifcheck/gates.py | Machine-readable technical gate checks |
| src/motifcheck/report.py | Tables/figures from frozen result files |
| tests/ | Deterministic fixtures plus separate synthetic training tests |
| reports/ | Protocol, data card, model card, final report, limitations |

Each prediction row contains window_id, split, model_id, seed, logit, probability definition, checkpoint hash, data hash, and config hash. Each edit row links to its original window/component and includes validity checks. A checkpoint must include architecture/config and preprocessing version, not weights alone.

Keep notebooks for exploration and presentation. Every reported result must also be regenerable by a script/CLI. Pin an environment lockfile, record Python/PyTorch/CUDA versions, and keep CPU inference functional for the tiny CNN. Avoid custom CUDA kernels and unnecessary framework dependencies.

Suggested command contract, to be implemented; these commands do not exist yet:

```bash
motifcheck validate-sources --config configs/protocol.yaml
motifcheck build-data --config configs/protocol.yaml
motifcheck check-splits --config configs/protocol.yaml
motifcheck self-test --suite core
motifcheck fit-baselines --split validation
motifcheck train-pairs --models cnn --seeds 17 29 43 71 101
motifcheck audit --split validation
motifcheck lock-analysis --config configs/protocol.yaml
motifcheck evaluate --split test --lock reports/analysis-lock.json
motifcheck report --from-frozen-results
```

Test evaluation must refuse to run without a matching analysis-lock hash and successful technical gate records. This is a workflow guard, not a security boundary. It should not require repeated user permission for routine implementation.

Gate output contract:

```json
{
  "gate": "G2",
  "status": "pass",
  "data_hash": "computed-at-runtime",
  "config_hash": "computed-at-runtime",
  "checks": {
    "cross_split_gene_overlap": 0,
    "cross_split_exact_duplicate_overlap": 0
  },
  "warnings": [],
  "evidence_paths": ["reports/split_audit.json"]
}
```

Every real gate must include all required checks, actual counts, timestamps, and code version. The abbreviated example does not authorize omitting near-duplicate or locus checks.

## 12. Compute and run budget

Design for one GPU with 8–12 GB memory, 16 GB host RAM, and CPU fallback for baselines/inference. These are planning targets; measure the real footprint before committing to a run matrix. The core CNN has no large checkpoint or embedding cache.

| Work item | Planned runs | Budget rule |
|---|---|---|
| Synthetic/self-tests | Small fixed fixtures | CPU/small GPU; complete before real training |
| Linear baselines | B0–B4, small regularization grid | CPU; reuse cached feature matrices |
| CNN pilot | C0/C1, one seed, five epochs | Estimate full-run cost and memory |
| Core CNN matrix | Two conditions x five seeds | Ten full runs; mandatory |
| Permuted-label controls | Three small runs | Use a fixed reduced validation-development cohort |
| ISM | 64 windows x two conditions x five seeds | About 386,000 mutant predictions; batch and stream |
| Optional transformer | Two conditions x three seeds | Only after core report and within remaining budget |

Set an initial total core cap of **24 GPU-hours**, including audits. This is a stop-and-review cap, not a runtime prediction. If the pilot forecast exceeds it, reduce the frozen dataset size before full runs or omit optional work. Retain five paired core seeds and valid independent test support. Do not save compute by quietly reducing replication after seeing unfavorable results.

Cap source downloads at an initial 20 GB; inspect file sizes first. If processed assets exceed this, revise the acquisition plan. Raw FASTQ/BAM processing is outside core scope. Store compact sequences, feature tables, checksums, checkpoints, and predictions; avoid duplicating whole reference genomes per run.

## 13. Ordered implementation backlog

| Milestone | Work | Exit artifact | Dependency |
|---|---|---|---|
| M0 | Freeze question, task distinction, proposed endpoints; time-box SPIDRnet audit | protocol.md and reproduction_status.md | None |
| M1 | Resolve source files; implement extraction and toy coordinate tests | source_manifest.json; data card | G0 |
| M2 | Build candidate universe, leakage components, immutable split; validate support | split_audit.json | G1 |
| M3 | Implement CPU baselines, TinyCNN, weights, deterministic tests | Passing G3; synthetic report | G2 |
| M4 | Pilot training, numerical checks, balancing QC, resource forecast | pilot_report.md | G3 |
| M5 | Run ten paired CNN fits and baselines; inspect validation only | validation_results.parquet | G4, G5 |
| M6 | Validate motif-edit panels and attribution; finalize bootstrap/report scripts | analysis-lock.json | Validation complete |
| M7 | Execute frozen test evaluation once; classify findings | final results and CIs | G6 |
| M8 | Generate report, clean-environment replay, short demo | Tagged repository and research note | G7 |

Expected focused effort: roughly 15–20 working days, with preprocessing and source compatibility the main uncertainty. This is more likely to be 4–6 calendar weeks alongside other commitments. If time is short, deliver the valid one-RBP core with exact ISM and omit the transformer, transfer study, and elaborate UI.

## 14. Failure handling

| Observation | First interpretation/check | Next action |
|---|---|---|
| Baseline cannot overfit a tiny synthetic set | Wiring, loss, labels, optimizer, gradient issue | Stop real-data runs |
| Composition model matches CNN | Could reflect simple real signal or a shortcut | Examine matched panel and motif controls; do not claim bias yet |
| Random-label model performs suspiciously well | Leakage, bug, or finite-sample fluctuation | Investigate before accepting any main result |
| Balancing loses many examples or ESS collapses | Poor common support | Report narrowed population or intervention infeasibility |
| Motif-order controls cannot be constructed | Low-complexity motif constraint | Report exclusions; use natural-window evidence; no fabricated controls |
| ISM and gradients disagree | Nonlinearity or attribution implementation issue | Check analytic fixtures; retain exact ISM if code is correct |
| Better MPA but worse AP | Potential trade-off or removal of genuine context | Report both; do not optimize on test until both look good |
| No CNN advantage over k-mers | A legitimate small-data finding | Document computational efficiency and limits |
| SPIDRnet behavior cannot be reproduced exactly | Missing artifacts or task mismatch | Label Track A partial; keep Track B independent |
| A test-aware code fix changes results | Original confirmatory analysis compromised | Preserve old results, describe amendment, mark rerun exploratory or use a new untouched cohort |

## 15. Deliverables and definition of done

Mandatory outputs:

1. A reproducible repository with locked environment, source acquisition recipe, unit/integration fixtures, and one-command report generation.
2. A data card explaining target population, assembly, background uncertainty, exclusions, split logic, and benchmark prevalence.
3. A model card with exact capacity, optimization, seeds, known limits, and runtime/memory measurements.
4. Frozen prediction/audit tables and code producing all confidence intervals.
5. A two-page research note plus technical appendix.
6. Three central figures: predictive comparison; paired motif-preference effects with controls; representative exact-ISM examples selected by fixed rules.
7. A results table containing all seeds and planned comparisons, including failures and negative results.
8. A claim-level statement using Section 9 and a transparent list of what was not reproduced or tested.

Optional UI: a simple local viewer of precomputed examples, showing observed label, model score, reference motif annotation, and exact model response. Label model outputs and computational edits clearly. The viewer does not retrain models or select examples interactively to improve the scientific result.

**Done means the experiment is traceable, technically valid, and honestly interpretable. It does not mean the intervention won.**

## 16. Extensions, in priority order

Only start after the core report is complete:

1. Replicate the locked protocol for a second human RBP with adequate measured motif/data support; confirm resource availability before selecting it.
2. Test PTBP1 transfer to HepG2 using gene/sequence components assigned consistently across both cell types. Exclude all training-exposed loci from the external generalization panel. Separate this from a same-locus cell-state comparison.
3. Add the tiny transformer under the same primary intervention and evaluate its extra cost.
4. Integrate a pretrained representation as a separately registered study; document pretraining exposure and domain mismatch. This is not needed to make the present demo credible.
5. Extend to base-resolution signal modeling with explicit assay-background modeling, or functional perturbation outcomes. These are thesis-scale directions, not additions to the initial application demo.

## 17. Source notes

Sources support the motivation, existing methods, and candidate resources. The experimental architecture, thresholds, gates, and budget in this plan are proposed design decisions.

- **[S1] SPIDRnet public teaching project.** Reports its model, data processing, interpretation observations, and limitations. Read as a project report rather than an independently replicated publication. https://github.com/sasselab-teaching/SPIDRnet
- **[S2] Horlacher et al., RBPNet, Genome Biology (2023).** Sequence-to-signal learning and explicit modeling of control/background signal. https://doi.org/10.1186/s13059-023-03015-7 ; code: https://github.com/mhorlacher/rbpnet
- **[S3] ENCODE RBP inventory and ENCORE matrix.** Candidate experiment mapping; actual file metadata and access must be validated during acquisition. https://encyclopedia.encodeproject.org/tables/supplementary_table_2 ; https://www.encodeproject.org/encore-matrix/?internal_tags=ENCORE&status=released&type=Experiment
- **[S4] Rosado-Tristani et al., CisBP-RNA.** Provides measured and inferred motifs with provenance. https://doi.org/10.1093/nar/gkaf1081 ; https://www.cisbp.org/rna/
- **[S5] Sasse et al., TISM, iScience (2024).** Gradient-based approximation to computational substitution effects. https://doi.org/10.1016/j.isci.2024.110807 ; https://github.com/LXsasse/TISM
- **[S6] Van Nostrand et al., Nature (2020).** Public binding and perturbation resource; contextual basis for future extensions. https://doi.org/10.1038/s41586-020-2077-3

During planning, some ENCODE pages returned access errors through the browsing interface. The candidate accessions were previously visible in the indexed ENCODE inventory, but no source-file download or checksum validation has been performed. This is why G1 is a mandatory acquisition gate rather than a box already marked complete.
