# Data card — MotifCheck PTBP1 / K562 benchmark

*Generated 2026-09-29T18:45:26+00:00 · dataset version `motifcheck-ptbp1-k562-grch38-v1` · protocol sha256 `d60baba1cfb141ac`*

## Target population

201-nucleotide, strand-oriented genomic windows centred on the midpoint of a reproducible PTBP1 eCLIP peak, or on an eligible sampled background position, restricted to unambiguously annotated genes expressed in K562 (gene TPM >= 1.0).

**Background means *not detected by this assay under these rules*. It does not mean unbound RNA.** No positive control establishes that background windows are genuinely free of PTBP1 occupancy.

## Composition

- Reproducible peaks available in the source file: 7,797
- Positive windows after all filters: **5,352**
- Background windows: **21,936**
- Realised positive:background ratio: **4.099:1**
- Genes represented: 6,939
- Assembly: GRCh38 (201-nt windows, half-open coordinates)

### Split composition

| split | rows | positives | background | components | genes |
|---|---|---|---|---|---|
| test | 3,878 | 754 | 3,124 | 1,011 | 1,020 |
| train | 19,232 | 3,786 | 15,446 | 4,822 | 4,893 |
| validation | 4,178 | 812 | 3,366 | 1,015 | 1,026 |

Target splits are 70/15/15 by **component**, not by window. 128 near-duplicate pairs (>= 0.9 identity over >= 0.8 of the shorter window, reverse-complement duplicates included for leakage detection only) were found and merged before splitting.

## Exclusions

| reason | count |
|---|---|
| ambiguous_gene_or_context | 3,887 |
| collapsed_into_existing_positive | 1,562 |
| background_non_canonical_bases | 8 |

## Known limitations of this dataset

1. **No processed control signal.** ENCSR981WKN releases no size-matched input bigWig. The input-coverage feature was therefore *removed* from baseline B2 rather than zero-filled (amendment A2). Absence of an input file is never encoded as zero coverage.
2. **Positive ceiling.** Only 7,797 reproducible peaks exist, so the plan's 10,000-positive target is unattainable; 5,352 were realised after collapsing overlapping loci (amendment A1).
3. **Artificial prevalence.** The 1:4 ratio is a benchmark convention, not transcriptome-wide peak prevalence. Every precision-recall figure states the prevalence it was computed at.
4. **Selection bias from expression filtering.** Requiring gene TPM >= 1 removes binding at unexpressed loci and biases the population toward transcribed regions.
5. **Class GC is close but not identical** (positives 0.4276, background 0.4168). A strong GC baseline would not by itself establish a shortcut.
6. **Peak strand is an alignment annotation**, not a statement that the bound RNA runs in that direction. Orientation uses it as a prespecified convention.

## Provenance

| asset | id | bytes | sha256 | status |
|---|---|---|---|---|
| annotation | gencode.v49.annotation.gtf.gz | 93,374,019 | `d6e6fe0515c95b2a…` | ok |
| expression | ENCFF255BPX | 31,297,301 | `27d33c96623e1c84…` | ok |
| genome | hg38.fa.gz | 983,659,424 | `c1dd87068c254eb5…` | ok |
| motif_inferred_sensitivity | M12970_2.00 | 312 | `7a062a0acac3b7af…` | ok |
| motif_primary | M00249_2.00 | 545 | `a94af776c2ed414f…` | ok |
| motif_secondary | M00250_2.00 | 543 | `4f3985620487e8c4…` | ok |
| peaks | ENCFF907HNN | 162,406 | `6b4438f26ddbf868…` | ok |
| peaks_replicate_1 | ENCFF594PWG | 1,004,814 | `885780ea183b1acc…` | ok |
| peaks_replicate_2 | ENCFF894KLP | 1,476,599 | `0a934d420e58e8a1…` | ok |

Full manifest with URLs, retrieval dates, licenses and per-asset validation: `data/source_manifest.json`.
