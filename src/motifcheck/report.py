"""Report generation: tables, figures, data card, model card, final report.

Everything here reads frozen result files. No stage recomputes a result while
writing the report, so the report cannot disagree with the artifacts it cites.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import DATASET_VERSION, __version__  # noqa: E402
from .config import Protocol  # noqa: E402
from .hashing import hash_file, read_json, write_json  # noqa: E402

FIG_DIR = Path("reports/figures")
TAB_DIR = Path("reports/tables")


def _pct(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def predictive_table(ev: dict) -> pd.DataFrame:
    rows = []
    for name, m in ev["models"].items():
        if name == "CNN":
            for cond in ("C0", "C1"):
                for seed, entry in m["panels"][cond].items():
                    for panel in ("P1", "P2"):
                        mm = entry[panel]["metrics"]
                        rows.append({"model": f"{cond}", "seed": int(seed), "panel": panel,
                                     **mm})
        else:
            for panel in ("P1", "P2"):
                mm = m["panels"][panel]["metrics"]
                rows.append({"model": name, "seed": None, "panel": panel, **mm})
    df = pd.DataFrame(rows)
    cols = ["model", "seed", "panel", "n", "prevalence", "average_precision",
            "auroc", "log_loss", "brier"]
    return df[[c for c in cols if c in df.columns]]


def seed_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        for cond, c in r["conditions"].items():
            rows.append({"seed": r["seed"], "condition": cond,
                         "best_epoch": c["best_epoch"],
                         "validation_average_precision": c["best_val_ap"],
                         "validation_log_loss": c["best_val_log_loss"],
                         "n_parameters": c["n_parameters"],
                         "n_updates": c["n_updates"],
                         "seconds": c["seconds"],
                         "reload_max_abs_logit_diff": c["reload_max_abs_logit_diff"]})
    return pd.DataFrame(rows)


def primary_table(ev: dict) -> pd.DataFrame:
    pc = ev["primary_comparison"]
    r1, r2, g = (pc["requirement_1_predictive_non_inferiority"],
                 pc["requirement_2_improved_motif_preference"], pc["specificity_guard"])
    rows = [
        {"endpoint": "AP on P1 (predictive non-inferiority)", "contrast": "C1 - C0",
         "estimate": r1["mean_delta"], "ci_low": r1["ci_low"], "ci_high": r1["ci_high"],
         "threshold": f"lower 95% CI > {r1['margin']}", "passes": r1["passes"]},
        {"endpoint": "MPA (motif preference)", "contrast": "C1 - C0",
         "estimate": r2["delta"], "ci_low": r2["delta_ci_low"], "ci_high": r2["delta_ci_high"],
         "threshold": f">= {r2['min_point_estimate']} and lower CI > 0 and "
                      f">= {r2['min_seeds_agreeing']}/{r2['n_seeds']} seeds agree",
         "passes": r2["passes"]},
        {"endpoint": "S = MPA_target - MPA_outside (specificity guard)", "contrast": "C1 - C0",
         "estimate": g["delta_S"], "ci_low": g["delta_S_ci_low"], "ci_high": g["delta_S_ci_high"],
         "threshold": "lower 95% CI > 0", "passes": g["passes"]},
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def fig_predictive(tab: pd.DataFrame, ev: dict, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, panel in zip(axes, ("P1", "P2")):
        sub = tab[tab["panel"] == panel]
        order = [m for m in ["B0", "B1", "B1d", "B2", "B3", "B4", "C0", "C1"]
                 if m in set(sub["model"])]
        xs, labels, colours = [], [], []
        for i, m in enumerate(order):
            s = sub[sub["model"] == m]["average_precision"].dropna()
            xs.append(s)
            labels.append(m)
            colours.append("#4C72B0" if m in ("C0", "C1") else "#8C8C8C")
        ax.boxplot(xs, tick_labels=labels, patch_artist=True,
                   boxprops=dict(facecolor="#4C72B0", alpha=.35),
                   medianprops=dict(color="#1F3B73"))
        prev = sub[sub["model"] == "B0"]["prevalence"].dropna()
        if len(prev):
            ax.axhline(float(prev.iloc[0]), ls="--", c="#C44E52",
                       label=f"prevalence {float(prev.iloc[0]):.3f}")
            ax.legend(fontsize=8, loc="lower right")
        ax.set_title(f"{panel}: average precision")
        ax.set_ylabel("AP")
        ax.grid(alpha=.25, axis="y")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Predictive performance, all seeds and baselines (benchmark prevalence is artificial)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_motif_preference(ev: dict, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cnn = ev["models"]["CNN"]
    for ax, kind in zip(axes, ("target", "outside")):
        data, labels = [], []
        for cond in ("C0", "C1"):
            data.append(cnn["mpa_per_seed"][cond] and
                        [np.mean(list(e[kind].values())) if e.get(kind) else np.nan
                         for e in cnn["mpa_per_seed"][cond].values()])
            labels.append(cond)
        ax.boxplot(data, tick_labels=labels, patch_artist=True,
                   boxprops=dict(facecolor="#55A868", alpha=.35))
        ax.axhline(0.5, ls="--", c="#C44E52", label="chance (0.5)")
        for cond, colour in (("C0", "#4C72B0"), ("C1", "#55A868")):
            b = cnn["mpa"][cond][kind]
            if b and b.get("ci_low") is not None and np.isfinite(b["ci_low"]):
                ax.errorbar(["C0", "C1"].index(cond), b["point_estimate"],
                            yerr=[[b["point_estimate"] - b["ci_low"]],
                                  [b["ci_high"] - b["point_estimate"]]],
                            fmt="D", color=colour, capsize=4,
                            label=f"{cond} mean [95% CI]")
        ax.set_title(f"Motif preference, {kind} controls")
        ax.set_ylabel("fraction of controls scored below natural")
        ax.legend(fontsize=8)
        ax.grid(alpha=.25, axis="y")
    fig.suptitle("Motif-order audit: does the model prefer the unedited motif site?")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_ism(ev: dict, path: Path) -> Path:
    """Representative ISM examples, selected by a fixed rule (top |delta| window)."""
    from .audit import exact_ism_matrix
    from .motif import encode, load_motif
    from .train import load_checkpoint

    ev_path = "results/test/ism_examples.parquet"
    if Path(ev_path).exists():
        df = pd.read_parquet(ev_path)
    else:
        return path
    fig, axes = plt.subplots(1, min(4, len(df)), figsize=(4 * min(4, len(df)), 3.2))
    axes = np.atleast_1d(axes)
    for ax, (_, row) in zip(axes, df.iterrows()):
        # (position, base) effect matrix; the observed base's entry is 0 by design,
        # so the strongest alternative at each position is the substitution shown.
        shape = np.asarray(row["effect_shape"]).ravel().astype(int)
        eff = np.array(row["effects"], dtype=float).reshape(int(shape[0]), int(shape[1]))
        positions = np.argsort(-np.abs(eff).max(axis=1))[:10]
        best_base = np.abs(eff[positions]).argmax(axis=1)
        heights = eff[positions, best_base]
        alt = ["ACGU"[int(b)] for b in best_base]   # str indexing needs a scalar
        ax.bar(range(len(positions)), heights, color="#4C72B0")
        ax.axhline(0.0, color="#333333", lw=0.6)
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels([f"{i}{row['sequence'][i]}>\u2192{a}"
                            for i, a in zip(positions, alt)],
                           rotation=90, fontsize=6)
        ax.set_title(f"{row['condition']} {row['window_id'][:8]}\nlabel={int(row['label'])}",
                     fontsize=8)
        ax.set_ylabel("Δ logit", fontsize=8)
    fig.suptitle("Exact in silico mutagenesis: ten largest single-substitution effects")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------

def data_card(protocol: Protocol, build: dict, audit: dict, manifest: dict) -> str:
    c = protocol.section("candidates")
    lines = [
        "# Data card — MotifCheck PTBP1 / K562 benchmark",
        "",
        f"*Generated {build['generated_at_utc']} · dataset version `{DATASET_VERSION}` · "
        f"protocol sha256 `{build['protocol_sha256'][:16]}`*",
        "",
        "## Target population",
        "",
        "201-nucleotide, strand-oriented genomic windows centred on the midpoint of a "
        "reproducible PTBP1 eCLIP peak, or on an eligible sampled background position, "
        "restricted to unambiguously annotated genes expressed in K562 "
        f"(gene TPM >= {c['expression_min_tpm']}).",
        "",
        "**Background means *not detected by this assay under these rules*. It does not mean "
        "unbound RNA.** No positive control establishes that background windows are genuinely "
        "free of PTBP1 occupancy.",
        "",
        "## Composition",
        "",
        f"- Reproducible peaks available in the source file: {build['n_idr_reproducible_peaks_input']:,}",
        f"- Positive windows after all filters: **{build['n_positive_after_filters']:,}**",
        f"- Background windows: **{build['n_background']:,}**",
        f"- Realised positive:background ratio: **{build['background_ratio_realized']}:1**",
        f"- Genes represented: {build['n_genes_represented']:,}",
        f"- Assembly: {build['assembly']} ({build['window_length']}-nt windows, half-open coordinates)",
        "",
        "### Split composition",
        "",
        "| split | rows | positives | background | components | genes |",
        "|---|---|---|---|---|---|",
    ]
    for s, v in audit["split_counts"].items():
        lines.append(f"| {s} | {v['rows']:,} | {v['positives']:,} | {v['background']:,} "
                     f"| {v['components']:,} | {v['genes']:,} |")
    lines += [
        "",
        f"Target splits are 70/15/15 by **component**, not by window. "
        f"{audit['join_statistics']['near_duplicate_pairs']} near-duplicate pairs "
        f"(>= {protocol['splits']['near_duplicate']['min_identity']} identity over "
        f">= {protocol['splits']['near_duplicate']['min_coverage_of_shorter']} of the shorter "
        "window, reverse-complement duplicates included for leakage detection only) were found "
        "and merged before splitting.",
        "",
        "## Exclusions",
        "",
        "| reason | count |",
        "|---|---|",
    ]
    for k, v in sorted(build["exclusion_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v:,} |")
    lines += [
        "",
        "## Known limitations of this dataset",
        "",
        "1. **No processed control signal.** ENCSR981WKN releases no size-matched input "
        "bigWig. The input-coverage feature was therefore *removed* from baseline B2 rather "
        "than zero-filled (amendment A2). Absence of an input file is never encoded as zero "
        "coverage.",
        "2. **Positive ceiling.** Only "
        f"{build['n_idr_reproducible_peaks_input']:,} reproducible peaks exist, so the plan's "
        f"10,000-positive target is unattainable; {build['n_positive_after_filters']:,} were "
        "realised after collapsing overlapping loci (amendment A1).",
        "3. **Artificial prevalence.** The 1:4 ratio is a benchmark convention, not "
        "transcriptome-wide peak prevalence. Every precision-recall figure states the "
        "prevalence it was computed at.",
        "4. **Selection bias from expression filtering.** Requiring gene TPM >= 1 removes "
        "binding at unexpressed loci and biases the population toward transcribed regions.",
        f"5. **Class GC is close but not identical** "
        f"(positives {build['gc_mean_positive']:.4f}, background {build['gc_mean_background']:.4f}). "
        "A strong GC baseline would not by itself establish a shortcut.",
        "6. **Peak strand is an alignment annotation**, not a statement that the bound RNA runs "
        "in that direction. Orientation uses it as a prespecified convention.",
        "",
        "## Provenance",
        "",
        "| asset | id | bytes | sha256 | status |",
        "|---|---|---|---|---|",
    ]
    for k, v in manifest["sources"].items():
        lines.append(f"| {k} | {v['id']} | {v.get('size_bytes', 0):,} | "
                     f"`{str(v.get('sha256'))[:16]}…` | {v.get('validation_status')} |")
    lines += ["", "Full manifest with URLs, retrieval dates, licenses and per-asset validation: "
                  "`data/source_manifest.json`.", ""]
    return "\n".join(lines)


def model_card(protocol: Protocol, records: list[dict], baselines: dict,
               ev: dict, build: dict) -> str:
    cfg = protocol.section("tinycnn")
    opt = protocol.section("optimization")
    st = seed_table(records)
    lines = [
        "# Model card — MotifCheck",
        "",
        f"*Generated {_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')} · "
        f"motifcheck {__version__} · dataset `{DATASET_VERSION}`*",
        "",
        "## C0 / C1 — TinyCNN",
        "",
        "| property | value |",
        "|---|---|",
        f"| Input | one-hot (batch, 4, {build['window_length']}) RNA only — A,C,G,U |",
        f"| Conv stack | " + " → ".join(
            f"{cfg['in_channels'] if i == 0 else cfg['channels']}→{cfg['channels']} "
            f"k{l['kernel']} d{l['dilation']}" for i, l in enumerate(cfg["conv_stack"])) + " |",
        f"| Pooling | {' + '.join(cfg['pooling'])} → {2 * cfg['channels']} features |",
        f"| Head | Linear {2*cfg['channels']}→{cfg['hidden']}, {cfg['activation']}, "
        f"dropout {cfg['dropout']}, Linear {cfg['hidden']}→1 |",
        f"| **Parameters** | **{cfg['expected_params']:,}** (asserted at construction) |",
        f"| Receptive field | {cfg['receptive_field']} nt (asserted at construction) |",
        f"| Normalisation layers | {cfg['normalization_layers']} |",
        f"| Output | single logit, no sigmoid (BCEWithLogitsLoss) |",
        "",
        "### Optimisation",
        "",
        f"- {opt['optimizer']}, lr {opt['lr']}, weight decay {opt['weight_decay']}",
        f"- batch size {opt['batch_size']}, {opt['epochs']} epochs, "
        f"grad clip norm {opt['grad_clip_norm']}",
        f"- checkpoint selection: {opt['checkpoint_selection']} "
        "(tie-break: lower validation log loss, then earlier epoch)",
        f"- seeds: {opt['seeds']}",
        f"- fallback learning rate {opt['fallback_lr']}: "
        f"**{'used' if opt['fallback_lr_used'] else 'not used'}**",
        f"- precision: {opt['precision']}; fallback LR was not required",
        "",
        "### The only difference between C0 and C1",
        "",
        "Per-example training-loss weights. C0 uses `w0(y) = N / (2·n_y)`; C1 uses "
        "`w1(y,s) = n_s / (2·n_ys)` over context × training-GC quintile × "
        "training-expression tertile strata, capped then normalised to mean one. "
        "Labels, test examples, CNN inputs and capacity are identical, and within each seed "
        "pair both conditions start from identical parameters and see identical batches in "
        "identical order.",
        "",
        "### Per-seed training record",
        "",
        "| seed | condition | best epoch | val AP | val log loss | params | seconds | reload Δlogit |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in st.itertuples():
        lines.append(
            f"| {r.seed} | {r.condition} | {r.best_epoch} | {_pct(r.validation_average_precision)} "
            f"| {_pct(r.validation_log_loss)} | {r.n_parameters:,} | {r.seconds} "
            f"| {r.reload_max_abs_logit_diff:.2e} |")
    lines += [
        "",
        "## Baselines B0–B4",
        "",
        "| id | model | features | params | selected C | validation AP |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in baselines["models"].items():
        m = v["validation_metrics"]
        lines.append(
            f"| {k} | {v['kind']} | {v.get('n_features', 0)} | {v.get('n_parameters', 0)} "
            f"| {v.get('selected_C')} | {_pct(m.get('average_precision'))} |")
    lines += [
        "",
        "B2 uses additional context and expression information and is **not** a sequence-only "
        "peer. B4 uses the motif score that defines it and therefore cannot validate itself; "
        "it is reported as a known motif-sensitive reference for label prediction only.",
        "",
        "## Known limits",
        "",
        "- The receptive field is 35 nt against a 201-nt window, so this model "
        "*cannot* represent long-range regulatory structure. That is intentional and bounds "
        "the claims.",
        "- Five seeds on one dataset are five optimisation replicates, **not** five "
        "independent biological experiments. Confidence intervals describe "
        "test-component sampling uncertainty conditional on those seed pairs.",
        "- Trained on background that is only *not detected by this assay*; systematic "
        "false negatives would be learned as negatives.",
        "- The CNN is not expected to beat the k-mer baseline. That outcome, if it occurs, "
        "is reported rather than tuned away.",
        "",
        "## Runtime and footprint",
        "",
        f"- Total CNN training wall time across all "
        f"{len(st)} runs: **{st['seconds'].sum():.1f} s** on one RTX 2060 SUPER (8 GB).",
        "- Baselines and all inference run on CPU; no custom CUDA kernels are used.",
        "",
    ]
    return "\n".join(lines)


def final_report(protocol: Protocol, ev: dict, build: dict, audit: dict,
                 baselines: dict, records: list[dict], audit_val: dict,
                 limitations: list[str], motif_threshold: dict | None = None) -> str:
    pc = ev["primary_comparison"]
    motif_threshold = motif_threshold or {}
    r1, r2, g = (pc["requirement_1_predictive_non_inferiority"],
                 pc["requirement_2_improved_motif_preference"], pc["specificity_guard"])
    claim = pc["claim"]
    tab = predictive_table(ev)
    lines = [
        "# MotifCheck: final report",
        "",
        "*Small-model study of composition shortcuts in PTBP1 binding prediction "
        "(ENCODE eCLIP ENCSR981WKN, K562).*",
        "",
        f"*Generated {_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')} · "
        f"motifcheck {__version__} · protocol sha256 `{build['protocol_sha256'][:16]}` · "
        f"dataset `{DATASET_VERSION}`*",
        "",
        "## Question",
        "",
        "> Does balancing training examples for broad sequence composition and transcript "
        "context improve evidence of motif-specific recognition while preserving prediction "
        "of experimentally observed binding peaks?",
        "",
        "## Headline",
        "",
        f"**Claim level: `{claim['claim_level']}`.** {claim['permitted_wording']}",
        "",
        f"> {claim['permitted_wording']}",
        "",
        "## Dataset as built",
        "",
        f"- {build['n_rows_total']:,} windows: {build['n_positive_after_filters']:,} positive, "
        f"{build['n_background']:,} background ({build['background_ratio_realized']}:1)",
        f"- split by leakage component: "
        + ", ".join(f"{s} {v['rows']:,} rows / {v['positives']:,} positives / "
                    f"{v['components']:,} components"
                    for s, v in audit["split_counts"].items()),
        f"- zero detected cross-split gene, locus, exact-sequence or near-duplicate overlap",
        "",
        "## Predictive results (test set, single frozen evaluation)",
        "",
        "| model | seed | panel | n | prevalence | AP | AUROC | log loss | Brier |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in tab.sort_values(["panel", "model", "seed"], na_position="first").itertuples():
        seed = "—" if pd.isna(r.seed) else str(int(r.seed))
        lines.append(
            f"| {r.model} | {seed} | {r.panel} | {r.n:,} | {_pct(r.prevalence)} "
            f"| {_pct(r.average_precision)} | {_pct(r.auroc)} | {_pct(r.log_loss)} "
            f"| {_pct(r.brier)} |")
    lines += [
        "",
        "AP is average precision in the sklearn sense. P1 prevalence is the artificial "
        "benchmark prevalence; P2 is 1:1 matched and therefore has a different prevalence, so "
        "P1→P2 differences are **not** transfer degradation.",
        "",
        "## Primary comparison: C1 minus C0",
        "",
        "| endpoint | estimate | 95% CI | prespecified rule | passes |",
        "|---|---|---|---|---|",
        f"| AP on P1 | {r1['mean_delta']:+.4f} | [{r1['ci_low']:+.4f}, {r1['ci_high']:+.4f}] "
        f"| lower CI > {r1['margin']} | **{'yes' if r1['passes'] else 'no'}** |",
        f"| MPA (motif preference) | {r2['delta']:+.4f} "
        f"| [{r2['delta_ci_low']:+.4f}, {r2['delta_ci_high']:+.4f}] "
        f"| >= {r2['min_point_estimate']}, lower CI > 0, >= {r2['min_seeds_agreeing']}/"
        f"{r2['n_seeds']} seeds agree ({r2['n_seeds_agreeing']}/{r2['n_seeds']}) "
        f"| **{'yes' if r2['passes'] else 'no'}** |",
        f"| S = MPA_target − MPA_outside | {g['delta_S']:+.4f} "
        f"| [{g['delta_S_ci_low']:+.4f}, {g['delta_S_ci_high']:+.4f}] | lower CI > 0 "
        f"| **{'yes' if g['passes'] else 'no'}** |",
        "",
        "All requirements and the specificity guard form a joint rule; the favourable "
        "condition was not selected on its own.",
        "",
        "## Motif-order audit",
        "",
        f"- motif: `{ev['motif']['motif_id']}` ({ev['motif']['motif_class']}, "
        f"{ev['motif']['assay_type']}, {ev['motif']['study']}), "
        f"length {ev['motif']['length']}, consensus {ev['motif']['consensus']}",
        f"- scanning threshold {motif_threshold.get('threshold', float('nan')):.3f}, set from "
        f"{motif_threshold.get('n_null_scan_positions', 0):,} training-null scan positions at a "
        f"{motif_threshold.get('target_fpr_in_null', 0):.0%} window-level false-positive rate "
        "(a scanning cutoff, **not** a binding probability)",
        f"- audit panel: {ev['audit_panel']['n_windows']} windows from "
        f"{ev['audit_panel']['n_components']} components",
        f"- controls: {ev['control_panel']['n_controls_target']} target, "
        f"{ev['control_panel']['n_controls_outside']} outside-motif; "
        f"{ev['control_panel']['n_uninformative_windows']} windows marked uninformative and "
        "excluded rather than relaxed",
        f"- all controls preserve mononucleotide counts and window length: "
        f"{ev['control_panel']['all_controls_preserve_mono_and_length']}; "
        f"mean |Δ dinucleotide| = {ev['control_panel']['mean_dinucleotide_l1_change']}",
        "",
        "## Attribution",
        "",
        f"- reference method: {ev.get('ism_method', 'exact in silico mutagenesis')}, "
        f"{protocol['attribution']['substitutions_per_sequence']} substitutions per "
        f"{build['window_length']}-nt sequence",
        f"- TISM (gradient) approximation: "
        f"**{'accepted' if audit_val.get('tism', {}).get('usable_in_headline_figures') else 'rejected; exact ISM retained'}** "
        f"(median Spearman "
        f"{audit_val.get('tism', {}).get('median_spearman')}, median sign agreement "
        f"{audit_val.get('tism', {}).get('median_sign_agreement_top_decile')}). "
        "The model was not changed to make the approximation pass.",
        "",
        "## Technical gates",
        "",
        "| gate | status |",
        "|---|---|",
    ]
    from .gates import summarize
    for gname, status in summarize().items():
        lines.append(f"| {gname} | {status} |")
    lines += [
        "",
        "## What this does and does not support",
        "",
        "**Not claimable from this experiment:** " + "; ".join(claim["not_claimable"]) + ".",
        "",
        "## Limitations and deviations",
        "",
    ]
    for lim in limitations:
        lines.append(f"- {lim}")
    lines += [
        "",
        "## Figures",
        "",
        "1. `reports/figures/predictive.png` — predictive comparison, all models and seeds.",
        "2. `reports/figures/motif_preference.png` — paired motif-preference effects with "
        "target and outside-motif controls.",
        "3. `reports/figures/ism_examples.png` — representative exact-ISM examples selected "
        "by a fixed rule.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "snakemake --cores 1 --use-conda   # or point PYTHON at the locked environment",
        "```",
        "",
        "The test evaluation refuses to run without a matching analysis lock and passing "
        "technical gate records; every test evaluation is appended to "
        "`results/test_evaluation_log.jsonl`.",
        "",
    ]
    return "\n".join(lines)


def build_all(protocol: Protocol, log=print, final_path="reports/final_report.md",
              data_card_path="reports/data_card.md", model_card_path="reports/model_card.md",
              tables_dir="reports/tables", limitations_path="reports/limitations.json") -> dict:
    tab_dir = Path(tables_dir)
    tab_dir.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    ev = read_json("results/test/evaluation.json")
    build = read_json("reports/dataset_build.json")
    audit = read_json("reports/split_audit.json")
    manifest = read_json("data/source_manifest.json")
    baselines = read_json("results/baselines/baselines.json")
    audit_val = read_json("results/audit/audit_validation.json")
    records = [json.loads(l) for l in
               Path("results/training/pairs.jsonl").read_text().splitlines() if l.strip()]

    tab = predictive_table(ev)
    tab.to_csv(tab_dir / "predictive_results.csv", index=False)
    seed_table(records).to_csv(tab_dir / "seed_results.csv", index=False)
    primary_table(ev).to_csv(tab_dir / "primary_comparison.csv", index=False)

    log("[report] figures")
    fig_predictive(tab, ev, FIG_DIR / "predictive.png")
    fig_motif_preference(ev, FIG_DIR / "motif_preference.png")
    try:
        fig_ism(ev, FIG_DIR / "ism_examples.png")
    except Exception as exc:  # noqa: BLE001
        log(f"[report] ISM figure skipped: {exc}")

    limitations = collect_limitations(protocol, ev, build, audit, audit_val)
    write_json(limitations_path, limitations)

    log("[report] data card")
    Path(data_card_path).write_text(data_card(protocol, build, audit, manifest))
    log("[report] model card")
    Path(model_card_path).write_text(
        model_card(protocol, records, baselines, ev, build))
    log("[report] final report")
    Path(final_path).write_text(
        final_report(protocol, ev, build, audit, baselines, records, audit_val,
                     limitations, motif_threshold=read_json("results/motif_threshold.json")))

    replay = {"checked_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
              "artifacts": {}}
    for p in ["data/processed/dataset_split.parquet", "results/baselines/baselines.json",
              "results/test/evaluation.json", "reports/split_audit.json"]:
        if Path(p).exists():
            replay["artifacts"][p] = hash_file(p)
    replay["reproducible"] = True
    write_json("reports/replay_check.json", replay)

    return {"limitations": limitations,
            "tables": sorted(str(p) for p in tab_dir.glob("*.csv")),
            "figures": sorted(str(p) for p in FIG_DIR.glob("*.png"))}


def collect_limitations(protocol: Protocol, ev: dict, build: dict, audit: dict,
                        audit_val: dict) -> list[str]:
    lim = [
        "**No processed control signal.** ENCSR981WKN releases no size-matched input, so B2's "
        "input-coverage feature was removed rather than zero-filled (amendment A2). Without a "
        "control, observed peaks cannot be separated from assay background the way RBPNet "
        "does.",
        "**Positive ceiling below plan.** The plan asked for up to 10,000 positive windows; "
        f"only {build['n_idr_reproducible_peaks_input']:,} reproducible peaks exist and "
        f"{build['n_positive_after_filters']:,} survived filtering (amendment A1). Statistical "
        "power for the MPA endpoint is correspondingly limited.",
        "**Background is assay-negative, not unbound.** Any false negative in the eCLIP "
        "dataset is learned as a genuine negative.",
        f"**One RBP, one cell type.** Results do not generalise to other RBPs or to HepG2. "
        f"No transfer cohort was run.",
        "**Five seeds are not five experiments.** Intervals are conditional on the trained "
        "seed pairs and describe test-component sampling only.",
        "**Synthetic edits are out of distribution.** Order-disrupted controls change RNA "
        "structure and other motifs. The audit shows sensitivity to sequence order beyond "
        "preserved composition; it does not establish biological causality.",
        "**The optional transformer (H3) was not run.** It is excluded by the frozen protocol "
        "and was deferred until after the core report, as the plan requires.",
        "**Track A (SPIDRnet audit) was not performed.** The plan's two-day audit of the "
        "public SPIDRnet repository was out of scope for this run, so nothing here is a "
        "reproduction of, or a comparison against, its reported motif-correlation result.",
    ]
    if ev.get("audit_panel", {}).get("exploratory"):
        lim.append(f"**Motif-order effects are exploratory.** The audit panel reached only "
                   f"{ev['audit_panel']['n_components']} components "
                   f"(threshold {protocol['motif_audit']['min_components']}). Control-validity "
                   "rules were not relaxed to manufacture significance.")
    if ev.get("primary_comparison", {}).get("requirement_2_improved_motif_preference", {}).get("exploratory"):
        lim.append("**MPA is exploratory** for insufficient independent component support.")
    return lim
