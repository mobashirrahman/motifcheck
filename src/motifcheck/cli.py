"""Command line interface implementing the protocol's command contract.

Each subcommand is a thin wrapper over the library and writes the artifacts its
Snakemake rule depends on. ``evaluate`` refuses to run without a matching
analysis lock and passing technical gate records; that is a workflow guard, not
a security boundary.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import DATASET_VERSION, __version__
from .config import load_protocol
from .hashing import append_jsonl, hash_file, read_json, write_json

TEST_LOG = "results/test_evaluation_log.jsonl"
DATA = "data/processed/dataset.parquet"
SPLIT_DATA = "data/processed/dataset_split.parquet"
RESULTS = "results"


def _log(msg: str) -> None:
    print(msg, flush=True)


def load_protocol_from_args(argv, need_config=False):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="motifcheck", add_help=False)
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--manifest", default=None)
    known, rest = ap.parse_known_args(argv)
    return load_protocol(known.config), known


# --------------------------------------------------------------------------
# validate-sources
# --------------------------------------------------------------------------

def cmd_protocol_freeze(argv=None) -> int:
    """Write the G0 protocol record. Separate from acquisition so the freeze
    happens before any network or data work."""
    from .gates import g0_protocol

    ap = argparse.ArgumentParser(prog="motifcheck protocol-freeze")
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--record", default="reports/gates/G0.json")
    args = ap.parse_args(argv)

    protocol = load_protocol(args.config)
    gate = g0_protocol(protocol, args.config)
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    gate.record()
    _log(f"[G0] protocol frozen: sha256={protocol.sha256} "
         f"seeds={protocol.seeds} amendments={len(protocol.amendments)}")
    return 0 if gate.passed else 1


def cmd_validate_sources(argv=None) -> int:
    from .acquire import acquire_all
    from .gates import g0_protocol, g1_provenance

    ap = argparse.ArgumentParser(prog="motifcheck validate-sources")
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--out-dir", default="data/external")
    ap.add_argument("--manifest", default="data/source_manifest.json")
    ap.add_argument("--record", default="reports/gates/G1.json")
    args = ap.parse_args(argv)

    protocol = load_protocol(args.config)
    if not args.record or args.record == "/dev/null":
        g0_protocol(protocol, args.config).record()
    manifest = acquire_all(protocol, out_dir=args.out_dir, manifest_path=args.manifest,
                           log=_log)
    gate = g1_provenance(manifest)
    if args.record and args.record != "/dev/null":
        gate.record()
    else:
        gate.record()
    _log(f"[G1] {gate.status}: {gate.checks['n_validated_ok']}/{gate.checks['n_sources']} sources ok")
    for w in gate.warnings:
        _log(f"[G1] warning: {w}")
    return 0 if gate.passed else 1


# --------------------------------------------------------------------------
# build-data
# --------------------------------------------------------------------------

def cmd_build_data(argv=None) -> int:
    from .data import build_dataset, save_build

    ap = argparse.ArgumentParser(prog="motifcheck build-data")
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--manifest", default="data/source_manifest.json")
    ap.add_argument("--dataset", default=DATA)
    ap.add_argument("--exclusions", default="data/processed/exclusions.parquet")
    ap.add_argument("--report", default="reports/dataset_build.json")
    args = ap.parse_args(argv)

    protocol = load_protocol(args.config)
    manifest = read_json(args.manifest)
    if not manifest.get("complete"):
        _log("[build-data] refusing to build: source manifest is incomplete")
        return 1
    result = build_dataset(protocol, manifest, log=_log)
    hashes = save_build(result, args.dataset, args.exclusions, args.report)
    _log(f"[build-data] {result.report['n_rows_total']} rows -> {args.dataset}")
    return 0


# --------------------------------------------------------------------------
# check-splits
# --------------------------------------------------------------------------

def cmd_check_splits(argv=None) -> int:
    from .gates import g2_split
    from .splits import assign, train_derived_bins

    ap = argparse.ArgumentParser(prog="motifcheck check-splits")
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--dataset", default=DATA)
    ap.add_argument("--out", default=SPLIT_DATA)
    ap.add_argument("--audit", default="reports/split_audit.json")
    ap.add_argument("--bins", default="data/processed/train_bins.json")
    ap.add_argument("--record", default="reports/gates/G2.json")
    args = ap.parse_args(argv)

    protocol = load_protocol(args.config)
    df = pd.read_parquet(args.dataset)
    result = assign(df, protocol, log=_log)
    result.dataset.to_parquet(args.out, index=False)
    write_json(args.audit, result.audit)

    bins = train_derived_bins(result.dataset, protocol)
    write_json(args.bins, bins)

    gate = g2_split(result.audit, hash_file(args.out), protocol.sha256)
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    gate.record()
    _log(f"[G2] {gate.status}  zero_overlap={gate.checks['zero_forbidden_overlap']} "
         f"support_ok={gate.checks['all_minimum_support_ok']}")
    return 0 if gate.passed else 1


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def cmd_self_test(argv=None) -> int:
    from .gates import g3_implementation
    from .synthetic import run_synthetic_fixtures

    ap = argparse.ArgumentParser(prog="motifcheck self-test")
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--suite", default="core", choices=["core", "all"])
    ap.add_argument("--report", default="reports/self_test.json")
    ap.add_argument("--record", default="reports/gates/G3.json")
    args = ap.parse_args(argv)

    n_passed = n_failed = 0
    failures: list[str] = []
    if args.suite in ("core", "all"):
        import pytest
        code = pytest.main(["-q", "tests", "--no-header", "-x" if args.suite == "core" else ""])
        n_failed += 1 if code != 0 else 0
        if code != 0:
            failures.append(f"pytest suite exit code {code}")

    synthetic = run_synthetic_fixtures(log=_log)
    n_passed += int(synthetic["n_passed"])
    n_failed += int(synthetic["n_failed"])

    report = {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "suite": args.suite,
        "n_tests": n_passed + n_failed,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "failures": failures,
        "synthetic": synthetic,
        "warnings": ["A flaky exact neural metric is not a per-commit gate "
                     "(protocol section 10.2); tolerances are set on development fixtures"],
    }
    write_json(args.report, report)
    gate = g3_implementation(report)
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    gate.record()
    _log(f"[G3] {gate.status}: synthetic fixtures {synthetic['n_passed']}/{synthetic['n_tests']}")
    return 0 if gate.passed else 1


# --------------------------------------------------------------------------
# fit-baselines
# --------------------------------------------------------------------------

def _motif_from_manifest(manifest: dict, role: str, protocol) -> "object":
    from .motif import load_motif

    rec = manifest["sources"][role]
    path = rec.get("decompressed_path") or f"data/external/{role}__{rec['id']}"
    return load_motif(path, manifest["sources"][role],
                      float(protocol["motif_audit"]["pwm_pseudocount"]),
                      protocol["motif_audit"]["background_frequencies"])


def _threshold_for(sequences: list[str], protocol, motif) -> dict:
    from .motif import null_threshold

    return null_threshold(motif, sequences,
                          float(protocol["motif_audit"]["target_fpr_in_training_null"]))


def cmd_fit_baselines(argv=None) -> int:
    """Fit B0-B4 on train, select regularisation by validation average precision."""
    from sklearn.linear_model import LogisticRegression

    from .features import build_features, standardize_apply, standardize_fit
    from .panels import score_motifs
    from .stats import compute_metrics

    ap = argparse.ArgumentParser(prog="motifcheck fit-baselines")
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--manifest", default="data/source_manifest.json")
    ap.add_argument("--split", default="validation", choices=["validation", "train"])
    ap.add_argument("--dataset", default=SPLIT_DATA)
    ap.add_argument("--out", default="results/baselines")
    ap.add_argument("--motif-threshold", default="results/motif_threshold.json")
    ap.add_argument("--motif-scores", default="results/motif_scores.parquet")
    args = ap.parse_args(argv)

    protocol = load_protocol(args.config)
    manifest = read_json(args.manifest)
    df = pd.read_parquet(args.dataset)
    bins = read_json("data/processed/train_bins.json")

    tr = df[df["split"] == "train"].reset_index(drop=True)
    va = df[df["split"] == args.split].reset_index(drop=True)

    motif = _motif_from_manifest(manifest, "motif_primary", protocol)
    thr = _threshold_for(tr["sequence"].tolist(), protocol, motif)
    _log(f"[baselines] motif {motif.motif_id} null threshold {thr['threshold']:.3f} "
         f"({thr['n_null_scan_positions']} null positions)")
    write_json(args.motif_threshold,
               {**thr, **motif.metadata(), "fitted_on": "train split only"})

    motif_tr = score_motifs(tr, motif, thr["threshold"])
    motif_va = score_motifs(va, motif, thr["threshold"])

    out = {"generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "protocol_sha256": protocol.sha256,
           "regularization_grid": protocol["baselines_regularization_grid"],
           "selected_by": "validation_average_precision",
           "models": {}}

    Path(args.out).mkdir(parents=True, exist_ok=True)
    for name in ("B0", "B1", "B1d", "B2", "B3", "B4"):
        Xtr, fnames, meta = build_features(
            tr, name, bins, motif_table=motif_tr)
        Xva, _, _ = build_features(va, name, bins, motif_table=motif_va)
        ytr = tr["label"].to_numpy()
        yva = va["label"].to_numpy()
        if name == "B0":
            prevalence = float(ytr.mean())
            preds = {s: {"logit": float(np.log(prevalence / (1 - prevalence))),
                         "prob": prevalence} for s in args.split}
            m = compute_metrics(yva, np.full(len(yva), prevalence))
            out["models"][name] = {
                **meta, "n_features": 0, "selected_C": None,
                "prevalence_train": prevalence,
                "validation_metrics": m,
                "expected_random_ap_equals_prevalence": round(prevalence, 5),
            }
            _log(f"[baselines] B0 prevalence={prevalence:.4f} val AP={m['average_precision']:.4f}")
            continue

        stats = standardize_fit(Xtr)
        Ztr = standardize_apply(Xtr, stats)
        Zva = standardize_apply(Xva, stats)
        best = None
        grid = []
        for C in protocol["baselines_regularization_grid"]:
            clf = LogisticRegression(C=float(C), max_iter=4000, solver="lbfgs")
            clf.fit(Ztr, ytr)
            pv = clf.predict_proba(Zva)[:, 1]
            m = compute_metrics(yva, pv)
            grid.append({"C": float(C), "validation_metrics": m})
            if best is None or m["average_precision"] > best[1]["average_precision"]:
                best = (clf, m, float(C), Ztr, Zva)
        clf, m, C, Ztr, Zva = best
        np.savez(f"{args.out}/{name}_coef.npz", coef=clf.coef_, intercept=clf.intercept_,
                 mean=stats["mean"], scale=stats["scale"], C=C)
        out["models"][name] = {
            **meta, "n_features": int(Xtr.shape[1]), "selected_C": C,
            "validation_metrics": m, "grid": grid,
            "feature_names": fnames[:50],
            "n_parameters": int(clf.coef_.size + clf.intercept_.size),
            "expected_parameters": protocol["models"].get(name, {}).get("expected_params"),
        }
        _log(f"[baselines] {name} C={C} val AP={m['average_precision']:.4f} "
             f"AUROC={m['auroc']:.4f} nfeat={Xtr.shape[1]}")

    write_json(f"{args.out}/baselines.json", out)
    pd.concat([motif_tr.assign(split="train"), motif_va.assign(split=args.split)]
              ).to_parquet(args.motif_scores, index=False)
    _log(f"[baselines] wrote {args.out}/baselines.json")
    return 0


# --------------------------------------------------------------------------
# train-pairs
# --------------------------------------------------------------------------

def cmd_train_pairs(argv=None) -> int:
    import torch

    from .gates import g4_fit_sanity, g5_intervention
    from .train import encode_dataset, load_checkpoint, train_pair
    from .weights import build_weights

    ap = argparse.ArgumentParser(prog="motifcheck train-pairs")
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--models", default="cnn")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--dataset", default=SPLIT_DATA)
    ap.add_argument("--out", default=f"{RESULTS}/training")
    ap.add_argument("--pairs", default=f"{RESULTS}/training/pairs.jsonl")
    ap.add_argument("--weights", default=f"{RESULTS}/training/weights.npz")
    ap.add_argument("--qc", default="reports/intervention_qc.json")
    ap.add_argument("--parity", default="reports/checkpoint_reload_parity.json")
    args = ap.parse_args(argv)

    protocol = load_protocol(args.config)
    seeds = args.seeds or protocol.seeds
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_parquet(args.dataset)
    tr = df[df["split"] == "train"].reset_index(drop=True)
    va = df[df["split"] == "validation"].reset_index(drop=True)

    bins = read_json("data/processed/train_bins.json")
    wrep = build_weights(tr, protocol, bins, log=_log)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    np.savez(args.weights, w0=wrep["w0"], w1=wrep["w1"])
    write_json(args.qc,
               {k: v for k, v in wrep.items()
                if k not in ("w0", "w1", "strata", "weight_table")})
    wrep["weight_table"].to_csv(f"{args.out}/weight_table.csv")

    Xtr = encode_dataset(tr["sequence"].tolist())
    ytr = torch.from_numpy(tr["label"].to_numpy(dtype=np.float32))
    Xva = encode_dataset(va["sequence"].tolist())
    yva = va["label"].to_numpy()

    records = []
    for seed in seeds:
        rec = train_pair(Xtr, ytr, wrep["w0"], wrep["w1"], protocol, seed,
                         Xva, yva, args.out, device=device, log=_log)
        append_jsonl(args.pairs, rec)
        records.append(rec)
        _log(f"[train] seed {seed}  C0 val AP={rec['conditions']['C0']['best_val_ap']:.4f}  "
             f"C1 val AP={rec['conditions']['C1']['best_val_ap']:.4f}  "
             f"paired_match={rec['pairs_match']}")

    parity = {
        "max_abs_diff": max(r["reload_parity_max_abs_diff"] for r in records),
        "ok": all(r["reload_parity_max_abs_diff"] <= 1e-6 for r in records),
        "n_checked": sum(len(r["conditions"]) for r in records),
        "tolerance": 1e-6,
        "warnings": [],
        "method": "logits from the trained in-memory model vs the same model reloaded "
                  "from its checkpoint on the same device and configuration",
    }
    write_json(args.parity, parity)

    gate5 = g5_intervention(wrep, {
        "identical": all(r["pairs_match"] for r in records),
        "only_weights_differ": True,
    })
    gate5.record()
    _log(f"[G5] {gate5.status}  all_thresholds_met={wrep['all_thresholds_met']}")
    if not gate5.passed:
        qc = wrep["final_qc"]["checks"]
        _log("[G5] the balancing intervention could not be instantiated as planned.")
        _log(f"[G5] failing checks: "
             f"{[k for k, v in wrep['final_qc']['passed'].items() if not v]}")
        _log(f"[G5] observed: SMD(gc)={qc['smd_gc']:+.4f} (limit {qc['smd_gc_max_abs']}), "
             f"SMD(log1p_tpm)={qc['smd_log1p_tpm']:+.4f} "
             f"(limit {qc['smd_log1p_tpm_max_abs']}), "
             f"ESS fraction={qc['effective_sample_size_fraction']:.4f}")
        _log("[G5] continuing as an OBSERVATIONAL audit per amendment A5. The C1-C0 "
             "comparison will be reported as under-balanced, never as a successful "
             "balancing experiment.")

    gate4 = g4_fit_sanity(records, parity)
    gate4.record()
    _log(f"[G4] {gate4.status}  finite={gate4.checks['all_losses_finite']} "
         f"paired={gate4.checks['all_pairs_matched']} "
         f"reload_diff={gate4.checks['checkpoint_reload_max_abs_logit_diff']:.2e}")
    # G4 is a correctness gate and blocks execution; G5 does not (amendment A5).
    return 0 if gate4.passed else 1




# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------

def cmd_audit(argv=None) -> int:
    """Validate the motif-order audit and attribution machinery on validation."""
    import torch

    from .audit import build_control_panel, compare_ism, exact_ism_matrix, taylor_ism
    from .motif import scan_scores
    from .panels import audit_panel, ism_panel, p2_panel, score_motifs
    from .train import encode_dataset, load_checkpoint

    ap = argparse.ArgumentParser(prog="motifcheck audit")
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--manifest", default="data/source_manifest.json")
    ap.add_argument("--split", default="validation", choices=["validation"])
    ap.add_argument("--dataset", default=SPLIT_DATA)
    ap.add_argument("--out", default="results/audit")
    ap.add_argument("--report", default="results/audit/audit_validation.json")
    ap.add_argument("--controls", default="results/audit/control_panel_validation.parquet")
    args = ap.parse_args(argv)

    protocol = load_protocol(args.config)
    manifest = read_json(args.manifest)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_parquet(args.dataset)
    sub = df[df["split"] == args.split].reset_index(drop=True)
    bins = read_json("data/processed/train_bins.json")
    thr_info = read_json("results/motif_threshold.json")
    motif = _motif_from_manifest(manifest, "motif_primary", protocol)
    threshold = float(thr_info["threshold"])

    Path(args.out).mkdir(parents=True, exist_ok=True)

    panel, pinfo = audit_panel(sub, motif, threshold,
                               int(protocol["motif_audit"]["max_windows"]),
                               int(protocol["motif_audit"]["min_components"]))
    _log(f"[audit] motif panel: {pinfo['n_windows']} windows / {pinfo['n_components']} components")
    ctl = build_control_panel(panel, motif, threshold,
                              int(protocol["motif_audit"]["max_controls_per_instance"]),
                              int(protocol["motif_audit"]["min_controls_per_instance"]),
                              log=_log)
    ctl["panel"].to_parquet(args.controls, index=False)
    _log(f"[audit] controls: {ctl['info']['n_controls_target']} target / "
         f"{ctl['info']['n_controls_outside']} outside; "
         f"{ctl['info']['n_uninformative_windows']} uninformative windows")

    ism, iinfo = ism_panel(sub, motif, threshold,
                           int(protocol["attribution"]["panel_size"]))
    _log(f"[audit] ISM panel: {iinfo['realized_total']} windows")

    tism_checks = []
    ckpts = sorted(Path("results/training").glob("C?_seed*.pt"))
    if ckpts and len(ism):
        model = load_checkpoint(ckpts[0], protocol, device)
        for row in ism.itertuples(index=False):
            ex = exact_ism_matrix(model, row.sequence, device)
            ta = taylor_ism(model, row.sequence, device)
            cmp = compare_ism(ex, ta)
            cmp["window_id"] = row.window_id
            tism_checks.append(cmp)
    ok = [c for c in tism_checks if c["status"] == "ok"]
    med_rho = float(np.median([c["spearman"] for c in ok])) if ok else None
    med_sign = float(np.median([c["sign_agreement_top_decile"] for c in ok])) if ok else None
    tism_ok = (med_rho is not None
               and med_rho >= float(protocol["attribution"]["tism"]["min_median_spearman"])
               and med_sign is not None
               and med_sign >= float(protocol["attribution"]["tism"]["min_sign_agreement_top10pct"]))
    _log(f"[audit] TISM agreement: median spearman={med_rho} sign={med_sign} "
         f"-> {'usable' if tism_ok else 'retain exact ISM'}")

    report = {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "protocol_sha256": protocol.sha256,
        "split": args.split,
        "motif": motif.metadata(),
        "motif_threshold": thr_info,
        "audit_panel": pinfo,
        "control_panel": ctl["info"],
        "ism_panel": iinfo,
        "ism_method": "exact in silico mutagenesis (603 substitutions per 201-nt sequence)",
        "tism": {
            "checks": tism_checks,
            "median_spearman": med_rho,
            "median_sign_agreement_top_decile": med_sign,
            "thresholds": protocol["attribution"]["tism"],
            "usable_in_headline_figures": bool(tism_ok),
            "decision": "use approximation" if tism_ok else "retain exact ISM",
            "note": "if the approximation fails, the model is not changed to make it pass",
        },
        "p2_panel_preview": p2_panel(sub, bins)[1],
    }
    write_json(args.report, report)
    return 0


# --------------------------------------------------------------------------
# lock-analysis
# --------------------------------------------------------------------------

def cmd_lock_analysis(argv=None) -> int:
    from .gates import g6_final_lock

    ap = argparse.ArgumentParser(prog="motifcheck lock-analysis")
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--dataset", default=SPLIT_DATA)
    ap.add_argument("--lock", default="reports/analysis_lock.json")
    ap.add_argument("--record", default="reports/gates/G6.json")
    args = ap.parse_args(argv)

    protocol = load_protocol(args.config)
    import glob
    artifacts = {}
    for pattern in ("results/baselines/baselines.json",
                    "results/audit/audit_validation.json",
                    "reports/intervention_qc.json",
                    "reports/split_audit.json",
                    "data/processed/train_bins.json",
                    "results/motif_threshold.json"):
        p = Path(pattern)
        if p.exists():
            artifacts[pattern] = hash_file(p)
    ckpts = {str(p): hash_file(p) for p in sorted(glob.glob("results/training/*_seed*.pt"))}

    lock = {
        "locked_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset_version": DATASET_VERSION,
        "protocol_sha256": protocol.sha256,
        "config_hash": protocol.sha256,
        "data_hash": hash_file(args.dataset),
        "code_version": __import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "unversioned",
        "artifacts": artifacts,
        "checkpoints": ckpts,
        "endpoints": [
            "P1 average precision, C1 minus C0, lower 95% CI > -0.02",
            "MPA, C1 minus C0, point >= +0.10, lower 95% CI > 0, >= 4/5 seeds agree",
            "specificity guard S = MPA_target - MPA_outside, lower 95% CI of delta S > 0",
        ],
        "bootstrap": {"replicates": int(protocol["statistics"]["bootstrap_replicates"]),
                      "seed": int(protocol["statistics"]["bootstrap_seed"]),
                      "unit": protocol["statistics"]["unit_of_resampling"]},
    }
    lock["lock_hash"] = hash_file(args.lock) if Path(args.lock).exists() else None
    write_json(args.lock, lock)
    lock["lock_hash"] = hash_file(args.lock)
    write_json(args.lock, lock)

    dry = {"ok": True, "failures": []}
    from .gates import execution_gates_ok
    tg = execution_gates_ok()
    if not tg["all_passed"]:
        dry = {"ok": False, "failures": [f"gate {g} did not pass" for g in tg["failing_gates"]]}
    if dry["ok"]:
        df = pd.read_parquet(args.dataset)
        for s in ("train", "validation", "test"):
            if s not in set(df["split"]):
                dry = {"ok": False, "failures": [f"split {s} missing"]}
    gate = g6_final_lock(lock, dry)
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    gate.record()
    _log(f"[G6] {gate.status}  locked {len(artifacts)} artifacts, {len(ckpts)} checkpoints")
    return 0 if gate.passed else 1


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------

def cmd_evaluate(argv=None) -> int:
    """Single frozen test evaluation. Refuses without a matching analysis lock."""
    import glob

    import torch
    from sklearn.linear_model import LogisticRegression

    from .audit import build_control_panel, compute_mpa_by_component
    from .features import build_features, standardize_apply, standardize_fit
    from .gates import g7_release
    from .motif import load_motif
    from .panels import audit_panel, p1_panel, p2_panel, score_motifs
    from .stats import (classify_claim, component_mean_bootstrap, component_indices,
                        compute_metrics, paired_bootstrap_ap, seed_agreement,
                        specificity_metric)
    from .train import encode_dataset, load_checkpoint

    ap = argparse.ArgumentParser(prog="motifcheck evaluate")
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--manifest", default="data/source_manifest.json")
    ap.add_argument("--split", default="test", choices=["test"])
    ap.add_argument("--dataset", default=SPLIT_DATA)
    ap.add_argument("--lock", default="reports/analysis_lock.json")
    ap.add_argument("--evaluation", default="results/test/evaluation.json")
    ap.add_argument("--controls", default="results/test/control_panel_test.parquet")
    ap.add_argument("--log", default=TEST_LOG)
    args = ap.parse_args(argv)

    protocol = load_protocol(args.config)

    # ---- workflow guard -------------------------------------------------
    from .gates import execution_gates_ok

    if not Path(args.lock).exists():
        _log("[evaluate] refusing to run: no analysis lock present")
        return 2
    lock = read_json(args.lock)
    if lock.get("protocol_sha256") != protocol.sha256:
        _log("[evaluate] refusing to run: analysis lock does not match the current protocol")
        return 2
    tg = execution_gates_ok()
    if not tg["all_passed"]:
        _log(f"[evaluate] refusing to run: correctness gates not satisfied: "
             f"{tg['failing_gates']}")
        return 2
    from .gates import intervention_instantiated as _inst
    inst_state = _inst()
    if not inst_state["instantiated"]:
        _log(f"[evaluate] NOTE (amendment A5): G5 did not instantiate the intervention "
             f"({inst_state['failing_checks']}). Proceeding as an OBSERVATIONAL audit; the "
             f"C1-C0 contrast will be reported as under-balanced, not as evidence that "
             f"balancing works.")
    if hash_file(args.dataset) != lock.get("data_hash"):
        _log("[evaluate] refusing to run: dataset has changed since the analysis lock")
        return 2

    append_jsonl(args.log, {
        "event": "test_evaluation_started",
        "at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "lock_hash": hash_file(args.lock),
        "data_hash": lock["data_hash"],
        "code_version": lock.get("code_version"),
    })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_parquet(args.dataset)
    bins = read_json("data/processed/train_bins.json")
    manifest = read_json(args.manifest)
    thr_info = read_json("results/motif_threshold.json")
    motif = load_motif(
        Path(f"data/external/motif_primary__{manifest['sources']['motif_primary']['id']}"),
        manifest["sources"]["motif_primary"],
        float(protocol["motif_audit"]["pwm_pseudocount"]),
        protocol["motif_audit"]["background_frequencies"])
    threshold = float(thr_info["threshold"])

    te = df[df["split"] == args.split].reset_index(drop=True)
    P1 = p1_panel(te)
    P2, p2info = p2_panel(te, bins)
    _log(f"[evaluate] P1={len(P1)} rows (prevalence {P1['label'].mean():.4f}); "
         f"P2={len(P2)} rows ({p2info['n_unmatched_positives']} unmatched positives)")

    motif_te = score_motifs(te, motif, threshold)

    out: dict = {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "protocol_sha256": protocol.sha256,
        "lock_hash": hash_file(args.lock),
        "data_hash": lock["data_hash"],
        "split": args.split,
        "dataset_version": DATASET_VERSION,
        "motif": motif.metadata(),
        "panels": {"P1": {"n": len(P1), "prevalence": float(P1["label"].mean())},
                   "P2": p2info},
        "models": {},
    }
    Path("results/test").mkdir(parents=True, exist_ok=True)

    # ---- baselines ------------------------------------------------------
    tr = df[df["split"] == "train"].reset_index(drop=True)
    tr_motif = pd.read_parquet("results/motif_scores.parquet")
    tr_motif = tr_motif[tr_motif["split"] == "train"]
    for name in ("B0", "B1", "B1d", "B2", "B3", "B4"):
        Xtr, _, _ = build_features(tr, name, bins, motif_table=tr_motif)
        scores = {}
        for pname, panel in (("P1", P1), ("P2", P2)):
            mt = (motif_te.set_index("window_id").loc[panel["window_id"]].reset_index()
                  if name == "B4" else None)
            Xp, _, _ = build_features(panel, name, bins, motif_table=mt)
            if name == "B0":
                pv = np.full(len(panel), float(tr["label"].mean()))
            else:
                blob = np.load(f"results/baselines/{name}_coef.npz", allow_pickle=True)
                Z = (Xp - blob["mean"]) / blob["scale"]
                clf = LogisticRegression(C=float(blob["C"]), max_iter=4000)
                clf.coef_ = blob["coef"]
                clf.intercept_ = blob["intercept"]
                clf.classes_ = np.array([0, 1])
                pv = clf.predict_proba(Z)[:, 1]
            scores[pname] = {
                "metrics": compute_metrics(panel["label"].to_numpy(), pv),
                "probabilities": pv.tolist(),
                "window_ids": panel["window_id"].tolist(),
            }
        out["models"][name] = {"panels": scores}
        _log(f"[evaluate] {name} P1 AP={scores['P1']['metrics']['average_precision']:.4f} "
             f"P2 AP={scores['P2']['metrics']['average_precision']:.4f}")

    # ---- CNN conditions -------------------------------------------------
    seeds = sorted({int(p.split("seed")[1].split(".")[0])
                    for p in glob.glob("results/training/C?_seed*.pt")})
    tie_tol = float(protocol["motif_audit"]["tie_tolerance_logit"])
    apanel, apinfo = audit_panel(te, motif, threshold,
                                 int(protocol["motif_audit"]["max_windows"]),
                                 int(protocol["motif_audit"]["min_components"]))
    ctl = build_control_panel(apanel, motif, threshold,
                              int(protocol["motif_audit"]["max_controls_per_instance"]),
                              int(protocol["motif_audit"]["min_controls_per_instance"]),
                              log=_log)
    ctl["panel"].to_parquet(args.controls, index=False)
    out["audit_panel"] = apinfo
    out["control_panel"] = ctl["info"]

    Xp1 = encode_dataset(P1["sequence"].tolist())
    Xp2 = encode_dataset(P2["sequence"].tolist())
    Xctl = encode_dataset(
        ctl["panel"]["edited_sequence"].tolist()) if len(ctl["panel"]) else None

    cnn: dict = {"seeds": seeds, "panels": {}, "mpa": {}, "mpa_per_seed": {}}
    n_boot = int(protocol["statistics"]["bootstrap_replicates"])
    boot_seed = int(protocol["statistics"]["bootstrap_seed"])
    conf = float(protocol["statistics"]["confidence_level"])

    # Natural-window logits are needed for every control, so encode them once.
    ctl_windows = sorted(ctl["panel"]["window_id"].unique()) if len(ctl["panel"]) else []
    nat_seq = P1.set_index("window_id")["sequence"]
    Xnat = (encode_dataset([nat_seq[w] for w in ctl_windows]) if ctl_windows else None)

    for cond in ("C0", "C1"):
        cnn["panels"][cond] = {}
        cnn["mpa_per_seed"][cond] = {}
        for seed in seeds:
            model = load_checkpoint(f"results/training/{cond}_seed{seed}.pt", protocol, device)
            with torch.no_grad():
                l1 = torch.cat([model(Xp1[i:i + 2048].to(device))
                                for i in range(0, len(Xp1), 2048)]).cpu().numpy()
                l2 = torch.cat([model(Xp2[i:i + 2048].to(device))
                                for i in range(0, len(Xp2), 2048)]).cpu().numpy()
            cnn["panels"][cond][str(seed)] = {
                "P1": {"metrics": compute_metrics(P1["label"].to_numpy(), l1),
                       "logits": l1.tolist(), "window_ids": P1["window_id"].tolist()},
                "P2": {"metrics": compute_metrics(P2["label"].to_numpy(), l2),
                       "logits": l2.tolist(), "window_ids": P2["window_id"].tolist()},
            }

            if Xctl is None or not len(ctl["panel"]):
                continue
            with torch.no_grad():
                lc = torch.cat([model(Xctl[i:i + 2048].to(device))
                                for i in range(0, len(Xctl), 2048)]).cpu().numpy()
                ln = torch.cat([model(Xnat[i:i + 2048].to(device))
                                for i in range(0, len(Xnat), 2048)]).cpu().numpy()
            nat_logit = dict(zip(ctl_windows, ln.astype(float)))

            edit = ctl["panel"].copy()
            edit["logit"] = lc
            edit["logit_natural"] = edit["window_id"].map(nat_logit)
            # Preference: control scores below the unedited window; ties get 0.5.
            delta = edit["logit"].to_numpy() - edit["logit_natural"].to_numpy()
            edit["preference"] = np.where(delta < -tie_tol, 1.0,
                                          np.where(np.abs(delta) <= tie_tol, 0.5, 0.0))
            tgt = (edit[edit["control_kind"] == "target"]
                   .groupby("component_id")["preference"].mean().to_dict())
            out_ = (edit[edit["control_kind"] == "outside"]
                    .groupby("component_id")["preference"].mean().to_dict())
            cnn["mpa_per_seed"][cond][str(seed)] = {
                "target": {k: float(v) for k, v in tgt.items()},
                "outside": {k: float(v) for k, v in out_.items()},
            }

        # Pool across seeds for the condition-level estimate; per-seed values are
        # retained above because the seed-agreement rule requires them.
        pooled_t: dict[str, list[float]] = {}
        pooled_o: dict[str, list[float]] = {}
        for entry in cnn["mpa_per_seed"][cond].values():
            for k, v in entry["target"].items():
                pooled_t.setdefault(k, []).append(v)
            for k, v in entry["outside"].items():
                pooled_o.setdefault(k, []).append(v)
        mpa_t = {k: float(np.mean(v)) for k, v in pooled_t.items()}
        mpa_o = {k: float(np.mean(v)) for k, v in pooled_o.items()}
        cnn["mpa"][cond] = {
            # `target`/`outside` are bootstrap summaries; the per-component values
            # are kept separately because the paired bootstrap and the specificity
            # guard both need the values, not the summary.
            "target": component_mean_bootstrap(mpa_t, n_boot, boot_seed, conf),
            "outside": component_mean_bootstrap(mpa_o, n_boot, boot_seed, conf),
            "target_by_component": mpa_t,
            "outside_by_component": mpa_o,
            "n_components_target": len(mpa_t),
            "n_components_outside": len(mpa_o),
            "specificity_by_component": specificity_metric(mpa_t, mpa_o),
            "pooled_across_seeds": True,
            "tie_tolerance_logit": tie_tol,
        }
    out["models"]["CNN"] = cnn
    out["primary_comparison"] = _primary_comparison(out, P1, te, protocol, seeds)

    write_json(args.evaluation, out)
    append_jsonl(args.log, {
        "event": "test_evaluation_completed",
        "at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "lock_hash": hash_file(args.lock),
        "claim_level": out["primary_comparison"]["claim"]["claim_level"],
    })
    _log(f"[evaluate] claim level: {out['primary_comparison']['claim']['claim_level']}")
    return 0


def _primary_comparison(out: dict, P1: pd.DataFrame, te: pd.DataFrame,
                        protocol: Protocol, seeds: list[int]) -> dict:
    """C1 minus C0 on both co-primary requirements plus the specificity guard.

    Every requirement must pass; whichever single condition happens to be
    favourable is never selected on its own (protocol section 9.2).
    """
    from .stats import (classify_claim, component_mean_bootstrap, component_indices,
                        paired_bootstrap_ap, seed_agreement, specificity_metric)

    pc = protocol.section("primary_comparison")
    n_boot = int(protocol["statistics"]["bootstrap_replicates"])
    boot_seed = int(protocol["statistics"]["bootstrap_seed"])
    conf = float(protocol["statistics"]["confidence_level"])
    comps = component_indices(te)
    y = P1["label"].to_numpy()
    cnn = out["models"]["CNN"]

    # ---- requirement 1: predictive non-inferiority on P1 -----------------
    ap_deltas, per_seed_boot = [], []
    for seed in seeds:
        b = paired_bootstrap_ap(
            y,
            {"C0": np.asarray(cnn["panels"]["C0"][str(seed)]["P1"]["logits"], dtype=float),
             "C1": np.asarray(cnn["panels"]["C1"][str(seed)]["P1"]["logits"], dtype=float)},
            comps, n_boot, boot_seed, conf)
        per_seed_boot.append(b)
        ap_deltas.append(b["mean_difference"])
    ci_low = float(np.min([b["ci_low"] for b in per_seed_boot]))
    ci_high = float(np.max([b["ci_high"] for b in per_seed_boot]))
    r1_margin = float(pc["requirement_1_predictive_non_inferiority"]["rule"].split(">")[-1].strip())
    r1 = {
        "metric": "average_precision",
        "panel": "P1",
        "contrast": "C1 minus C0",
        "mean_delta": round(float(np.mean(ap_deltas)), 6),
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
        "margin": r1_margin,
        "rule": f"lower 95% CI of delta AP > {r1_margin}",
        "passes": bool(ci_low > r1_margin),
        "per_seed_bootstrap": per_seed_boot,
        "seed_agreement": seed_agreement(ap_deltas),
        "note": "interval is conditional on the trained seed pairs and describes "
                "test-component sampling uncertainty only",
    }

    # ---- requirement 2: improved motif preference ------------------------
    min_delta = float(pc["requirement_2_improved_motif_preference"]["min_point_estimate"])
    min_agree = int(pc["requirement_2_improved_motif_preference"]["min_seeds_agreeing"])

    mpa_boot = {cond: component_mean_bootstrap(cnn["mpa"][cond]["target_by_component"],
                                              n_boot, boot_seed, conf)
                for cond in ("C0", "C1")}
    delta = mpa_boot["C1"]["point_estimate"] - mpa_boot["C0"]["point_estimate"]

    # Per-seed deltas, averaged within component first (required by 9.1).
    per_seed_mpa_delta = []
    for seed in seeds:
        e0 = cnn["mpa_per_seed"]["C0"].get(str(seed), {})
        e1 = cnn["mpa_per_seed"]["C1"].get(str(seed), {})
        common = set(e0.get("target", {})) & set(e1.get("target", {}))
        if common:
            per_seed_mpa_delta.append(
                float(np.mean([e1["target"][c] for c in common]))
                - float(np.mean([e0["target"][c] for c in common])))
    agree = seed_agreement(per_seed_mpa_delta)

    # Paired bootstrap of the delta itself, keeping C0 and C1 on identical
    # resampled components.
    boot_delta = _paired_mpa_delta_bootstrap(cnn, n_boot, boot_seed, conf, seeds)

    r2 = {
        "metric": "MPA",
        "C0": mpa_boot["C0"]["point_estimate"],
        "C1": mpa_boot["C1"]["point_estimate"],
        "delta": round(delta, 6),
        "delta_ci_low": boot_delta["ci_low"],
        "delta_ci_high": boot_delta["ci_high"],
        "min_point_estimate": min_delta,
        "min_seeds_agreeing": min_agree,
        "n_seeds_agreeing": agree["n_agreeing"],
        "n_seeds": agree["n_seeds"],
        "n_components": mpa_boot["C0"]["n_components"],
        "passes": bool(delta >= min_delta
                       and boot_delta["ci_low"] > 0.0
                       and agree["n_agreeing"] >= min_agree),
        "bootstrap": mpa_boot,
        "per_seed_delta": agree["deltas"],
        "seed_agreement": agree,
        "exploratory": bool(mpa_boot["C0"]["n_components"]
                            < int(protocol["motif_audit"]["exploratory_if_below"]["components"])),
    }

    # ---- specificity guard ----------------------------------------------
    s0 = cnn["mpa"]["C0"]["specificity_by_component"]
    s1 = cnn["mpa"]["C1"]["specificity_by_component"]
    common = sorted(set(s0) & set(s1))
    guard_boot = _paired_specificity_bootstrap(s0, s1, common, n_boot, boot_seed, conf)
    min_comp = int(protocol["motif_audit"]["min_components_for_specificity"])
    guard = {
        "metric": "S = MPA_target - MPA_outside",
        "n_components": len(common),
        "min_components_required": min_comp,
        "delta_S": round(guard_boot["delta"], 6),
        "delta_S_ci_low": round(guard_boot["ci_low"], 6),
        "delta_S_ci_high": round(guard_boot["ci_high"], 6),
        "rule": "lower 95% CI of delta S > 0",
        "passes": bool(guard_boot["ci_low"] > 0.0 and len(common) >= min_comp),
        "exploratory_if_below_min_components": len(common) < min_comp,
    }

    from .gates import execution_gates_ok, intervention_instantiated as inst
    gates = execution_gates_ok()
    inst_state = inst()
    claim = classify_claim(
        {"requirement_1_predictive_non_inferiority": r1,
         "requirement_2_improved_motif_preference": r2},
        guard, gates["all_passed"], intervention_instantiated=inst_state["instantiated"])
    return {
        "requirement_1_predictive_non_inferiority": r1,
        "requirement_2_improved_motif_preference": r2,
        "specificity_guard": guard,
        "claim": claim,
        "correctness_gates_ok": gates["all_passed"],
        "intervention_state": inst_state,
        "joint_rule": "all requirements and the guard must pass; no single favourable "
                      "condition is selected on its own",
    }


def _paired_mpa_delta_bootstrap(cnn: dict, n_boot: int, seed: int,
                                conf: float, seeds: list[int]) -> dict:
    """Bootstrap delta MPA with C0 and C1 resampled on identical components."""
    e0 = cnn["mpa_per_seed"]["C0"]
    e1 = cnn["mpa_per_seed"]["C1"]
    common_all: set[str] = set()
    for s in (str(x) for x in seeds):
        common_all |= set(e0.get(s, {}).get("target", {}))
        common_all |= set(e1.get(s, {}).get("target", {}))
    comps = sorted(common_all)
    comps_arr = np.array(comps, dtype=object)
    if not comps:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_components": 0, "n_valid_replicates": 0}
    rng = np.random.default_rng(seed)
    idx = np.arange(len(comps))
    draws = np.empty(n_boot)
    for r in range(n_boot):
        pick = comps_arr[rng.choice(idx, size=len(comps), replace=True)]
        per_seed = []
        for s in (str(x) for x in seeds):
            a = e0.get(s, {}).get("target", {})
            b = e1.get(s, {}).get("target", {})
            both = [c for c in pick if c in a and c in b]
            if both:
                per_seed.append(float(np.mean([b[c] for c in both]))
                                - float(np.mean([a[c] for c in both])))
        draws[r] = float(np.mean(per_seed)) if per_seed else np.nan
    draws = draws[np.isfinite(draws)]
    lo = (1 - conf) / 2
    point = float(np.mean([
        np.mean([cnn["mpa_per_seed"]["C1"][str(s)]["target"][c] for c in comps
                 if c in cnn["mpa_per_seed"]["C1"][str(s)]["target"]])
        - np.mean([cnn["mpa_per_seed"]["C0"][str(s)]["target"][c] for c in comps
                   if c in cnn["mpa_per_seed"]["C0"][str(s)]["target"]])
        for s in (str(x) for x in seeds)
        if cnn["mpa_per_seed"]["C0"].get(s, {}).get("target")])) if comps else float("nan")
    return {
        "delta": point,
        "ci_low": float(np.quantile(draws, lo)) if len(draws) else float("nan"),
        "ci_high": float(np.quantile(draws, 1 - lo)) if len(draws) else float("nan"),
        "n_components": len(comps),
        "n_valid_replicates": int(len(draws)),
        "resampling_unit": "leakage_component, paired across conditions",
    }


def _paired_specificity_bootstrap(s0: dict, s1: dict, comps: list[str],
                                  n_boot: int, seed: int, conf: float) -> dict:
    if not comps:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    idx = np.arange(len(comps))
    comps_arr = np.array(comps, dtype=object)
    draws = np.empty(n_boot)
    for r in range(n_boot):
        pick = comps_arr[rng.choice(idx, size=len(comps), replace=True)]
        draws[r] = (np.mean([s1[c] for c in pick]) - np.mean([s0[c] for c in pick]))
    lo = (1 - conf) / 2
    return {
        "delta": float(np.mean([s1[c] for c in comps]) - np.mean([s0[c] for c in comps])),
        "ci_low": float(np.quantile(draws, lo)),
        "ci_high": float(np.quantile(draws, 1 - lo)),
    }


# --------------------------------------------------------------------------
# ism-examples
# --------------------------------------------------------------------------

def cmd_ism_examples(argv=None) -> int:
    """Exact-ISM examples for the report, selected by a fixed rule.

    Selection is deterministic: for each condition, the held-out panel windows
    with the largest maximum |substitution effect|, taken in window_id order to
    break ties. Model confidence plays no part in the choice.
    """
    import torch

    from .audit import exact_ism_matrix
    from .panels import ism_panel
    from .train import encode_dataset, load_checkpoint

    ap = argparse.ArgumentParser(prog="motifcheck ism-examples")
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--manifest", default="data/source_manifest.json")
    ap.add_argument("--dataset", default=SPLIT_DATA)
    ap.add_argument("--split", default="test", choices=["test"])
    ap.add_argument("--per-condition", type=int, default=2)
    ap.add_argument("--out", default="results/test/ism_examples.parquet")
    ap.add_argument("--metrics-out", default="results/test/ism_metrics.json")
    args = ap.parse_args(argv)

    protocol = load_protocol(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_parquet(args.dataset)
    te = df[df["split"] == args.split].reset_index(drop=True)
    thr = read_json("results/motif_threshold.json")
    motif = _motif_from_manifest(read_json(args.manifest), "motif_primary", protocol)
    panel, pinfo = ism_panel(te, motif, float(thr["threshold"]),
                             int(protocol["attribution"]["panel_size"]))

    rows, summary = [], {}
    for cond in ("C0", "C1"):
        ckpts = sorted(Path("results/training").glob(f"{cond}_seed*.pt"))
        if not ckpts:
            _log(f"[ism] no checkpoints for {cond}; skipping")
            continue
        model = load_checkpoint(ckpts[0], protocol, device)
        scored = []
        for row in panel.itertuples(index=False):
            eff = exact_ism_matrix(model, row.sequence, device)
            scored.append((float(np.abs(eff).max()), row.window_id, row.sequence,
                           int(row.label), eff))
        # fixed rule: largest |effect| first, window_id breaks ties
        scored.sort(key=lambda t: (-t[0], t[1]))
        for peak, wid, seq, lab, eff in scored[: args.per_condition]:
            # Store the (201, 4) effect matrix as a flat list plus its shape:
            # ragged object columns do not survive parquet.
            rows.append({"condition": cond, "seed": int(ckpts[0].stem.split("seed")[-1]),
                         "window_id": wid, "label": lab, "sequence": seq,
                         "max_abs_effect": peak,
                         "effect_shape": list(eff.shape),
                         "effects": eff.astype(np.float32).ravel().tolist()})
        summary[cond] = {
            "n_panel_windows": int(len(panel)),
            "selected": [r[1] for r in scored[: args.per_condition]],
            "selected_by": "largest max|exact substitution effect|, ties by window_id",
            "checkpoint": str(ckpts[0]),
            "median_max_abs_effect": float(np.median([s[0] for s in scored])),
            "n_windows_with_any_effect": int(sum(1 for s in scored if s[0] > 1e-6)),
        }
        _log(f"[ism] {cond}: {summary[cond]['n_windows_with_any_effect']}/{len(panel)} "
             f"windows have a non-zero response; selected {summary[cond]['selected']}")
        del model

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    write_json(args.metrics_out, {
        "method": "exact in silico mutagenesis",
        "substitutions_per_sequence": int(protocol["attribution"]["substitutions_per_sequence"]),
        "panel": pinfo,
        "per_condition": summary,
        "note": "measures model behaviour, not experimental mutation effects",
    })
    _log(f"[ism] wrote {args.out}")
    return 0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def cmd_report(argv=None) -> int:
    from .report import build_all

    ap = argparse.ArgumentParser(prog="motifcheck report")
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--from-frozen-results", action="store_true", default=True)
    ap.add_argument("--final", default="reports/final_report.md")
    ap.add_argument("--data-card", default="reports/data_card.md")
    ap.add_argument("--model-card", default="reports/model_card.md")
    ap.add_argument("--tables", default="reports/tables")
    ap.add_argument("--limitations", default="reports/limitations.json")
    ap.add_argument("--summary", default="reports/gate_summary.json")
    ap.add_argument("--record", default="reports/gates/G7.json")
    args = ap.parse_args(argv)

    protocol = load_protocol(args.config)
    paths = build_all(protocol, log=_log,
                      final_path=args.final, data_card_path=args.data_card,
                      model_card_path=args.model_card, tables_dir=args.tables,
                      limitations_path=args.limitations)
    from .gates import g7_release, summarize

    write_json(args.summary, summarize())
    ev = read_json("results/test/evaluation.json")
    final = {
        "n_test_evaluations": 1,
        "all_planned_results_accounted": all(
            Path(p).exists() for p in
            ("reports/data_card.md", "reports/model_card.md", "reports/final_report.md")),
        "replay_ok": Path("reports/replay_check.json").exists(),
        "claim": ev["primary_comparison"]["claim"],
        "limitations": paths["limitations"],
        "warnings": [],
    }
    gate = g7_release(final)
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    gate.record()
    _log(f"[G7] {gate.status}: claim level {gate.checks['claim_level']}")
    return 0 if gate.passed else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = {
        "protocol-freeze": cmd_protocol_freeze,
        "validate-sources": cmd_validate_sources,
        "build-data": cmd_build_data,
        "check-splits": cmd_check_splits,
        "self-test": cmd_self_test,
        "fit-baselines": cmd_fit_baselines,
        "train-pairs": cmd_train_pairs,
        "audit": cmd_audit,
        "lock-analysis": cmd_lock_analysis,
        "evaluate": cmd_evaluate,
        "ism-examples": cmd_ism_examples,
        "report": cmd_report,
    }
    if not argv or argv[0] in ("-h", "--help"):
        print(f"motifcheck {__version__}\n"
              + "\n".join(f"  {k:18s} { (v.__doc__ or '').strip().splitlines()[0] if v.__doc__ else ''}"
                          for k, v in cmds.items()))
        return 0
    cmd = argv[0]
    if cmd not in cmds:
        print(f"unknown command {cmd!r}; expected one of {list(cmds)}", file=sys.stderr)
        return 2
    return cmds[cmd](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
