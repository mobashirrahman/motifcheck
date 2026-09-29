"""Metrics, paired component bootstrap, and claim classification.

The independent unit is the leakage component, not the window (protocol 9.1).
Bootstrapping windows would treat correlated windows from one gene as
independent evidence and produce intervals that are far too narrow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)

METRICS = ("average_precision", "auroc", "log_loss", "brier")


def compute_metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    """All four prespecified metrics. One-class inputs are reported, not hidden."""
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    out: dict[str, float | None] = {}
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    out["n"] = int(len(y))
    out["prevalence"] = round(n_pos / len(y), 6) if len(y) else None
    if len(y) == 0:
        return {m: None for m in METRICS} | out
    if n_pos == 0 or n_neg == 0:
        out["error"] = "one_class_input"
        out.update({m: None for m in METRICS})
        return out
    out["average_precision"] = float(average_precision_score(y, s))
    out["auroc"] = float(roc_auc_score(y, s))
    out["log_loss"] = float(log_loss(y, np.clip(s, 1e-15, 1 - 1e-15), labels=[0, 1]))
    out["brier"] = float(brier_score_loss(y, np.clip(s, 1e-15, 1 - 1e-15)))
    return out


# --------------------------------------------------------------------------
# Component bootstrap
# --------------------------------------------------------------------------

def component_indices(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Map component_id -> positional row indices.

    Positional (not label-based) so these index directly into aligned arrays such
    as `df["label"].to_numpy()`.
    """
    return {c: np.asarray(g) for c, g in
            df.groupby("component_id", sort=True).indices.items()}


def paired_bootstrap_ap(y: np.ndarray, scores: dict[int, np.ndarray],
                        components: dict[str, np.ndarray],
                        n_boot: int, seed: int,
                        confidence: float = 0.95) -> dict:
    """Paired component bootstrap for delta AP between two conditions.

    ``scores`` maps condition -> aligned score vector. Both conditions are
    scored on exactly the same resampled components in each replicate, so the
    difference is paired and the interval reflects component sampling only.
    """
    comp_names = np.array(sorted(components))
    idx_by_comp = [components[c] for c in comp_names]
    rng = np.random.default_rng(seed)
    deltas = []
    invalid = 0
    per_replicate = {c: [] for c in scores}
    for _ in range(n_boot):
        pick = rng.integers(0, len(comp_names), size=len(comp_names))
        rows = np.concatenate([idx_by_comp[i] for i in pick])
        yy = y[rows]
        if yy.min() == yy.max():
            invalid += 1
            continue
        ds = {}
        for cond, sc in scores.items():
            m = compute_metrics(yy, sc[rows])
            per_replicate[cond].append(m["average_precision"])
            ds[cond] = m["average_precision"]
        deltas.append(ds)
    if not deltas:
        return {"n_valid_replicates": 0, "n_invalid_replicates": invalid,
                "warning": "no valid bootstrap replicate contained both classes"}
    keys = list(scores)
    diffs = np.array([d[keys[1]] - d[keys[0]] for d in deltas], dtype=float)
    lo_q = (1 - confidence) / 2
    return {
        "n_valid_replicates": int(len(diffs)),
        "n_invalid_replicates": int(invalid),
        "invalid_fraction": round(invalid / n_boot, 5),
        "mean_difference": float(diffs.mean()),
        "ci_low": float(np.quantile(diffs, lo_q)),
        "ci_high": float(np.quantile(diffs, 1 - lo_q)),
        "interval_type": "percentile",
        "confidence": confidence,
        "n_components_resampled": int(len(comp_names)),
        "resampling_unit": "leakage_component",
        "interpretation": "conditional on the trained seed pairs; not independent experiments",
    }


def component_mean_bootstrap(values_by_component: dict[str, float],
                             n_boot: int, seed: int,
                             confidence: float = 0.95) -> dict:
    """Bootstrap of a per-component mean (used for MPA, which is averaged
    within component before resampling)."""
    names = np.array(sorted(values_by_component))
    v = np.array([values_by_component[n] for n in names], dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for r in range(n_boot):
        pick = rng.integers(0, len(names), size=len(names))
        draws[r] = v[pick].mean()
    lo_q = (1 - confidence) / 2
    return {
        "point_estimate": float(v.mean()),
        "ci_low": float(np.quantile(draws, lo_q)),
        "ci_high": float(np.quantile(draws, 1 - lo_q)),
        "mean": float(draws.mean()),
        "n_components": int(len(names)),
        "n_replicates": int(n_boot),
        "interval_type": "percentile",
        "confidence": confidence,
        "resampling_unit": "leakage_component",
    }


def seed_agreement(deltas: list[float], threshold: float = 0.0) -> dict:
    """Fraction of seed pairs agreeing with the pooled direction."""
    pos = sum(1 for d in deltas if d > threshold)
    return {
        "n_seeds": len(deltas),
        "n_agreeing": int(pos),
        "fraction_agreeing": round(pos / len(deltas), 4) if deltas else None,
        "deltas": [round(float(d), 6) for d in deltas],
        "median": float(np.median(deltas)) if deltas else None,
        "min": float(np.min(deltas)) if deltas else None,
        "max": float(np.max(deltas)) if deltas else None,
    }


# --------------------------------------------------------------------------
# Motif preference accuracy
# --------------------------------------------------------------------------

def motif_preference(natural_logits: np.ndarray, control_logits: np.ndarray,
                     tie_tolerance: float) -> float:
    """Fraction of valid controls scored below the unedited window.

    Ties (|delta logit| <= tie_tolerance) receive 0.5. The tolerance is frozen
    before scoring; it exists because exact float ties arise from genuinely
    order-invariant edits, not from an error.
    """
    d = np.asarray(control_logits, dtype=np.float64) - float(natural_logits)
    if d.size == 0:
        return float("nan")
    return float(((d < -tie_tolerance).sum() + (np.abs(d) <= tie_tolerance).sum() * 0.5) / d.size)


# --------------------------------------------------------------------------
# Claim classification (protocol section 9.3)
# --------------------------------------------------------------------------

def classify_claim(primary: dict, specificity: dict | None,
                   technical_gates_ok: bool,
                   intervention_instantiated: bool = True) -> dict:
    """All co-primary requirements and the specificity guard must pass.

    Whichever single condition is favourable is never selected on its own.

    If the intervention never reached its prespecified balance (amendment A5),
    the level is capped at "valid_demo" no matter how the endpoints fall: an
    under-balanced C1 is not evidence about balanced training.
    """
    r1 = primary.get("requirement_1_predictive_non_inferiority", {})
    r2 = primary.get("requirement_2_improved_motif_preference", {})
    r1_ok = bool(r1.get("passes"))
    r2_ok = bool(r2.get("passes"))
    spec_ok = bool(specificity and specificity.get("passes"))
    spec_required = r1_ok and r2_ok

    if not technical_gates_ok:
        level = "inconclusive"
        wording = ("Technical validity gates did not all pass; the result is not "
                   "interpretable as a confirmatory finding.")
        evidence = {"technical_gates_ok": False}
    elif r1_ok and r2_ok and spec_ok:
        level = "intervention_supported"
        wording = ("Balancing improved motif-order evidence while preserving "
                   "performance in this benchmark")
        evidence = {"requirement_1": r1, "requirement_2": r2, "specificity_guard": specificity}
    elif r1_ok and r2_ok and not spec_ok:
        level = "valid_demo"
        wording = ("A reproducible audit of a small PTBP1 peak predictor. The co-primary "
                   "requirements were met but the specificity guard did not, so improved "
                   "motif recognition cannot be claimed.")
        evidence = {"requirement_1": r1, "requirement_2": r2,
                    "specificity_guard": specificity,
                    "withheld": "specificity guard"}
    else:
        level = "valid_demo"
        failed = []
        if not r1_ok:
            failed.append("predictive non-inferiority on P1")
        if not r2_ok:
            failed.append("improved motif preference")
        wording = ("A reproducible audit of a small PTBP1 peak predictor. The prespecified "
                   f"intervention claim is not supported ({'; '.join(failed)} failed). "
                   "This is a valid, reportable negative or inconclusive result.")
        evidence = {"requirement_1": r1, "requirement_2": r2, "failed_requirements": failed}

    capped_for_uninstantiated_intervention = False
    if not intervention_instantiated and level == "intervention_supported":
        capped_for_uninstantiated_intervention = True
        level = "valid_demo"
        wording = (
            "A reproducible audit of a small PTBP1 peak predictor. The co-primary "
            "requirements were met, but the balancing intervention never reached its "
            "prespecified balance thresholds (amendment A5), so C1 is an under-balanced "
            "condition and the result is reported as observational, not as a successful "
            "balancing experiment."
        )
        evidence = {**evidence, "capped": "intervention not instantiated (amendment A5)"}

    return {
        "claim_level": level,
        "permitted_wording": wording,
        "capped_for_uninstantiated_intervention": capped_for_uninstantiated_intervention,
        "intervention_instantiated": intervention_instantiated,
        "not_claimable": [
            "causal biological recognition",
            "universal debiasing",
            "disease prediction",
            "resolution of the lab's reported problem",
            "mechanism establishment (not available from this computational demo alone)",
        ],
        "evidence": evidence,
        "validity_independent_of_outcome": True,
    }


def specificity_metric(mpa_target: dict[str, float],
                       mpa_outside: dict[str, float]) -> dict:
    """S = MPA_target - MPA_outside, per component."""
    shared = sorted(set(mpa_target) & set(mpa_outside))
    return {c: mpa_target[c] - mpa_outside[c] for c in shared}
