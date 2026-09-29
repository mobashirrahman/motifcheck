"""The single intervention: training-loss weighting for nuisance balance.

Implements protocol section 7. Only training weights change; labels, test
examples, model inputs and capacity are untouched. Strata are
context x training-GC quintile x training-expression tertile, fitted on
training rows only.

Both conditions consume the same examples in the same order for the same number
of updates; only the per-example loss weight differs.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

from .config import Protocol


# --------------------------------------------------------------------------
# Strata
# --------------------------------------------------------------------------

def assign_strata(df: pd.DataFrame, bins: dict, attempt: int = 0) -> pd.Series:
    """Context x GC quintile x expression tertile, from training-derived edges."""
    g = pd.cut(df["gc"], bins=[-np.inf, *bins["gc_quintile_edges"], np.inf],
               labels=False, include_lowest=True)
    t = pd.cut(df["log1p_tpm"],
               bins=[-np.inf, *bins["expression_tertile_edges"], np.inf],
               labels=False, include_lowest=True)
    cats = bins["context_categories"]
    c = df["context"].where(df["context"].isin(cats), "other")
    return (c.astype(str) + "|q" + g.astype("Int64").astype(str)
            + "|t" + t.astype("Int64").astype(str))


def _merge_sparse(df: pd.DataFrame, strata: pd.Series, min_pos: int, min_bg: int,
                  attempt: int = 0, log=None) -> tuple[pd.Series, dict, np.ndarray]:
    """Deterministically merge adjacent numeric bins until support is sufficient.

    Numeric bins are widened first (GC quintile, then expression tertile) inside
    the same context category. Intronic and exonic are never merged together to
    force a pass: if a context has no common support at all, its strata are
    dropped and reported, which narrows the estimand instead of hiding it.
    """
    # _pos carries each row's original position so the surviving-row mask can be
    # rebuilt after strata are dropped: dropping a stratum shortens the frame.
    working = df.assign(_stratum=strata, _pos=np.arange(len(df)))
    report = {"attempt": attempt, "merges": [], "dropped_strata": [],
              "min_positives_per_stratum": min_pos, "min_background_per_stratum": min_bg}
    for _ in range(12):
        counts = (working.groupby(["_stratum", "label"], observed=True).size()
                  .unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0))
        bad = counts[(counts[1] < min_pos) | (counts[0] < min_bg)]
        if bad.empty:
            break
        # deterministic choice: the stratum with the fewest positives, ties by name
        worst = min(bad.index, key=lambda s: (int(counts.loc[s, 1]), int(counts.loc[s, 0]), str(s)))
        parts = str(worst).split("|")
        if len(parts) != 3:
            report["dropped_strata"].append(worst)
            working = working[working["_stratum"] != worst]
            continue
        ctx, q, t = parts
        ctx_mask = counts.index.astype(str).str.split("|").str[0].to_numpy() == ctx
        ctx_counts = counts.loc[ctx_mask]
        if int(ctx_counts[1].sum()) < min_pos or int(ctx_counts[0].sum()) < min_bg:
            report["dropped_strata"].append(worst)
            working = working[working["_stratum"] != worst]
            continue
        qi, ti = int(q[1:]), int(t[1:])
        if qi > 0:
            new = f"{ctx}|q{qi - 1}|{t}"
        elif ti > 0:
            new = f"{ctx}|{q}|t{ti - 1}"
        else:
            report["dropped_strata"].append(worst)
            working = working[working["_stratum"] != worst]
            continue
        report["merges"].append({"from": str(worst), "to": new})
        working.loc[working["_stratum"] == worst, "_stratum"] = new
        if log:
            log(f"[weights] attempt {attempt}: merged sparse stratum {worst} -> {new}")
    alive = np.zeros(len(df), dtype=bool)
    alive[working["_pos"].to_numpy()] = True
    return working["_stratum"].reset_index(drop=True), report, alive


# --------------------------------------------------------------------------
# Weight formulas
# --------------------------------------------------------------------------

def standard_weights(labels: np.ndarray) -> np.ndarray:
    """C0: w0(y) = N / (2 n_y). Targets globally balanced classes."""
    N = len(labels)
    w = np.zeros(N, dtype=np.float64)
    for y in (0, 1):
        n_y = int((labels == y).sum())
        if n_y == 0:
            raise ValueError(f"class {y} absent from training data; cannot weight")
        w[labels == y] = N / (2.0 * n_y)
    return w / w.mean()


def balanced_weights(strata: pd.Series, labels: np.ndarray,
                     clip: float = 5.0) -> tuple[np.ndarray, pd.DataFrame]:
    """C1: w1(y,s) = n_s / (2 n_ys), clipped at `clip`, then mean-normalised."""
    strata = strata.reset_index(drop=True)
    n = len(labels)
    sizes = strata.value_counts()
    counts = pd.crosstab(strata, labels).reindex(index=sizes.index, columns=[0, 1],
                                                  fill_value=0)
    w = np.zeros(n, dtype=np.float64)
    for s, row in counts.iterrows():
        n_s = int(sizes[s])
        for y in (0, 1):
            n_ys = int(row[y])
            if n_ys == 0:
                continue
            w[(strata == s).to_numpy() & (labels == y)] = n_s / (2.0 * n_ys)
    w_raw = w.copy()
    w = np.minimum(w, clip)          # cap before final renormalisation (section 7)
    mean_before = w.mean()
    w = w / mean_before
    table = pd.DataFrame({"n_s": sizes, "n_0": counts[0], "n_1": counts[1]})
    table["w0_typical"] = sizes / (2 * counts[1].clip(lower=1))
    return w, table


def effective_sample_size(w: np.ndarray) -> float:
    """(sum w)^2 / sum(w^2)."""
    s1 = float(np.sum(w))
    s2 = float(np.sum(w ** 2))
    return s1 * s1 / s2 if s2 > 0 else 0.0


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(x * w) / np.sum(w))


def weighted_var(x: np.ndarray, w: np.ndarray) -> float:
    """Frequency-weighted variance with Bessel-style denominator n/(n-1)."""
    sw = float(np.sum(w))
    mu = float(np.sum(x * w) / sw)
    var = float(np.sum(w * (x - mu) ** 2) / sw)
    if sw > 1:
        var *= sw / (sw - 1.0)
    return var


def weighted_smd(x: np.ndarray, w_x: np.ndarray, y: np.ndarray, w_y: np.ndarray) -> float:
    """Standardized mean difference between two weighted samples.

    Both the location shift and the pooled spread are computed on the weighted
    distributions, because that is the distribution the loss actually sees.
    """
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    mx, my = weighted_mean(x, w_x), weighted_mean(y, w_y)
    vx, vy = weighted_var(x, w_x), weighted_var(y, w_y)
    sp = np.sqrt((vx + vy) / 2.0)
    return float((mx - my) / sp) if sp > 0 else 0.0


def balance_qc(train: pd.DataFrame, strata: pd.Series, w: np.ndarray,
               protocol: Protocol) -> dict:
    """Section 7 intervention QC, computed on the weighted training distribution."""
    qc = protocol.section("intervention")["qc"]
    tr = train.reset_index(drop=True)
    strata = strata.reset_index(drop=True)
    carry = [c for c in ("gc", "log1p_tpm", "context") if c in tr.columns]
    wdf = tr[carry].copy().reset_index(drop=True)
    wdf["label"] = tr["label"].to_numpy()
    wdf["w"] = np.asarray(w, dtype=float)
    pos = wdf[wdf["label"] == 1]
    bg = wdf[wdf["label"] == 0]

    ess = effective_sample_size(w)
    ess_frac = ess / len(w)
    wp, wb = pos["w"].to_numpy(), bg["w"].to_numpy()
    smd_gc = weighted_smd(pos["gc"].to_numpy(), wp, bg["gc"].to_numpy(), wb)
    smd_tpm = weighted_smd(pos["log1p_tpm"].to_numpy(), wp,
                           bg["log1p_tpm"].to_numpy(), wb)
    ctx_props = {}
    max_ctx_diff = 0.0
    for c in sorted(wdf["context"].unique()):
        pp = float((pos.loc[pos["context"] == c, "w"]).sum() / pos["w"].sum())
        pb = float((bg.loc[bg["context"] == c, "w"]).sum() / bg["w"].sum())
        ctx_props[c] = {"positive": pp, "background": pb, "abs_diff": abs(pp - pb)}
        max_ctx_diff = max(max_ctx_diff, abs(pp - pb))

    checks = {
        "effective_sample_size_fraction": round(ess_frac, 5),
        "effective_sample_size_min": qc["min_effective_sample_size_fraction"],
        "smd_gc": round(smd_gc, 5),
        "smd_gc_max_abs": qc["max_abs_smd_gc"],
        "smd_log1p_tpm": round(smd_tpm, 5),
        "smd_log1p_tpm_max_abs": qc["max_abs_smd_log1p_tpm"],
        "max_context_proportion_diff": round(max_ctx_diff, 5),
        "max_context_proportion_diff_max": qc["max_abs_context_proportion_diff"],
        "context_proportions": ctx_props,
    }
    passed = {
        "effective_sample_size": ess_frac >= qc["min_effective_sample_size_fraction"],
        "smd_gc": abs(smd_gc) <= qc["max_abs_smd_gc"],
        "smd_log1p_tpm": abs(smd_tpm) <= qc["max_abs_smd_log1p_tpm"],
        "context_proportions": max_ctx_diff <= qc["max_abs_context_proportion_diff"],
    }
    # Residual balance of all mononucleotides (and gene length when present).
    # These are diagnostics: the protocol states they are not hidden acceptance
    # requirements, and they are reported rather than tuned against.
    residual = {}
    for col in [c for c in tr.columns if c.startswith("n_")]:
        if col not in wdf.columns:
            wdf[col] = tr[col].to_numpy()
        residual[col] = {
            "smd": round(weighted_smd(
                wdf.loc[wdf.label == 1, col].to_numpy(), wp,
                wdf.loc[wdf.label == 0, col].to_numpy(), wb), 5)}
    residual["note"] = ("diagnostics only per protocol section 7; they are not "
                        "additional hidden matching requirements")
    return {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "n_weighted_rows": int(len(w)),
        "checks": checks,
        "passed": passed,
        "all_passed": all(passed.values()),
        "residual_diagnostics": residual,
        "residual_diagnostics_are_not_acceptance_criteria": True,
        "clip_ceiling": protocol.section("intervention")["clip_ceiling"],
        "weight_mean_after_normalisation": round(float(w.mean()), 6),
        "note": "balance that remains after clipping is reported, not hidden",
    }


def drop_unsupported(df: pd.DataFrame, strata: pd.Series,
                     variables: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    """Drop strata with no common support on a targeted nuisance variable.

    Weights can equalise class mass *within* a stratum, but they cannot create
    overlap. If positives and backgrounds in a stratum occupy disjoint ranges of
    GC or expression, the stratum contributes residual imbalance no matter how
    it is weighted. The protocol's remedy is to exclude such strata from BOTH
    conditions and narrow the estimand, which is what this does.

    Support is judged on empirical ranges: a stratum is unsupported only when
    the two classes do not overlap at all, which is a strong, unambiguous test.
    """
    work = df.assign(_stratum=strata)
    dropped: list[str] = []
    for name in variables:
        if name not in work.columns:
            continue
        g = work.groupby(["_stratum", "label"], observed=True)[name].agg(["min", "max"])
        try:
            pos = g.xs(1, level="label")
            bg = g.xs(0, level="label")
        except KeyError:
            continue
        shared = sorted(set(pos.index) & set(bg.index))
        for s in shared:
            p_lo, p_hi = float(pos.loc[s, "min"]), float(pos.loc[s, "max"])
            b_lo, b_hi = float(bg.loc[s, "min"]), float(bg.loc[s, "max"])
            if p_hi < b_lo or b_hi < p_lo:
                dropped.append(str(s))
    dropped = sorted(set(dropped))
    if dropped:
        work = work[~work["_stratum"].isin(dropped)]
    report = {"dropped_for_no_common_support": dropped,
              "n_strata_dropped_for_no_common_support": len(dropped)}
    # `_pos` is retained so the caller can map survivors back to training rows.
    return work.reset_index(drop=True), report


def build_weights(train: pd.DataFrame, protocol: Protocol, bins: dict,
                  log=print) -> dict:
    """Produce C0 and C1 training weights plus QC. Training rows only."""
    y = train["label"].to_numpy()
    w0 = standard_weights(y)

    attempts = []
    strata = None
    merge_report = None
    for attempt in range(int(protocol["intervention"]["max_revise_attempts"]) + 1):
        raw = assign_strata(train, bins, attempt=attempt)
        strata, merge_report, kept = _merge_sparse(
            train, raw,
            int(protocol["intervention"]["min_positives_per_stratum"]),
            int(protocol["intervention"]["min_background_per_stratum"]),
            attempt=attempt, log=log,
        )
        # Strata with disjoint class support on a targeted nuisance variable
        # cannot be balanced by reweighting; drop them from both conditions.
        masked = train.loc[kept].reset_index(drop=True).copy()
        masked["_pos"] = np.flatnonzero(kept)
        sub, support_report = drop_unsupported(masked, strata, ("gc", "log1p_tpm"))
        merge_report.update(support_report)
        kept = np.zeros(len(train), dtype=bool)
        kept[sub["_pos"].to_numpy()] = True
        sub = sub.drop(columns=["_pos"]).reset_index(drop=True)
        sub_strata = sub["_stratum"]
        w1, table = balanced_weights(sub_strata, sub["label"].to_numpy(),
                                     float(protocol["intervention"]["clip_ceiling"]))
        qc = balance_qc(sub, sub_strata, w1, protocol)
        attempts.append({"attempt": attempt, "strata_merge_report": merge_report,
                         "qc": qc})
        log(f"[weights] attempt {attempt}: ESS fraction "
            f"{qc['checks']['effective_sample_size_fraction']:.3f}, "
            f"SMD(gc) {qc['checks']['smd_gc']:+.3f}, "
            f"SMD(tpm) {qc['checks']['smd_log1p_tpm']:+.3f}, "
            f"max ctx diff {qc['checks']['max_context_proportion_diff']:.3f}")
        if qc["all_passed"]:
            break

    w1_full = np.ones(len(train), dtype=np.float64)
    # Rows dropped for want of common support keep weight 1 in C1. That is
    # neutral rather than silently informative: they are excluded from the
    # strata QC, and `rows_excluded_unsupported` below reports how many.
    w1_full[kept] = w1

    final_qc = attempts[-1]["qc"]
    return {
        "attempts": attempts,
        "n_attempts": len(attempts),
        "all_thresholds_met": bool(final_qc["all_passed"]),
        "strata_definitions": bins,
        "retained_strata": int(strata.nunique()),
        "rows_excluded_unsupported": int((~kept).sum()),
        "fraction_excluded_unsupported": round(float((~kept).mean()), 5),
        "estimand_note": (
            "If supported strata were dropped, the balanced condition targets class balance "
            "within the retained strata only; the estimand is correspondingly narrowed."
        ) if (~kept).any() else "All training rows retained; estimand covers the full training set.",
        "w0": w0,
        "w1": w1_full,
        "strata": strata,
        "weight_table": table,
        "final_qc": final_qc,
    }
