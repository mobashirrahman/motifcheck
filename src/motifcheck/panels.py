"""Evaluation panels: P1, P2, the motif-order audit panel, and the ISM panel.

All panels are selected from the frozen candidate master list using
prespecified metadata rules and fixed hashes. No panel is ever chosen by
searching for a favourable model result, and no selection uses model
confidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .hashing import stable_unit_interval
from .motif import Motif, best_hits, scan_scores

PANEL_SEED = 20260929


def _bin_keys(df: pd.DataFrame, bins: dict) -> pd.Series:
    g = pd.cut(df["gc"], bins=[-np.inf, *bins["gc_quintile_edges"], np.inf],
               labels=False, include_lowest=True)
    t = pd.cut(df["log1p_tpm"],
               bins=[-np.inf, *bins["expression_tertile_edges"], np.inf],
               labels=False, include_lowest=True)
    return (df["context"].astype(str) + "|q" + g.astype("Int64").astype(str)
            + "|t" + t.astype("Int64").astype(str))


def p1_panel(df: pd.DataFrame) -> pd.DataFrame:
    """All eligible held-out windows after the predefined sampling policy."""
    return df.copy()


def p2_panel(df: pd.DataFrame, bins: dict, seed: int = PANEL_SEED) -> tuple[pd.DataFrame, dict]:
    """Prespecified 1:1 positive/background match on context, GC and expression.

    This panel asks whether sequence models still discriminate once simple
    nuisance information is reduced. It is not a representative sample of the
    transcriptome and its prevalence differs from P1 by construction, so an AP
    change between panels is not a transfer-degradation signal.
    """
    keys = _bin_keys(df, bins)
    work = df.assign(_key=keys)
    order = {w: stable_unit_interval(w, seed=seed) for w in work["window_id"]}
    work = work.assign(_h=work["window_id"].map(order)).sort_values("_h")
    pos = work[work["label"] == 1]
    bg = work[work["label"] == 0]

    keep_pos, keep_bg = [], []
    unmatched = 0
    for k in sorted(pos["_key"].unique()):
        p = pos[pos["key"] == k] if False else pos[pos["_key"] == k]
        b = bg[bg["_key"] == k]
        n = min(len(p), len(b))
        if n == 0:
            unmatched += len(p)
            continue
        unmatched += len(p) - n
        keep_pos += list(p.index[:n])
        keep_bg += list(b.index[:n])
    out = work.loc[sorted(keep_pos + keep_bg)].drop(columns=["_key", "_h"])
    info = {
        "matching_variables": ["context", "gc_quintile", "expression_tertile"],
        "bin_source": "training-derived edges (train split only)",
        "ratio": "1:1 positive:background",
        "n_input_rows": int(len(df)),
        "n_selected": int(len(out)),
        "n_positive": int((out["label"] == 1).sum()) if len(out) else 0,
        "n_background": int((out["label"] == 0).sum()) if len(out) else 0,
        "n_unmatched_positives": int(unmatched),
        "bins": bins,
        "interpretation": "diagnostic panel; prevalence differs from P1 by design",
    }
    return out, info


def score_motifs(df: pd.DataFrame, motif: Motif, threshold: float) -> pd.DataFrame:
    """Attach motif score and hit count. Lives in the audit table only.

    Plan section 4.4: motif scores must not enter CNN inputs or the balancing
    intervention, so this frame is joined at audit time and never fed to
    training.
    """
    rows = []
    for wid, seq in zip(df["window_id"], df["sequence"]):
        sc = scan_scores(seq, motif)
        if sc.size == 0:
            rows.append((wid, float("nan"), 0, -1))
            continue
        hits = best_hits(seq, motif, threshold)
        top = int(sc.argmax())
        rows.append((wid, float(sc.max()), len(hits), top))
    return pd.DataFrame(rows, columns=["window_id", "motif_max_score",
                                        "motif_hit_count", "motif_best_position"])


def audit_panel(df: pd.DataFrame, motif: Motif, threshold: float,
                max_windows: int, min_components: int,
                seed: int = PANEL_SEED) -> tuple[pd.DataFrame, dict]:
    """Held-out windows with a supported motif occurrence, chosen without models.

    Selection basis is label, annotation, motif score and a fixed hash. Windows
    are spread across components so the audit does not reuse one gene's windows
    as if they were independent.
    """
    scored = df.assign(_h=[stable_unit_interval(w, seed=seed) for w in df["window_id"]])
    scored = scored.sort_values(["component_id", "_h"]).reset_index(drop=True)
    per_comp_cap = max(1, int(np.ceil(max_windows / max(min_components, 1))))
    out_rows: list[int] = []
    seen_comp: dict[str, int] = {}
    for pos, row in enumerate(scored.itertuples(index=False)):
        sc = scan_scores(row.sequence, motif)
        if sc.size == 0 or float(sc.max()) < threshold:
            continue
        c = row.component_id
        if seen_comp.get(c, 0) >= per_comp_cap:
            continue
        seen_comp[c] = seen_comp.get(c, 0) + 1
        out_rows.append(pos)
        if len(out_rows) >= max_windows:
            break
    panel = scored.iloc[out_rows].drop(columns=["_h"]).reset_index(drop=True)
    n_comp = int(panel["component_id"].nunique()) if len(panel) else 0
    info = {
        "motif_id": motif.motif_id,
        "motif_class": motif.motif_class,
        "threshold": float(threshold),
        "n_windows": int(len(panel)),
        "n_components": n_comp,
        "n_positive": int((panel["label"] == 1).sum()) if len(panel) else 0,
        "n_background": int((panel["label"] == 0).sum()) if len(panel) else 0,
        "selection_basis": ["label", "annotation", "motif_score", "fixed_hash"],
        "model_confidence_used": False,
        "per_component_cap": per_comp_cap,
        "exploratory": bool(n_comp < min_components or len(panel) < 2 * min_components),
        "exploratory_reason": (
            f"fewer than {min_components} independent components or too few valid windows"
            if (n_comp < min_components or len(panel) < 2 * min_components) else None),
    }
    return panel, info


def ism_panel(df: pd.DataFrame, motif: Motif, threshold: float, size: int,
              seed: int = PANEL_SEED) -> tuple[pd.DataFrame, dict]:
    """16 windows per positive/background x motif-present/absent stratum.

    If a stratum is short, all available examples are used and the shortfall is
    reported rather than backfilled from another stratum.
    """
    scored = df.assign(
        _score=[float(scan_scores(s, motif).max()) if scan_scores(s, motif).size
                else float("nan") for s in df["sequence"]],
    )
    scored["_present"] = scored["_score"] >= threshold
    scored = scored.assign(_h=[stable_unit_interval(w, seed=seed) for w in scored["window_id"]])
    per = size // 4
    picks, counts = [], {}
    for label in (1, 0):
        for present in (True, False):
            sub = scored[(scored["label"] == label) & (scored["_present"] == present)] \
                .sort_values(["component_id", "_h"])
            take = sub.head(per)
            key = f"label{label}_motif_{'present' if present else 'absent'}"
            counts[key] = {"available": int(len(sub)), "selected": int(len(take))}
            picks += list(take.index)
    panel = df.loc[sorted(picks)].copy()
    info = {
        "requested_total": size,
        "realized_total": int(len(panel)),
        "per_stratum_target": per,
        "strata_counts": counts,
        "short_strata": [k for k, v in counts.items() if v["selected"] < v["available"]
                         or v["selected"] < per],
        "same_panel_for_all_models_and_seeds": True,
        "selection_basis": "stratum + fixed hash; never model confidence",
    }
    return panel, info
