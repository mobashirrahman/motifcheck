"""Motif-order audit and exact attribution.

The primary mechanistic endpoint is motif preference accuracy (MPA): for each
natural window containing the target motif, we build local controls that
preserve the edited segment's mononucleotide counts and the overall window
length while destroying sequence order, and ask whether the model scores the
unedited window above its own controls.

Exact in silico mutagenesis (ISM) is the attribution reference and measures
model behaviour, not experimental mutation effects. A gradient (Taylor)
approximation is implemented and checked against exact ISM, never assumed
equal to it.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .hashing import stable_hash_int
from .motif import Motif, best_hits, encode, scan_scores
from .stats import motif_preference

_COMP = str.maketrans("ACGU", "UGCA")


def _rng_for(*parts) -> np.random.Generator:
    seed = stable_hash_int(*parts, nbits=63)
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------
# Order-disrupted controls
# --------------------------------------------------------------------------

@dataclass
class Control:
    sequence: str
    start: int
    end: int
    original_target_score: float
    edited_target_score: float
    new_hits: int
    valid: bool
    reason: str | None
    mono_preserved: bool
    length_preserved: bool
    dinucleotide_l1_change: float


def _dinuc_vector(seq: str) -> np.ndarray:
    v = np.zeros(16)
    if len(seq) < 2:
        return v
    for i in range(len(seq) - 1):
        k = seq[i:i + 2]
        idx = "ACGU".find(k[0]) * 4 + "ACGU".find(k[1])
        if idx >= 0:
            v[idx] += 1
    return v


def build_order_controls(sequence: str, motif: Motif, hit_start: int, hit_end: int,
                         threshold: float, n_controls: int, min_controls: int,
                         key: str) -> tuple[list[Control], dict]:
    """Mononucleotide-count- and length-preserving local shuffles of one motif instance.

    Validity rules, fixed before scoring:
      * the edited segment keeps its exact multiset of bases and its length;
      * the full window keeps its length and its global mononucleotide counts;
      * the target motif score at the same position strictly decreases;
      * no new above-threshold motif occurrence is introduced anywhere in the
        window (otherwise the control could be explained by a gained site).
    """
    seg = sequence[hit_start:hit_end]
    multiset = sorted(seg)
    original_target = float(scan_scores(sequence, motif)[hit_start])
    original_hits = {h[0] for h in best_hits(sequence, motif, threshold)}

    rng = _rng_for(key, hit_start, hit_end, "order")
    controls: list[Control] = []
    uninformative_reason = None
    attempts = 0
    max_attempts = max(n_controls * 200, 2000)

    while len(controls) < n_controls and attempts < max_attempts:
        attempts += 1
        shuffled = list(multiset)
        rng.shuffle(shuffled)
        shuffled = "".join(shuffled)
        if shuffled == seg:
            continue
        edited = sequence[:hit_start] + shuffled + sequence[hit_end:]
        if len(edited) != len(sequence):
            continue
        mono_ok = all(edited.count(b) == sequence.count(b) for b in "ACGU")
        if not mono_ok:
            continue
        scores = scan_scores(edited, motif)
        if scores.size == 0:
            continue
        edited_target = float(scores[hit_start])
        if not (edited_target < original_target):
            continue
        new_hits = sum(1 for h in best_hits(edited, motif, threshold) if h[0] not in original_hits)
        if new_hits:
            continue
        dn_change = float(np.abs(_dinuc_vector(edited) - _dinuc_vector(sequence)).sum())
        controls.append(Control(
            sequence=edited, start=hit_start, end=hit_end,
            original_target_score=original_target, edited_target_score=edited_target,
            new_hits=0, valid=True, reason=None, mono_preserved=True,
            length_preserved=True, dinucleotide_l1_change=dn_change,
        ))

    info = {
        "attempts": attempts,
        "valid_controls": len(controls),
        "min_controls_required": min_controls,
        "informative": len(controls) >= min_controls,
        "original_target_score": original_target,
    }
    if len(controls) < min_controls:
        info["uninformative_reason"] = (
            f"fewer than {min_controls} valid order-disrupted controls could be constructed "
            "for this site (low-complexity or highly constrained motif instance)")
        uninformative_reason = info["uninformative_reason"]
    return controls, info


def build_outside_controls(sequence: str, motif: Motif, hit_start: int, hit_end: int,
                          threshold: float, n_controls: int, min_controls: int,
                          key: str) -> tuple[list[Control], dict]:
    """Equal-length order changes placed OUTSIDE the motif instance.

    Used by the specificity guard: a model that merely prefers all unedited
    natural sequences will score these no better than the target controls, so
    S = MPA_target - MPA_outside removes that confound. Control sites are chosen
    without reference to any model score.
    """
    L = len(sequence)
    outside = [(0, hit_start), (hit_end, L)]
    outside = [(s, e) for s, e in outside if e - s >= 4]
    if not outside:
        return [], {"valid_controls": 0, "informative": False,
                    "uninformative_reason": "no region outside the motif is long enough"}
    target_len = hit_end - hit_start
    original_target = float(scan_scores(sequence, motif)[hit_start])
    original_hits = {h[0] for h in best_hits(sequence, motif, threshold)}

    rng = _rng_for(key, "outside")
    controls: list[Control] = []
    max_attempts = max(n_controls * 200, 2000)
    for _ in range(max_attempts):
        if len(controls) >= n_controls:
            break
        s, e = outside[int(rng.integers(0, len(outside)))]
        width = min(target_len, e - s)
        if width < 3:
            continue
        st = int(rng.integers(s, e - width + 1))
        seg = sequence[st:st + width]
        shuffled = list(seg)
        rng.shuffle(shuffled)
        shuffled = "".join(shuffled)
        if shuffled == seg:
            continue
        edited = sequence[:st] + shuffled + sequence[st + width:]
        if not all(edited.count(b) == sequence.count(b) for b in "ACGU"):
            continue
        new_hits = sum(1 for h in best_hits(edited, motif, threshold) if h[0] not in original_hits)
        if new_hits:
            continue
        controls.append(Control(
            sequence=edited, start=st, end=st + width,
            original_target_score=original_target,
            edited_target_score=float("nan"), new_hits=0, valid=True, reason=None,
            mono_preserved=True, length_preserved=True,
            dinucleotide_l1_change=float(np.abs(_dinuc_vector(edited) - _dinuc_vector(sequence)).sum()),
        ))
    info = {
        "valid_controls": len(controls),
        "informative": len(controls) >= min_controls,
        "control_length": target_len,
        "selection": "outside-motif positions chosen by hash, never by model score",
    }
    if len(controls) < min_controls:
        info["uninformative_reason"] = (
            f"fewer than {min_controls} valid outside-motif controls could be constructed")
    return controls, info


def decoy_motifs(motif: Motif, seed_key: str = "decoys") -> list[Motif]:
    """Composition-matched decoys, fixed before any model is scored.

    Each decoy keeps the primary motif's exact column composition but destroys
    its information content, so a model that responds to composition alone
    should not show a preference for the genuine target over these.
    """
    out = []
    rev = motif.matrix[::-1]
    out.append(Motif(motif_id=f"{motif.motif_id}_revcomp", matrix=rev,
                     background=motif.background, pseudocount=motif.pseudocount,
                     motif_class="decoy", assay_type="reverse_composition_matched",
                     study="constructed", study_id="revcomp", cisbp_tf_id=motif.cisbp_tf_id,
                     database_build=motif.database_build))
    rng = _rng_for(seed_key)
    for i in range(2):
        flat = motif.matrix.ravel().copy()
        rng.shuffle(flat)
        out.append(Motif(motif_id=f"{motif.motif_id}_shuffled{i+1}",
                         matrix=flat.reshape(motif.matrix.shape),
                         background=motif.background, pseudocount=motif.pseudocount,
                         motif_class="decoy", assay_type="composition_matched_shuffle",
                         study="constructed", study_id=f"shuffle{i+1}",
                         cisbp_tf_id=motif.cisbp_tf_id, database_build=motif.database_build))
    return out


# --------------------------------------------------------------------------
# Exact and Taylor ISM
# --------------------------------------------------------------------------

def _substitution_sites(sequence: str) -> tuple[list[int], list[int]]:
    """Positions and alternative bases: L*3 sites, observed base excluded."""
    base = encode(sequence)
    rows, cols = [], []
    for i in range(base.shape[0]):
        obs = int(base[i].argmax())
        for a in range(4):
            if a != obs:
                rows.append(i)
                cols.append(a)
    return rows, cols


@torch.no_grad()
def exact_ism(model, sequence: str, device: str = "cuda",
              chunk: int = 128) -> np.ndarray:
    """Logit change for all L*3 single-base substitutions (603 for L=201).

    The input tensor is never mutated; the observed base's own effect is exactly
    zero by definition because it is not among the mutant sites.
    """
    base = encode(sequence)                       # (L, 4)
    x0 = torch.from_numpy(base.T.astype(np.float32).copy())[None].to(device)
    f0 = float(model(x0)[0])
    rows, cols = _substitution_sites(sequence)
    out = np.zeros(len(rows), dtype=np.float32)
    for s in range(0, len(rows), chunk):
        sel = slice(s, min(s + chunk, len(rows)))
        batch = x0.repeat(len(rows[sel]), 1, 1)
        r = torch.tensor(rows[sel], device=device)
        c = torch.tensor(cols[sel], device=device)
        pos = torch.arange(len(rows[sel]), device=device)
        # clear every channel at the substituted position, then set the new base
        batch[pos, :, r] = 0.0
        batch[pos, c, r] = 1.0
        out[sel] = (model(batch) - f0).float().cpu().numpy()
    return out


def exact_ism_matrix(model, sequence: str, device: str = "cuda") -> np.ndarray:
    """(L, 4) matrix of substitution effects; the observed base's entry is 0.

    Reconstructed from the same (position, alternative base) site list used to
    build the mutant batches, so the two views cannot drift out of alignment.
    """
    L = len(sequence)
    M = np.zeros((L, 4), dtype=np.float32)
    flat = exact_ism(model, sequence, device=device)
    rows, cols = _substitution_sites(sequence)
    for k, (i, a) in enumerate(zip(rows, cols)):
        M[i, a] = flat[k]
    return M


def taylor_ism(model, sequence: str, device: str = "cuda") -> np.ndarray:
    """First-order effect df/dx[i,a] - df/dx[i,r] at the observed input."""
    base = encode(sequence)
    x = torch.from_numpy(base.T.copy())[None].to(device).requires_grad_(True)
    f = model(x).sum()
    (grad,) = torch.autograd.grad(f, x)
    g = grad[0].cpu().numpy()                 # (channels, L)
    obs = base.argmax(axis=1)                 # (L,)
    g_obs = g[obs, np.arange(base.shape[0])]  # df/dx[i, observed_base_at_i]
    # taylor_delta(i, a) = df/dx[i, a] - df/dx[i, r]
    return (g.T - g_obs[:, None]).astype(np.float32)


def compare_ism(exact: np.ndarray, taylor: np.ndarray,
                top_fraction: float = 0.10) -> dict:
    """Agreement between exact and Taylor substitution effects.

    Constant or near-zero sequences are reported as undefined rather than being
    assigned a correlation of zero or one.
    """
    from scipy.stats import spearmanr

    e = np.asarray(exact).ravel()
    t = np.asarray(taylor).ravel()
    if len(e) < 3:
        return {"status": "undefined", "reason": "too few substitutions"}
    if np.std(e) < 1e-8:
        return {"status": "undefined", "reason": "exact effects are constant",
                "max_abs_exact": float(np.max(np.abs(e)))}
    rho = float(spearmanr(e, t).statistic)
    if not np.isfinite(rho):
        return {"status": "undefined", "reason": "spearman undefined",
                "max_abs_exact": float(np.max(np.abs(e)))}
    k = max(1, int(len(e) * top_fraction))
    top = np.argsort(-np.abs(e))[:k]
    agree = float(np.mean(np.sign(e[top]) == np.sign(t[top])))
    return {
        "status": "ok",
        "spearman": round(rho, 5),
        "sign_agreement_top_decile": round(agree, 5),
        "n_substitutions": int(len(e)),
        "n_in_top_decile": int(k),
        "max_abs_exact": float(np.max(np.abs(e))),
    }


# --------------------------------------------------------------------------
# Panel construction
# --------------------------------------------------------------------------

def build_control_panel(panel_df: pd.DataFrame, motif: Motif, threshold: float,
                        n_controls: int, min_controls: int,
                        include_outside: bool = True, log=print) -> dict:
    """Construct the fixed edit panel for every window in the audit panel."""
    rows = []
    n_uninformative = 0
    n_instances = 0
    for row in panel_df.itertuples(index=False):
        seq = row.sequence
        hits = best_hits(seq, motif, threshold)
        if not hits:
            continue
        hit_start, hit_end, _ = hits[0]
        n_instances += 1
        tgt, tgt_info = build_order_controls(
            seq, motif, hit_start, hit_end, threshold, n_controls, min_controls,
            key=f"{row.window_id}:target")
        out = []
        if include_outside:
            out, out_info = build_outside_controls(
                seq, motif, hit_start, hit_end, threshold, n_controls, min_controls,
                key=f"{row.window_id}:outside")
        if not tgt_info["informative"]:
            n_uninformative += 1
            continue
        for kind, ctrls in (("target", tgt), ("outside", out)):
            for k, c in enumerate(ctrls):
                rows.append({
                    "window_id": row.window_id,
                    "component_id": row.component_id,
                    "label": int(row.label),
                    "control_kind": kind,
                    "control_index": k,
                    "natural_sequence": seq,
                    "edited_sequence": c.sequence,
                    "edit_start": c.start, "edit_end": c.end,
                    "original_target_score": c.original_target_score,
                    "edited_target_score": c.edited_target_score,
                    "new_high_scoring_occurrences": c.new_hits,
                    "mono_counts_preserved": c.mono_preserved,
                    "length_preserved": c.length_preserved,
                    "dinucleotide_l1_change": c.dinucleotide_l1_change,
                    "valid": c.valid,
                    "exclusion_reason": c.reason,
                })
    panel = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["window_id", "component_id", "label", "control_kind", "control_index",
                 "natural_sequence", "edited_sequence", "edit_start", "edit_end",
                 "original_target_score", "edited_target_score",
                 "new_high_scoring_occurrences", "mono_counts_preserved",
                 "length_preserved", "dinucleotide_l1_change", "valid", "exclusion_reason"])
    info = {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "n_windows_considered": int(len(panel_df)),
        "n_motif_instances": n_instances,
        "n_windows_with_controls": int(panel["window_id"].nunique()) if len(panel) else 0,
        "n_uninformative_windows": int(n_uninformative),
        "n_controls_target": int((panel["control_kind"] == "target").sum()) if len(panel) else 0,
        "n_controls_outside": int((panel["control_kind"] == "outside").sum()) if len(panel) else 0,
        "max_controls_per_instance": n_controls,
        "min_controls_per_instance": min_controls,
        "all_controls_preserve_mono_and_length": bool(
            panel["mono_counts_preserved"].all() and panel["length_preserved"].all()
        ) if len(panel) else None,
        "max_dinucleotide_l1_change": float(panel["dinucleotide_l1_change"].max())
        if len(panel) else None,
        "mean_dinucleotide_l1_change": float(panel["dinucleotide_l1_change"].mean())
        if len(panel) else None,
    }
    return {"panel": panel, "info": info}


def compute_mpa_by_component(pred_df: pd.DataFrame, control_panel: pd.DataFrame,
                             tie_tolerance: float,
                             control_kind: str = "target") -> dict[str, float]:
    """Average within component, then return the per-component MPA values."""
    sub = control_panel[control_panel["control_kind"] == control_kind]
    if not len(sub):
        return {}
    nat = pred_df.set_index(["window_id", "control_kind"])["logit"]
    values: dict[str, list[float]] = {}
    for (wid, kind, _), grp in sub.groupby(["window_id", "control_kind", "control_index"]):
        natural = float(nat.get((wid, control_kind), np.nan))
        if not np.isfinite(natural):
            continue
        edited = float(grp["logit"].iloc[0])
        values.setdefault(grp["component_id"].iloc[0], []).append(
            motif_preference(natural, np.array([edited]), tie_tolerance))
    return {c: float(np.mean(v)) for c, v in values.items()}
