"""Synthetic learning fixtures (protocol section 10.2).

These test the pipeline independently of any biological claim. Each fixture uses
a fixed generated dataset and a broad tolerance established on development
fixtures; none is a per-commit gate on an exact neural metric.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from .motif import encode

L = 201
BASES = np.array(list("ACGU"))
MOTIF = "ACGUACGU"


def _background(n: int, rng: np.random.Generator, gc: float = 0.5) -> np.ndarray:
    p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])
    return rng.choice(BASES, size=(n, L), p=p)


def _plant(seqs: np.ndarray, motif: str, positions: np.ndarray, rng: np.random.Generator) -> None:
    for i, p in enumerate(positions):
        if p + len(motif) <= L:
            seqs[i, p:p + len(motif)] = list(motif)


def _shuffled_motif(motif: str, rng: np.random.Generator) -> str:
    s = list(motif)
    rng.shuffle(s)
    out = "".join(s)
    return motif if out == motif else out


def _split(y: np.ndarray, rng: np.random.Generator, frac: float = 0.5):
    """Random stratified split. The generated data is blocked (positives then
    negatives), so a positional slice would put a single class on each side."""
    idx = rng.permutation(len(y))
    cut = int(len(y) * frac)
    return idx[:cut], idx[cut:]


def _to_str(seqs: np.ndarray) -> list[str]:
    return ["".join(row) for row in seqs]


def _train_tinycnn(seqs_tr, y_tr, seqs_ev, y_ev, epochs: int = 30, seed: int = 0,
                   dropout_off: bool = False):
    from .config import load_protocol
    from .models import build_model

    protocol = load_protocol()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    model = build_model("C0", protocol, L).to(dev)
    if dropout_off:
        model.drop.p = 0.0
    X = torch.from_numpy(np.stack([encode(s).T for s in seqs_tr]).astype(np.float32)).to(dev)
    y = torch.from_numpy(np.asarray(y_tr, dtype=np.float32)).to(dev)
    Xv = torch.from_numpy(np.stack([encode(s).T for s in seqs_ev]).astype(np.float32)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    bs = 64
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]
            loss = crit(model(X[b]), y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        lv = model(Xv).cpu().numpy()
        lt = model(X).cpu().numpy()
    return lv, lt


def _mono_logreg(seqs_tr, y_tr, seqs_ev):
    from .features import mono_features, standardize_apply, standardize_fit

    Xtr, _ = mono_features(seqs_tr)
    Xev, _ = mono_features(seqs_ev)
    st = standardize_fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(standardize_apply(Xtr, st), y_tr)
    return clf.predict_proba(standardize_apply(Xev, st))[:, 1]


# --------------------------------------------------------------------------

def fixture_true_order(rng) -> dict:
    """Order carries the label; composition is matched across classes."""
    n = 1500
    seqs = _background(2 * n, rng, gc=0.5)
    y = np.concatenate([np.ones(n), np.zeros(n)])
    pos_pos = rng.integers(20, L - len(MOTIF) - 20, size=n)
    pos_neg = rng.integers(20, L - len(MOTIF) - 20, size=n)
    _plant(seqs[:n], MOTIF, pos_pos, rng)
    for i in range(n):
        alt = _shuffled_motif(MOTIF, rng)
        p = pos_neg[i]
        seqs[n + i, p:p + len(alt)] = list(alt)
    s = _to_str(seqs)
    itr, iev = _split(y, np.random.default_rng(11))

    lv, _ = _train_tinycnn([s[i] for i in itr], y[itr], [s[i] for i in iev], y[iev])
    lv2, _ = _train_tinycnn([s[i] for i in itr], y[itr], [s[i] for i in iev], y[iev], seed=1)
    cnn_auc = float(np.mean([roc_auc_score(y[iev], lv), roc_auc_score(y[iev], lv2)]))
    mono_pred = _mono_logreg([s[i] for i in itr], y[itr], [s[i] for i in iev])
    mono_auc = float(roc_auc_score(y[iev], mono_pred))

    # Composition gap, normalised by window length. An absolute threshold would
    # only be testing Monte-Carlo noise: two random halves of one class already
    # differ by O(sqrt(L)) bases, so the same-class null is reported alongside.
    comp_pos = np.mean([[s[i].count(b) for b in "ACGU"] for i in range(n)], axis=0)
    comp_neg = np.mean([[s[n + i].count(b) for b in "ACGU"] for i in range(n)], axis=0)
    composition_gap = float(np.abs(comp_pos - comp_neg).sum())
    half = n // 2
    null_a = np.mean([[s[i].count(b) for b in "ACGU"] for i in range(half)], axis=0)
    null_b = np.mean([[s[n + i].count(b) for b in "ACGU"] for i in range(half)], axis=0)
    null_gap = float(np.abs(null_a - null_b).sum())
    composition_gap_fraction = composition_gap / L

    return {
        "fixture": "true_order",
        "cnn_auroc": round(cnn_auc, 4),
        "cnn_auroc_target": 0.95,
        "cnn_passed": bool(cnn_auc >= 0.95),
        "mono_only_auroc": round(mono_auc, 4),
        "mono_only_near_chance": bool(abs(mono_auc - 0.5) < 0.10),
        "max_abs_composition_difference": round(composition_gap, 4),
        "composition_gap_fraction_of_window": round(composition_gap_fraction, 5),
        "composition_gap_tolerance_fraction": 0.05,
        "null_split_half_gap_same_class": round(null_gap, 4),
        "composition_matched": bool(composition_gap_fraction < 0.05),
        "passed": bool(cnn_auc >= 0.95 and abs(mono_auc - 0.5) < 0.10
                       and composition_gap_fraction < 0.05),
    }


def fixture_composition_only(rng) -> dict:
    """The label depends only on composition."""
    n = 1500
    pos = _background(n, rng, gc=0.75)
    neg = _background(n, rng, gc=0.30)
    seqs = _to_str(np.vstack([pos, neg]))
    y = np.concatenate([np.ones(n), np.zeros(n)])
    itr, iev = _split(y, np.random.default_rng(12))
    mono_pred = _mono_logreg([seqs[i] for i in itr], y[itr], [seqs[i] for i in iev])
    mono_auc = float(roc_auc_score(y[iev], mono_pred))
    lv, _ = _train_tinycnn([seqs[i] for i in itr], y[itr], [seqs[i] for i in iev], y[iev])
    return {
        "fixture": "composition_only",
        "mono_only_auroc": round(mono_auc, 4),
        "mono_only_target": 0.90,
        "cnn_auroc": round(float(roc_auc_score(y[iev], lv)), 4),
        "passed": bool(mono_auc >= 0.90),
        "note": "the audit must not attribute this to a planted order pattern; "
                "the composition model succeeding here is the expected result",
    }


def fixture_shortcut_shift(rng) -> dict:
    """A composition cue is correlated with the label in training and reversed in test."""
    n = 1200
    train = _background(2 * n, rng, gc=0.5)
    y = np.concatenate([np.ones(n), np.zeros(n)])
    # label-correlated composition cue in training
    train[:n] = _background(n, rng, gc=0.70)
    pos_p = rng.integers(20, L - len(MOTIF) - 20, size=n)
    _plant(train[:n], MOTIF, pos_p, rng)
    for i in range(n):
        alt = _shuffled_motif(MOTIF, rng)
        train[n + i, pos_p[i]:pos_p[i] + len(alt)] = list(alt)

    # test: composition cue reversed, order pattern preserved
    test = _background(2 * n, rng, gc=0.5)
    test[:n] = _background(n, rng, gc=0.30)      # reversed: positives now GC-poor
    _plant(test[:n], MOTIF, pos_p, rng)
    for i in range(n):
        alt = _shuffled_motif(MOTIF, rng)
        test[n + i, pos_p[i]:pos_p[i] + len(alt)] = list(alt)

    s_tr, s_ev = _to_str(train), _to_str(test)
    mono_pred = _mono_logreg(s_tr, y, s_ev)
    mono_auc = float(roc_auc_score(y, mono_pred))
    lv, _ = _train_tinycnn(s_tr, y, s_ev, y)
    cnn_auc = float(roc_auc_score(y, lv))
    return {
        "fixture": "shortcut_shift",
        "mono_only_auroc_on_shifted_test": round(mono_auc, 4),
        "cnn_auroc_on_shifted_test": round(cnn_auc, 4),
        "composition_model_collapses": bool(mono_auc < 0.6),
        "pipeline_reveals_shift": bool(mono_auc < 0.6),
        "passed": bool(mono_auc < 0.6),
        "note": "no requirement that balancing solves this toy problem; the fixture only "
                "requires that the reporting pipeline exposes the change",
    }


def fixture_random_labels(rng) -> dict:
    """Labels are independent of sequence; held-out discrimination is chance-compatible."""
    n = 1200
    seqs = _background(2 * n, rng, gc=0.5)
    y = np.concatenate([np.ones(n), np.zeros(n)])
    rng.shuffle(y)
    s = _to_str(seqs)
    itr, iev = _split(y, np.random.default_rng(14))
    mono_pred = _mono_logreg([s[i] for i in itr], y[itr], [s[i] for i in iev])
    mono_auc = float(roc_auc_score(y[iev], mono_pred))
    lv, _ = _train_tinycnn([s[i] for i in itr], y[itr], [s[i] for i in iev], y[iev], epochs=15)
    cnn_auc = float(roc_auc_score(y[iev], lv))
    return {
        "fixture": "random_label",
        "mono_only_auroc": round(mono_auc, 4),
        "cnn_auroc": round(cnn_auc, 4),
        "chance_compatible": bool(abs(cnn_auc - 0.5) < 0.15),
        "passed": bool(abs(cnn_auc - 0.5) < 0.15),
        "note": "tolerance reflects finite-sample uncertainty, not a requirement of "
                "exactly 0.5",
    }


def fixture_overfit_wiring() -> dict:
    """Overfit 128 fixed, non-conflicting examples with dropout disabled."""
    rng = np.random.default_rng(7)
    n = 128
    seqs = _background(n, rng, gc=0.5)
    pos = rng.choice(np.arange(0, n), size=n // 2, replace=False)
    y = np.zeros(n)
    y[pos] = 1.0
    for i in pos:
        p = int(rng.integers(0, L - len(MOTIF)))
        seqs[i, p:p + len(MOTIF)] = list(MOTIF)
    s = _to_str(seqs)
    _, lt = _train_tinycnn(s, y, s, y, epochs=300, seed=3, dropout_off=True)
    acc = float(np.mean((lt > 0) == (y > 0.5)))
    return {
        "fixture": "overfit_wiring",
        "train_accuracy": round(acc, 4),
        "target": 0.99,
        "passed": bool(acc >= 0.99),
        "note": "checks wiring, not generalisation",
    }


def run_synthetic_fixtures(log=print) -> dict:
    results = []
    for name, fn in (
        ("true_order", lambda: fixture_true_order(np.random.default_rng(1))),
        ("composition_only", lambda: fixture_composition_only(np.random.default_rng(2))),
        ("shortcut_shift", lambda: fixture_shortcut_shift(np.random.default_rng(3))),
        ("random_label", lambda: fixture_random_labels(np.random.default_rng(4))),
        ("overfit_wiring", fixture_overfit_wiring),
    ):
        try:
            r = fn()
            log(f"[synthetic] {name:20s} passed={r['passed']}")
        except Exception as exc:  # noqa: BLE001
            r = {"fixture": name, "passed": False,
                 "error": f"{type(exc).__name__}: {exc}"}
            log(f"[synthetic] {name:20s} ERROR {exc}")
        results.append(r)
    n_passed = sum(1 for r in results if r.get("passed"))
    return {
        "n_tests": len(results),
        "n_passed": n_passed,
        "n_failed": len(results) - n_passed,
        "all_passed": n_passed == len(results),
        "overfit_accuracy": next(
            (r.get("train_accuracy") for r in results if r["fixture"] == "overfit_wiring"), None),
        "fixtures": results,
        "note": "broad tolerances on fixed generated data; not per-commit gates "
                "(protocol section 10.2)",
    }
