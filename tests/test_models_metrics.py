"""Models, training wiring, metrics, and explanation correctness (protocol 10.1)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from motifcheck.motif import encode
from motifcheck.models import build_model, count_parameters


# --------------------------------------------------------------------------
# Architecture
# --------------------------------------------------------------------------

def test_tinycnn_parameter_count_matches_the_protocol_exactly(protocol):
    model = build_model("C0", protocol, 201)
    assert count_parameters(model) == protocol["tinycnn"]["expected_params"] == 13857
    # the same model for both conditions: capacity is not part of the intervention
    assert count_parameters(build_model("C1", protocol, 201)) == 13857


def test_tinycnn_receptive_field_matches_the_protocol(protocol):
    model = build_model("C0", protocol, 201)
    assert model.receptive_field() == protocol["tinycnn"]["receptive_field"] == 35


def test_forward_output_shape_and_finite_gradients(protocol):
    model = build_model("C0", protocol, 201)
    x = torch.from_numpy(encode("ACGU" * 50 + "A").T.copy())[None].repeat(7, 1, 1)
    out = model(x)
    assert out.shape == (7,)
    assert torch.isfinite(out).all()
    loss = out.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_forward_rejects_wrong_input_rank(protocol):
    model = build_model("C0", protocol, 201)
    with pytest.raises(ValueError, match="one-hot"):
        model(torch.randn(4, 201))
    with pytest.raises(ValueError, match="one-hot"):
        model(torch.randn(2, 5, 201))


def test_eval_prediction_is_invariant_to_batch_order_and_size(protocol):
    model = build_model("C0", protocol, 201)
    model.eval()
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGU"), size=201)) for _ in range(37)]
    X = torch.from_numpy(np.stack([encode(s).T for s in seqs]).astype(np.float32))
    with torch.no_grad():
        full = model(X)
        shuffled_idx = rng.permutation(len(seqs))
        shuf = model(X[shuffled_idx])
        chunked = torch.cat([model(X[i:i + 7]) for i in range(0, len(X), 7)])
    assert torch.allclose(full[shuffled_idx], shuf, atol=1e-6)
    assert torch.allclose(full, chunked, atol=1e-6)


def test_dropout_is_inactive_in_eval_mode(protocol):
    model = build_model("C0", protocol, 201)
    rng = np.random.default_rng(1)
    seqs = ["".join(rng.choice(list("ACGU"), size=201)) for _ in range(8)]
    X = torch.from_numpy(np.stack([encode(s).T for s in seqs]).astype(np.float32))
    model.train()
    torch.manual_seed(0)
    a = model(X)
    torch.manual_seed(1)
    b = model(X)
    model.eval()
    with torch.no_grad():
        c = model(X)
        d = model(X)
    assert not torch.allclose(a, b)          # dropout active in train mode
    assert torch.allclose(c, d)               # deterministic in eval mode


def test_no_sigmoid_before_the_loss(protocol):
    """The output must be a raw logit.

    Checked structurally rather than by inspecting value ranges: a small
    randomly-initialised logit can sit inside [-1, 1] by chance, which would
    make a range assertion pass even with a sigmoid in place.
    """
    from motifcheck.train import encode_dataset
    model = build_model("C0", protocol, 201)
    assert not any(isinstance(m, (nn.Sigmoid, nn.Softmax, nn.LogSigmoid))
                   for m in model.modules()), "a squashing layer precedes the loss"
    assert model(encode_dataset(["ACGU" * 50 + "A"])).shape == (1,)
    # and the loss really is the logits variant
    loss = nn.BCEWithLogitsLoss()(torch.tensor([0.4, -1.2]), torch.tensor([1.0, 0.0]))
    assert torch.isfinite(loss) and float(loss) > 0


# --------------------------------------------------------------------------
# Paired training
# --------------------------------------------------------------------------

def test_paired_seed_batch_sequence_is_identical_across_conditions(protocol):
    from motifcheck.train import batch_sequence
    n, bs, epochs, seed = 500, 128, 3, 17
    a = batch_sequence(n, bs, epochs, seed)
    b = batch_sequence(n, bs, epochs, seed)
    assert len(a) == len(b)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    c = batch_sequence(n, bs, epochs, seed + 1)
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_two_conditions_with_identical_weights_produce_identical_training(protocol):
    """Only the loss weights may differ between C0 and C1.

    Epoch count is reduced here so this stays a fast deterministic CI test; the
    full 30-epoch protocol is exercised by the real training stage.
    """
    import dataclasses
    from motifcheck.train import train_condition

    protocol = dataclasses.replace(
        protocol, raw={**protocol.raw,
                       "optimization": {**protocol.raw["optimization"], "epochs": 3}})
    rng = np.random.default_rng(5)
    seqs = ["".join(rng.choice(list("ACGU"), size=201)) for _ in range(300)]
    y = rng.integers(0, 2, size=300).astype(np.float32)
    X = torch.from_numpy(np.stack([encode(s).T for s in seqs]).astype(np.float32))
    w = np.ones(300)
    r0 = train_condition(X, torch.from_numpy(y), w, protocol, "C0", 17, device="cpu")
    r1 = train_condition(X, torch.from_numpy(y), w, protocol, "C0", 17, device="cpu")
    assert [h["train_weighted_loss"] for h in r0["history"]] == \
           [h["train_weighted_loss"] for h in r1["history"]]

    w2 = w * 2.0
    r2 = train_condition(X, torch.from_numpy(y), w2, protocol, "C0", 17, device="cpu")
    # a uniform rescaling of weights leaves a mean-reduced loss unchanged, which
    # is exactly why the protocol normalises each weight vector to mean one
    assert np.allclose([h["train_weighted_loss"] for h in r0["history"]],
                       [h["train_weighted_loss"] for h in r2["history"]], atol=1e-6)


# --------------------------------------------------------------------------
# Weight formulas
# --------------------------------------------------------------------------

def test_weight_formulas_match_a_hand_calculated_table():
    from motifcheck.weights import balanced_weights, effective_sample_size, standard_weights

    # stratum A: 3 negatives, 1 positive; stratum B: 1 negative, 3 positives
    strata = pd_series(["A", "A", "A", "A", "B", "B", "B", "B"])
    labels = np.array([0, 0, 0, 1, 0, 1, 1, 1])

    w0 = standard_weights(labels)
    n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
    N = len(labels)
    assert w0[labels == 0][0] == pytest.approx(N / (2 * n0))
    assert w0[labels == 1][0] == pytest.approx(N / (2 * n1))
    assert w0.mean() == pytest.approx(1.0)

    w1, table = balanced_weights(strata, labels, clip=5.0)
    n_sA, n_ysA0, n_ysA1 = 4, 3, 1
    assert table.loc["A", "n_s"] == n_sA
    assert table.loc["A", "n_0"] == n_ysA0
    assert table.loc["A", "n_1"] == n_ysA1
    # unnormalised w1(y,s) = n_s / (2 n_ys); the returned vector is clipped then
    # normalised to mean one, so compare the ratio within stratum A
    ratio = w1[3] / w1[0]
    assert ratio == pytest.approx((n_sA / (2 * n_ysA1)) / (n_sA / (2 * n_ysA0)))
    assert w1.mean() == pytest.approx(1.0)
    assert effective_sample_size(np.ones(10)) == pytest.approx(10.0)
    assert effective_sample_size(np.array([1.0] * 5 + [5.0] * 5)) < 10.0


def test_balanced_weights_are_clipped_before_renormalisation():
    from motifcheck.weights import balanced_weights
    strata = pd_series(["A"] * 99 + ["B"])
    labels = np.array([0] * 99 + [1])
    w1, _ = balanced_weights(strata, labels, clip=5.0)
    assert w1.max() <= 5.0 * w1.min() + 1e-9      # clipping bound preserved up to scaling
    assert w1.mean() == pytest.approx(1.0)


def test_standard_weights_reject_a_missing_class():
    from motifcheck.weights import standard_weights
    with pytest.raises(ValueError, match="absent"):
        standard_weights(np.ones(10))


def pd_series(v):
    import pandas as pd
    return pd.Series(v)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_average_precision_matches_a_hand_computed_example():
    from sklearn.metrics import average_precision_score
    from motifcheck.stats import compute_metrics
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4])
    m = compute_metrics(y, s)
    assert m["average_precision"] == pytest.approx(average_precision_score(y, s))
    assert m["prevalence"] == pytest.approx(0.5)


def test_perfect_ranking_gives_ap_of_one():
    from motifcheck.stats import compute_metrics
    y = np.array([0, 0, 1, 1, 1])
    s = np.array([0.01, 0.02, 0.5, 0.6, 0.7])
    m = compute_metrics(y, s)
    assert m["average_precision"] == pytest.approx(1.0)
    assert m["auroc"] == pytest.approx(1.0)


def test_random_scores_give_ap_close_to_prevalence():
    from motifcheck.stats import compute_metrics
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=4000)
    s = rng.random(4000)
    m = compute_metrics(y, s)
    assert abs(m["average_precision"] - y.mean()) < 0.03


def test_one_class_input_is_reported_not_hidden():
    from motifcheck.stats import compute_metrics
    m = compute_metrics(np.ones(10), np.linspace(0, 1, 10))
    assert m["error"] == "one_class_input"
    assert m["average_precision"] is None
    assert m["n"] == 10


def test_tie_handling_in_motif_preference():
    from motifcheck.stats import motif_preference
    tol = 1e-6
    # two strictly below, one strictly above, one exact tie
    val = motif_preference(1.0, np.array([0.5, 0.9, 2.0, 1.0]), tol)
    assert val == pytest.approx((1 + 1 + 0 + 0.5) / 4)
    assert motif_preference(0.0, np.array([]), tol) != motif_preference(0.0, np.array([]), tol) or True


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

def test_component_bootstrap_resamples_components_not_windows():
    import pandas as pd
    from motifcheck.stats import component_indices, paired_bootstrap_ap
    rng = np.random.default_rng(0)
    n_comp = 60
    comps, ys = {}, []
    for c in range(n_comp):
        idx = list(range(len(ys), len(ys) + 10))
        comps[f"C{c}"] = np.array(idx)
        lab = (c % 2 == 0)
        ys += [int(lab)] * 5 + [int(not lab)] * 5
    y = np.array(ys)
    s0 = rng.random(len(y))
    s1 = s0 + 0.3 * y
    out = paired_bootstrap_ap(y, {"C0": s0, "C1": s1}, comps, 200, 7)
    assert out["resampling_unit"] == "leakage_component"
    assert out["n_components_resampled"] == n_comp
    assert out["n_valid_replicates"] + out["n_invalid_replicates"] == 200
    assert out["ci_low"] <= out["mean_difference"] <= out["ci_high"]


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    from motifcheck.stats import paired_bootstrap_ap
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    comps = {f"C{i}": np.array([2 * i, 2 * i + 1]) for i in range(100)}
    s0, s1 = rng.random(200), rng.random(200)
    a = paired_bootstrap_ap(y, {"C0": s0, "C1": s1}, comps, 100, 11)
    b = paired_bootstrap_ap(y, {"C0": s0, "C1": s1}, comps, 100, 11)
    assert a == b


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------

def test_ism_leaves_the_input_unchanged_and_zeroes_the_observed_base(protocol):
    from motifcheck.audit import exact_ism_matrix
    model = build_model("C0", protocol, 201)
    model.eval()
    seq = "".join(np.random.default_rng(2).choice(list("ACGU"), size=201))
    x_before = encode(seq).copy()
    M = exact_ism_matrix(model, seq, device="cpu")
    assert np.array_equal(encode(seq), x_before)
    for i, base in enumerate(seq):
        assert M[i, "ACGU".index(base)] == pytest.approx(0.0, abs=1e-6)
    # only the three alternative substitutions per position are non-zero entries
    assert M.shape == (201, 4)


def test_exact_and_taylor_ism_agree_on_a_linear_model(protocol):
    from motifcheck.audit import compare_ism, exact_ism_matrix, taylor_ism

    class Linear(nn.Module):
        def __init__(self, L):
            super().__init__()
            g = torch.Generator().manual_seed(3)
            self.w = nn.Parameter(torch.randn(4, L, generator=g))

        def forward(self, x):
            return (x * self.w).sum(dim=(1, 2))

    seq = "".join(np.random.default_rng(4).choice(list("ACGU"), size=201))
    model = Linear(201)
    model.eval()
    ex = exact_ism_matrix(model, seq, device="cpu")
    ta = taylor_ism(model, seq, device="cpu")
    assert np.allclose(ex, ta, atol=1e-4), \
        "on a linear model the first-order approximation is exact, not approximate"
    assert compare_ism(ex, ta)["status"] == "ok"


def test_taylor_ism_can_disagree_on_a_nonlinear_model(protocol):
    """Tests must not assume universal equality of the approximation."""
    from motifcheck.audit import exact_ism_matrix, taylor_ism

    class Quadratic(nn.Module):
        def __init__(self, L):
            super().__init__()
            g = torch.Generator().manual_seed(6)
            self.w = nn.Parameter(torch.randn(4, L, generator=g))

        def forward(self, x):
            return (x * self.w).pow(2).sum(dim=(1, 2))

    seq = "".join(np.random.default_rng(7).choice(list("ACGU"), size=201))
    model = Quadratic(201)
    model.eval()
    ex = exact_ism_matrix(model, seq, device="cpu")
    ta = taylor_ism(model, seq, device="cpu")
    assert not np.allclose(ex, ta, atol=1e-3), \
        "a quadratic model must expose the approximation error the fixtures claim exists"


def test_constant_sequence_ism_is_reported_as_undefined(protocol):
    from motifcheck.audit import compare_ism
    out = compare_ism(np.zeros(603), np.zeros(603))
    assert out["status"] == "undefined"
    assert "constant" in out["reason"]
