"""Dataset and leakage controls (protocol 10.1, second block).

The central test deliberately injects same-gene windows, overlaps, exact
duplicates and a detectable near-duplicate into different partitions and
requires the checker to fail. A leakage screen that cannot fail is not a screen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motifcheck.splits import (assign, assign_splits, audit_splits, build_components,
                               near_duplicate_stats)

W = 201


def _row(wid, chrom, start, strand, gene, seq, label, ctx="intronic", gc=0.5, tpm=5.0):
    n = len(seq) // 2
    return {
        "window_id": wid, "chrom": chrom, "start": start, "end": start + W,
        "strand": strand, "gene_id": gene, "gene_name": gene, "context": ctx,
        "sequence": seq, "label": label, "peak_start": start, "peak_end": start + W,
        "peak_strand": strand, "signal": 3.0, "gc": gc,
        "n_A": n, "n_C": n, "n_G": n, "n_U": n,
        "log1p_tpm": float(np.log1p(tpm)), "tpm": tpm, "gene_length": 5000,
        "input_signal": np.nan, "source": "synthetic",
        "assembly": "GRCh38", "dataset_version": "test",
    }


def _seq(seed: int, length: int = W) -> str:
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("ACGU"), size=length))


def _frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------

def test_injected_leakage_is_detected(protocol):
    """Same gene, overlap, exact duplicate and near-duplicate across partitions
    must all be caught."""
    a, b, c, d = _seq(1), _seq(2), _seq(3), _seq(4)
    near = a[:190] + _seq(99, 11)          # 190/201 identical, one substitution block

    df = _frame([
        _row("W1", "chr1", 1000, "+", "GENE_A", a, 1),
        _row("W2", "chr1", 5000, "+", "GENE_A", b, 0),      # same gene as W1
        _row("W3", "chr1", 1000, "-", "GENE_B", a, 0),      # exact duplicate of W1
        _row("W4", "chr2", 2000, "+", "GENE_C", c, 1),
        _row("W5", "chr2", 9000, "+", "GENE_D", near, 0),   # near-duplicate of W1
        _row("W6", "chr3", 3000, "+", "GENE_E", d, 1),
    ])
    comp, stats, nd = build_components(df, protocol)
    g = {c: set(df.loc[comp == c, "window_id"]) for c in comp.unique()}

    joined_a = comp[0] == comp[2]        # same gene
    joined_b = comp[0] == comp[3]        # identical sequence (and same locus)
    joined_c = comp[0] == comp[4]        # near duplicate
    assert joined_a, "same-gene windows were not joined"
    assert joined_b, "exact duplicate was not joined"
    assert joined_c, "near duplicate was not joined"
    assert any(i in (0, 3, 4) and j in (0, 3, 4) for i, j, _, _ in nd)

    # Now split them into different partitions and require the audit to fail.
    df["component_id"] = comp.values
    df["split"] = ["train" if w in {"W1", "W6"} else "validation" for w in df["window_id"]]
    audit = audit_splits(df, protocol)
    assert audit["zero_forbidden_overlap"] is False
    assert audit["status"] == "fail"
    assert any(v > 0 for v in audit["forbidden_overlap_checks"].values())


def test_split_assignment_is_independent_of_row_order(protocol):
    rows = [
        _row(f"W{i}", "chr1", 1000 + 500 * i, "+", f"G{i}", _seq(100 + i), i % 2)
        for i in range(40)
    ]
    df = _frame(rows)
    comp, _, _ = build_components(df, protocol)
    s1 = assign_splits(df, comp, protocol)

    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    comp2, _, _ = build_components(shuffled, protocol)
    s2 = assign_splits(shuffled, comp2, protocol)

    m1 = dict(zip(df["window_id"], s1))
    m2 = dict(zip(shuffled["window_id"], s2))
    assert m1 == m2, "split assignment changed under row reordering"


def test_component_id_does_not_depend_on_row_order(protocol):
    rows = [_row(f"W{i}", "chr1", 1000 + 500 * i, "+", f"G{i}", _seq(200 + i), i % 2)
            for i in range(30)]
    df = _frame(rows)
    c1, _, _ = build_components(df, protocol)
    df2 = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    c2, _, _ = build_components(df2, protocol)
    assert dict(zip(df["window_id"], c1)) == dict(zip(df2["window_id"], c2))


def test_near_duplicate_rule_behaviour_on_known_fixtures():
    a = _seq(7)
    exact = a
    one_sub = a[:100] + ("A" if a[100] != "A" else "C") + a[101:]
    unrelated = _seq(8)
    unrelated_revcomp = unrelated.translate(str.maketrans("ACGU", "UGCA"))[::-1]

    i_e, c_e = near_duplicate_stats(a, exact)
    assert i_e == pytest.approx(1.0) and c_e == pytest.approx(1.0)

    i_1, c_1 = near_duplicate_stats(a, one_sub)
    assert 0.98 <= i_1 <= 1.0 and 0.98 <= c_1 <= 1.0

    i_u, c_u = near_duplicate_stats(a, unrelated)
    assert i_u < 0.9 and c_u < 0.8
    i_r, _ = near_duplicate_stats(a, unrelated_revcomp)
    assert i_r < 0.9


def test_train_only_bin_edges_are_fitted_on_train_only(protocol):
    from motifcheck.splits import train_derived_bins
    rows = [
        _row(f"W{i}", "chr1", 1000 + 300 * i, "+", f"G{i}", _seq(300 + i), i % 2,
             gc=0.2 + 0.001 * i)
        for i in range(120)
    ]
    df = _frame(rows)
    comp, _, _ = build_components(df, protocol)
    df["component_id"] = comp.values
    df["split"] = assign_splits(df, comp, protocol).values

    bins = train_derived_bins(df, protocol)
    assert bins["fitted_on"] == "train"
    assert bins["n_train_rows"] == int((df["split"] == "train").sum())
    # A test-only extreme value must not move a single edge.
    before = list(bins["gc_quintile_edges"])
    df2 = df.copy()
    extreme = _row("WEXT", "chrZ", 100, "+", "GEXTREME", _seq(999), 1, gc=0.999)
    df2 = pd.concat([df2, pd.DataFrame([extreme])], ignore_index=True)
    df2["component_id"] = df2["component_id"].fillna("C_EXTREME")
    df2.loc[df2["window_id"] == "WEXT", "split"] = "test"
    bins2 = train_derived_bins(df2, protocol)
    assert bins2["gc_quintile_edges"] == before


def test_cnn_inputs_cannot_carry_metadata(protocol, split_data):
    """CNN input tensors must be sequence only. Proven by reconstruction."""
    import torch
    from motifcheck.train import encode_dataset
    from motifcheck.motif import encode

    sub = split_data.head(64)
    X = encode_dataset(sub["sequence"].tolist())
    assert X.shape == (64, 4, protocol.window_length)
    assert X.dtype == torch.float32
    for i in range(8):
        expected = encode(sub["sequence"].iloc[i])
        assert torch.allclose(X[i], torch.from_numpy(expected.T.copy()))
    # exactly one channel set per position
    assert torch.allclose(X.sum(dim=1), torch.ones(64, protocol.window_length))
    # input must not change when every metadata field is scrambled
    sub2 = sub.copy()
    for col in ("gene_id", "label", "gc", "chrom", "start", "log1p_tpm"):
        sub2[col] = "SCRAMBLED"
    X2 = encode_dataset(sub2["sequence"].tolist())
    assert torch.equal(X, X2)


def test_background_never_overlaps_an_excluded_peak_neighbourhood(split_data):
    """Background windows must sit outside every retained peak's buffer."""
    if len(split_data) == 0:
        pytest.skip("empty dataset")
    bg = split_data[split_data["label"] == 0]
    pos = split_data[split_data["label"] == 1]
    if len(bg) == 0 or len(pos) == 0:
        pytest.skip("dataset lacks both classes")
    # no background window may contain a positive window's locus
    pos_loci = set(zip(pos["chrom"], pos["start"], pos["end"]))
    assert not (set(zip(bg["chrom"], bg["start"], bg["end"])) & pos_loci)
    # and no background window may overlap any positive peak interval
    bad = 0
    for chrom, grp in bg.groupby("chrom"):
        p = pos[pos["chrom"] == chrom]
        if not len(p):
            continue
        starts = np.sort(grp["start"].to_numpy())
        p_starts = p["start"].to_numpy()
        p_ends = p["end"].to_numpy()
        j = np.searchsorted(p_starts, grp["end"].to_numpy(), side="left")
        for i, (_, row) in enumerate(grp.iterrows()):
            k = j[i]
            while k > 0 and p_ends[k - 1] > row["start"]:
                k -= 1
            for m in range(k, len(p_starts)):
                if p_starts[m] >= row["end"]:
                    break
                if p_ends[m] > row["start"]:
                    bad += 1
    assert bad == 0, f"{bad} background windows overlap a positive peak"


def test_dataset_rows_all_have_201_canonical_rna_bases(split_data):
    if len(split_data) == 0:
        pytest.skip("empty dataset")
    assert split_data["sequence"].str.len().nunique() == 1
    assert int(split_data["sequence"].str.len().iloc[0]) == 201
    assert set("".join(split_data["sequence"].head(500))) <= set("ACGU")


def test_input_signal_is_never_zero_filled(split_data):
    """Amendment A2: an unavailable control must stay missing, not become 0."""
    if len(split_data) == 0:
        pytest.skip("empty dataset")
    assert "input_signal" in split_data.columns
    assert split_data["input_signal"].isna().all()


def test_split_data_has_no_cross_split_gene_or_locus_overlap(split_data):
    if len(split_data) == 0:
        pytest.skip("empty dataset")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        ga = split_data[split_data["split"] == a]
        gb = split_data[split_data["split"] == b]
        if not len(ga) or not len(gb):
            continue
        assert not (set(ga["gene_id"]) & set(gb["gene_id"]))
        assert not (set(ga["component_id"]) & set(gb["component_id"]))
        assert not (set(ga["sequence"]) & set(gb["sequence"]))
