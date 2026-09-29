"""Motif-order control construction and panel selection (protocol 10.1)."""

from __future__ import annotations

import numpy as np
import pytest

from motifcheck.audit import (build_order_controls, build_outside_controls,
                             decoy_motifs, _dinuc_vector)
from motifcheck.motif import Motif, best_hits, load_motif, scan_scores


def _test_motif() -> Motif:
    m = np.array([[0.05, 0.05, 0.05, 0.85],
                  [0.05, 0.05, 0.05, 0.85],
                  [0.05, 0.80, 0.10, 0.05],
                  [0.05, 0.80, 0.10, 0.05],
                  [0.05, 0.80, 0.10, 0.05],
                  [0.70, 0.10, 0.10, 0.10]])
    return Motif(motif_id="TEST", matrix=m, background=np.full(4, .25),
                 pseudocount=0.25, motif_class="test", assay_type="synthetic",
                 study="t", study_id="t", cisbp_tf_id="t", database_build="t")


# Score of the planted CUCUCU instance under this matrix is ~6.4 bits. A low
# threshold would make random background positions count as hits, and the
# "introduces no new occurrence" rule would then reject every control.
THRESHOLD = 5.0


def _sequence_with_motif(seed: int = 0, length: int = 201) -> tuple[str, int]:
    rng = np.random.default_rng(seed)
    motif = _test_motif()
    seq = list("".join(rng.choice(list("ACGU"), size=length)))
    pos = 60
    seq[pos:pos + motif.length] = list("CUCUCU")
    s = "".join(seq)
    return s, pos


def test_controls_preserve_mono_counts_and_length():
    motif = _test_motif()
    seq, pos = _sequence_with_motif(1)
    controls, info = build_order_controls(seq, motif, pos, pos + motif.length,
                                          threshold=THRESHOLD, n_controls=20,
                                          min_controls=5, key="t")
    assert info["informative"]
    assert len(controls) >= 5
    for c in controls:
        assert len(c.sequence) == len(seq)
        for b in "ACGU":
            assert c.sequence.count(b) == seq.count(b)


def test_controls_strictly_decrease_the_target_motif_score():
    motif = _test_motif()
    seq, pos = _sequence_with_motif(2)
    controls, _ = build_order_controls(seq, motif, pos, pos + motif.length,
                                       threshold=THRESHOLD, n_controls=20,
                                       min_controls=5, key="t")
    for c in controls:
        assert c.edited_target_score < c.original_target_score


def test_controls_introduce_no_new_high_scoring_occurrences():
    motif = _test_motif()
    seq, pos = _sequence_with_motif(3)
    controls, _ = build_order_controls(seq, motif, pos, pos + motif.length,
                                       threshold=THRESHOLD, n_controls=20,
                                       min_controls=5, key="t")
    for c in controls:
        assert c.new_hits == 0


def test_dinucleotide_change_is_documented_not_zero():
    """Order-disrupting edits must change dinucleotides; reporting zero would
    mean the control did not actually disrupt anything."""
    motif = _test_motif()
    seq, pos = _sequence_with_motif(4)
    controls, info = build_order_controls(seq, motif, pos, pos + motif.length,
                                          threshold=THRESHOLD, n_controls=20,
                                          min_controls=5, key="t")
    assert all(c.dinucleotide_l1_change > 0 for c in controls)
    assert info and np.mean([c.dinucleotide_l1_change for c in controls]) > 0


def test_low_complexity_segment_is_marked_uninformative_not_forced():
    """A motif instance made only of one repeated base cannot yield 5 valid
    order-disrupted controls. The fixture must report that, not relax rules."""
    motif = _test_motif()
    seq = list("C" * 201)
    pos = 50
    s = "".join(seq)
    controls, info = build_order_controls(s, motif, pos, pos + motif.length,
                                          threshold=-50.0, n_controls=20,
                                          min_controls=5, key="degenerate")
    # a homopolymer segment admits few distinct shuffles that also lower the score
    assert not info["informative"]
    assert "uninformative_reason" in info
    assert len(controls) < 5


def test_outside_controls_preserve_composition_and_stay_outside_the_motif():
    motif = _test_motif()
    seq, pos = _sequence_with_motif(5)
    controls, info = build_outside_controls(seq, motif, pos, pos + motif.length,
                                            threshold=THRESHOLD, n_controls=20,
                                            min_controls=5, key="t")
    assert info["informative"]
    for c in controls:
        assert len(c.sequence) == len(seq)
        for b in "ACGU":
            assert c.sequence.count(b) == seq.count(b)
        assert c.start >= pos + motif.length or c.end <= pos, \
            "an outside control must not overlap the motif instance"


def test_control_construction_is_deterministic():
    motif = _test_motif()
    seq, pos = _sequence_with_motif(6)
    a, _ = build_order_controls(seq, motif, pos, pos + motif.length,
                                THRESHOLD, 10, 5, "same-key")
    b, _ = build_order_controls(seq, motif, pos, pos + motif.length,
                                THRESHOLD, 10, 5, "same-key")
    assert [c.sequence for c in a] == [c.sequence for c in b]
    c2, _ = build_order_controls(seq, motif, pos, pos + motif.length,
                                 THRESHOLD, 10, 5, "other-key")
    assert [c.sequence for c in a] != [c.sequence for c in c2]


def test_decoy_motifs_are_composition_matched_and_not_the_target():
    motif = _test_motif()
    decoys = decoy_motifs(motif)
    assert len(decoys) >= 2
    for d in decoys:
        assert d.motif_class == "decoy"
        assert d.matrix.shape == motif.matrix.shape
        # each decoy keeps the target's multiset of per-position base masses
        assert np.isclose(d.matrix.sum(), motif.matrix.sum())
        assert d.motif_id != motif.motif_id
    assert any(d.matrix[0, :].argmax() != motif.matrix[0, :].argmax() or
               d.matrix[1, :].argmax() != motif.matrix[1, :].argmax() for d in decoys)


def test_dinuc_vector_counts_overlaps():
    v = _dinuc_vector("AAAA")
    assert v.sum() == 3          # AAA appears three times
    assert v[0] == 3
    assert _dinuc_vector("AC").sum() == 1


def test_null_threshold_is_monotone_in_the_target_fpr():
    motif = _test_motif()
    seqs = ["".join(np.random.default_rng(s).choice(list("ACGU"), size=201))
            for s in range(60)]
    from motifcheck.motif import null_threshold
    loose = null_threshold(motif, seqs, 0.05)
    tight = null_threshold(motif, seqs, 0.001)
    assert tight["threshold"] > loose["threshold"]
    assert loose["target_fpr_in_null"] == 0.05


def test_audit_panel_selection_ignores_model_confidence(protocol, split_data):
    """Panel selection must depend only on metadata and fixed hashes."""
    from motifcheck.panels import audit_panel
    if len(split_data) == 0:
        pytest.skip("dataset not built")
    motif = _test_motif()
    thr = 5.0
    a, ia = audit_panel(split_data, motif, thr, 60, 10)
    b, ib = audit_panel(split_data.sample(frac=1.0, random_state=3), motif, thr, 60, 10)
    assert set(a["window_id"]) == set(b["window_id"])
    assert ia["model_confidence_used"] is False


def test_ism_panel_strata_and_shortfalls_are_reported(protocol, split_data):
    from motifcheck.panels import ism_panel
    if len(split_data) == 0:
        pytest.skip("dataset not built")
    motif = _test_motif()
    panel, info = ism_panel(split_data, motif, THRESHOLD, 64)
    assert set(info["strata_counts"]) == {
        "label1_motif_present", "label1_motif_absent",
        "label0_motif_present", "label0_motif_absent"}
    assert info["realized_total"] <= 64
    assert info["same_panel_for_all_models_and_seeds"] is True
    # never backfill a short stratum from another one
    assert info["realized_total"] == sum(v["selected"] for v in info["strata_counts"].values())
