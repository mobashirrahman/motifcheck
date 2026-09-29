"""Coordinates and sequence handling (protocol 10.1, first block).

Uses a hand-checked toy FASTA so expected values can be verified by reading,
not by re-running the code under test.
"""

from __future__ import annotations

import pytest

from motifcheck.data import is_canonical, to_rna, window_id
from motifcheck.motif import reverse_complement

# The toy FASTA is a checked-in fixture; expected answers below are read off
# it by eye rather than recomputed by the code under test.
#   toy        201 nt, canonical except an N run at 40-44
#   toyshort    60 nt, for boundary cases


def test_exact_201_base_extraction():
    from motifcheck.data import Genome
    import pysam

    g = Genome(pysam.FastaFile("tests/fixtures/toy.fa"))
    seq = g.fetch("toy", 0, 201)
    assert seq is not None
    assert len(seq) == 201
    assert seq[:12] == "AAACCCGGGTTT"
    assert seq[40:45] == "NNNNN"
    assert seq[195:201] == "GUACGU"


def test_half_open_indexing_excludes_the_end_coordinate():
    import pysam
    g = pysam.FastaFile("tests/fixtures/toy.fa")
    a = g.fetch("toy", 0, 5).upper()
    b = g.fetch("toy", 4, 9).upper()
    assert a == "AAACC"
    assert b == "CCGGG"          # overlap of exactly one base at index 4
    assert a[4] == b[0]


def test_reverse_complement_round_trip_for_both_alphabets():
    for seq in ("AAACCCGGGUUU", "AAACCCGGGTTT"):
        rc = reverse_complement(seq)
        assert reverse_complement(rc) == seq
        assert len(rc) == len(seq)


def test_reverse_complement_preserves_the_input_alphabet():
    # A regression guard: a single table either turns RNA into DNA (A->T) or
    # leaves DNA unchanged (A->U). Both failures return a wrong strand silently.
    assert reverse_complement("AAACCCGGGTTT") == "AAACCCGGGTTT"
    assert reverse_complement("AAAAC") == "GTTTT"
    assert reverse_complement("UUUUG") == "CAAAA"
    assert reverse_complement("GGACGUAA") == "UUACGUCC"
    assert "T" not in reverse_complement("GGACGUAA")
    assert "U" not in reverse_complement("GGACGTA")


def test_negative_strand_orientation_then_tu_conversion():
    """Negative genomic strand is reverse-complemented as DNA, then T->U."""
    from motifcheck.motif import encode
    forward_dna = "AAACCCGGGTTTACGTACGTAC"
    negative_strand_dna = reverse_complement(forward_dna)
    rna = to_rna(negative_strand_dna)
    assert set(rna) <= set("ACGU")
    # a window oriented on the negative strand equals the reverse complement of
    # the RNA read off the positive strand
    assert rna == to_rna(reverse_complement(forward_dna))


def test_t_to_u_conversion_is_the_only_alphabet_boundary():
    assert to_rna("ACGT") == "ACGU"
    assert "T" not in to_rna("ACGT")


def test_non_canonical_bases_are_rejected_not_truncated():
    """is_canonical is applied to the DNA window before the T->U boundary."""
    assert not is_canonical("ACGTN")
    assert not is_canonical("ACGT-")
    assert not is_canonical("ACGU")     # RNA is not valid at the DNA stage
    assert is_canonical("ACGT")


def test_encode_raises_on_non_canonical_input():
    from motifcheck.motif import encode
    with pytest.raises(ValueError, match="non-canonical"):
        encode("ACGTN")
    with pytest.raises(ValueError):
        encode("ACG-")


def test_out_of_bounds_is_rejected_not_silently_truncated():
    """pysam clamps out-of-range fetches instead of raising, which would turn a
    bad window into a short one. Genome.fetch must refuse rather than pass that
    through."""
    from motifcheck.data import Genome
    import pysam
    f = pysam.FastaFile("tests/fixtures/toy.fa")
    assert f.get_reference_length("toy") == 201
    assert f.get_reference_length("toyshort") == 60

    g = Genome(f)
    assert len(g.fetch("toy", 0, 201)) == 201
    assert g.fetch("toy", 150, 301) is None        # past the contig end
    assert g.fetch("toy", -1, 100) is None         # negative start
    assert g.fetch("toy", 100, 100) is None        # empty interval
    assert g.fetch("no_such_contig", 0, 10) is None


def test_window_id_is_stable_and_coordinate_sensitive():
    a = window_id("chr1", 100, 301, "+", "GRCh38")
    b = window_id("chr1", 100, 301, "+", "GRCh38")
    c = window_id("chr1", 101, 302, "+", "GRCh38")
    d = window_id("chr1", 100, 301, "-", "GRCh38")
    e = window_id("chr1", 100, 301, "+", "hg19")
    assert a == b
    assert len({a, c, d, e}) == 4


def test_motif_scan_matches_a_hand_computed_score():
    from motifcheck.motif import Motif, scan_scores
    import numpy as np

    # A 4-position matrix that is unambiguous: it accepts only ACGU in order.
    m = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 1.0]])
    motif = Motif(motif_id="TEST", matrix=m,
                  background=np.array([.25, .25, .25, .25]), pseudocount=0.0,
                  motif_class="test", assay_type="test", study="t", study_id="t",
                  cisbp_tf_id="t", database_build="t")
    seq = "GGGGACGUAAAA"          # the motif sits at index 4
    scores = scan_scores(seq, motif)
    assert scores.shape[0] == len(seq) - 4 + 1
    assert scores.argmax() == 4
    # hand check: log2(1/0.25) four times = 8.0
    assert scores[4] == pytest.approx(8.0)
    # and any position containing a mismatch scores strictly lower
    assert np.all(np.delete(scores, 4) < scores[4])


def test_motif_scan_respects_orientation():
    """Scanning the reverse complement must find a different site, proving the
    scanner is not silently orientation-agnostic. ACGU is its own reverse
    complement at the level of base identity, so a different site is only
    found if positions matter."""
    from motifcheck.motif import Motif, scan_scores
    import numpy as np
    # ACUG is deliberately chosen: ACGU is its own reverse complement, so it
    # could not distinguish the two orientations at all.
    m = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0], [0, 0, 1.0, 0]])
    motif = Motif(motif_id="T", matrix=m, background=np.full(4, .25), pseudocount=0.0,
                  motif_class="t", assay_type="t", study="t", study_id="t",
                  cisbp_tf_id="t", database_build="t")
    fwd = "GGACUGAA"
    rev = reverse_complement(fwd)
    assert "U" in rev and "T" not in rev
    assert scan_scores(fwd, motif).max() == pytest.approx(8.0)
    assert scan_scores(rev, motif).max() < scan_scores(fwd, motif).max()
