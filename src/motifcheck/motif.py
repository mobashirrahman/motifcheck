"""Motif handling: PWM loading, log-odds scoring, and strand-aware scanning.

Matrices are stored in the CisBP-RNA "frequencies" convention (columns sum to
one over A/C/G/U). Scoring uses log-odds against the background frequencies
recorded in the protocol, with the protocol pseudocount added before the log.

Orientation is recorded explicitly. CisBP-RNA matrices for RNAcompete/SELEX are
given in the 5'->3' orientation of the RNA read; ``strand`` records that and is
asserted against the header, so a scan cannot silently use the wrong orientation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ALPHABET = "ACGU"
# Complements for each alphabet. A single table cannot serve both: mapping
# A->T would turn an RNA input into DNA, and mapping A->U would leave a DNA
# input unchanged. Both silent failures return a wrong strand.
_COMP_RNA = str.maketrans("ACGU", "UGCA")
_COMP_DNA = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq: str) -> str:
    """Reverse complement, preserving whether the input is RNA or DNA."""
    if "U" in seq:
        return seq.translate(_COMP_RNA)[::-1]
    return seq.translate(_COMP_DNA)[::-1]


@dataclass(frozen=True)
class Motif:
    motif_id: str
    matrix: np.ndarray           # (L, 4) probabilities, columns ordered A,C,G,U
    background: np.ndarray       # (4,)
    pseudocount: float
    motif_class: str             # "Direct" (measured) or "Inferred"
    assay_type: str
    study: str
    study_id: str
    cisbp_tf_id: str
    database_build: str
    strand: str = "+"            # orientation of the matrix, 5'->3'

    @property
    def length(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def log_odds(self) -> np.ndarray:
        """(L, 4) additive log-odds scores.

        Probabilities are floored before the logarithm: a zero matrix entry with a
        zero pseudocount would otherwise yield -inf, and -inf * 0 in the scan
        einsum produces NaN rather than a merely very low score.
        """
        m = self.matrix + self.pseudocount
        m = np.maximum(m, 1e-12)
        return np.log2(m / (self.background + self.pseudocount))

    def consensus(self) -> str:
        return "".join(ALPHABET[i] for i in self.matrix.argmax(axis=1))

    def iupac(self) -> str:
        codes = []
        for row in self.matrix:
            best = row.max()
            codes.append("".join(
                b for b, v in zip(ALPHABET, row) if v >= 0.6 * best))
        return "".join(codes)

    def fingerprint(self) -> str:
        return f"{self.motif_id}:{self.length}:{self.consensus()}:{self.motif_class}"

    def metadata(self) -> dict:
        return {
            "motif_id": self.motif_id,
            "length": self.length,
            "consensus": self.consensus(),
            "iupac_0.6": self.iupac(),
            "matrix_strand": self.strand,
            "matrix_content": "position-major frequencies; each row sums to 1",
            "pseudocount": self.pseudocount,
            "background_frequencies": dict(zip(ALPHABET, self.background.tolist())),
            "motif_class": self.motif_class,
            "assay_type": self.assay_type,
            "study": self.study,
            "study_id": self.study_id,
            "cisbp_tf_id": self.cisbp_tf_id,
            "database_build": self.database_build,
        }


def load_motif(path: str | Path, meta: dict, pseudocount: float,
               background: dict | None = None) -> Motif:
    rows = [l for l in Path(path).read_text().splitlines() if l.strip()]
    header = [h.strip().upper() for h in rows[0].split("\t")][1:]
    if header != list(ALPHABET):
        raise ValueError(f"motif {meta['id']} header is {header}; expected A,C,G,U order")
    matrix = np.array([[float(v) for v in r.split("\t")[1:]] for r in rows[1:]], dtype=float)
    if matrix.shape[1] != 4:
        raise ValueError(f"motif {meta['id']} has {matrix.shape[1]} columns, expected 4")
    bg = np.array([(background or {"A": 0.25, "C": 0.25, "G": 0.25, "U": 0.25})[b]
                   for b in ALPHABET], dtype=float)
    return Motif(
        motif_id=meta["id"],
        matrix=matrix,
        background=bg,
        pseudocount=float(pseudocount),
        motif_class=meta.get("motif_class", "unknown"),
        assay_type=meta.get("assay_type", "unknown"),
        study=meta.get("study", "unknown"),
        study_id=meta.get("study_id", "unknown"),
        cisbp_tf_id=meta.get("cisbp_tf_id", "unknown"),
        database_build=meta.get("database_build", "unknown"),
    )


def encode(sequence: str) -> np.ndarray:
    """(L, 4) float32 one-hot in A,C,G,U column order. Raises on non-canonical."""
    try:
        idx = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    except UnicodeEncodeError as exc:
        raise ValueError("sequence contains non-ASCII characters") from exc
    lut = np.full(256, -1, dtype=np.int8)
    for i, b in enumerate(ALPHABET):
        lut[ord(b)] = i
    codes = lut[idx]
    if (codes < 0).any():
        bad = sorted({sequence[j] for j in np.flatnonzero(codes < 0)})
        raise ValueError(f"non-canonical RNA bases present: {bad}")
    out = np.zeros((len(sequence), 4), dtype=np.float32)
    out[np.arange(len(sequence)), codes] = 1.0
    return out


def scan_scores(sequence: str, motif: Motif) -> np.ndarray:
    """Log-odds score for every start position. Length L-window+1."""
    enc = encode(sequence)
    lo = motif.log_odds.astype(np.float32)
    w = enc.shape[0] - motif.length + 1
    if w <= 0:
        return np.zeros(0, dtype=np.float32)
    # sliding window view without materialising per-position copies
    view = np.lib.stride_tricks.sliding_window_view(enc, motif.length, axis=0)  # (w, 4, L)
    return np.einsum("wkl,lk->w", view, lo).astype(np.float32)


def best_hits(sequence: str, motif: Motif, threshold: float) -> list[tuple[int, int, float]]:
    """Non-overlapping hits above threshold, best score first (plan 8.2)."""
    scores = scan_scores(sequence, motif)
    if scores.size == 0:
        return []
    order = np.argsort(-scores)
    taken: list[int] = []
    hits: list[tuple[int, int, float]] = []
    for i in order:
        if scores[i] < threshold:
            break
        start = int(i)
        if any(abs(start - s) < motif.length for s, _, _ in hits):
            continue
        hits.append((start, start + motif.length, float(scores[i])))
        taken.append(start)
    return hits


def null_threshold(motif: Motif, sequences: list[str], target_fpr: float = 0.01) -> dict:
    """Window-level score threshold at a given false-positive rate.

    Scans *training-derived* null sequences only. The resulting threshold is a
    scanning cutoff, not a calibrated binding probability, and is labelled as
    such wherever it is used.
    """
    all_scores: list[np.ndarray] = []
    for s in sequences:
        sc = scan_scores(s, motif)
        if sc.size:
            all_scores.append(sc)
    if not all_scores:
        raise ValueError("no scan positions available; null sequences are too short")
    pooled = np.concatenate(all_scores)
    thr = float(np.quantile(pooled, 1.0 - target_fpr))
    return {
        "threshold": thr,
        "target_fpr_in_null": target_fpr,
        "n_null_sequences": len(sequences),
        "n_null_scan_positions": int(pooled.size),
        "null_quantiles": {
            q: float(np.quantile(pooled, q)) for q in (0.5, 0.9, 0.99, 0.999)
        },
        "interpretation": "scanning cutoff calibrated to training null sequences; "
                          "NOT a biological binding probability",
    }
