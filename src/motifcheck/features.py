"""Baseline models B0-B4 and their feature matrices.

B1 (composition) is the exact-invariance control for the motif audit: it is
mononucleotide-only, so a mononucleotide-count-preserving edit cannot change
its prediction by construction. B1's richer sibling with dinucleotides is
reported separately because dinucleotide content does change under the audit's
edits, and that response is expected rather than an implementation bug
(protocol section 8.2).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

ALPHABET = "ACGU"

MONO = list(ALPHABET)                                            # 4
DINO = ["".join(p) for p in itertools.product(ALPHABET, repeat=2)]  # 16
KMERS = {k: ["".join(p) for p in itertools.product(ALPHABET, repeat=k)]
         for k in (3, 4, 5)}


def mono_features(sequences: list[str]) -> tuple[np.ndarray, list[str]]:
    X = np.zeros((len(sequences), len(MONO)), dtype=np.float64)
    for i, s in enumerate(sequences):
        for j, b in enumerate(MONO):
            X[i, j] = s.count(b) / len(s)
    return X, [f"mono_{b}" for b in MONO]


def mono_dinucleotide_features(sequences: list[str]) -> tuple[np.ndarray, list[str]]:
    m, mnames = mono_features(sequences)
    d = np.zeros((len(sequences), len(DINO)), dtype=np.float64)
    for i, s in enumerate(sequences):
        for j, kmer in enumerate(DINO):
            d[i, j] = _count_overlap(s, kmer) / max(len(s) - 1, 1)
    return np.hstack([m, d]), mnames + [f"dino_{k}" for k in DINO]


def _count_overlap(s: str, kmer: str) -> int:
    return sum(s[i:i + len(kmer)] == kmer for i in range(len(s) - len(kmer) + 1))


def kmer_features(sequences: list[str]) -> tuple[np.ndarray, list[str]]:
    blocks, names = [], []
    for k in (3, 4, 5):
        ks = KMERS[k]
        block = np.zeros((len(sequences), len(ks)), dtype=np.float64)
        for i, s in enumerate(sequences):
            total = 0
            counts = np.zeros(len(ks))
            for pos in range(len(s) - k + 1):
                sub = s[pos:pos + k]
                idx = _kmer_index(sub, k)
                if idx is not None:
                    counts[idx] += 1
                    total += 1
            if total:
                block[i] = counts / total
        blocks.append(block)
        names += [f"k{k}_{x}" for x in ks]
    return np.hstack(blocks), names


_KMER_LUT = {k: {"".join(p): i for i, p in enumerate(itertools.product(ALPHABET, repeat=k))}
             for k in (3, 4, 5)}


def _kmer_index(sub: str, k: int) -> int | None:
    return _KMER_LUT[k].get(sub)


def context_features(df: pd.DataFrame, bins: dict) -> tuple[np.ndarray, list[str]]:
    """Context category indicators plus log1p TPM.

    The input-coverage feature required by the original plan is absent under
    amendment A2: ENCSR981WKN releases no processed size-matched input signal,
    and encoding a missing control as zero is explicitly forbidden.
    """
    cats = bins["context_categories"]
    cols = [f"context_{c}" for c in cats]
    C = np.zeros((len(df), len(cats)), dtype=np.float64)
    for j, c in enumerate(cats):
        C[:, j] = (df["context"].to_numpy() == c).astype(float)
    tpm = df["log1p_tpm"].to_numpy(dtype=np.float64).reshape(-1, 1)
    return np.hstack([C, tpm]), cols + ["log1p_tpm"]


def build_features(df: pd.DataFrame, which: str, bins: dict,
                   motif_table: pd.DataFrame | None = None
                   ) -> tuple[np.ndarray, list[str], dict]:
    """Return (X, feature_names, metadata). Scalers are fitted on train only."""
    seqs = df["sequence"].tolist()
    if which == "B0":
        return np.zeros((len(df), 0)), [], {"kind": "constant_prevalence"}
    if which == "B1":
        X, names = mono_features(seqs)
        return X, names, {"kind": "composition_mono_only",
                          "exact_invariant_to_mono_preserving_edits": True}
    if which == "B1d":
        X, names = mono_dinucleotide_features(seqs)
        return X, names, {"kind": "composition_mono_plus_dinucleotide",
                          "exact_invariant_to_mono_preserving_edits": False,
                          "note": "dinucleotides change under audit edits; response is expected"}
    if which == "B2":
        Xs, snames = mono_dinucleotide_features(seqs)
        Xc, cnames = context_features(df, bins)
        return np.hstack([Xs, Xc]), snames + cnames, {
            "kind": "composition_plus_context",
            "amendment": "A2: input-coverage feature removed (no released control signal)",
            "uses_extra_information": True}
    if which == "B3":
        X, names = kmer_features(seqs)
        return X, names, {"kind": "kmer_3_4_5_normalized"}
    if which == "B4":
        if motif_table is None:
            raise ValueError("B4 requires a motif score table")
        mt = motif_table.set_index("window_id")
        cols = ["motif_max_score", "motif_hit_count"]
        X = mt.loc[df["window_id"], cols].to_numpy(dtype=np.float64)
        return X, cols, {
            "kind": "motif_reference",
            "limitation": "B4 is not independent evidence about its own input; it is reported "
                          "as a known motif-sensitive reference for label prediction only"}
    raise ValueError(f"unknown baseline {which}")


def standardize_fit(X: np.ndarray) -> dict:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return {"mean": mu, "scale": sd, "n_fit_rows": int(len(X))}


def standardize_apply(X: np.ndarray, stats: dict) -> np.ndarray:
    return (X - stats["mean"]) / stats["scale"]
