"""Leakage control: connected components, immutable splits, and the split audit.

Windows are joined into components when they share a gene, overlap at a
genomic locus, have identical sequence, or meet a prespecified near-duplicate
criterion. Entire components are then assigned to splits by stable hashing of
the component's canonical identifier, so the assignment cannot depend on
filesystem or dataframe order.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Protocol
from .hashing import stable_hash_int, stable_unit_interval

KMER_K = 8
MINHASH_N = 64
MINHASH_BANDS = 16          # 16 bands x 4 rows
_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return int(a)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# --------------------------------------------------------------------------
# Near-duplicate detection
# --------------------------------------------------------------------------

def _minhash_signatures(sequences: list[str], k: int = KMER_K,
                        n_perm: int = MINHASH_N) -> np.ndarray:
    """(n_seq, n_perm) uint64 MinHash matrix over character k-mers."""
    import hashlib

    coeffs = np.arange(1, n_perm + 1, dtype=np.uint64)
    out = np.full((len(sequences), n_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, s in enumerate(sequences):
        if len(s) < k:
            continue
        kmers = [s[j:j + k] for j in range(len(s) - k + 1)]
        # deterministic 64-bit k-mer hash, independent of PYTHONHASHSEED
        base = np.array([
            int.from_bytes(hashlib.blake2b(x.encode(), digest_size=8).digest(), "big")
            for x in kmers
        ], dtype=np.uint64)
        h = np.bitwise_xor(
            base[:, None],
            (coeffs[None, :] * np.uint64(0x9E3779B97F4A7C15)) & _MASK,
        )
        out[i] = h.min(axis=0)
    return out


def candidate_pairs(sequences: list[str], bands: int = MINHASH_BANDS) -> set[tuple[int, int]]:
    """LSH candidate pairs from MinHash signatures."""
    sig = _minhash_signatures(sequences)
    n_perm = sig.shape[1]
    rows = n_perm // bands
    seen: set[tuple[int, int]] = set()
    for b in range(bands):
        sl = sig[:, b * rows:(b + 1) * rows]
        keys = {}
        for i in range(sig.shape[0]):
            keys.setdefault(sl[i].tobytes(), []).append(i)
        for members in keys.values():
            if len(members) < 2 or len(members) > 4000:
                continue
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    seen.add((members[x], members[y]))
    return seen


def near_duplicate_stats(a: str, b: str) -> tuple[float, float]:
    """Ungapped best match: (identity, coverage_of_shorter).

    identity  = matched positions / max(len_a, len_b)
    coverage  = matched positions / min(len_a, len_b)
    """
    from difflib import SequenceMatcher

    m = SequenceMatcher(None, a, b, autojunk=False)
    matched = sum(block.size for block in m.get_matching_blocks())
    lo, hi = min(len(a), len(b)), max(len(a), len(b))
    if hi == 0:
        return 0.0, 0.0
    return matched / hi, matched / lo


_COMP = str.maketrans("ACGU", "UGCA")


def find_near_duplicates(sequences: list[str], min_identity: float,
                         min_coverage: float, count_rc: bool = True) -> list[tuple[int, int, float, float]]:
    """Pairs passing the prespecified near-duplicate rule, plus both statistics."""
    pairs = candidate_pairs(sequences)
    revcomp = [s.translate(_COMP)[::-1] for s in sequences] if count_rc else None
    found: list[tuple[int, int, float, float]] = []
    checked: set[tuple[int, int]] = set()
    for i, j in sorted(pairs):
        if (i, j) in checked:
            continue
        ident, cov = near_duplicate_stats(sequences[i], sequences[j])
        if ident >= min_identity and cov >= min_coverage:
            checked.add((i, j))
            found.append((i, j, ident, cov))
        if revcomp is not None:
            ident_rc, cov_rc = near_duplicate_stats(sequences[i], revcomp[j])
            if ident_rc >= min_identity and cov_rc >= min_coverage:
                checked.add((i, j))
                found.append((i, j, ident_rc, cov_rc))
    return found


# --------------------------------------------------------------------------
# Component construction
# --------------------------------------------------------------------------

@dataclass
class SplitResult:
    dataset: pd.DataFrame
    audit: dict
    near_duplicates: list[tuple[int, int, float, float]]


def build_components(df: pd.DataFrame, protocol: Protocol, log=print) -> tuple[pd.Series, dict]:
    """Return a component_id per row plus the join statistics."""
    n = len(df)
    uf = UnionFind(n)
    stats: dict[str, int] = {}

    # (a) shared gene
    by_gene: dict[str, list[int]] = {}
    for i, g in enumerate(df["gene_id"].tolist()):
        by_gene.setdefault(g, []).append(i)
    joined = 0
    for members in by_gene.values():
        for k in range(1, len(members)):
            uf.union(members[0], members[k])
            joined += 1
    stats["gene_joins"] = joined
    stats["n_gene_groups"] = len(by_gene)

    # (b) overlapping genomic locus
    joined = 0
    for _, grp in df.groupby("chrom", sort=True):
        rows = grp.sort_values(["start", "end"]).index.tolist()
        prev_end = None
        for r in rows:
            s = int(df.at[r, "start"])
            e = int(df.at[r, "end"])
            if prev_end is not None and s < prev_end:
                uf.union(int(r), prev_idx)  # type: ignore[name-defined]
                joined += 1
            prev_end, prev_idx = e, int(r)
    stats["locus_joins"] = joined

    # (c) identical sequence
    joined = 0
    by_seq: dict[str, list[int]] = {}
    for i, s in enumerate(df["sequence"].tolist()):
        by_seq.setdefault(s, []).append(i)
    for members in by_seq.values():
        for k in range(1, len(members)):
            uf.union(members[0], members[k])
            joined += 1
    stats["identical_sequence_groups"] = sum(1 for m in by_seq.values() if len(m) > 1)
    stats["identical_sequence_joins"] = joined

    # (d) near duplicates
    nd_cfg = protocol["splits"]["near_duplicate"]
    log(f"[splits] scanning {n} sequences for near duplicates "
        f"(>= {nd_cfg['min_identity']} identity over >= {nd_cfg['min_coverage_of_shorter']} of shorter)")
    nd = find_near_duplicates(df["sequence"].tolist(),
                              float(nd_cfg["min_identity"]),
                              float(nd_cfg["min_coverage_of_shorter"]),
                              bool(nd_cfg["count_reverse_complement"]))
    for i, j, _, _ in nd:
        uf.union(i, j)
    stats["near_duplicate_pairs"] = len(nd)
    stats["near_duplicate_impl"] = nd_cfg["implementation"]
    stats["near_duplicate_settings"] = dict(nd_cfg)

    roots = np.array([uf.find(i) for i in range(n)])
    uniq = np.unique(roots)
    remap = {int(r): i for i, r in enumerate(uniq)}
    # component id is the canonical (smallest) window_id in the component, so it
    # does not depend on row order
    win_ids = df["window_id"].tolist()
    canon: dict[int, str] = {}
    for i, r in enumerate(roots):
        key = int(r)
        wid = win_ids[i]
        if key not in canon or wid < canon[key]:
            canon[key] = wid
    comp_ids = pd.Series([f"C_{canon[int(r)][2:18]}" for r in roots], index=df.index)
    stats["n_components"] = len(uniq)
    stats["largest_component_size"] = int(np.bincount(roots).max())
    return comp_ids, stats, nd


def assign_splits(df: pd.DataFrame, comp_ids: pd.Series, protocol: Protocol) -> pd.Series:
    seed = int(protocol["splits"]["hash_seed"])
    fr = protocol["splits"]["fractions"]
    order = ["train", "validation", "test"]
    out = []
    for cid in comp_ids:
        u = stable_unit_interval(cid, seed=seed)
        if u < fr["train"]:
            out.append("train")
        elif u < fr["train"] + fr["validation"]:
            out.append("validation")
        else:
            out.append("test")
    return pd.Series(out, index=df.index)


def audit_splits(df: pd.DataFrame, protocol: Protocol, log=print) -> dict:
    """Verify zero forbidden overlap and minimum support. Never self-certifies."""
    checks: dict[str, int] = {}
    by_split = {s: df[df["split"] == s] for s in ("train", "validation", "test")}
    comps = {s: set(g["component_id"]) for s, g in by_split.items()}

    names = ["train", "validation", "test"]
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            sa, sb = names[a], names[b]
            checks[f"cross_split_gene_overlap__{sa}__{sb}"] = len(
                set(by_split[sa]["gene_id"]) & set(by_split[sb]["gene_id"]))
            checks[f"cross_split_component_overlap__{sa}__{sb}"] = len(
                comps[sa] & comps[sb])
            checks[f"cross_split_exact_duplicate_overlap__{sa}__{sb}"] = len(
                set(by_split[sa]["sequence"]) & set(by_split[sb]["sequence"]))
            oa = _locus_keys(by_split[sa])
            ob = _locus_keys(by_split[sb])
            checks[f"cross_split_locus_overlap__{sa}__{sb}"] = len(oa & ob)

    split_cfg = protocol.section("splits")
    counts = {
        s: {
            "rows": int(len(g)),
            "positives": int(g["label"].sum()),
            "background": int((g["label"] == 0).sum()),
            "components": int(len(comps[s])),
            "genes": int(g["gene_id"].nunique()),
            "positives_in_unique_components": int(
                g[g["label"] == 1]["component_id"].nunique()),
        }
        for s, g in by_split.items()
    }
    support = {
        "train_positives_ok": counts["train"]["positives"] >= split_cfg["min_train_positives"],
        "train_components_ok": counts["train"]["components"] >= split_cfg["min_train_components"],
        "validation_positives_ok":
            counts["validation"]["positives"] >= split_cfg["min_eval_positives"],
        "validation_components_ok":
            counts["validation"]["components"] >= split_cfg["min_eval_components"],
        "test_positives_ok": counts["test"]["positives"] >= split_cfg["min_eval_positives"],
        "test_components_ok": counts["test"]["components"] >= split_cfg["min_eval_components"],
    }
    overlap_clean = all(v == 0 for v in checks.values())
    return {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "protocol_sha256": protocol.sha256,
        "split_counts": counts,
        "forbidden_overlap_checks": checks,
        "zero_forbidden_overlap": overlap_clean,
        "minimum_support_checks": support,
        "all_support_ok": all(support.values()),
        "status": "pass" if (overlap_clean and all(support.values())) else "fail",
        "exploratory": not (overlap_clean and all(support.values())),
    }


def _locus_keys(g: pd.DataFrame) -> set:
    return set(zip(g["chrom"], g["start"], g["end"]))


def assign(df: pd.DataFrame, protocol: Protocol, log=print) -> SplitResult:
    comp_ids, join_stats, nd = build_components(df, protocol, log=log)
    log(f"[splits] {join_stats['n_components']} components "
        f"(largest {join_stats['largest_component_size']} windows)")

    sizes = comp_ids.value_counts()
    giant_frac = float(sizes.max()) / len(df)
    giant_max = float(protocol["splits"]["giant_component_max_fraction"])
    out = df.assign(component_id=comp_ids.values)
    dropped_frac = 0.0
    if giant_frac > giant_max:
        giant = sizes.index[0]
        dropped_frac = float((out["component_id"] == giant).sum()) / len(df)
        log(f"[splits] giant component {giant} holds {giant_frac:.1%} of candidates; removing")
        out = out[out["component_id"] != giant].reset_index(drop=True)
    if dropped_frac > float(protocol["splits"]["giant_component_max_loss_fraction"]):
        raise ValueError(
            f"giant-component removal would drop {dropped_frac:.1%} of candidates, above the "
            "protocol limit; revise the candidate-universe or similarity protocol before training"
        )

    out["split"] = assign_splits(out, out["component_id"], protocol).values
    audit = audit_splits(out, protocol, log=log)
    audit.update({
        "join_statistics": join_stats,
        "giant_component_fraction": round(giant_frac, 5),
        "giant_component_max_allowed": giant_max,
        "giant_component_removed": giant_frac > giant_max,
        "candidate_loss_fraction_from_giant_removal": round(dropped_frac, 5),
        "near_duplicate_examples": [
            {"i": int(i), "j": int(j), "identity": round(ident, 4), "coverage": round(cov, 4)}
            for i, j, ident, cov in nd[:20]
        ],
    })
    return SplitResult(dataset=out, audit=audit, near_duplicates=nd)


def train_derived_bins(df: pd.DataFrame, protocol: Protocol) -> dict:
    """Bin edges and category sets fitted on training rows only."""
    tr = df[df["split"] == "train"]
    q = tr["gc"].quantile([0.2, 0.4, 0.6, 0.8]).tolist()
    t = tr["log1p_tpm"].quantile([1 / 3, 2 / 3]).tolist()
    return {
        "gc_quintile_edges": [round(float(x), 6) for x in q],
        "expression_tertile_edges": [round(float(x), 6) for x in t],
        "context_categories": sorted(tr["context"].unique().tolist()),
        "fitted_on": "train",
        "n_train_rows": int(len(tr)),
    }
