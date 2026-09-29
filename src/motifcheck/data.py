"""Dataset construction: candidate universe, window extraction, labels.

Implements protocol section 4. The prediction unit is a 201-nt,
strand-oriented genomic window. Positives are windows centred on a retained
IDR-reproducible PTBP1 peak midpoint; background windows are drawn from the
same expressed-gene universe with a conservative exclusion buffer around every
available peak.

Nothing in this module looks at motif scores or model predictions.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

from . import DATASET_VERSION
from .config import Protocol
from .fileio import open_text
from .hashing import hash_file, stable_hash_int, write_json

CANONICAL = set("ACGT")


# --------------------------------------------------------------------------
# Source parsing
# --------------------------------------------------------------------------

def load_peaks(path: str | Path) -> pd.DataFrame:
    """narrowPeak -> frame. Strand is used for window orientation only."""
    cols = ["chrom", "start", "end", "name", "score", "strand",
            "signal", "p", "q", "summit_offset"]
    with open_text(path) as fh:
        df = pd.read_csv(
            fh, sep="\t", header=None, comment="track", names=cols,
            engine="python", on_bad_lines="skip",
        )
    df = df.dropna(subset=["chrom", "start", "end"])
    df["chrom"] = df["chrom"].astype(str)
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["strand"] = df["strand"].astype(str)
    df = df[df["chrom"].str.startswith("chr")].reset_index(drop=True)
    return df


def load_genes(
    gtf_path: str | Path,
    gene_types: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, str]]:
    """GENCODE gene records plus per-gene sorted exon start coordinates.

    Returns the gene table and an exon lookup restricted to each gene's own
    span, so intronic/exonic classification is exact rather than approximate.
    """
    wanted = ("gene", "exon")
    genes: dict[str, dict] = {}
    exon_starts: dict[str, list[tuple[int, int]]] = {}
    tx2gene: dict[str, str] = {}

    with open(gtf_path) as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] not in wanted:
                continue
            attrs = dict(
                (m.group(1), m.group(2)) for m in _ATTR_RE.finditer(f[8])
            )
            gid = attrs.get("gene_id")
            if not gid:
                continue
            if f[2] == "gene":
                genes[gid] = {
                    "gene_id": gid,
                    "chrom": f[0],
                    "start": int(f[3]),
                    "end": int(f[4]),
                    "strand": f[6],
                    "gene_type": attrs.get("gene_type", attrs.get("transcript_type", "")),
                    "gene_name": attrs.get("gene_name", ""),
                }
            else:
                tx = attrs.get("transcript_id")
                if tx:
                    tx2gene[tx.split(".")[0]] = gid
                exon_starts.setdefault(gid, []).append((int(f[3]), int(f[4])))

    table = pd.DataFrame(list(genes.values()))
    if gene_types:
        table = table[table["gene_type"].isin(gene_types)].reset_index(drop=True)

    exons: dict[str, np.ndarray] = {}
    for gid, iv in exon_starts.items():
        if gid not in genes:
            continue
        arr = np.array(sorted(iv), dtype=np.int64)
        exons[gid] = arr
    return table, exons, tx2gene


import re as _re
_ATTR_RE = _re.compile(r'(\w+)\s+"([^"]*)"')


def load_expression(tsv_path: str | Path, tx2gene: dict[str, str]) -> pd.DataFrame:
    """ENCODE RSEM quantification -> gene-level TPM by summing transcript TPM.

    Summing transcript TPM is an approximation; it is monotone in expression
    and is the only gene-level quantity the released file supports, so the
    aggregation is recorded rather than presented as an exact gene TPM.
    """
    df = pd.read_csv(tsv_path, sep="\t")
    df["transcript_id"] = df["transcript_id"].astype(str).str.split(".").str[0]
    df["gene_id"] = df["transcript_id"].map(tx2gene)
    n_unmapped = int(df["gene_id"].isna().sum())
    df = df.dropna(subset=["gene_id"])
    grouped = (df.groupby("gene_id", as_index=False)
                 .agg(tpm=("TPM", "sum"), n_transcripts=("TPM", "size")))
    return grouped, n_unmapped


# --------------------------------------------------------------------------
# Interval arithmetic (per chromosome, sorted, merged)
# --------------------------------------------------------------------------

def merge_intervals(intervals: np.ndarray) -> np.ndarray:
    """(N,2) int array -> sorted, non-overlapping merged intervals."""
    if len(intervals) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    iv = intervals[np.argsort(intervals[:, 0], kind="stable")]
    out: list[list[int]] = []
    for row in iv:
        if out and row[0] <= out[-1][1]:
            out[-1][1] = max(out[-1][1], int(row[1]))
        else:
            out.append([int(row[0]), int(row[1])])
    return np.array(out, dtype=np.int64) if out else np.zeros((0, 2), dtype=np.int64)


class ChromIntervals:
    """Fast 'does [s,e) hit any interval?' over one chromosome."""

    def __init__(self, intervals: np.ndarray):
        self.iv = merge_intervals(intervals)
        self.starts = self.iv[:, 0] if len(self.iv) else np.zeros(0, dtype=np.int64)
        self.ends = self.iv[:, 1] if len(self.iv) else np.zeros(0, dtype=np.int64)

    def hits(self, starts: np.ndarray, length: int) -> np.ndarray:
        """True where [s, s+length) intersects any interval.

        Merged intervals are disjoint, so both starts and ends increase
        monotonically. The last interval starting before the query end is
        therefore the only one that can still reach the query start: if its end
        is at or before the start, every earlier interval ends even earlier.
        That makes this an O(log n) lookup rather than a scan.
        """
        if len(self.starts) == 0 or len(starts) == 0:
            return np.zeros(len(starts), dtype=bool)
        idx = np.searchsorted(self.starts, starts + length, side="left") - 1
        valid = idx >= 0
        safe = np.clip(idx, 0, len(self.ends) - 1)
        return valid & (self.ends[safe] > starts)

    def subtract(self, span: tuple[int, int]) -> np.ndarray:
        """Segments of `span` not covered by any interval."""
        s, e = span
        if len(self.starts) == 0:
            return np.array([[s, e]], dtype=np.int64) if e > s else np.zeros((0, 2), np.int64)
        lo = max(0, int(np.searchsorted(self.ends, s, side="right")))
        hi = int(np.searchsorted(self.starts, e, side="left"))
        blockers = self.iv[lo:hi]
        out: list[list[int]] = []
        cur = s
        for b in blockers:
            if int(b[0]) > cur:
                out.append([cur, min(int(b[0]), e)])
            cur = max(cur, int(b[1]))
            if cur >= e:
                break
        if cur < e:
            out.append([cur, e])
        return np.array([o for o in out if o[1] > o[0]], dtype=np.int64).reshape(-1, 2)


# --------------------------------------------------------------------------
# Window extraction
# --------------------------------------------------------------------------

@dataclass
class Genome:
    fasta: pysam.FastaFile

    def fetch(self, chrom: str, start: int, end: int) -> str | None:
        """Fetch [start, end). Returns None rather than a truncated string.

        pysam silently clamps out-of-range requests, which would turn an
        out-of-bounds window into a short window and corrupt every downstream
        length-dependent calculation. The bounds are checked here instead.
        """
        if chrom not in self.fasta.references:
            return None
        if start < 0 or end > self.fasta.get_reference_length(chrom):
            return None
        if end <= start:
            return None
        try:
            return self.fasta.fetch(chrom, start, end).upper()
        except (ValueError, KeyError):
            return None

    def length(self, chrom: str) -> int:
        return self.fasta.get_reference_length(chrom)


def to_rna(seq: str) -> str:
    """Single T-to-U conversion boundary (plan section 4.2)."""
    return seq.replace("T", "U")


def is_canonical(seq: str) -> bool:
    return all(b in CANONICAL for b in seq)


def window_id(chrom: str, start: int, end: int, strand: str, assembly: str,
              version: str = DATASET_VERSION) -> str:
    return f"W_{stable_hash_int(chrom, start, end, strand, assembly, version, nbits=128):032x}"


# --------------------------------------------------------------------------
# Candidate universe
# --------------------------------------------------------------------------

@dataclass
class BuildResult:
    dataset: pd.DataFrame
    exclusions: pd.DataFrame
    report: dict


def build_dataset(protocol: Protocol, manifest: dict, log=print) -> BuildResult:
    ext = Path("data/external")
    W = protocol.window_length
    half = W // 2
    assembly = protocol["task"]["assembly"]
    cand = protocol.section("candidates")

    def _path(role: str) -> str:
        rec = manifest["sources"][role]
        p = rec.get("decompressed_path") or str(ext / f"{role}__{rec['id']}")
        return p

    log("[data] loading peaks")
    idr = load_peaks(_path("peaks"))
    rep_frames = []
    for role in ("peaks_replicate_1", "peaks_replicate_2"):
        if role in manifest["sources"]:
            rep_frames.append(load_peaks(_path(role)))
    log(f"[data] IDR reproducible peaks: {len(idr)}")

    log("[data] loading GENCODE genes and exons")
    genes, exons, tx2gene = load_genes(_path("annotation"))
    log(f"[data] genes: {len(genes)}")

    log("[data] loading K562 expression")
    expr, n_unmapped_tx = load_expression(_path("expression"), tx2gene)
    expr_map = dict(zip(expr["gene_id"], expr["tpm"]))
    log(f"[data] genes with quantified expression: {len(expr_map)} "
        f"({n_unmapped_tx} transcript rows unmapped)")

    min_tpm = float(cand["expression_min_tpm"])
    expressed = genes[genes["gene_id"].isin(expr_map)].copy()
    expressed["tpm"] = expressed["gene_id"].map(expr_map).astype(float)
    expressed = expressed[expressed["tpm"] >= min_tpm].reset_index(drop=True)
    log(f"[data] expressed genes (TPM >= {min_tpm}): {len(expressed)}")

    log("[data] opening genome")
    genome = Genome(pysam.FastaFile(_path("genome")))

    buffer_nt = int(cand["background_exclusion_buffer"])
    exclusions: list[dict] = []

    def _excl(kind: str, chrom, start, end, **kw):
        exclusions.append({"exclusion_reason": kind, "chrom": chrom,
                           "start": start, "end": end, **kw})

    # ---- exclusion zones: every available peak expanded by the buffer -----
    # The buffer expands the interval: [start - b, end + b). Subtracting the
    # buffer from both endpoints would only extend leftwards and would let
    # background windows sit immediately downstream of a peak.
    zones: dict[str, np.ndarray] = {}
    for frame, tag in [(idr, "idr")] + [(f, f"rep{i+1}") for i, f in enumerate(rep_frames)]:
        for chrom, grp in frame.groupby("chrom"):
            arr = grp[["start", "end"]].to_numpy(dtype=np.int64).copy()
            arr[:, 0] -= buffer_nt
            arr[:, 1] += buffer_nt
            arr[:, 0] = np.maximum(arr[:, 0], 0)
            prev = zones.get(chrom)
            zones[chrom] = arr if prev is None else np.vstack([prev, arr])
    blockers = {c: ChromIntervals(v) for c, v in zones.items()}

    # ---- gene index for ambiguity detection -------------------------------
    gene_by_chrom: dict[str, pd.DataFrame] = {
        c: g.reset_index(drop=True) for c, g in expressed.groupby("chrom")
    }

    def classify_context(gene_id: str, wstart: int, wend: int) -> str:
        ex = exons.get(gene_id)
        if ex is not None and len(ex):
            j = int(np.searchsorted(ex[:, 1], wstart, side="right"))
            if j < len(ex) and ex[j, 0] < wend:
                return "exonic"
        return "intronic"

    # ---- positives -------------------------------------------------------
    log("[data] extracting positive windows")
    idr = idr.copy()
    idr["midpoint"] = (idr["start"] + idr["end"]) // 2
    idr["wstart"] = idr["midpoint"] - half
    idr["wend"] = idr["midpoint"] + half + 1

    keep: list[dict] = []
    for row in idr.itertuples(index=False):
        chrom, ws, we = row.chrom, row.wstart, row.wend
        if chrom not in blockers and chrom not in genome.fasta.references:
            _excl("chromosome_not_in_genome", chrom, ws, we)
            continue
        if ws < 0 or we > genome.length(chrom):
            _excl("window_out_of_bounds", chrom, ws, we, peak_start=row.start, peak_end=row.end)
            continue
        seq = genome.fetch(chrom, ws, we)
        if seq is None or not is_canonical(seq):
            _excl("non_canonical_bases", chrom, ws, we,
                  peak_start=row.start, peak_end=row.end)
            continue
        keep.append({
            "chrom": chrom, "wstart": ws, "wend": we, "strand": row.strand,
            "peak_start": row.start, "peak_end": row.end, "peak_strand": row.strand,
            "peak_width": row.end - row.start, "signal": row.signal,
            "label": 1, "source": "idr_reproducible",
        })
    pos = pd.DataFrame(keep, columns=["chrom", "wstart", "wend", "strand",
                                      "peak_start", "peak_end", "peak_strand",
                                      "peak_width", "signal", "label", "source"])
    log(f"[data] positives after sequence/bounds filters: {len(pos)}")
    if len(pos) == 0:
        raise ValueError(
            "no positive windows survived bounds/sequence filtering; refusing to "
            "build a dataset with an empty positive class (check the genome index "
            "and assembly before continuing)"
        )

    # collapse overlapping positives to one deterministic representative per locus
    if cand.get("collapse_overlapping_positives", True) and len(pos):
        pos = pos.sort_values(["chrom", "wstart", "wend"]).reset_index(drop=True)
        keep_rows: list[int] = []
        prev_end = -1
        prev_chrom = None
        for i, row in enumerate(pos.itertuples(index=False)):
            if prev_chrom == row.chrom and row.wstart < prev_end:
                _excl("collapsed_into_existing_positive", row.chrom, row.wstart, row.wend)
                continue
            prev_chrom, prev_end = row.chrom, row.wend
            keep_rows.append(i)
        log(f"[data] collapsed overlapping positives: "
            f"{len(pos) - len(keep_rows)} removed, {len(keep_rows)} loci")
        pos = pos.loc[keep_rows].reset_index(drop=True)

    # ---- background ------------------------------------------------------
    log("[data] sampling background windows from expressed gene spans")
    target_n_bg = int(len(pos) * int(cand["background_ratio"]))
    # Each entry stores the segment start and the NUMBER OF VALID START POSITIONS,
    # not the segment length. Weighting by segment length would let a sampled
    # start land within one window length of the segment end, pushing the window
    # out of the usable segment and back into an excluded peak neighbourhood.
    usable: list[tuple[str, int, int]] = []
    for row in expressed.itertuples(index=False):
        chrom = row.chrom
        if chrom not in genome.fasta.references:
            continue
        glen = genome.length(chrom)
        span = (max(0, row.start), min(glen, row.end))
        usable_seg = (blockers[chrom].subtract(span) if chrom in blockers
                      else np.array([[span[0], span[1]]], dtype=np.int64))
        for s, e in usable_seg:
            s, e = max(int(s), half), min(int(e), glen - half - 1)
            n_starts = (e - s) - W + 1
            if n_starts >= 1:
                usable.append((chrom, s, n_starts))
    lengths = np.array([n for _, _, n in usable], dtype=np.int64)
    cum = np.cumsum(lengths)
    total = int(cum[-1]) if len(cum) else 0
    log(f"[data] usable expressed-gene sequence: {total/1e6:.1f} Mb of valid window "
        f"starts over {len(usable)} segments")

    rng = np.random.default_rng(int(protocol["splits"]["hash_seed"]))
    draws = rng.integers(0, total, size=int(target_n_bg * 3.0) + 1000)
    bg_rows: list[dict] = []
    seen: set[int] = set()
    pos_locus = set(zip(pos["chrom"], pos["wstart"]))
    for d in draws:
        if len(bg_rows) >= target_n_bg:
            break
        j = int(np.searchsorted(cum, d))
        chrom, s0, n0 = usable[j]
        offset = int(d) - (int(cum[j - 1]) if j > 0 else 0)
        start = s0 + offset
        wstart, wend = start, start + W
        key = stable_hash_int(chrom, wstart, nbits=64)
        if key in seen:
            continue
        if (chrom, wstart) in pos_locus:
            continue
        seq = genome.fetch(chrom, wstart, wend)
        if seq is None or not is_canonical(seq):
            _excl("background_non_canonical_bases", chrom, wstart, wend)
            continue
        seen.add(key)
        bg_rows.append({"chrom": chrom, "wstart": wstart, "wend": wend,
                        "strand": ".", "label": 0, "source": "eligible_background",
                        "peak_start": -1, "peak_end": -1, "peak_strand": ".",
                        "peak_width": 0, "signal": np.nan})
    bg = pd.DataFrame(bg_rows)
    log(f"[data] background windows: {len(bg)}")

    # ---- unify, orient, annotate -----------------------------------------
    df = pd.concat([pos, bg], ignore_index=True).reset_index(drop=True)
    df["gc"] = np.nan
    df["gene_id"] = ""
    df["context"] = "ambiguous"
    df["tpm"] = np.nan

    log("[data] extracting sequences and assigning gene context")
    gene_table = genes.set_index("gene_id")
    gene_span_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _spans(chrom: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if chrom not in gene_span_cache:
            g = gene_by_chrom.get(chrom)
            if g is None or len(g) == 0:
                gene_span_cache[chrom] = (np.zeros(0, np.int64), np.zeros(0, np.int64),
                                           np.zeros(0, dtype=object))
            else:
                g = g.sort_values("end").reset_index(drop=True)
                gene_span_cache[chrom] = (g["start"].to_numpy(), g["end"].to_numpy(),
                                           g["gene_id"].to_numpy())
        return gene_span_cache[chrom]

    seqs: list[str] = []
    gene_ids: list[str] = []
    contexts: list[str] = []
    tpms: list[float] = []
    n_ambiguous = 0
    for row in df.itertuples(index=False):
        raw = genome.fetch(row.chrom, row.wstart, row.wend)
        oriented = (raw if row.strand != "-"
                    else raw.translate(str.maketrans("ACGT", "TGCA"))[::-1])
        seqs.append(to_rna(oriented))

        gs, ge, gids = _spans(row.chrom)
        j = int(np.searchsorted(ge, row.wstart, side="right")) if len(ge) else -1
        n_over = 0
        gid = None
        if j >= 0:
            while j + n_over < len(gs) and gs[j + n_over] < row.wend:
                if ge[j + n_over] > row.wstart:
                    n_over += 1
                if n_over >= 2:
                    break
            if n_over == 1:
                gid = str(gids[j])
        if gid is None:
            gene_ids.append("")
            contexts.append("ambiguous")
            tpms.append(0.0)
            n_ambiguous += 1
            continue
        gene_ids.append(gid)
        contexts.append(classify_context(gid, row.wstart, row.wend))
        tpms.append(float(expr_map.get(gid, 0.0)))

    df["sequence"] = seqs
    df["gene_id"] = gene_ids
    df["context"] = contexts
    df["tpm"] = tpms
    df["gc"] = df["sequence"].map(lambda s: (s.count("G") + s.count("C")) / len(s))
    for b in "ACGU":
        df[f"n_{b}"] = df["sequence"].map(lambda s, b=b: s.count(b))
    df["log1p_tpm"] = np.log1p(df["tpm"].fillna(0.0))
    df["gene_length"] = df["gene_id"].map(gene_table["end"] - gene_table["start"]) \
        if "end" in gene_table else np.nan
    df["gene_name"] = df["gene_id"].map(gene_table["gene_name"])
    df["input_signal"] = np.nan   # amendment A2: no released control signal; never zero-filled

    before = len(df)
    amb = df["context"].eq("ambiguous")
    for row in df[amb].itertuples(index=False):
        _excl("ambiguous_gene_or_context", row.chrom, row.wstart, row.wend)
    df = df.loc[~amb].reset_index(drop=True)
    log(f"[data] excluded {before - len(df)} ambiguous-gene/context windows "
        f"({n_ambiguous} windows overlapped zero or >1 expressed gene)")
    log(f"[data] dataset: {len(df)} rows ({int((df['label']==1).sum())} positive, "
        f"{int((df['label']==0).sum())} background)")

    df["window_id"] = [
        window_id(r.chrom, r.wstart, r.wend, r.strand, assembly)
        for r in df.itertuples(index=False)
    ]
    if df["window_id"].duplicated().any():
        raise ValueError("window_id collision: the same locus received two identities")
    df["dataset_version"] = DATASET_VERSION
    df["assembly"] = assembly
    df = df.drop(columns=[c for c in ("peak_width",) if c not in df.columns and c != "peak_width"])
    cols = ["window_id", "chrom", "wstart", "wend", "strand", "gene_id", "gene_name",
            "context", "sequence", "label", "peak_start", "peak_end", "peak_strand",
            "signal", "gc", "n_A", "n_C", "n_G", "n_U", "log1p_tpm", "tpm",
            "gene_length", "input_signal", "source", "assembly", "dataset_version"]
    df = df[cols]
    df = df.rename(columns={"wstart": "start", "wend": "end"})

    excl = pd.DataFrame(exclusions, columns=["exclusion_reason", "chrom", "start", "end"])

    report = {
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset_version": DATASET_VERSION,
        "protocol_sha256": protocol.sha256,
        "assembly": assembly,
        "window_length": W,
        "n_idr_reproducible_peaks_input": int(len(idr)),
        "n_positive_after_filters": int((df["label"] == 1).sum()),
        "n_background": int((df["label"] == 0).sum()),
        "n_rows_total": int(len(df)),
        "background_ratio_realized": round(float((df['label'] == 0).sum()) /
                                           max(1, int((df['label'] == 1).sum())), 3),
        "context_counts": df["context"].value_counts().to_dict(),
        "label_context_crosstab": df.groupby(["label", "context"]).size()
                                  .unstack(fill_value=0).to_dict(),
        "n_genes_represented": int(df["gene_id"].nunique()),
        "exclusion_counts": excl["exclusion_reason"].value_counts().to_dict(),
        "n_expression_unmapped_transcripts": int(n_unmapped_tx),
        "expression_min_tpm": min_tpm,
        "usable_expressed_sequence_mb": round(total / 1e6, 2),
        "gc_mean_positive": float(df.loc[df.label == 1, "gc"].mean()),
        "gc_mean_background": float(df.loc[df.label == 0, "gc"].mean()),
        "known_amendment_impact": {
            "A1_positive_cap": int(cand["max_positives"]),
            "A1_realized_positives": int((df["label"] == 1).sum()),
            "A2_input_signal": "unavailable; column present and NaN, never zero-filled",
        },
    }
    return BuildResult(dataset=df, exclusions=excl, report=report)


def save_build(result: BuildResult, dataset_path="data/processed/dataset.parquet",
               excl_path="data/processed/exclusions.parquet",
               report_path="reports/dataset_build.json") -> dict:
    Path(dataset_path).parent.mkdir(parents=True, exist_ok=True)
    result.dataset.to_parquet(dataset_path, index=False)
    result.exclusions.to_parquet(excl_path, index=False)
    h = write_json(report_path, result.report)
    return {"dataset_sha256": hash_file(dataset_path),
            "exclusions_sha256": hash_file(excl_path),
            "report_sha256": h}
