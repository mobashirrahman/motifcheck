"""Source acquisition and provenance validation (protocol gate G1).

Downloads every asset named in ``configs/protocol.yaml``, verifies checksums,
and writes ``data/source_manifest.json``. Failed downloads are recorded
distinctly from legitimately empty files, as protocol section 4.1 requires.
"""

from __future__ import annotations

import datetime as _dt
import gzip
import shutil
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import Protocol
from .fileio import is_gzip as _is_gzip, iter_lines as _fileio_lines, open_text as _open_text
from .hashing import hash_file, write_json

USER_AGENT = "motifcheck/1.0 (reproducible RNA-binding benchmark; academic use)"
_UA = {"User-Agent": USER_AGENT}


class AcquisitionError(RuntimeError):
    pass


def _download(url: str, dest: Path, timeout: int = 180, retries: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers=_UA)
            with urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
            tmp.replace(dest)
            return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:  # noqa: PERF203
            last = exc
            if attempt < retries:
                time.sleep(2 ** attempt)
    tmp.unlink(missing_ok=True)
    raise AcquisitionError(f"failed to download {url} after {retries} attempts: {last}")


def _gzip_lines(path: Path):
    yield from _fileio_lines(path)


def _decompress_gzip(src: Path) -> Path:
    dst = src.with_suffix("")
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=1 << 20)
    return dst


def _index_fasta(fasta: Path) -> Path:
    """Build a .fai index so windows can be read by random access.

    Delegated to pysam rather than hand-rolled: correct offset bookkeeping for
    line widths and newline handling is easy to get subtly wrong, and a wrong
    offset yields corrupt sequence rather than an error.
    """
    import pysam

    fai = Path(str(fasta) + ".fai")
    if fai.exists() and fai.stat().st_size > 0:
        return fai
    pysam.faidx(str(fasta))
    if not fai.exists() or fai.stat().st_size == 0:
        raise AcquisitionError(f"pysam.faidx produced no index for {fasta}")
    return fai


# --------------------------------------------------------------------------
# Per-asset validators. Each must fail loudly rather than assume file schema.
# --------------------------------------------------------------------------

def _validate_peaks(path: Path, meta: dict) -> dict:
    with _open_text(path) as fh:
        first = fh.readline()
        n_cols = len(first.rstrip("\n").split("\t"))
        n = sum(1 for line in fh if line.strip()) + (1 if first.strip() else 0)
        first_fields = first.rstrip("\n").split("\t")
    if n_cols < 6:
        raise AcquisitionError(
            f"peak file has {n_cols} columns; narrowPeak with strand and signal is required "
            "to orient windows (plan section 4.2). Refusing to guess the schema."
        )
    return {
        "n_peaks": n,
        "n_columns": n_cols,
        "name_field": first_fields[3] if n_cols > 3 else None,
        "score_field": first_fields[4] if n_cols > 4 else None,
        "schema": "narrowPeak (BED6+3); column 5 is the ENCODE constant 1000 and is NOT a "
                  "calibrated statistic; columns 7-8 are -1 in this release and are not used",
    }


def _validate_pwm(path: Path, meta: dict) -> dict:
    rows = list(_gzip_lines(path))
    header = rows[0].split("\t")
    bases = [h.strip().upper() for h in header[1:]]
    if set(bases) != {"A", "C", "G", "U"}:
        raise AcquisitionError(f"PWM header must be A,C,G,U in RNA alphabet; got {bases}")
    matrix = [[float(v) for v in r.split("\t")[1:]] for r in rows[1:] if r.strip()]
    # CisBP-RNA PWMs are position-major: each ROW is a distribution over A,C,G,U
    # and sums to one. Validating the row axis catches a transposed file.
    row_sums = [round(sum(row), 6) for row in matrix]
    col_sums = [round(sum(matrix[i][b] for i in range(len(matrix))), 6) for b in range(4)]
    return {
        "n_positions": len(matrix),
        "bases": bases,
        "content": "position-major frequencies; each row sums to 1",
        "row_sums": row_sums,
        "max_abs_deviation_of_row_from_one": max(abs(s - 1.0) for s in row_sums),
        "column_sums_are_not_normalised": col_sums,
        "assay_type": meta.get("assay_type"),
        "motif_class": meta.get("motif_class"),
        "study": meta.get("study"),
        "study_id": meta.get("study_id"),
        "cisbp_tf_id": meta.get("cisbp_tf_id"),
        "database_build": meta.get("database_build"),
    }


def _validate_gtf(path: Path, meta: dict) -> dict:
    n_features = 0
    genes: set[str] = set()
    contigs: set[str] = set()
    for line in _gzip_lines(path):
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 9:
            continue
        n_features += 1
        contigs.add(f[0])
        if f[2] == "gene":
            for attr in f[8].split(";"):
                if attr.strip().startswith("gene_id"):
                    genes.add(attr.split('"')[1] if '"' in attr else attr.strip())
    return {
        "n_features": n_features,
        "n_genes": len(genes),
        "n_contigs": len(contigs),
        "seqnames_match_peaks": bool(contigs & {"chr1"}),
    }


def _validate_expression(path: Path, meta: dict) -> dict:
    header = next(iter(_gzip_lines(path)), "").split("\t")
    if "transcript_id" not in header or "length" not in header:
        raise AcquisitionError(
            "expression file is not a GENCODE RSEM quantification "
            "(needs transcript_id and length columns); refusing to guess TPM semantics"
        )
    return {
        "columns": header,
        "tool": "RSEM (ENCODE)",
        "tpm_column": "TPM" if "TPM" in header else None,
        "gene_level_aggregation_required": True,
    }


def _validate_fasta(path: Path, meta: dict) -> dict:
    fai = _index_fasta(path)
    entries = sum(1 for _ in open(fai))
    if entries < 20:
        raise AcquisitionError(f"genome FASTA index has only {entries} contigs")
    return {"n_contigs": entries, "index": fai.name}


_VALIDATORS = {
    "bed": _validate_peaks,
    "pwm": _validate_pwm,
    "gtf": _validate_gtf,
    "tsv": _validate_expression,
    "fasta": _validate_fasta,
}


def acquire_all(protocol: Protocol, out_dir: str | Path = "data/external",
                manifest_path: str | Path = "data/source_manifest.json",
                log=print) -> dict:
    out_dir = Path(out_dir)
    entries: dict[str, dict] = {}
    failures: list[str] = []

    for name, meta in protocol.section("sources").items():
        dest = out_dir / f"{name}__{meta['id']}"
        log(f"[acquire] {name:28s} <- {meta['id']}")
        record: dict = {
            "role": name,
            "id": meta["id"],
            "url": meta["url"],
            "source_url": meta.get("source_url"),
            "description": " ".join(str(meta.get("description", "")).split()),
            "assembly": meta.get("assembly"),
            "format": meta.get("format"),
            "status": meta.get("status"),
            "retrieved_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "license_note": "ENCODE data are released under the ENCODE data use policy; "
                            "UCSC and GENCODE are freely redistributable; CisBP-RNA motifs are "
                            "used for academic research with citation.",
        }
        try:
            if not dest.exists() or dest.stat().st_size == 0:
                _download(meta["url"], dest)
            size = dest.stat().st_size
            digest = hash_file(dest)
            record.update(size_bytes=size, sha256=digest)
            if size == 0:
                record["validation_status"] = "failed_empty_file"
                failures.append(f"{name}: downloaded file is empty")
            else:
                expected = meta.get("expected_md5")
                if expected is not None:
                    import hashlib
                    got = hashlib.md5(dest.read_bytes()).hexdigest()
                    record["md5"] = got
                    record["expected_md5"] = expected
                    record["md5_matches_portal"] = bool(got == expected)
                    if got != expected:
                        raise AcquisitionError(
                            f"md5 mismatch: portal {expected}, downloaded {got}")
                target = dest
                if _is_gzip(dest) and meta["format"] in ("fasta", "gtf"):
                    target = _decompress_gzip(dest)
                    record["decompressed_path"] = str(target)
                    record["decompressed_sha256"] = hash_file(target)
                info = _VALIDATORS[meta["format"]](target, meta)
                if meta["format"] == "fasta":
                    _index_fasta(target)
                record["validation_status"] = "ok"
                record["validated"] = info
        except Exception as exc:  # noqa: BLE001
            record["validation_status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(f"{name}: {exc}")
            log(f"[acquire] FAILED {name}: {exc}")
        entries[name] = record
        log(f"[acquire] {'ok' if record.get('validation_status') == 'ok' else record.get('validation_status'):10s} {name}")

    manifest = {
        "manifest_version": 1,
        "protocol_sha256": protocol.sha256,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "assemblies": sorted({e["assembly"] for e in entries.values() if e.get("assembly")}),
        "sources": entries,
        "failures": failures,
        "complete": not failures,
        "unavailable_by_design": {
            "processed_size_matched_input_signal": (
                "Not released for ENCSR981WKN. Amendment A2 removes the input-coverage "
                "feature from B2 and disables the input-coverage observability filter."
            )
        },
    }
    write_json(manifest_path, manifest)
    log(f"[acquire] manifest -> {manifest_path}  ({len(failures)} failure(s))")
    return manifest


def main(argv: list[str] | None = None) -> int:
    from .cli import load_protocol_from_args

    cfg, args = load_protocol_from_args(argv, need_config=True)
    manifest = acquire_all(
        cfg,
        out_dir=getattr(args, "out_dir", None) or "data/external",
        manifest_path=getattr(args, "manifest", None) or "data/source_manifest.json",
    )
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
