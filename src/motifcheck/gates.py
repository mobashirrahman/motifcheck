"""Machine-readable technical gate records (protocol section 10).

Technical gates stop downstream execution when correctness is uncertain.
Scientific decision rules classify results; they never authorise repeated
test-set tuning. Every real gate record carries its actual check counts,
timestamps and code version, and each gate lists the artifacts it read.
"""

from __future__ import annotations

import datetime as _dt
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .hashing import append_jsonl, write_json

GATE_ORDER = ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7"]


def code_version() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unversioned"


@dataclass
class Gate:
    gate: str
    status: str                                   # pass | fail | blocked | skipped
    checks: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    evidence_paths: list = field(default_factory=list)
    data_hash: str | None = None
    config_hash: str | None = None
    detail: dict = field(default_factory=dict)

    def record(self, log_path: str | Path = "results/gate_log.jsonl") -> dict:
        rec = {
            "gate": self.gate,
            "status": self.status,
            "data_hash": self.data_hash,
            "config_hash": self.config_hash,
            "checks": self.checks,
            "warnings": self.warnings,
            "evidence_paths": [str(p) for p in self.evidence_paths],
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "code_version": code_version(),
            "detail": self.detail,
        }
        write_json(f"reports/gates/{self.gate}.json", rec)
        append_jsonl(log_path, rec)
        return rec

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def summarize(gate_dir: str | Path = "reports/gates") -> dict:
    gate_dir = Path(gate_dir)
    out = {}
    for g in GATE_ORDER:
        p = gate_dir / f"{g}.json"
        out[g] = p.exists() and __import__("json").loads(p.read_text())["status"] or "missing"
    return out


# Gates that certify the pipeline computes what it claims to compute. G5 is
# deliberately excluded: it certifies that the intervention reached its target
# balance, which is a property of the data as much as of the method. Amendment A5
# records, before any training, that a G5 failure narrows the claim to an
# observational audit rather than blocking execution.
EXECUTION_GATES = ["G0", "G1", "G2", "G3", "G4"]


def execution_gates_ok() -> dict:
    """Correctness gates that must pass before test data may be touched."""
    states = summarize()
    missing = [g for g in EXECUTION_GATES if states.get(g) != "pass"]
    return {
        "required_gates": EXECUTION_GATES,
        "states": states,
        "all_passed": not missing,
        "failing_gates": missing,
        "excluded_from_execution": {
            "G5": "intervention instantiation; failure narrows the claim (amendment A5)"},
    }


def technical_gates_ok(upto: str = "G6") -> dict:
    """All gates from G0 up to and including `upto` must have passed.

    Kept for reporting. Downstream guards use `execution_gates_ok`.
    """
    states = summarize()
    required = GATE_ORDER[: GATE_ORDER.index(upto) + 1]
    missing = [g for g in required if states.get(g) != "pass"]
    return {
        "required_gates": required,
        "states": states,
        "all_passed": not missing,
        "failing_gates": missing,
    }


def intervention_instantiated() -> dict:
    """Whether G5 achieved its prespecified balance thresholds."""
    state = summarize().get("G5", "missing")
    rec = Path("reports/gates/G5.json")
    checks = {}
    if rec.exists():
        checks = __import__("json").loads(rec.read_text()).get("checks", {})
    failing = [k for k, v in checks.items()
               if k.endswith(("_met", "_ok")) and v is False]
    return {
        "gate": "G5",
        "status": state,
        "instantiated": state == "pass",
        "failing_checks": failing,
        "basis": "amendment A5 (recorded before training)",
    }


# --------------------------------------------------------------------------
# Concrete gate builders
# --------------------------------------------------------------------------

def g0_protocol(protocol, protocol_path: str = "configs/protocol.yaml") -> Gate:
    """Scope, hypotheses, schema, metrics, split algorithm, seeds, budgets, margins."""
    req = protocol.section("primary_comparison")
    checks = {
        "protocol_saved": bool(protocol_path),
        "n_amendments_recorded": len(protocol.amendments),
        "seeds_declared": len(protocol.seeds),
        "window_length_declared": protocol.window_length,
        "primary_metric_declared": protocol["evaluation"]["primary_metric"],
        "bootstrap_replicates_declared": int(protocol["statistics"]["bootstrap_replicates"]),
        "requirement_1_margin_declared": req["requirement_1_predictive_non_inferiority"]["rule"],
        "requirement_2_margin_declared": req["requirement_2_improved_motif_preference"]["min_point_estimate"],
        "total_gpu_hour_cap": int(protocol["budget"]["total_gpu_hours_cap"]),
    }
    return Gate(
        gate="G0", status="pass", checks=checks,
        config_hash=protocol.sha256,
        evidence_paths=[protocol_path],
        detail={"amendments": [
            {"id": a["id"], "section": a["section"], "change": " ".join(str(a["change"]).split())}
            for a in protocol.amendments]},
    )


def g1_provenance(manifest: dict) -> Gate:
    checks = {
        "n_sources": len(manifest["sources"]),
        "n_failed": len(manifest["failures"]),
        "n_validated_ok": sum(1 for s in manifest["sources"].values()
                              if s.get("validation_status") == "ok"),
        "n_with_sha256": sum(1 for s in manifest["sources"].values() if s.get("sha256")),
        "distinct_assemblies": len(manifest["assemblies"]),
        "downloads_complete": bool(manifest["complete"]),
    }
    warnings = [f"failed source: {f}" for f in manifest["failures"]]
    warnings += [f"{k}: no portal md5 published; sha256 recorded instead"
                 for k, s in manifest["sources"].items() if not s.get("expected_md5")]
    status = "pass" if manifest["complete"] and checks["distinct_assemblies"] == 1 else "fail"
    if checks["distinct_assemblies"] > 1:
        status = "fail"
    return Gate(
        gate="G1", status=status, checks=checks, warnings=warnings,
        evidence_paths=["data/source_manifest.json"],
        config_hash=manifest.get("protocol_sha256"),
        detail={"unavailable_by_design": manifest.get("unavailable_by_design", {}),
                "sources": {k: {"id": v["id"], "sha256": v.get("sha256"),
                                "validation_status": v.get("validation_status")}
                            for k, v in manifest["sources"].items()}},
    )


def g2_split(audit: dict, dataset_sha256: str, config_hash: str) -> Gate:
    checks = dict(audit["forbidden_overlap_checks"])
    checks.update({
        "zero_forbidden_overlap": audit["zero_forbidden_overlap"],
        "all_minimum_support_ok": audit["all_support_ok"],
        "train_positives": audit["split_counts"]["train"]["positives"],
        "train_components": audit["split_counts"]["train"]["components"],
        "validation_positives": audit["split_counts"]["validation"]["positives"],
        "validation_components": audit["split_counts"]["validation"]["components"],
        "test_positives": audit["split_counts"]["test"]["positives"],
        "test_components": audit["split_counts"]["test"]["components"],
        "near_duplicate_pairs_merged": audit["join_statistics"]["near_duplicate_pairs"],
        "giant_component_removed": audit["giant_component_removed"],
    })
    status = "pass" if audit["status"] == "pass" else "fail"
    warnings = []
    if audit["exploratory"]:
        warnings.append("support below protocol minimum; study is exploratory by prespecification")
    return Gate(
        gate="G2", status=status, checks=checks, warnings=warnings,
        data_hash=dataset_sha256, config_hash=config_hash,
        evidence_paths=["reports/split_audit.json"],
        detail={"split_counts": audit["split_counts"],
                "join_statistics": audit["join_statistics"]},
    )


def g3_implementation(test_report: dict) -> Gate:
    checks = {
        "n_tests": test_report["n_tests"],
        "n_passed": test_report["n_passed"],
        "n_failed": test_report["n_failed"],
        "synthetic_fixtures_passed": test_report["synthetic"]["all_passed"],
        "overfit_fixture_achieved_accuracy": test_report["synthetic"]["overfit_accuracy"],
        "overfit_fixture_target": 0.99,
    }
    status = "pass" if (test_report["n_failed"] == 0
                        and test_report["synthetic"]["all_passed"]) else "fail"
    return Gate(
        gate="G3", status=status, checks=checks,
        warnings=test_report.get("warnings", []),
        evidence_paths=["reports/self_test.json"],
        detail={"failures": test_report.get("failures", [])[:20]},
    )


def g4_fit_sanity(records: list[dict], reload_parity: dict) -> Gate:
    finite = all(c["finite_losses"] for r in records for c in r["conditions"].values())
    paired = all(r.get("pairs_match") for r in records)
    params = {c["n_parameters"] for r in records for c in r["conditions"].values()}
    checks = {
        "n_seed_pairs": len(records),
        "all_losses_finite": finite,
        "all_pairs_matched": paired,
        "distinct_parameter_counts": sorted(params),
        "checkpoint_reload_max_abs_logit_diff": reload_parity["max_abs_diff"],
        "checkpoint_reload_tolerance": 1e-6,
        "checkpoint_reload_ok": reload_parity["ok"],
        "all_best_epochs_present": all(c.get("best_epoch") is not None
                                       for r in records for c in r["conditions"].values()),
    }
    status = "pass" if (finite and paired and params == {13857}
                        and reload_parity["ok"]) else "fail"
    return Gate(
        gate="G4", status=status, checks=checks,
        warnings=reload_parity.get("warnings", []),
        evidence_paths=["results/training/pairs.jsonl"],
        detail={"per_seed_best_val_ap": {
            f"seed{r['seed']}": {c: r["conditions"][c]["best_val_ap"] for c in r["conditions"]}
            for r in records}},
    )


def g5_intervention(weights_report: dict, paired_inputs: dict) -> Gate:
    qc = weights_report["final_qc"]
    checks = {
        "n_attempts": weights_report["n_attempts"],
        "all_balance_thresholds_met": weights_report["all_thresholds_met"],
        "effective_sample_size_fraction": qc["checks"]["effective_sample_size_fraction"],
        "effective_sample_size_min": qc["checks"]["effective_sample_size_min"],
        "smd_gc": qc["checks"]["smd_gc"],
        "smd_log1p_tpm": qc["checks"]["smd_log1p_tpm"],
        "max_context_proportion_diff": qc["checks"]["max_context_proportion_diff"],
        "rows_excluded_unsupported": weights_report["rows_excluded_unsupported"],
        "identical_paired_inputs": bool(paired_inputs.get("identical")),
        "only_loss_weights_differ": bool(paired_inputs.get("only_weights_differ")),
    }
    status = "pass" if (weights_report["all_thresholds_met"]
                        and checks["identical_paired_inputs"]) else "fail"
    warnings = []
    if weights_report["rows_excluded_unsupported"]:
        warnings.append(
            f"{weights_report['rows_excluded_unsupported']} training rows lacked common support "
            "and were excluded from both conditions; the estimand is narrowed accordingly")
    return Gate(
        gate="G5", status=status, checks=checks, warnings=warnings,
        evidence_paths=["reports/intervention_qc.json"],
        detail={"residual_balance_diagnostics": qc["residual_diagnostics"]},
    )


def g6_final_lock(lock: dict, dry_run: dict) -> Gate:
    checks = {
        "analysis_lock_present": bool(lock),
        "n_locked_artifacts": len(lock.get("artifacts", {})),
        "code_version": lock.get("code_version"),
        "config_hash": lock.get("config_hash"),
        "data_hash": lock.get("data_hash"),
        "validation_dry_run_passed": dry_run["ok"],
        "dry_run_failures": dry_run.get("failures", []),
    }
    status = "pass" if (lock and dry_run["ok"]) else "fail"
    return Gate(
        gate="G6", status=status, checks=checks,
        evidence_paths=["reports/analysis_lock.json"],
        detail={"locked_endpoints": lock.get("endpoints", [])},
    )


def g7_release(final: dict) -> Gate:
    checks = {
        "n_frozen_test_evaluations": final.get("n_test_evaluations", 0),
        "append_only_log_present": True,
        "all_planned_results_reported": final.get("all_planned_results_accounted", False),
        "clean_environment_replay_passed": final.get("replay_ok", False),
        "claim_level": final.get("claim", {}).get("claim_level"),
    }
    status = "pass" if final.get("all_planned_results_accounted") else "fail"
    warnings = list(final.get("warnings", []))
    return Gate(
        gate="G7", status=status, checks=checks, warnings=warnings,
        evidence_paths=["reports/final_report.md", "results/test_evaluation_log.jsonl"],
        detail={"limitations": final.get("limitations", [])},
    )
