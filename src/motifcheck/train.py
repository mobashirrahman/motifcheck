"""Seeded training of paired CNN conditions C0 and C1.

Within a seed pair the two conditions start from identical parameters and
receive identical batches in identical order. Only the per-example loss weight
differs, so any measured difference is attributable to the intervention and not
to optimisation noise.
"""

from __future__ import annotations

import datetime as _dt
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import DATASET_VERSION
from .config import Protocol
from .hashing import hash_file
from .models import build_model, count_parameters
from .motif import encode

CONDITIONS = ("C0", "C1")


def encode_dataset(sequences: list[str]) -> torch.Tensor:
    """(N, 4, L) one-hot float32. CNN inputs carry sequence and nothing else."""
    X = np.stack([encode(s).T for s in sequences])
    return torch.from_numpy(X.astype(np.float32))


def batch_sequence(n: int, batch_size: int, epochs: int, seed: int) -> list[np.ndarray]:
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).numpy()
        for i in range(0, n, batch_size):
            batches.append(perm[i:i + batch_size])
    return batches


def train_condition(X: torch.Tensor, y: torch.Tensor, w: np.ndarray,
                    protocol: Protocol, model_id: str, seed: int,
                    X_val: torch.Tensor | None = None, y_val: np.ndarray | None = None,
                    device: str = "cuda", log=print) -> dict:
    """Train one condition/seed. Returns the best-validation-AP checkpoint and history."""
    from sklearn.metrics import average_precision_score, log_loss

    opt_cfg = protocol.section("optimization")
    epochs = int(opt_cfg["epochs"])
    bs = int(opt_cfg["batch_size"])
    lr = float(opt_cfg["fallback_lr"]) if opt_cfg.get("fallback_lr_used") else float(opt_cfg["lr"])

    torch.manual_seed(seed)
    # Both tensors live on the training device: a CPU validation tensor against a
    # CUDA model raises a type error deep inside the first forward pass.
    X = X.to(device)
    if X_val is not None:
        X_val = X_val.to(device)
    model = build_model(model_id, protocol, X.shape[2]).to(device)
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=float(opt_cfg["weight_decay"]))
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    wt = torch.from_numpy(np.asarray(w, dtype=np.float32)).to(device)
    yt = y.to(device)
    batches = batch_sequence(len(X), bs, epochs, seed)

    best = {"ap": -np.inf, "logloss": np.inf, "epoch": -1, "state": None}
    history = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        running = 0.0
        for idx in batches:
            bi = torch.from_numpy(idx).to(device)
            logits = model(X[bi])
            per_example = criterion(logits, yt[bi])
            # mean of weighted per-example BCE with fixed global normalisation
            loss = (per_example * wt[bi]).sum() / wt.sum()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg["grad_clip_norm"]))
            optimiser.step()
            running += float(loss.detach()) * len(bi)
        row = {"epoch": ep + 1, "train_weighted_loss": running / len(X)}
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                vl = torch.cat([model(X_val[i:i + 2048]) for i in range(0, len(X_val), 2048)])
            vp = torch.sigmoid(vl).cpu().numpy()
            ap = float(average_precision_score(y_val, vp))
            ll = float(log_loss(y_val, np.clip(vp, 1e-15, 1 - 1e-15), labels=[0, 1]))
            row.update(val_ap=ap, val_log_loss=ll)
            better = (ap > best["ap"]) or (ap == best["ap"] and ll < best["logloss"])
            if better:
                best = {"ap": ap, "logloss": ll, "epoch": ep + 1,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        history.append(row)
        if not np.isfinite(row["train_weighted_loss"]):
            raise FloatingPointError(f"non-finite training loss at epoch {ep+1}")
    elapsed = time.time() - t0

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {
        "model": model,
        "initial_state": init_state,
        "history": history,
        "best_epoch": best["epoch"],
        "best_val_ap": None if best["ap"] == -np.inf else best["ap"],
        "best_val_log_loss": None if best["logloss"] == np.inf else best["logloss"],
        "n_parameters": count_parameters(model),
        "n_updates": len(batches),
        "learning_rate": lr,
        "seconds": elapsed,
        "finite": bool(np.isfinite([h["train_weighted_loss"] for h in history]).all()),
    }


def train_pair(X: torch.Tensor, y: torch.Tensor, w0: np.ndarray, w1: np.ndarray,
               protocol: Protocol, seed: int, X_val: torch.Tensor,
               y_val: np.ndarray, out_dir: str | Path, device: str = "cuda",
               log=print) -> dict:
    """Train C0 and C1 for one seed with identical batches and initialization."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec: dict = {
        "seed": seed,
        "dataset_version": DATASET_VERSION,
        "protocol_sha256": protocol.sha256,
        "trained_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "paired_inputs_identical": True,
        "conditions": {},
    }
    initial_ref = None
    batch_ref = None
    for cond in CONDITIONS:
        w = w0 if cond == "C0" else w1
        log(f"[train] {cond} seed={seed}")
        r = train_condition(X, y, w, protocol, "C0", seed,
                            X_val=X_val, y_val=y_val, device=device, log=log)
        ckpt_path = out_dir / f"{cond}_seed{seed}.pt"
        torch.save({
            "architecture": protocol.section("tinycnn"),
            "model_id": cond,
            "seed": seed,
            "dataset_version": DATASET_VERSION,
            "protocol_sha256": protocol.sha256,
            "state_dict": r["model"].state_dict(),
            "preprocessing_version": DATASET_VERSION,
            "best_epoch": r["best_epoch"],
        }, ckpt_path)

        # Save/reload parity (protocol 10.1): logits from the trained in-memory
        # model must reproduce those from the reloaded checkpoint on the same
        # device and configuration.
        reload_diff = 0.0
        if X_val is not None:
            Xv = X_val.to(device)
            with torch.no_grad():
                in_memory = torch.cat([r["model"](Xv[i:i + 2048])
                                       for i in range(0, len(Xv), 2048)])
            reloaded = load_checkpoint(ckpt_path, protocol, device)
            with torch.no_grad():
                from_disk = torch.cat([reloaded(Xv[i:i + 2048])
                                       for i in range(0, len(Xv), 2048)])
            reload_diff = float(torch.max(torch.abs(in_memory - from_disk)))
            del reloaded

        same_init = True
        if initial_ref is None:
            initial_ref = r["initial_state"]
        else:
            same_init = all(torch.equal(initial_ref[k], r["initial_state"][k])
                            for k in initial_ref)
        batches = batch_sequence(len(X), int(protocol.section("optimization")["batch_size"]),
                                 int(protocol.section("optimization")["epochs"]), seed)
        same_batches = True
        if batch_ref is None:
            batch_ref = batches
        else:
            same_batches = all(np.array_equal(a, b) for a, b in zip(batch_ref, batches))

        rec["conditions"][cond] = {
            "identical_initial_parameters": bool(same_init),
            "identical_batch_sequence": bool(same_batches),
            "n_parameters": r["n_parameters"],
            "n_updates": r["n_updates"],
            "best_epoch": r["best_epoch"],
            "best_val_ap": r["best_val_ap"],
            "best_val_log_loss": r["best_val_log_loss"],
            "seconds": round(r["seconds"], 2),
            "finite_losses": r["finite"],
            "learning_rate": r["learning_rate"],
            "history": r["history"],
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": hash_file(ckpt_path),
            "reload_max_abs_logit_diff": reload_diff,
        }
        del r
    rec["reload_parity_max_abs_diff"] = max(
        rec["conditions"][c]["reload_max_abs_logit_diff"] for c in CONDITIONS)
    rec["pairs_match"] = all(
        rec["conditions"][c].get("identical_initial_parameters") and
        rec["conditions"][c].get("identical_batch_sequence") for c in CONDITIONS)
    return rec


def load_checkpoint(path: str | Path, protocol: Protocol, device: str = "cuda") -> nn.Module:
    blob = torch.load(path, map_location=device, weights_only=False)
    if blob.get("protocol_sha256") != protocol.sha256:
        raise ValueError(
            f"checkpoint {path} was trained under protocol {blob.get('protocol_sha256')} "
            f"but the current protocol is {protocol.sha256}; refusing to mix"
        )
    model = build_model(blob["model_id"], protocol, 201).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model
