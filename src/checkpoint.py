"""Versioned, atomic, full-state checkpointing for Cortex.

Two on-disk formats are recognised:

``weights_only``
    A plain ``state_dict`` (what ``best_model.pt`` / ``best_model_2.pt`` are).
    Loadable into the model for inference or as an initialisation point, but it
    carries **no** optimizer/scheduler/epoch information.

``full``
    The Cortex training checkpoint written by :func:`save_checkpoint`. It
    contains every piece of state required to continue training (model,
    optimizer, scheduler, AMP scaler, RNG, sampler, progress counters, config
    and provenance metadata).

Saves are atomic: a temporary file is written, flushed and then ``os.replace``d
over the destination. The previous valid file is preserved as ``<name>.bak``
via a hard link so an interrupted write cannot corrupt the last good state.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import random
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch

CORTEX_VERSION = "2.0.0"
FORMAT_VERSION = 2

FULL_REQUIRED_KEYS = ("model_state", "optimizer_state", "epoch", "next_epoch")


# ----------------------------------------------------------------------
# Atomic IO
# ----------------------------------------------------------------------
def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def atomic_torch_save(obj: Any, path: str) -> None:
    """Atomically save ``obj`` to ``path``, preserving the previous file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    if os.path.exists(path):
        bak = f"{path}.bak"
        try:
            if os.path.exists(bak):
                os.remove(bak)
            os.link(path, bak)
        except OSError:
            import shutil

            shutil.copy2(path, bak)
    os.replace(tmp, path)
    _fsync_dir(os.path.dirname(os.path.abspath(path)) or ".")


# ----------------------------------------------------------------------
# RNG state
# ----------------------------------------------------------------------
def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if state.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[checkpoint] WARNING: could not fully restore RNG state: {exc}", file=sys.stderr)


def _capture_sampler_state(sampler) -> Optional[Any]:
    """Capture the resumable state of a sampler or RNG.

    ``torch.Generator`` exposes ``get_state``/``set_state`` while PyTorch
    ``Sampler`` objects use ``state_dict``/``load_state_dict``; support both.
    """
    if sampler is None:
        return None
    if hasattr(sampler, "state_dict"):
        return sampler.state_dict()
    if hasattr(sampler, "get_state"):
        return sampler.get_state()
    return None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# Environment / provenance
# ----------------------------------------------------------------------
def _env_versions() -> Dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        versions["cuda"] = torch.version.cuda or "unknown"
        versions["gpu"] = torch.cuda.get_device_name(0)
    for mod in ("transformers", "peft"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "unknown"
    return versions


def dataset_fingerprint(paths) -> str:
    """Cheap fingerprint of dataset files (size + mtime + md5 of first 1 MiB)."""
    import hashlib

    h = hashlib.sha256()
    for p in sorted(paths):
        if not os.path.exists(p):
            h.update(f"{p}:MISSING".encode())
            continue
        st = os.stat(p)
        h.update(f"{p}:{st.st_size}:{int(st.st_mtime)}".encode())
        with open(p, "rb") as fh:
            h.update(fh.read(1024 * 1024))
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------
# Save / load
# ----------------------------------------------------------------------
def build_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    next_epoch: int,
    global_step: int,
    micro_step: int,
    best_val_loss: float,
    best_val_metrics: Optional[Dict[str, float]],
    patience_counter: int,
    config,
    dataset_fp: str,
    sampler,
    ohem_state: Optional[Dict[str, Any]] = None,
    history: Optional[list] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    state = {
        "format_version": FORMAT_VERSION,
        "checkpoint_type": "full",
        "cortex_version": CORTEX_VERSION,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "config": config.to_dict(),
        "config_hash": config.config_hash(),
        "seed": config.seed,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "next_epoch": int(next_epoch),
        "global_step": int(global_step),
        "micro_step": int(micro_step),
        "best_val_loss": float(best_val_loss),
        "best_val_metrics": best_val_metrics,
        "patience_counter": int(patience_counter),
        "rng": capture_rng_state(),
        "sampler_state": _capture_sampler_state(sampler),
        "ohem_state": ohem_state,
        "history": history,
        "dataset_fingerprint": dataset_fp,
        "env": _env_versions(),
    }
    if extra:
        state.update(extra)
    return state


def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    atomic_torch_save(state, path)


def load_checkpoint(path: str, map_location="cpu") -> Dict[str, Any]:
    """Load any Cortex checkpoint.

    Full checkpoints are loaded with ``weights_only=False`` (they contain
    Python objects such as RNG states). Weights-only ``state_dict`` files are
    detected and wrapped into a minimal full-like dict.
    """
    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, dict) and obj.get("checkpoint_type") == "full":
        obj["_source"] = path
        return obj

    # Could be a raw state_dict, or a dict with model_state but no marker.
    if isinstance(obj, dict) and "model_state" in obj:
        obj.setdefault("checkpoint_type", "full")
        obj.setdefault("format_version", 1)
        obj["_source"] = path
        return obj

    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return {
            "checkpoint_type": "weights_only",
            "format_version": 1,
            "model_state": obj,
            "_source": path,
        }
    raise ValueError(f"Unrecognised checkpoint format at {path}")


def is_full_checkpoint(ckpt: Dict[str, Any]) -> bool:
    return ckpt.get("checkpoint_type") == "full" and all(k in ckpt for k in FULL_REQUIRED_KEYS)


# ----------------------------------------------------------------------
# Compatibility
# ----------------------------------------------------------------------
def compatibility_report(ckpt: Dict[str, Any], config, strict_hash: bool = False) -> Dict[str, Any]:
    """Compare a full checkpoint's config against the current config."""
    report = {"compatible": True, "warnings": [], "errors": []}
    if not is_full_checkpoint(ckpt):
        report["compatible"] = False
        report["errors"].append("not a full checkpoint")
        return report

    saved = ckpt.get("config", {})
    mismatches = []
    for key in (
        "drug_model_name",
        "target_model_name",
        "hidden_dim",
        "num_attention_heads",
        "lora_r",
        "lora_alpha",
        "target_modules",
        "compression_factor",
        "pooling",
        "fusion",
        "max_drug_len",
        "max_target_len",
        "finetune_mode",
        "label_col",
    ):
        if key not in saved:
            continue
        a, b = saved[key], getattr(config, key)
        if isinstance(a, list):
            a = tuple(a)
        if isinstance(b, list):
            b = tuple(b)
        if a != b:
            mismatches.append((key, a, b))
    if mismatches:
        report["compatible"] = False
        for k, a, b in mismatches:
            msg = f"config mismatch '{k}': checkpoint={a!r} current={b!r}"
            report["errors"].append(msg)

    if strict_hash and not mismatches:
        if saved_hash := ckpt.get("config_hash"):
            if saved_hash != config.config_hash():
                report["warnings"].append(
                    f"config_hash differs ({saved_hash} != {config.config_hash()}); "
                    "non-architectural settings changed."
                )
    return report


def load_model_state(model, ckpt: Dict[str, Any], strict: bool = True) -> Dict[str, Any]:
    """Load model weights, tolerating architecture differences.

    Returns a report dict with ``missing``, ``unexpected`` and ``mismatched``
    key lists. Tensors whose shape does not match the current model (for
    example a checkpoint trained with a different LoRA rank or target-module
    set) are skipped rather than raising, so weights-only checkpoints from an
    older configuration can still be used to warm-start compatible layers.

    When ``strict`` is true, a mismatch raises instead of being skipped.
    """
    if "model_state" not in ckpt:
        raise ValueError("checkpoint has no model_state")
    state = ckpt["model_state"]
    target = model.state_dict()

    missing = [k for k in target if k not in state]
    unexpected = [k for k in state if k not in target]
    mismatched = [
        k for k in target
        if k in state and tuple(state[k].shape) != tuple(target[k].shape)
    ]
    if strict and (missing or unexpected or mismatched):
        raise RuntimeError(
            f"state_dict incompatible: missing={missing[:5]} "
            f"unexpected={unexpected[:5]} mismatched={mismatched[:5]}"
        )

    loadable = {
        k: v for k, v in state.items()
        if k in target and tuple(v.shape) == tuple(target[k].shape)
    }
    model.load_state_dict(loadable, strict=False)
    return {"missing": missing, "unexpected": unexpected, "mismatched": mismatched}


# ----------------------------------------------------------------------
# Recovery manifest
# ----------------------------------------------------------------------
def write_manifest(ckpt_dir: str, entries: Dict[str, Any]) -> str:
    path = os.path.join(ckpt_dir, "manifest.json")
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(entries, fh, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def read_manifest(ckpt_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(ckpt_dir, "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def resolve_resume(mode: str, ckpt_dir: str) -> Optional[str]:
    """Resolve a ``--resume`` mode to a concrete checkpoint path.

    ``auto``  -> latest full checkpoint (manifest, then latest.pt, then best.pt)
    ``none``  -> returns None (fresh run)
    otherwise ``mode`` is treated as a path.
    """
    if mode == "none":
        return None
    if mode == "auto":
        manifest = read_manifest(ckpt_dir)
        if manifest:
            latest = manifest.get("latest")
            if latest and os.path.exists(latest):
                return latest
        for candidate in ("latest.pt", "best.pt"):
            p = os.path.join(ckpt_dir, candidate)
            if os.path.exists(p):
                return p
        return None
    if not os.path.exists(mode):
        raise FileNotFoundError(f"--resume path does not exist: {mode}")
    return mode
