"""Phase A preflight for a fresh Cortex training campaign.

Runs, before any expensive training, a set of fail-fast checks:

1. **Environment** — torch/CUDA/GPU, versions, git commit and dirty state.
2. **Data** — fingerprint + integrity + leakage audit of the configured splits.
3. **Fresh initialisation** — verifies that a freshly constructed model does
   **not** contain weights from any historical Cortex checkpoint
   (``checkpoints/best_model*.pt``), while the encoder *base* weights do match
   their Hugging Face pretrained initialisation.
4. **Config snapshot** — resolved config + hash, written to the report.

The result is a machine-readable ``preflight_report.json`` and a process exit
code (non-zero if any hard check fails).

Usage::

    python -m src.preflight --set train_path=data/splits/v1/cold_target/train.parquet \
        --set val_path=data/splits/v1/cold_target/val.parquet \
        --set test_path=data/splits/v1/cold_target/test.parquet \
        --historical checkpoints/best_model.pt checkpoints/best_model_2.pt
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config, apply_overrides  # noqa: E402
from src import checkpoint as ckpt_utils  # noqa: E402
from src import data_audit  # noqa: E402
from src.logger import get_logger  # noqa: E402

logger = get_logger("src.preflight")

# Cortex-specific (non-encoder) parameter prefixes. A genuinely fresh model
# must have random values here; if any historical checkpoint supplied them, the
# tensors would match exactly.
CORTEX_PARAM_PREFIXES = (
    "drug_proj.", "target_proj.", "target_compressor.",
    "cross_attention.", "mlp.", "lora_",
)


def _git(args: List[str]) -> str:
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        return subprocess.check_output(["git", *args], cwd=cwd,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def git_info() -> Dict[str, str]:
    return {
        "commit": _git(["rev-parse", "--short", "HEAD"]),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "status_porcelain": _git(["status", "--porcelain"]),
    }


def is_cortex_param(key: str) -> bool:
    return any(key.startswith(p) or f".{p}" in key for p in CORTEX_PARAM_PREFIXES)


def compare_to_checkpoints(model, paths: List[str]) -> Dict:
    """For each historical checkpoint, report how similar the *shared* tensors are.

    A fresh model shows large differences (and **no** exactly-equal tensor) on
    every shared Cortex parameter. If adapters/head had been loaded, those
    tensors would be bit-identical.
    """
    target = model.state_dict()
    out = {}
    for path in paths:
        if not path or not os.path.exists(path):
            out[path] = {"exists": False}
            continue
        try:
            ckpt = ckpt_utils.load_checkpoint(path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - defensive
            out[path] = {"exists": True, "error": str(exc)}
            continue
        state = ckpt.get("model_state", {})
        shared = [k for k in target if k in state and tuple(state[k].shape) == tuple(target[k].shape)]
        cortex_shared = [k for k in shared if is_cortex_param(k)]
        # Only Cortex-specific tensors indicate contamination: the frozen encoder
        # base weights are identical to HF pretrained *by design*, so equality
        # there is expected and not a concern. Degenerate tensors that are all
        # zero in the fresh model (e.g. unused pooler LoRA-B, which PEFT
        # zero-initialises and never trains) are also excluded: they carry no
        # information about whether weights were loaded.
        informative = [k for k in cortex_shared if target[k].abs().max().item() > 0]
        identical_cortex = [
            k for k in informative
            if torch.allclose(state[k].float(), target[k].float())
        ]
        deltas = [
            (state[k].float() - target[k].float()).abs().max().item()
            for k in informative
        ]
        out[path] = {
            "exists": True,
            "n_shared_tensors": len(shared),
            "n_shared_cortex_tensors": len(cortex_shared),
            "n_informative_cortex_tensors": len(informative),
            "n_identical_cortex_tensors": len(identical_cortex),
            "identical_cortex_tensors": identical_cortex[:10],
            "min_cortex_max_abs_diff": (min(deltas) if deltas else None),
        }
    return out


def _input_embedding_weight(tower) -> Optional[torch.Tensor]:
    candidates = [tower]
    bm = getattr(tower, "base_model", None)
    if bm is not None:
        candidates.append(bm)
        inner = getattr(bm, "model", None)
        if inner is not None:
            candidates.append(inner)
    for obj in candidates:
        try:
            return obj.get_input_embeddings().weight.detach().float()
        except Exception:
            continue
    return None


def verify_encoder_pretrained(model, cfg: Config) -> Dict:
    """Check the encoder *base* input embeddings still match HF ``from_pretrained``."""
    from transformers import AutoModel

    result = {}
    for name, tower, hf_name, kwargs in (
        ("drug", model.drug_tower, cfg.drug_model_name,
         dict(deterministic_eval=True, trust_remote_code=True)),
        ("target", model.target_tower, cfg.target_model_name, {}),
    ):
        try:
            fresh = AutoModel.from_pretrained(hf_name, **kwargs)
            a = _input_embedding_weight(tower)
            b = _input_embedding_weight(fresh)
            if a is None or b is None or a.shape != b.shape:
                result[name] = {"checked": False, "reason": "embedding not accessible"}
            else:
                result[name] = {
                    "checked": True,
                    "matches_pretrained": bool(torch.allclose(a, b)),
                    "max_abs_diff": float((a - b).abs().max().item()),
                }
            del fresh
        except Exception as exc:
            result[name] = {"checked": False, "reason": str(exc)}
    return result


def verify_fresh_initialization(cfg: Config, historical: List[str]) -> Dict:
    from src.models import BindingModel

    ckpt_utils.seed_everything(cfg.seed)
    model = BindingModel(cfg)
    total, trainable = model.count_parameters()
    comparisons = compare_to_checkpoints(model, historical)
    encoders = verify_encoder_pretrained(model, cfg)

    # Fresh = no shared tensor is bit-identical to a historical checkpoint.
    contaminated = [
        p for p, r in comparisons.items()
        if r.get("exists") and r.get("n_identical_cortex_tensors", 0) > 0
    ]
    enc_ok = all(
        (not r.get("checked")) or r.get("matches_pretrained")
        for r in encoders.values()
    )
    return {
        "is_fresh": len(contaminated) == 0,
        "contaminated_checkpoints": contaminated,
        "comparisons": comparisons,
        "encoders_pretrained_ok": enc_ok,
        "encoders": encoders,
        "total_params": total,
        "trainable_params": trainable,
    }


def audit_splits(cfg: Config) -> Dict:
    paths = {"train": cfg.train_path, "val": cfg.val_path, "test": cfg.test_path}
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        return {"ok": False, "missing": missing, "paths": paths}
    report = data_audit.run(os.path.dirname(cfg.train_path) or ".")
    report["ok"] = True
    report["paths"] = paths
    report["fingerprint"] = ckpt_utils.dataset_fingerprint(list(paths.values()))
    return report


def run_preflight(cfg: Config, historical: List[str], skip_model: bool = False) -> Dict:
    report = {
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "git": git_info(),
        "env": ckpt_utils._env_versions(),
        "config": cfg.to_dict(),
        "config_hash": cfg.config_hash(),
        "checks": [],
    }

    def add(name, ok, detail):
        report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})

    # Data
    data = audit_splits(cfg)
    report["data"] = data
    add("data_files_exist", data.get("ok", False), data.get("missing", "ok"))
    if data.get("ok"):
        leak = data["leakage"]
        add("no_train_test_pair_leak", leak["pair_overlap_train_test"] == 0,
            f"pair_overlap_train_test={leak['pair_overlap_train_test']}")
        add("val_pairs_disjoint_from_train", leak["pair_overlap_train_val"] == 0,
            f"pair_overlap_train_val={leak['pair_overlap_train_val']} "
            "(hard check for the new pipeline; historical splits may fail)")

    # Fresh initialisation
    if not skip_model:
        init = verify_fresh_initialization(cfg, historical)
        report["init"] = init
        add("fresh_init_no_historical_weights", init["is_fresh"],
            init["contaminated_checkpoints"])
        add("encoders_pretrained", init["encoders_pretrained_ok"], init["encoders"])
    else:
        add("fresh_init_no_historical_weights", True, "skipped (--skip-model)")
        add("encoders_pretrained", True, "skipped (--skip-model)")

    report["ok"] = all(c["ok"] for c in report["checks"])
    return report


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Cortex fresh-training preflight")
    ap.add_argument("--out", default=None, help="path for preflight_report.json")
    ap.add_argument("--historical", nargs="*",
                    default=["checkpoints/best_model.pt", "checkpoints/best_model_2.pt"])
    ap.add_argument("--skip-model", action="store_true",
                    help="skip building the model (data/env checks only)")
    ap.add_argument("--set", action="append", default=[])
    return ap.parse_args(argv)


def _coerce(raw: str, like):
    if isinstance(like, bool):
        return raw.lower() in ("1", "true", "yes", "y")
    if isinstance(like, int) and not isinstance(like, bool):
        return int(raw)
    if isinstance(like, float):
        return float(raw)
    if isinstance(like, tuple):
        return tuple(raw.split(","))
    return raw


def main(argv=None):
    args = parse_args(argv)
    cfg = Config()
    overrides = {}
    for item in args.set:
        key, raw = item.split("=", 1)
        overrides[key] = _coerce(raw, getattr(cfg, key, None))
    cfg = apply_overrides(cfg, overrides)

    report = run_preflight(cfg, args.historical, skip_model=args.skip_model)
    out = args.out or os.path.join(cfg.resolved_checkpoint_dir(), "preflight_report.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    for c in report["checks"]:
        logger.info("[%s] %s -> %s", "OK" if c["ok"] else "FAIL", c["name"], c["detail"])
    logger.info("Preflight %s. Report -> %s", "PASSED" if report["ok"] else "FAILED", out)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
