"""Cortex evaluation / inference.

Loads either a **full** Cortex training checkpoint or a **weights-only**
``state_dict`` (such as the original ``best_model_2.pt``) and reports the full
regression metric suite on a chosen split. Never modifies checkpoints.

Examples::

    # Evaluate a full checkpoint's best model on the validation split
    python -m src.evaluate --checkpoint experiments/baseline/checkpoints/best.pt --split val

    # Evaluate the original weights-only checkpoint on the test split
    python -m src.evaluate --checkpoint checkpoints/best_model_2.pt --split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config, apply_overrides  # noqa: E402
from src import checkpoint as ckpt_utils  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.metrics import regression_metrics  # noqa: E402
from src.models import BindingModel  # noqa: E402
from src.tokenization import BindingDataset, dynamic_collate_fn  # noqa: E402

logger = get_logger("src.evaluate")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Cortex evaluation")
    ap.add_argument("--checkpoint", required=True,
                    help="path to a full or weights-only checkpoint")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--data-path", default=None, help="override the split parquet path")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--amp-dtype", choices=["fp32", "fp16", "bf16"], default=None)
    ap.add_argument("--json-out", default=None, help="write metrics JSON here")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N samples (smoke tests)")
    ap.add_argument("--set", action="append", default=[])
    return ap.parse_args(argv)


def build_config_from_checkpoint(path: str, args) -> Config:
    cfg = Config()
    ckpt = ckpt_utils.load_checkpoint(path, map_location="cpu")
    if ckpt.get("config"):
        overrides = {k: v for k, v in ckpt["config"].items() if k in set(cfg.field_names())}
        cfg = apply_overrides(cfg, overrides)
    # CLI overrides win.
    if args.amp_dtype is not None:
        cfg.amp_dtype = args.amp_dtype
    if args.batch_size is not None:
        cfg.eval_batch_size = args.batch_size
        cfg.batch_size = args.batch_size
    for item in args.set:
        key, raw = item.split("=", 1)
        from src.train import _coerce

        setattr(cfg, key, _coerce(raw, getattr(cfg, key, None)))
    return cfg


@torch.no_grad()
def run_eval(cfg, model, loader, device):
    model.eval()
    use_amp = cfg.uses_amp()
    amp_dtype = cfg.resolved_torch_dtype()
    preds, truths = [], []
    for batch in loader:
        drug_ids = batch["drug_input_ids"].to(device)
        drug_mask = batch["drug_attention_mask"].to(device)
        target_ids = batch["target_input_ids"].to(device)
        target_mask = batch["target_attention_mask"].to(device)
        labels = batch["label"].to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            predictions = model(drug_ids, drug_mask, target_ids, target_mask)
        p = predictions.float().cpu().numpy()
        t = labels.float().cpu().numpy()
        if cfg.normalize_target:
            p = p * cfg.target_std + cfg.target_mean
            t = t * cfg.target_std + cfg.target_mean
        preds.append(p)
        truths.append(t)
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return regression_metrics(truths, preds), preds, truths


def main(argv=None):
    args = parse_args(argv)
    cfg = build_config_from_checkpoint(args.checkpoint, args)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    if args.amp_dtype is None and not torch.cuda.is_available():
        cfg.amp_dtype = "fp32"

    split_path = args.data_path or {
        "train": cfg.train_path, "val": cfg.val_path, "test": cfg.test_path
    }[args.split]

    ckpt = ckpt_utils.load_checkpoint(args.checkpoint, map_location="cpu")
    kind = ckpt.get("checkpoint_type", "unknown")
    logger.info("Checkpoint: %s (type=%s)", args.checkpoint, kind)

    dataset = BindingDataset(split_path, cfg)
    if args.limit:
        from torch.utils.data import Subset

        dataset = Subset(dataset, list(range(min(args.limit, len(dataset)))))
    if cfg.normalize_target:
        from src.train import _unwrap

        _unwrap(dataset).set_label_normalization(cfg.target_mean, cfg.target_std)
    loader = DataLoader(dataset, batch_size=cfg.eval_batch_size, shuffle=False,
                        collate_fn=dynamic_collate_fn, num_workers=cfg.num_workers,
                        pin_memory=cfg.device.startswith("cuda"))

    model = BindingModel(cfg).to(device)
    report = ckpt_utils.load_model_state(model, ckpt, strict=False)
    if report["missing"]:
        logger.warning("Missing keys when loading checkpoint: %d (e.g. %s)",
                       len(report["missing"]), report["missing"][:5])
    if report["mismatched"]:
        logger.warning("Shape-mismatched keys skipped: %d (e.g. %s)",
                       len(report["mismatched"]), report["mismatched"][:5])
    if report["unexpected"]:
        logger.warning("Unexpected keys: %d (e.g. %s)",
                       len(report["unexpected"]), report["unexpected"][:5])

    metrics, preds, truths = run_eval(cfg, model, loader, device)
    logger.info("=== %s metrics (%s, n=%d) ===", args.split.upper(), args.checkpoint, metrics["n"])
    for k in ("rmse", "mae", "pearson", "spearman", "r2"):
        logger.info("  %-8s: %.4f", k, metrics[k])

    if args.json_out:
        parent = os.path.dirname(os.path.abspath(args.json_out))
        os.makedirs(parent, exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump({"checkpoint": args.checkpoint, "split": args.split,
                       "metrics": metrics}, fh, indent=2)
        logger.info("Wrote metrics to %s", args.json_out)
    return metrics


if __name__ == "__main__":
    main()
