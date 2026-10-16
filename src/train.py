"""Cortex training entry point with full-state resumable training.

Usage (see docs/CHECKPOINTS_AND_RECOVERY.md for the full guide)::

    # Fresh experiment
    python -m src.train --fresh-run --experiment-name baseline

    # Resume the latest checkpoint of an experiment (e.g. after a Kaggle reset)
    python -m src.train --resume auto --experiment-name baseline

    # Initialise from a weights-only checkpoint (fresh optimiser)
    python -m src.train --init-weights checkpoints/best_model_2.pt \
        --fresh-run --experiment-name from_weights

Resume behaviour is explicit and never silently falls back:
  * ``--resume none``        start a new run (subject to --fresh-run)
  * ``--resume auto``        resume the latest full checkpoint in the experiment dir
  * ``--resume PATH``        resume a specific full checkpoint
  * ``--init-weights PATH``  load model weights only, fresh optimiser/scheduler
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config, apply_overrides  # noqa: E402
from src import checkpoint as ckpt_utils  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.metrics import regression_metrics  # noqa: E402
from src.models import BindingModel  # noqa: E402
from src.tokenization import BindingDataset, dynamic_collate_fn  # noqa: E402

logger = get_logger("src.train")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Cortex DTI training")
    ap.add_argument("--resume", default="auto",
                    help="'auto' (default; resume latest if present, else start "
                         "fresh), 'none' (always start fresh), or a path to a "
                         "full checkpoint. Starting over an existing experiment "
                         "requires --fresh-run.")
    ap.add_argument("--init-weights", default=None,
                    help="path to a weights-only (or full) checkpoint to initialise model weights")
    ap.add_argument("--fresh-run", action="store_true",
                    help="start a brand-new run, discarding any existing "
                         "checkpoints in the experiment directory")
    ap.add_argument("--experiment-name", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--amp-dtype", choices=["fp32", "fp16", "bf16"], default=None)
    ap.add_argument("--loss", dest="loss_name", choices=["mse", "huber", "smoothl1"], default=None)
    ap.add_argument("--normalize-target", action="store_true", default=None)
    ap.add_argument("--hard-mining-start-epoch", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--save-every-steps", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="stop after this many optimizer steps (smoke tests)")
    ap.add_argument("--limit-train", type=int, default=None,
                    help="use only the first N training samples (smoke tests)")
    ap.add_argument("--limit-val", type=int, default=None,
                    help="use only the first N validation samples (smoke tests)")
    ap.add_argument("--set", action="append", default=[],
                    help="arbitrary config override key=value (repeatable)")
    return ap.parse_args(argv)


def build_config(args) -> Config:
    cfg = Config()
    overrides = {}
    if args.experiment_name is not None:
        overrides["experiment_name"] = args.experiment_name
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.checkpoint_dir is not None:
        overrides["checkpoint_dir"] = args.checkpoint_dir
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.grad_accum_steps is not None:
        overrides["grad_accum_steps"] = args.grad_accum_steps
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.amp_dtype is not None:
        overrides["amp_dtype"] = args.amp_dtype
    if args.loss_name is not None:
        overrides["loss_name"] = args.loss_name
    if args.normalize_target is not None:
        overrides["normalize_target"] = args.normalize_target
    if args.hard_mining_start_epoch is not None:
        overrides["hard_mining_start_epoch"] = args.hard_mining_start_epoch
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    if args.save_every_steps is not None:
        overrides["save_every_steps"] = args.save_every_steps

    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        overrides[key] = _coerce(raw, getattr(cfg, key, None))
    return apply_overrides(cfg, overrides)


def _coerce(raw: str, like):
    if isinstance(like, bool):
        return raw.lower() in ("1", "true", "yes", "y")
    if isinstance(like, int) and not isinstance(like, bool):
        return int(raw)
    if isinstance(like, float):
        return float(raw)
    if isinstance(like, tuple):
        return tuple(raw.split(","))
    if like is None:
        try:
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw
    return raw


# ----------------------------------------------------------------------
# Experiment / logging helpers
# ----------------------------------------------------------------------
class MetricsWriter:
    def __init__(self, path: str):
        self.path = path

    def append(self, record: Dict):
        record = {**record, "wall_time": time.time()}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")


def setup_experiment_dir(cfg: Config) -> str:
    exp_dir = cfg.resolved_checkpoint_dir()
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    return exp_dir


def build_scheduler(optimizer, cfg: Config, total_steps: int):
    if cfg.scheduler_name == "none" or total_steps <= 0:
        return None
    warmup = cfg.warmup_steps if cfg.warmup_steps >= 0 else int(cfg.warmup_ratio * total_steps)
    base_lr = cfg.learning_rate
    eta_min = cfg.eta_min

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(1.0, max(0.0, progress))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (eta_min / base_lr) + (1.0 - eta_min / base_lr) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_loss(cfg: Config):
    if cfg.loss_name == "mse":
        return nn.MSELoss(reduction="none"), nn.MSELoss()
    if cfg.loss_name == "huber":
        return nn.HuberLoss(reduction="none", delta=cfg.huber_delta), nn.HuberLoss(delta=cfg.huber_delta)
    if cfg.loss_name == "smoothl1":
        return nn.SmoothL1Loss(reduction="none"), nn.SmoothL1Loss()
    raise ValueError(f"Unknown loss: {cfg.loss_name}")


# Model-selection metric is decoupled from the loss so that runs trained with
# different objectives (e.g. MSE vs Huber) can be compared on one yardstick.
_SELECT_HIGHER_IS_BETTER = {"pearson", "spearman", "r2"}
_SELECT_CHOICES = ("val_loss", "rmse", "mae", "pearson", "spearman", "r2")


def selection_score(cfg: Config, val_loss: float, metrics: Dict) -> float:
    if cfg.select_metric == "val_loss":
        return float(val_loss)
    if cfg.select_metric not in metrics:
        raise ValueError(
            f"select_metric={cfg.select_metric!r} not available; choose {_SELECT_CHOICES}"
        )
    return float(metrics[cfg.select_metric])


def is_improvement(score: float, best_score: Optional[float], metric: str) -> bool:
    if best_score is None or not math.isfinite(best_score):
        return math.isfinite(score)
    if metric in _SELECT_HIGHER_IS_BETTER:
        return score > best_score
    return score < best_score


def build_optimizer(model, cfg: Config):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer_name == "adamw":
        return AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                     betas=cfg.betas, eps=cfg.eps)
    if cfg.optimizer_name == "adam":
        return Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                    betas=cfg.betas, eps=cfg.eps)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer_name}")


def _unwrap(ds):
    from torch.utils.data import Subset

    return ds.dataset if isinstance(ds, Subset) else ds


def git_commit(repo_dir: Optional[str] = None) -> str:
    """Short git commit of the working tree, or 'unknown' outside a repo."""
    import subprocess

    cwd = repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def make_loaders(cfg: Config, generator=None, limit_train=None, limit_val=None):
    train_ds = BindingDataset(cfg.train_path, cfg)
    val_ds = BindingDataset(cfg.val_path, cfg)
    if limit_train:
        from torch.utils.data import Subset

        train_ds = Subset(train_ds, list(range(min(limit_train, len(train_ds)))))
    if limit_val:
        from torch.utils.data import Subset

        val_ds = Subset(val_ds, list(range(min(limit_val, len(val_ds)))))
    common = dict(batch_size=cfg.batch_size, collate_fn=dynamic_collate_fn,
                  num_workers=cfg.num_workers, pin_memory=cfg.pin_memory and cfg.device.startswith("cuda"))
    # Never drop the whole (tiny) training set, e.g. a smoke run with
    # --limit-train 16 and batch_size 32 would otherwise yield zero batches.
    drop_last = bool(cfg.drop_last and len(train_ds) >= cfg.batch_size)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=drop_last,
                              generator=generator, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_ds, val_ds, train_loader, val_loader


# ----------------------------------------------------------------------
# Numerical guard
# ----------------------------------------------------------------------
def assert_finite(tensor, where: str):
    if not torch.isfinite(tensor).all():
        n_nan = int(torch.isnan(tensor).sum().item())
        n_inf = int(torch.isinf(tensor).sum().item())
        raise FloatingPointError(
            f"Non-finite values in {where}: nan={n_nan} inf={n_inf}. "
            "Training aborted (no silent skipping). See docs/OPTIMIZATION_GUIDE.md."
        )


# ----------------------------------------------------------------------
# Train / eval loops
# ----------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion_none,
                    cfg, device, epoch, progress, on_step=None, max_steps=None):
    """Run one epoch.

    ``progress`` is a mutable dict with keys ``global_step`` and ``micro_step``
    so that checkpoint callbacks observe up-to-date counters. ``on_step`` is an
    optional ``callable(global_step)`` invoked after each optimizer step (used
    for periodic step checkpoints). When ``max_steps`` is set the epoch stops as
    soon as ``progress["global_step"] >= max_steps`` (used by smoke runs).

    Returns ``(mean_loss, stopped_early)``.
    """
    model.train()
    accum = cfg.grad_accum_steps
    total_loss, n_batches = 0.0, 0
    use_amp = cfg.uses_amp()
    amp_dtype = cfg.resolved_torch_dtype()
    progress["epoch_samples"] = 0

    def _optimizer_step() -> bool:
        """Apply one optimizer update. Returns True if ``max_steps`` reached."""
        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), cfg.grad_clip_norm
        )
        finite_grad = bool(torch.isfinite(torch.as_tensor(grad_norm)))
        if not finite_grad and scaler is None:
            # Without AMP there is no loss-scaling safety net, so a non-finite
            # gradient is always a genuine bug: fail fast.
            raise FloatingPointError(
                f"Non-finite grad norm at step {progress['global_step']}"
            )
        if scaler is not None:
            # GradScaler detects inf/NaN during unscale_ and will skip this
            # step, lowering the scale for the next iteration. Surface a warning
            # rather than crashing so transient overflow is handled.
            if not finite_grad:
                logger.warning(
                    "Non-finite grad norm at step %d; GradScaler will skip "
                    "this update and reduce the scale.", progress["global_step"],
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        progress["global_step"] += 1
        progress["micro_step"] = 0
        if on_step is not None:
            on_step(progress["global_step"])
        return max_steps is not None and progress["global_step"] >= max_steps

    optimizer.zero_grad(set_to_none=True)
    for batch_idx, batch in enumerate(loader):
        drug_ids = batch["drug_input_ids"].to(device, non_blocking=True)
        drug_mask = batch["drug_attention_mask"].to(device, non_blocking=True)
        target_ids = batch["target_input_ids"].to(device, non_blocking=True)
        target_mask = batch["target_attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            predictions = model(drug_ids, drug_mask, target_ids, target_mask)
            sample_losses = criterion_none(predictions, labels)
            if epoch >= cfg.hard_mining_start_epoch:
                k = max(1, int(sample_losses.size(0) * cfg.hard_mining_fraction))
                loss = torch.topk(sample_losses, k=k).values.mean()
            else:
                loss = sample_losses.mean()

        assert_finite(loss, f"loss (epoch {epoch}, batch {batch_idx})")
        scaled_loss = loss / accum
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        total_loss += loss.item()
        n_batches += 1
        progress["micro_step"] += 1
        progress["epoch_samples"] += int(labels.size(0))

        if progress["micro_step"] % accum == 0:
            if _optimizer_step():
                logger.warning("Reached --max-steps=%d mid-epoch; stopping.", max_steps)
                return total_loss / max(1, n_batches), True

        if cfg.log_every_steps and batch_idx % cfg.log_every_steps == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "  epoch %d | batch %d/%d | loss %.4f | lr %.3e | step %d",
                epoch, batch_idx, len(loader), loss.item(), lr, progress["global_step"],
            )

    # Flush any gradients left over from a partial final accumulation group so
    # the tail of the epoch is not silently discarded.
    if progress["micro_step"] != 0:
        _optimizer_step()

    return total_loss / max(1, n_batches), False


@torch.no_grad()
def evaluate(model, loader, criterion, cfg, device, target_mean=0.0, target_std=1.0):
    model.eval()
    total_loss, n = 0.0, 0
    preds, truths = [], []
    use_amp = cfg.uses_amp()
    amp_dtype = cfg.resolved_torch_dtype()
    for batch in loader:
        drug_ids = batch["drug_input_ids"].to(device, non_blocking=True)
        drug_mask = batch["drug_attention_mask"].to(device, non_blocking=True)
        target_ids = batch["target_input_ids"].to(device, non_blocking=True)
        target_mask = batch["target_attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            predictions = model(drug_ids, drug_mask, target_ids, target_mask)
            loss = criterion(predictions, labels)
        total_loss += loss.item()
        n += 1

        p = predictions.float().cpu().numpy()
        t = labels.float().cpu().numpy()
        if cfg.normalize_target:
            p = p * target_std + target_mean
            t = t * target_std + target_mean
        preds.append(p)
        truths.append(t)

    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    metrics = regression_metrics(truths, preds)
    return total_loss / max(1, n), metrics


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    cfg = build_config(args)
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        cfg.device = "cpu"

    exp_dir = setup_experiment_dir(cfg)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    metrics_path = os.path.join(exp_dir, "metrics.jsonl")
    metrics_writer = MetricsWriter(metrics_path)

    # ---------------- Resume / overwrite decision (fail fast) ----------------
    explicit_resume_path = args.resume not in ("auto", "none")
    if args.fresh_run and explicit_resume_path:
        raise SystemExit(
            "--fresh-run cannot be combined with an explicit --resume PATH; pick one."
        )
    resume_mode = "none" if args.fresh_run else args.resume
    try:
        resume_path = ckpt_utils.resolve_resume(resume_mode, ckpt_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    manifest_exists = os.path.exists(os.path.join(ckpt_dir, "manifest.json"))
    if resume_path is None and manifest_exists and not args.fresh_run:
        raise SystemExit(
            f"Experiment '{cfg.experiment_name}' already exists at {exp_dir} and no "
            "full checkpoint was selected to resume from.\n"
            "Refusing to overwrite existing results. Choose one of:\n"
            "  --resume auto           continue the latest checkpoint\n"
            "  --fresh-run             start this experiment over (overwrites)\n"
            "  --experiment-name NAME  use a new experiment directory"
        )

    if args.fresh_run and (manifest_exists or os.path.isdir(ckpt_dir)):
        logger.warning("--fresh-run: clearing existing experiment artifacts in %s", exp_dir)
        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        for stale in ("metrics.jsonl", "final_report.json", "config.json", "manifest.json"):
            p = os.path.join(exp_dir, stale)
            if os.path.exists(p):
                os.remove(p)

    # Determinism / RNG
    ckpt_utils.seed_everything(cfg.seed)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info("Device: %s | AMP: %s (%s)", cfg.device, cfg.uses_amp(), cfg.amp_dtype)
    logger.info("Experiment dir: %s", exp_dir)
    logger.info("Selection metric: %s | seed %d | data protocol %s",
                cfg.select_metric, cfg.seed, cfg.data_protocol)
    git_sha = git_commit()
    logger.info("Git commit: %s", git_sha)

    train_ds, val_ds, train_loader, val_loader = make_loaders(
        cfg, generator, limit_train=args.limit_train, limit_val=args.limit_val)

    # Target normalisation statistics (from train split only).
    if cfg.normalize_target:
        labels = np.asarray(_unwrap(train_ds).labels, dtype=np.float64)
        cfg.target_mean = float(labels.mean())
        cfg.target_std = float(labels.std() or 1.0)
        logger.info("Target normalisation: mean=%.4f std=%.4f", cfg.target_mean, cfg.target_std)
        _unwrap(train_ds).set_label_normalization(cfg.target_mean, cfg.target_std)

    dataset_fp = ckpt_utils.dataset_fingerprint([cfg.train_path, cfg.val_path, cfg.test_path])

    model = BindingModel(cfg).to(cfg.device)
    total_p, train_p = model.count_parameters()
    logger.info("Parameters: total=%s trainable=%s", f"{total_p:,}", f"{train_p:,}")

    criterion_none, criterion_val = build_loss(cfg)
    optimizer = build_optimizer(model, cfg)

    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    scaler = (
        torch.amp.GradScaler("cuda", init_scale=cfg.amp_init_scale)
        if cfg.amp_dtype == "fp16" and cfg.device.startswith("cuda")
        else None
    )

    # ---------------- Resume / init resolution ----------------
    start_epoch = 1
    progress = {"global_step": 0, "micro_step": 0}
    best_val_loss = float("inf")
    best_val_metrics = None
    best_score = float("inf") if cfg.select_metric in ("val_loss", "rmse", "mae") else float("-inf")
    patience_counter = 0
    history = []
    # Provenance of where the model weights came from. A genuinely fresh
    # experiment reports "pretrained_hf" (encoders from HF + fresh Cortex
    # adapters/head) and must never reference best_model*.pt.
    init_source = "pretrained_hf"

    if args.init_weights:
        init_source = f"init_weights:{args.init_weights}"
        init_ckpt = ckpt_utils.load_checkpoint(args.init_weights, map_location="cpu")
        report = ckpt_utils.load_model_state(model, init_ckpt, strict=False)
        logger.info(
            "Initialised weights from %s (missing=%d, unexpected=%d, mismatched=%d)",
            args.init_weights, len(report["missing"]), len(report["unexpected"]),
            len(report["mismatched"]),
        )
        if report["mismatched"]:
            logger.warning(
                "Skipped %d shape-mismatched tensors (likely a different LoRA "
                "rank or target-module set): %s",
                len(report["mismatched"]), report["mismatched"][:5],
            )
        if report["missing"]:
            logger.warning("Missing keys on init: %s", report["missing"][:10])

    if resume_path is not None:
        logger.info("Resuming from checkpoint: %s", resume_path)
        init_source = f"resume:{resume_path}"
        ckpt = ckpt_utils.load_checkpoint(resume_path, map_location="cpu")
        if not ckpt_utils.is_full_checkpoint(ckpt):
            raise SystemExit(
                f"Checkpoint {resume_path} is weights-only and cannot be resumed as a full run. "
                "Use --init-weights for weights-only checkpoints."
            )
        report = ckpt_utils.compatibility_report(ckpt, cfg)
        if not report["compatible"]:
            raise SystemExit("Incompatible checkpoint:\n  " + "\n  ".join(report["errors"]))
        for w in report["warnings"]:
            logger.warning("Checkpoint warning: %s", w)

        ckpt_utils.load_model_state(model, ckpt, strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if scaler is not None and ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt["next_epoch"])
        progress["global_step"] = int(ckpt.get("global_step", 0))
        progress["micro_step"] = int(ckpt.get("micro_step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        best_val_metrics = ckpt.get("best_val_metrics")
        if ckpt.get("best_score") is not None:
            best_score = float(ckpt["best_score"])
        patience_counter = int(ckpt.get("patience_counter", 0))
        history = ckpt.get("history") or []
        # Restore RNG and sampler state.
        ckpt_utils.restore_rng_state(ckpt.get("rng"))
        if ckpt.get("sampler_state") is not None:
            generator.set_state(ckpt["sampler_state"])
        logger.info("Resumed at epoch %d, global_step %d, best_val_loss %.4f",
                    start_epoch, progress["global_step"], best_val_loss)
        # Restore normalisation stats if the run used them and CLI didn't override.
        saved_cfg = ckpt.get("config", {})
        if saved_cfg.get("normalize_target") and cfg.normalize_target:
            cfg.target_mean = float(saved_cfg.get("target_mean", cfg.target_mean))
            cfg.target_std = float(saved_cfg.get("target_std", cfg.target_std))
            _unwrap(train_ds).set_label_normalization(cfg.target_mean, cfg.target_std)
            _unwrap(val_ds).set_label_normalization(cfg.target_mean, cfg.target_std)

    # ---------------- Save resolved config ----------------
    with open(os.path.join(exp_dir, "config.json"), "w") as fh:
        json.dump(cfg.to_dict(), fh, indent=2, default=str)

    def do_checkpoint(tag: str, epoch: int, next_epoch: int):
        state = ckpt_utils.build_checkpoint(
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            epoch=epoch, next_epoch=next_epoch, global_step=progress["global_step"],
            micro_step=progress["micro_step"], best_val_loss=best_val_loss,
            best_val_metrics=best_val_metrics, patience_counter=patience_counter,
            config=cfg, dataset_fp=dataset_fp, sampler=generator, history=history,
            extra={
                "best_score": best_score,
                "select_metric": cfg.select_metric,
                "init_source": init_source,
                "total_params": total_p,
                "trainable_params": train_p,
                "git_commit": git_sha,
            },
        )
        path = os.path.join(ckpt_dir, f"{tag}.pt")
        ckpt_utils.save_checkpoint(path, state)
        manifest = ckpt_utils.read_manifest(ckpt_dir) or {}
        manifest[tag] = path
        if tag == "latest":
            manifest["latest"] = path
        if tag == "best":
            manifest["best"] = path
        manifest["epoch"] = epoch
        manifest["global_step"] = progress["global_step"]
        ckpt_utils.write_manifest(ckpt_dir, manifest)
        logger.info("Saved %s checkpoint -> %s", tag, path)
        return path

    def on_step(step: int):
        if cfg.save_every_steps and step % cfg.save_every_steps == 0:
            do_checkpoint(f"step_{step:07d}", epoch_holder[0], epoch_holder[0])

    epoch_holder = [start_epoch]

    # ---------------- Training loop ----------------
    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_holder[0] = epoch
        epoch_start = time.time()
        logger.info("--- Epoch %d/%d ---", epoch, cfg.epochs)

        train_loss, stopped_early = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            criterion_none, cfg, cfg.device, epoch, progress,
            on_step=on_step if cfg.save_every_steps else None,
            max_steps=args.max_steps,
        )

        val_loss, val_metrics = evaluate(model, val_loader, criterion_val, cfg, cfg.device,
                                         cfg.target_mean, cfg.target_std)
        epoch_time = time.time() - epoch_start
        score = selection_score(cfg, val_loss, val_metrics)

        peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        n_samples = int(progress.get("epoch_samples", len(train_loader) * cfg.batch_size))
        throughput = n_samples / epoch_time if epoch_time > 0 else 0.0
        lr = optimizer.param_groups[0]["lr"]
        logger.info("Epoch %d | train_loss %.4f | val_loss %.4f | rmse %.4f | mae %.4f | "
                    "r %.4f | rho %.4f | r2 %.4f | %s %.4f | %.1fs | %.1f samp/s | peak %.2fGB",
                    epoch, train_loss, val_loss, val_metrics["rmse"], val_metrics["mae"],
                    val_metrics["pearson"], val_metrics["spearman"], val_metrics["r2"],
                    cfg.select_metric, score, epoch_time, throughput, peak_mem)

        record = {"epoch": epoch, "global_step": progress["global_step"], "split": "val",
                  "train_loss": train_loss, "val_loss": val_loss, "lr": lr,
                  "select_metric": cfg.select_metric, "select_score": score,
                  "epoch_time_s": epoch_time, "throughput_samples_s": throughput,
                  "peak_gpu_gb": peak_mem, "amp_dtype": cfg.amp_dtype,
                  "total_params": total_p, "trainable_params": train_p,
                  "git_commit": git_sha, **val_metrics}
        history.append(record)
        metrics_writer.append(record)

        # Best / latest checkpoints, selected on ``cfg.select_metric``.
        if cfg.save_best and is_improvement(score, best_score, cfg.select_metric):
            best_score = score
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            patience_counter = 0
            do_checkpoint("best", epoch, epoch + 1)
        else:
            patience_counter += 1
            logger.info("No improvement in %s (%.4f). Patience %d/%d",
                        cfg.select_metric, score, patience_counter, cfg.early_stopping_patience)

        if cfg.save_latest:
            do_checkpoint("latest", epoch, epoch + 1)

        # Prune periodic step checkpoints.
        if cfg.keep_last_n > 0:
            _prune_step_checkpoints(ckpt_dir, cfg.keep_last_n)

        if patience_counter >= cfg.early_stopping_patience:
            logger.warning("Early stopping at epoch %d", epoch)
            break

        if stopped_early or (args.max_steps and progress["global_step"] >= args.max_steps):
            logger.warning("Reached --max-steps=%d; stopping.", args.max_steps)
            break

    logger.info("Training finished. Best %s: %.4f (val_loss %.4f)",
                cfg.select_metric, best_score, best_val_loss)
    with open(os.path.join(exp_dir, "final_report.json"), "w") as fh:
        json.dump({"best_score": best_score, "select_metric": cfg.select_metric,
                   "best_val_loss": best_val_loss, "best_val_metrics": best_val_metrics,
                   "init_source": init_source, "git_commit": git_sha,
                   "total_params": total_p, "trainable_params": train_p,
                   "dataset_fingerprint": dataset_fp,
                   "history": history, "config": cfg.to_dict()}, fh, indent=2, default=str)


def _prune_step_checkpoints(ckpt_dir: str, keep: int):
    steps = sorted(
        p for p in os.listdir(ckpt_dir)
        if p.startswith("step_") and p.endswith(".pt")
    )
    for name in steps[:-keep] if len(steps) > keep else []:
        try:
            os.remove(os.path.join(ckpt_dir, name))
        except OSError:
            pass


if __name__ == "__main__":
    main()
