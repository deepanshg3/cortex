"""Staged, compute-conscious experiment campaign runner.

Runs a *sequence* of training experiments that share one baseline configuration,
reusing the existing full-state training/checkpoint infrastructure. Each
experiment gets its own experiment directory, so every run is independently
resumable and auditable. After each run the (validation-only) metrics are
appended to a campaign summary for comparison.

Design rules honoured here:
* experiments change **one factor at a time** relative to the baseline;
* model selection uses validation metrics only — the test split is never read;
* a run already having ``final_report.json`` is skipped unless ``--force``;
* commands can be inspected with ``--dry-run`` before spending compute.

Usage::

    # See every command without running anything
    python -m src.campaign --split-dir data/splits/v1 --protocol cold_target \
        --out-dir experiments --dry-run

    # Run only stage 1 (learning rate / warmup), one seed
    python -m src.campaign --split-dir data/splits/v1 --protocol cold_target \
        --stage stage1_lr --seed 42

    # Run everything
    python -m src.campaign --split-dir data/splits/v1 --protocol cold_target
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.logger import get_logger  # noqa: E402

logger = get_logger("src.campaign")


# ----------------------------------------------------------------------
# Campaign definition: baseline + staged single-factor experiments
# ----------------------------------------------------------------------
BASELINE: Dict = {
    "epochs": 20,
    "batch_size": 32,
    "grad_accum_steps": 1,
    "learning_rate": 1e-4,
    "weight_decay": 0.05,
    "dropout_rate": 0.1,
    "warmup_ratio": 0.05,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": "query,key,value,dense",
    "finetune_mode": "lora",
    "loss_name": "mse",
    "hard_mining_start_epoch": 5,
    "amp_dtype": "fp16",
    "select_metric": "rmse",
    "max_drug_len": 200,
    "max_target_len": 1000,
    "early_stopping_patience": 3,
}

# Each experiment overrides exactly one or two related baseline keys.
CAMPAIGN: List[Dict] = [
    # -- Stage 1: learning rate & warmup ---------------------------------
    {"stage": "stage1_lr", "name": "lr_1e-4", "hypothesis":
        "Baseline LR is near-optimal for LoRA fine-tuning.",
     "overrides": {}},
    {"stage": "stage1_lr", "name": "lr_3e-4", "hypothesis":
        "A larger LR may escape an under-fitting plateau.",
     "overrides": {"learning_rate": 3e-4}},
    {"stage": "stage1_lr", "name": "lr_3e-5", "hypothesis":
        "A smaller LR may stabilise LoRA adapters and reduce overfitting.",
     "overrides": {"learning_rate": 3e-5}},
    {"stage": "stage1_lr", "name": "warmup_0.10", "hypothesis":
        "Longer warmup helps early stability with AdamW.",
     "overrides": {"warmup_ratio": 0.10}},
    # -- Stage 2: loss ---------------------------------------------------
    {"stage": "stage2_loss", "name": "huber", "hypothesis":
        "Huber is more robust to the conflicting duplicate-pair labels.",
     "overrides": {"loss_name": "huber"}},
    {"stage": "stage2_loss", "name": "smoothl1", "hypothesis":
        "SmoothL1 behaves like a fixed-delta Huber and may be more stable.",
     "overrides": {"loss_name": "smoothl1"}},
    # -- Stage 3: LoRA capacity -----------------------------------------
    {"stage": "stage3_lora", "name": "lora_r8", "hypothesis":
        "Lower rank regularises the adapters (matching legacy capacity).",
     "overrides": {"lora_r": 8, "lora_alpha": 16}},
    {"stage": "stage3_lora", "name": "lora_r32", "hypothesis":
        "Higher rank increases adapter capacity for the larger clean split.",
     "overrides": {"lora_r": 32, "lora_alpha": 64}},
    {"stage": "stage3_lora", "name": "lora_dropout_0.2", "hypothesis":
        "More adapter dropout reduces overfitting.",
     "overrides": {"lora_dropout": 0.2}},
    {"stage": "stage3_lora", "name": "lora_qv_only", "hypothesis":
        "Adapting only q/v is a cheaper, lower-variance alternative.",
     "overrides": {"target_modules": "query,value"}},
    # -- Stage 4: regularisation ----------------------------------------
    {"stage": "stage4_reg", "name": "wd_0.1", "hypothesis":
        "Stronger weight decay reduces head overfitting.",
     "overrides": {"weight_decay": 0.1}},
    {"stage": "stage4_reg", "name": "wd_0.01", "hypothesis":
        "Weaker weight decay may under-regularise; test the other direction.",
     "overrides": {"weight_decay": 0.01}},
    {"stage": "stage4_reg", "name": "head_dropout_0.2", "hypothesis":
        "More head dropout reduces overfitting in the fusion MLP.",
     "overrides": {"dropout_rate": 0.2}},
    # -- Stage 5: frozen vs selective encoders ---------------------------
    {"stage": "stage5_freeze", "name": "frozen", "hypothesis":
        "Frozen encoders (feature extraction) may generalise better on "
        "cold-target data by preventing encoder drift.",
     "overrides": {"finetune_mode": "frozen"}},
    {"stage": "stage5_freeze", "name": "freeze_first2", "hypothesis":
        "Warming up on frozen encoders before enabling LoRA improves stability.",
     "overrides": {"freeze_encoder_epochs": 2}},
    # -- Stage 6: sampling / mining -------------------------------------
    {"stage": "stage6_sampling", "name": "ohem_off", "hypothesis":
        "OHEM may hurt when labels are noisy; ordinary sampling may be better.",
     "overrides": {"hard_mining_start_epoch": 999}},
    {"stage": "stage6_sampling", "name": "ohem_early", "hypothesis":
        "Earlier OHEM focuses capacity on hard cold-target pairs sooner.",
     "overrides": {"hard_mining_start_epoch": 2}},
    # -- Stage 7: fusion -------------------------------------------------
    {"stage": "stage7_fusion", "name": "concat", "hypothesis":
        "Simple concatenation fusion may match cross-attention at lower cost.",
     "overrides": {"fusion": "concat"}},
]

STAGE_ORDER = [
    "stage0_baseline", "stage1_lr", "stage2_loss", "stage3_lora",
    "stage4_reg", "stage5_freeze", "stage6_sampling", "stage7_fusion",
]


def load_spec(path: Optional[str]) -> List[Dict]:
    if not path:
        return CAMPAIGN
    with open(path) as fh:
        data = json.load(fh)
    return data.get("experiments", data)


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return str(v)


def build_command(python: str, exp: Dict, base: Dict, out_dir: str, campaign: str,
                  data_paths: Optional[Dict[str, str]], seed: int) -> List[str]:
    overrides = {**base, **exp.get("overrides", {})}
    if data_paths:
        overrides.update({
            "train_path": data_paths["train"],
            "val_path": data_paths["val"],
            "test_path": data_paths["test"],
        })
    overrides["seed"] = seed
    cmd = [python, "-m", "src.train", "--fresh-run",
           "--experiment-name", f"{campaign}/{exp['name']}",
           "--output-dir", out_dir]
    for k, v in sorted(overrides.items()):
        cmd += ["--set", f"{k}={_fmt(v)}"]
    return cmd


def _read_report(exp_dir: str) -> Optional[Dict]:
    p = os.path.join(exp_dir, "final_report.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def run_campaign(args) -> Dict:
    experiments = load_spec(args.spec)
    if args.stage:
        stages = set(args.stage)
        experiments = [e for e in experiments if e.get("stage") in stages]
    if args.only:
        wanted = set(args.only)
        experiments = [e for e in experiments if e["name"] in wanted]

    base = dict(BASELINE)
    if args.base_config and os.path.exists(args.base_config):
        with open(args.base_config) as fh:
            base.update(json.load(fh))
    for item in args.base_set:
        if "=" not in item:
            raise SystemExit(f"--base-set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        base[k] = v

    data_paths = None
    if args.split_dir and args.protocol:
        sub = os.path.join(args.split_dir, args.protocol)
        data_paths = {s: os.path.join(sub, f"{s}.parquet") for s in ("train", "val", "test")}
        for s, p in data_paths.items():
            if not os.path.exists(p):
                raise SystemExit(f"split file missing: {p}")

    campaign_dir = os.path.join(args.out_dir, args.campaign)
    os.makedirs(campaign_dir, exist_ok=True)
    summary_path = os.path.join(campaign_dir, "summary.jsonl")
    python = args.python or sys.executable

    logger.info("Campaign '%s': %d experiment(s) | base config %s",
                args.campaign, len(experiments), args.base_config or "builtin")
    results = []
    for exp in experiments:
        exp_dir = os.path.join(args.out_dir, args.campaign, exp["name"])
        cmd = build_command(python, exp, base, args.out_dir, args.campaign, data_paths, args.seed)

        if args.dry_run:
            logger.info("DRY-RUN [%s] %s\n    %s", exp.get("stage"), exp["name"], " ".join(cmd))
            continue

        if os.path.exists(os.path.join(exp_dir, "final_report.json")) and not args.force:
            report = _read_report(exp_dir)
            logger.info("[skip] %s already finished (best %s=%s)", exp["name"],
                        report.get("select_metric"), report.get("best_score"))
            results.append(_summarise(exp, report, 0.0))
            continue

        logger.info("=== Running [%s] %s ===", exp.get("stage"), exp["name"])
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        elapsed = time.time() - t0
        report = _read_report(exp_dir)
        rec = _summarise(exp, report, elapsed, returncode=proc.returncode)
        results.append(rec)
        with open(summary_path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        if proc.returncode != 0:
            logger.error("Experiment %s failed (rc=%d); stopping campaign.", exp["name"], proc.returncode)
            break

    if not args.dry_run:
        logger.info("Campaign done. Summary -> %s", summary_path)
    return {"campaign": args.campaign, "results": results, "summary_path": summary_path}


def _summarise(exp: Dict, report: Optional[Dict], elapsed: float, returncode: int = 0) -> Dict:
    m = (report or {}).get("best_val_metrics") or {}
    return {
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "stage": exp.get("stage"),
        "name": exp["name"],
        "hypothesis": exp.get("hypothesis"),
        "overrides": exp.get("overrides", {}),
        "returncode": returncode,
        "runtime_s": round(elapsed, 1),
        "select_metric": (report or {}).get("select_metric"),
        "best_score": (report or {}).get("best_score"),
        "best_val_loss": (report or {}).get("best_val_loss"),
        "val_rmse": m.get("rmse"), "val_mae": m.get("mae"),
        "val_pearson": m.get("pearson"), "val_spearman": m.get("spearman"),
        "val_r2": m.get("r2"),
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Cortex staged experiment campaign")
    ap.add_argument("--campaign", default="campaign_v1")
    ap.add_argument("--out-dir", default="experiments")
    ap.add_argument("--spec", default=None, help="JSON campaign spec (default builtin)")
    ap.add_argument("--base-config", default="configs/baseline.json")
    ap.add_argument("--base-set", action="append", default=[])
    ap.add_argument("--split-dir", default=None)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--stage", nargs="*", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--python", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_campaign(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
