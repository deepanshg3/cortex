# Running Experiments

## One-liners

```bash
# Fresh run
python -m src.train --fresh-run --experiment-name baseline --output-dir experiments

# Resume after interruption
python -m src.train --resume auto --experiment-name baseline --output-dir experiments

# Warm-start from the original weights
python -m src.train --init-weights checkpoints/best_model_2.pt \
    --fresh-run --experiment-name from_weights --output-dir experiments
```

## CLI reference

| Flag | Meaning |
|------|---------|
| `--resume {auto,none,PATH}` | resume mode (default `auto`) |
| `--init-weights PATH` | load model weights only |
| `--fresh-run` | start over, clearing the experiment directory |
| `--experiment-name NAME` | subdirectory under `--output-dir` |
| `--output-dir DIR` | root for experiment directories (default `experiments`) |
| `--checkpoint-dir DIR` | explicit checkpoint dir (overrides the derived path) |
| `--seed N` / `--epochs N` / `--batch-size N` / `--grad-accum-steps N` / `--lr F` | core hyperparameters |
| `--amp-dtype {fp32,fp16,bf16}` | mixed-precision mode |
| `--loss {mse,huber,smoothl1}` | training loss |
| `--normalize-target` | standardise pKd using train stats |
| `--hard-mining-start-epoch N` | OHEM start epoch |
| `--num-workers N` | DataLoader workers |
| `--save-every-steps N` | periodic step checkpoint interval |
| `--max-steps N` / `--limit-train N` / `--limit-val N` | smoke-test aids |
| `--set key=value` | any config field (repeatable; values coerced by type) |

Examples:

```bash
# Huber loss + gradient accumulation, OHEM from epoch 3
python -m src.train --fresh-run --experiment-name huber_a2 \
    --loss huber --grad-accum-steps 2 --hard-mining-start-epoch 3

# Match the legacy architecture exactly (LoRA r=8 on q/k/v) if desired
python -m src.train --fresh-run --experiment-name legacy_arch \
    --set lora_r=8 --set lora_alpha=16 \
    --set target_modules=query,key,value
```

## What each run records

Under `experiments/<name>/`:

* `config.json` — the fully resolved config (reproducible snapshot).
* `metrics.jsonl` — one record per epoch: `epoch, global_step, val_loss, rmse,
  mae, pearson, spearman, r2, lr, epoch_time_s, peak_gpu_gb, amp_dtype`.
* `final_report.json` — best val loss/metrics.
* `checkpoints/` — `best.pt`, `latest.pt`, `step_*.pt`, `manifest.json`.

`manifest.json` records `best`, `latest`, `epoch`, `global_step` and every step
checkpoint path, so a run can be resumed or audited without guessing filenames.

## Reproducibility

* **Seeds.** `ckpt_utils.seed_everything(seed)` seeds Python, NumPy, Torch and
  CUDA; the DataLoader uses a seeded `torch.Generator`. Seed defaults to 42.
* **RNG in checkpoints.** Full checkpoints store Python/NumPy/Torch/CUDA RNG and
  the DataLoader generator state, so a resumed run continues the same random
  stream.
* **Config hash.** `Config.config_hash()` is a SHA-256 (16 hex) over all
  *scientific* fields; volatile fields (`device`, paths, `num_workers`,
  `target_mean/std`, `experiment_name`) are excluded so a checkpoint validates
  across machines. Full checkpoints store this hash and are compatibility-checked
  on resume.
* **Dataset fingerprint.** `dataset_fingerprint` hashes file size + mtime + the
  first MiB of each split; it is stored in every checkpoint so you can detect
  when the data changed under a run.

## Comparing runs

```bash
# Per-epoch metrics for two runs
column -t -s, experiments/baseline/metrics.jsonl
diff <(jq -S . experiments/a/config.json) <(jq -S . experiments/b/config.json)
```

Because `metrics.jsonl` is append-only JSONL, it loads directly into pandas:

```python
import pandas as pd
df = pd.read_json("experiments/baseline/metrics.jsonl", lines=True)
print(df[["epoch", "val_loss", "rmse", "pearson", "r2"]])
```

## Smoke test before a long run

```bash
python -m src.train --fresh-run --experiment-name smoke \
    --epochs 2 --limit-train 16 --limit-val 8 --max-steps 4 \
    --num-workers 0 --set log_every_steps=1
```

This exercises the full pipeline (data → AMP → backward → checkpoint → resume)
in under a minute.
