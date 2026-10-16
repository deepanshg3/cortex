# Checkpoints and Recovery

Cortex supports **full-state resumable training**: a checkpoint contains
everything needed to continue a run exactly where it stopped, even after the
machine is reset (e.g. a Kaggle notebook restart).

## Two on-disk formats

| `checkpoint_type` | Contents | Written by |
|-------------------|----------|------------|
| `full` | model + optimizer + scheduler + AMP scaler + RNG states + sampler state + counters + config + provenance | `src.train` |
| `weights_only` | a plain `state_dict` | legacy scripts / manual saves |

`checkpoints/best_model.pt` and `checkpoints/best_model_2.pt` are
`weights_only`. They carry **no** epoch, optimizer, or scheduler information.
Cortex never assigns them a fabricated epoch and never overwrites them.

## Atomic, crash-safe saves

`src.checkpoint.atomic_torch_save`:

1. writes to `path.tmp-<pid>`,
2. `flush()` + `os.fsync()`,
3. hard-links the previous valid file to `path.bak` (copy fallback),
4. `os.replace(tmp, path)` (atomic on POSIX),
5. fsyncs the directory.

An interrupted save therefore cannot corrupt the last good checkpoint; the
previous state remains available as `<name>.bak`.

## Files written per experiment

```
experiments/<experiment_name>/
├── config.json           # fully-resolved config used for the run
├── metrics.jsonl         # one JSON record per epoch (append-only)
├── final_report.json     # best val loss/metrics at the end
└── checkpoints/
    ├── manifest.json     # tag -> path, latest, best, epoch, global_step
    ├── best.pt           # best validation loss so far
    ├── latest.pt         # most recent epoch
    └── step_0000001.pt   # periodic (keep_last_n retained)
```

`manifest.json` is what `--resume auto` consults first.

## Resume modes (explicit; never silently fall back)

| Flag | Behaviour |
|------|-----------|
| `--resume auto` *(default)* | resume `latest` if a full checkpoint exists, else start fresh |
| `--resume none` | never resume |
| `--resume PATH` | resume a specific full checkpoint (error if missing) |
| `--init-weights PATH` | load **model weights only** (fresh optimizer/scheduler/epoch); works with weights-only files |
| `--fresh-run` | start a brand-new run, discarding existing checkpoints for the experiment |

Safety rules enforced by `src.train.main`:

* Starting over an existing experiment (a `checkpoints/manifest.json` exists)
  without a checkpoint to resume from is **refused** unless `--fresh-run` is
  given — no accidental overwrite of results.
* `--fresh-run` combined with an explicit `--resume PATH` is rejected.
* `--resume PATH` that does not exist raises a clear error.

When a checkpoint is resumed, Cortex:

1. validates the checkpoint against the current config
   (`compatibility_report` — architectural fields must match),
2. loads model, optimizer, scheduler and AMP-scaler state,
3. restores Python/NumPy/Torch/CUDA RNG and the DataLoader generator state,
4. restores target-normalisation statistics, best-val loss, patience and history,
5. continues from `next_epoch`.

## Typical workflows

```bash
# 1. Start a fresh run
python -m src.train --fresh-run --experiment-name baseline \
    --output-dir experiments --epochs 20 --batch-size 32

# 2. After a Kaggle reset, continue the same run (resumes optimizer + RNG)
python -m src.train --resume auto --experiment-name baseline \
    --output-dir experiments --epochs 20

# 3. Warm-start from the original weights with a fresh optimizer
python -m src.train --init-weights checkpoints/best_model_2.pt \
    --fresh-run --experiment-name from_weights --output-dir experiments

# 4. Start an existing experiment over from scratch
python -m src.train --fresh-run --experiment-name baseline --output-dir experiments
```

### Kaggle notes

* Copy `experiments/` (or at least `checkpoints/`) to persistent storage
  (`/kaggle/working` or a Dataset) between sessions; the runner will pick up
  `latest.pt` automatically with `--resume auto`.
* The T4 is sm_75 with **no bf16**. Keep `amp_dtype=fp16` (default) and the
  `amp_init_scale=4096` default; `drug_tower_fp32=True` is required for
  numerical stability (see `docs/ARCHITECTURE.md`).
* If a session is interrupted mid-epoch, `step_*.pt` files plus `latest.pt`
  limit re-work; set `--set save_every_steps=200` to tune granularity and
  `keep_last_n` to control disk usage.

## Non-finite values fail fast

* A non-finite **loss** raises `FloatingPointError` immediately — silent
  skipping is never allowed.
* Without AMP, a non-finite **gradient norm** also raises. With AMP, GradScaler
  owns this: the step is skipped and the scale reduced, and a warning is logged.
