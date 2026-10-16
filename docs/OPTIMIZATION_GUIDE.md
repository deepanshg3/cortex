# Optimization and Runtime Guide

Target hardware: a single **Kaggle T4** (16 GB, sm_75, **no bf16**) and, for
development, a 4 GB RTX 3050. The defaults in `config.Config` are chosen to run
on a T4. This guide lists the levers, in the order you should reach for them.

## 1. Numerical stability (do not touch these first)

* `drug_tower_fp32=True` (default) runs MoLFormer in fp32 inside AMP. Disabling
  it causes `NaN` (see `docs/ARCHITECTURE.md`). The cost is roughly one tower's
  worth of fp32 compute; the protein tower and head stay in fp16.
* `amp_init_scale=4096.0` (default). Never raise it toward the fp16 ceiling
  (65504); `65536` is exactly what poisons gradients.
* `amp_dtype="fp16"` on T4. Use `"bf16"` only on Ampere+ (RTX 30/40, A100, L4);
  bf16 needs no loss scaling and has no overflow issue.
* Non-finite loss always raises. If you see it, it is a real bug, not noise.

## 2. Memory

The dominant cost is the **protein sequence length** and the MoLFormer fp32
tower.

| Lever | Effect |
|-------|--------|
| `batch_size` (dyn. padding trims to batch max) | linear in peak memory |
| `grad_accum_steps` | keeps the *effective* batch large while lowering peak memory |
| `max_target_len` (default 1000) | quadratic in the compressor/attention only after ÷4; large proteins dominate |
| `gradient_checkpointing_enable()` | already enabled in `BindingModel.__init__` for both towers — trades ~20-30% speed for a large activation-memory saving |
| `finetune_mode="full"` | avoid on T4; keeps encoder activations for all params |
| `eval_batch_size` (default 64) | set to 128/256 at inference; no gradients |

If you still OOM: halve `batch_size` and double `grad_accum_steps` (same
effective batch), then reduce `max_target_len`.

## 3. Throughput

* **Dynamic padding** (already on): each batch is padded to its own max length,
  so short sequences stay cheap. Sorting/bucketing by length is a possible
  future speedup.
* `num_workers`: 2–4 on Kaggle (CPU cores are limited); `0` for debugging.
* `pin_memory=True` (default) overlaps host→device copies.
* Keep the drug tower in fp32 (required) but everything else in fp16.
* `torch.compile` is **not** enabled by default; it can help the projection/
  attention/head but may recompile on varying sequence lengths. Test with a
  fixed `max_target_len` before adopting it.
* Logging every `log_every_steps` (50) avoids per-step host syncs.

## 4. Optimisation quality

* `learning_rate=1e-4` with `warmup_ratio=0.05` and cosine decay is a solid
  default for LoRA fine-tuning.
* `weight_decay=0.05`, `grad_clip_norm=1.0`.
* `loss_name="huber"` (`huber_delta=1.0`) is more robust to the duplicate-pair
  label noise measured by the data audit than plain MSE.
* `normalize_target=True` stabilises training with Huber; metrics are always
  reported on the unnormalised pKd scale.
* OHEM (`hard_mining_start_epoch`, `hard_mining_fraction`) focuses on hard pairs
  once a baseline exists.
* `early_stopping_patience=3` (validation loss) stops unproductive runs; the
  best checkpoint is preserved regardless.

## 5. Checkpoint I/O

* `save_every_steps=0` writes only at epoch boundaries (best + latest).
  Periodic step checkpoints (`save_every_steps>0`) are pruned to `keep_last_n`
  to bound disk use.
* Saves are atomic (`docs/CHECKPOINTS_AND_RECOVERY.md`), so frequent saving is
  safe but costs time proportional to checkpoint size.
* On Kaggle, point `--output-dir` at `/kaggle/working/experiments` so artifacts
  survive the session; commit/persist between sessions for true crash recovery.

## 6. Suggested T4 profiles

| Profile | Settings |
|---------|----------|
| Safe baseline | `batch_size=32`, `grad_accum_steps=1`, `fp16`, `max_target_len=1000` |
| Low memory | `batch_size=16`, `grad_accum_steps=2`, `max_target_len=768` |
| Throughput (inference) | `eval_batch_size=256`, `fp16`, `num_workers=4` |
