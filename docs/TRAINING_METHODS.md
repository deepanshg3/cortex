# Training Methods

All hyperparameters live in `config.Config` (single source of truth) and can be
overridden from the CLI. Precedence: dataclass defaults → `--set key=value` →
dedicated flags (`--epochs`, `--lr`, …), with dedicated flags winning.

## Fine-tuning: LoRA by default

Both encoders are frozen and adapted with LoRA (`finetune_mode="lora"`):

```python
LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1,
           target_modules=["query","key","value","dense"],
           bias="none", task_type="FEATURE_EXTRACTION")
```

Only ~3.2 M of 55 M parameters are trainable. This fits a T4 and reduces
catastrophic forgetting. `finetune_mode="full"` unfreezes the encoders (memory
heavy; not recommended on a T4).

`freeze_encoder_epochs=N` (optional) keeps the encoders frozen for the first `N`
epochs before LoRA gradients are enabled — a gentle schedule for noisy labels.
Use `BindingModel.set_encoders_trainable(False/True)`.

## Optimiser and schedule

* **Optimiser:** `AdamW` (`optimizer_name="adamw"`), `lr=1e-4`,
  `weight_decay=0.05`, `betas=(0.9,0.999)`, `eps=1e-8`.
* **Schedule:** linear warmup (`warmup_ratio=0.05` of total steps, or an
  explicit `warmup_steps`) then cosine decay to `eta_min=1e-6`.
  `scheduler_name="none"` disables it.
* **Gradient clipping:** `clip_grad_norm=1.0` on trainable parameters.

## Effective batch size

`effective_batch = batch_size * grad_accum_steps`. Gradient accumulation lets a
small GPU emulate a large batch; the loss is divided by `grad_accum_steps`
before backward and the optimizer steps once every `grad_accum_steps`
micro-batches.

## Losses

`loss_name` ∈ {`mse` (default), `huber`, `smoothl1`}. Huber is robust to the
heavy tails and duplicate-pair label noise in BindingDB (`huber_delta=1.0`).

## Online hard-example mining (OHEM)

From `hard_mining_start_epoch` onward, only the hardest `hard_mining_fraction`
of each batch contributes to the gradient:

```python
k = max(1, int(batch * hard_mining_fraction))
loss = torch.topk(per_sample_losses, k).values.mean()
```

Defaults: start at epoch 5, fraction 0.5. This focuses capacity on difficult
pairs once a baseline exists.

## Target normalisation

`normalize_target=True` standardises pKd using **train-split** mean/std
(computed automatically and stored in the checkpoint). Metrics are always
reported after inverting the transform, so numbers stay on the pKd scale. Useful
when switching to Huber and wanting unit-variance residuals.

## Mixed precision

* `amp_dtype` ∈ {`fp32`, `fp16` (default), `bf16`}. T4 → `fp16`.
* `drug_tower_fp32=True` runs MoLFormer in fp32 inside an otherwise-AMP graph
  (required; see `docs/ARCHITECTURE.md`).
* `amp_init_scale=4096` (not 65536) avoids the fp16 overflow of the scale factor.

## Recipe: single-GPU (T4) baseline

```bash
python -m src.train --fresh-run --experiment-name t4_baseline \
    --output-dir experiments \
    --epochs 20 --batch-size 32 --grad-accum-steps 1 \
    --lr 1e-4 --amp-dtype fp16 --loss mse \
    --set save_every_steps=200 --set keep_last_n 3 --set num_workers 2
```

Start with `--epochs 1 --limit-train 200 --limit-val 100 --max-steps 20` to
validate the pipeline before launching a full run.
