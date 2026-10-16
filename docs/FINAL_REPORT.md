# Cortex Overhaul — Final Report

All commands assume the repo root and the project venv:

```bash
cd /home/deepanshgupta/Desktop/cortex && source venv/bin/activate
```

## 1. Bugs fixed

| # | Bug | Fix | Evidence |
|---|-----|-----|----------|
| 1 | AMP `NaN`: MoLFormer linear attention overflows fp16 | drug tower forced to fp32 (`drug_tower_fp32=True`) | finite loss + finite grads |
| 2 | `GradScaler` init scale `65536` > fp16 max `65504` poisoned every grad | `amp_init_scale=4096.0` | grad norm ≈ 11.6, finite |
| 3 | `loss.backward()` bypassed the scaler | `scaler.scale(loss).backward()` + `scaler.unscale_()` before clipping | resume/AMP tests |
| 4 | Masked mean pooling over compressed protein positions | `pooling="masked"` default; `pooling="mean"` reproduces legacy | see §3 |
| 5 | Sampler RNG state crash on `torch.Generator` | `_capture_sampler_state` | resume test |
| 6 | `manifest.json` written to wrong directory | manifest now in `checkpoints/` | resume/refuse flows |
| 7 | Weights-only checkpoint load failed on arch mismatch | `load_model_state` returns missing/unexpected/mismatched, skips incompatible | partial-load test |
| 8 | `--json-out` failed on missing parent dir | parent dir auto-created | nested-path run |

## 2. Reproducibility evidence (test split, n = 4,202)

`checkpoints/best_model_2.pt` is an exact architectural match for the current
model (0 missing / 0 unexpected / 0 mismatched keys).

```bash
python -m src.evaluate --checkpoint checkpoints/best_model_2.pt --split test            # pooling=masked
python -m src.evaluate --checkpoint checkpoints/best_model_2.pt --split test --set pooling=mean
```

| pooling | rmse | mae | pearson | spearman | r2 |
|---------|------|-----|---------|----------|----|
| `mean` (legacy, faithful) | 1.1502 | 0.8128 | 0.5877 | 0.5375 | 0.2645 |
| `masked` (corrected default) | 1.1988 | 0.8613 | 0.5801 | 0.5159 | 0.2010 |

The legacy weights were trained with unmasked pooling, so `pooling=mean`
reproduces their intended behaviour. `checkpoints/best_model.pt` is an older
architecture (LoRA r=8, q/k/v only, 2-layer head, no compressor) and loads
partially (backbones only).

## 3. Tests

```bash
python -m pytest -m "not slow" -q     # 72 passed
python -m pytest -m "slow" -q         # 9 passed  (includes GPU AMP regression)
python -m pytest -q                   # 81 passed
```

Coverage: label transform (`pkd = 9 - log10 Y`), data integrity + leakage,
tokenisation/compression, forward finiteness (CPU + GPU AMP), backward grads,
checkpoint atomicity + round-trip + compatibility, resume equivalence,
fresh-run clearing, best/latest, metrics, partial checkpoint load.

## 4. CLI quick reference

```bash
# Fresh run
python -m src.train --fresh-run --experiment-name baseline --output-dir experiments

# Resume after a Kaggle reset
python -m src.train --resume auto --experiment-name baseline --output-dir experiments

# Warm-start from original weights (fresh optimizer)
python -m src.train --init-weights checkpoints/best_model_2.pt \
    --fresh-run --experiment-name from_weights --output-dir experiments

# Smoke test (< 1 min)
python -m src.train --fresh-run --experiment-name smoke \
    --epochs 2 --limit-train 16 --limit-val 8 --max-steps 4 --num-workers 0

# Data audit
python -m src.data_audit --data-dir data/processed --json
```

## 5. Deliverables

* `src/checkpoint.py`, `src/metrics.py`, `src/data_audit.py` (new)
* `src/train.py`, `src/models.py`, `src/tokenization.py`, `src/evaluate.py`,
  `config.py` (overhauled)
* `tests/` + `pytest.ini` (81 tests)
* `docs/ARCHITECTURE.md`, `docs/TRAINING_METHODS.md`, `docs/DATA_AND_EVALUATION.md`,
  `docs/CHECKPOINTS_AND_RECOVERY.md`, `docs/EXPERIMENTS.md`,
  `docs/OPTIMIZATION_GUIDE.md`

Original artifacts `checkpoints/best_model.pt` and
`checkpoints/best_model_2.pt` are untouched; new runs write under
`experiments/<name>/`.

## 6. Open items / risks

* **Validation leakage**: 769 train pairs also appear in val — val is optimistic;
  trust test metrics.
* **Duplicate train pairs**: 2,815 duplicated (drug,target) pairs; deduplicate
  before a final headline run.
* **Cold-target only**: test has zero target overlap but ~49% drug overlap;
  add a cold-drug split for stronger generalisation claims.
* **No preprocessing script in repo**: split generation is a reproducibility gap.
* **Kaggle**: confirm HF cache/internet access and DVC remote credentials before
  launching; persist `experiments/` between sessions for resume.
