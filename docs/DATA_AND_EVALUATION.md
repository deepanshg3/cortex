# Data and Evaluation

## Provenance

The dataset is derived from TDC's `BindingDB_Kd` benchmark via
`src/download_data.py`. For every (drug, target) pair, `Y` is the dissociation
constant **Kd in nanomolar units**. The training label is the standard pKd
transform:

```
pkd = 9 - log10(Y)          # Y = Kd in nM  →  pkd is a log10(M) affinity + 9
```

This identity is verified for every row by `src/data_audit.py`
(`pkd_matches_9_minus_log10_Y: True`). Rows with `Y <= 0` are pathological; in
the current train split there are **2** such rows (they map to the sentinel
`pkd = 19.0`). A methyl-free, unit-consistent label makes RMSE directly
interpretable: 1.0 RMSE = one order of magnitude of affinity error.

## Processed splits

`data/processed/{train,val,test}.parquet`, DVC-tracked (see `.dvc` pointers).

Columns: `Drug_ID, Drug, Target_ID, Target, Y, Drug_Length, Target_Length, pkd`.

| Split | Rows | pkd mean | pkd std | min | max |
|-------|------|----------|---------|-----|-----|
| train | 31,371 | 5.8908 | 1.4256 | 2.00 | 19.00 |
| val   | 5,536  | 5.9130 | 1.4326 | 2.00 | 12.39 |
| test  | 4,202  | 5.7513 | 1.3412 | 2.00 | 12.10 |

## Integrity and leakage audit

Run the read-only audit (never modifies data):

```bash
python -m src.data_audit --data-dir data/processed          # human-readable
python -m src.data_audit --data-dir data/processed --json   # machine-readable
```

Findings on the shipped splits:

| Quantity | Value | Interpretation |
|----------|-------|----------------|
| train↔test **target** overlap | 0 | cold-target test split — no protein seen in training |
| train↔test **drug** overlap | 736 / 1,511 | ~49% of test drugs appear in train |
| train↔val **pair** overlap | 769 | validation pairs leak from train — validation is optimistic |
| train **duplicate (drug,target) pairs** | 2,815 | same pair, sometimes differing `Y` |
| test↔train pair overlap | 0 | |

Implications:

* **Test is the trustworthy split** (zero target leakage). Report test metrics.
* **Validation is optimistically biased** because 769 of its pairs are also in
  train. Use it for model selection/early stopping, not for headline numbers.
* The 2,815 duplicated train pairs should be deduplicated (e.g. mean `Y`) before
  a final training run to avoid double-weighting and label noise.
* Because ~49% of test drugs are seen in training, generalisation to *novel
  targets* (the harder, more realistic setting) is not measured by the current
  test split. Consider a "cold-drug / cold-target" split for stronger claims.

## Metrics

`src/metrics.regression_metrics` returns `rmse, mae, mse, pearson, spearman, r2,
n`, always on the **original pKd scale** (target normalisation is inverted first).

* `rmse`/`mae` — magnitude of error in log10(M) affinity units.
* `pearson` — linear correlation.
* `spearman` — rank correlation; robust to monotone miscalibration.
* `r2` — fraction of variance explained (can be negative).

## Evaluating a checkpoint

`src/evaluate.py` accepts either a **full** Cortex checkpoint or a **weights-only**
`state_dict`, and never writes to the checkpoint:

```bash
# Full checkpoint on the test split (uses the checkpoint's own config)
python -m src.evaluate --checkpoint experiments/baseline/checkpoints/best.pt --split test

# Original weights-only checkpoint
python -m src.evaluate --checkpoint checkpoints/best_model_2.pt --split test

# Reproduce a legacy checkpoint that was trained with unmasked mean pooling
python -m src.evaluate --checkpoint checkpoints/best_model_2.pt --split test \
    --set pooling=mean

# Machine-readable output
python -m src.evaluate --checkpoint checkpoints/best_model_2.pt --split test \
    --json-out reports/best_model_2_test.json
```

### Reference numbers on the test split

`checkpoints/best_model_2.pt` (weights-only, full 4,202-row test split):

| pooling | rmse | mae | pearson | spearman | r2 |
|---------|------|-----|---------|----------|----|
| `mean` (legacy, faithful) | **1.1502** | **0.8128** | **0.5877** | **0.5375** | **0.2645** |
| `masked` (corrected default) | 1.1988 | 0.8613 | 0.5801 | 0.5159 | 0.2010 |

The legacy weights were *trained* with unmasked pooling, so `pooling=mean`
reproduces their intended behaviour. New runs should train with the corrected
`pooling="masked"` default.
