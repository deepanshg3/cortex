"""Regression metrics for Cortex.

All metrics are computed on the *original* pKd scale (μM-free, log10 units)
after any target normalisation has been inverted. The target is
``pkd = 9 - log10(Kd_nM)`` (see docs/DATA_AND_EVALUATION.md), a dimensionless
log-affinity in units of log10(M) offset by 9.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

try:  # scipy is a hard dependency, but keep a graceful fallback.
    from scipy.stats import pearsonr, spearmanr

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if _HAVE_SCIPY:
        return float(pearsonr(y_true, y_pred)[0])
    if y_true.std() == 0 or y_pred.std() == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if _HAVE_SCIPY:
        return float(spearmanr(y_true, y_pred)[0])
    # Rank-based fallback (average ranks for ties).
    from scipy.stats import rankdata  # noqa: F401  (only reached if scipy present)

    rt = rankdata(y_true)
    rp = rankdata(y_pred)
    return float(np.corrcoef(rt, rp)[0, 1])


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Return RMSE, MAE, Pearson r, Spearman rho and R².

    Parameters
    ----------
    y_true, y_pred : array-like of the same length
        Values on the original pKd scale.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("empty inputs")

    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    pearson = _pearson(y_true, y_pred) if y_true.size > 1 else float("nan")
    spearman = _spearman(y_true, y_pred) if y_true.size > 1 else float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "mse": mse,
        "pearson": pearson,
        "spearman": spearman,
        "r2": r2,
        "n": int(y_true.size),
    }


METRIC_KEYS = ("rmse", "mae", "pearson", "spearman", "r2")
