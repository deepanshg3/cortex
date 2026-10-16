"""Tests for the regression metric suite (src.metrics)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.metrics import METRIC_KEYS, regression_metrics


def test_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = regression_metrics(y, y.copy())
    assert m["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert m["mae"] == pytest.approx(0.0, abs=1e-12)
    assert m["r2"] == pytest.approx(1.0, abs=1e-12)
    assert m["pearson"] == pytest.approx(1.0, abs=1e-9)
    assert m["spearman"] == pytest.approx(1.0, abs=1e-9)
    assert m["n"] == 5


def test_known_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([4.0, 3.0, 2.0, 1.0])
    m = regression_metrics(y_true, y_pred)
    # errors: 3, 1, -1, -3 -> mse = (9+1+1+9)/4 = 5
    assert m["mse"] == pytest.approx(5.0)
    assert m["rmse"] == pytest.approx(math.sqrt(5.0))
    assert m["mae"] == pytest.approx(2.0)
    # perfect anti-correlation
    assert m["pearson"] == pytest.approx(-1.0, abs=1e-9)


def test_metric_keys_present():
    m = regression_metrics([1, 2, 3], [1, 2, 3])
    for k in METRIC_KEYS:
        assert k in m
    assert "mse" in m and "n" in m


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        regression_metrics([1, 2, 3], [1, 2])


def test_empty_raises():
    with pytest.raises(ValueError):
        regression_metrics([], [])


def test_single_sample_correlation_nan():
    m = regression_metrics([3.0], [2.0])
    assert m["rmse"] == pytest.approx(1.0)
    assert math.isnan(m["pearson"])
    assert math.isnan(m["spearman"])
    assert math.isnan(m["r2"])  # ss_tot == 0


def test_spearman_monotonic_nonlinear():
    # Non-linear but strictly monotone -> Spearman = 1, Pearson < 1.
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.exp(y_true)
    m = regression_metrics(y_true, y_pred)
    assert m["spearman"] == pytest.approx(1.0, abs=1e-9)
    assert m["pearson"] < 1.0
