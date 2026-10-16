"""Fast unit tests for the training loop (no encoders, no download).

A stub model/loader exercises finite-loss enforcement, the step callback,
grad accumulation and the mid-epoch ``--max-steps`` cutoff.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from config import Config
from src.train import assert_finite, train_one_epoch


def _fake_loader(n_batches, batch=2, seq=4):
    for _ in range(n_batches):
        yield {
            "drug_input_ids": torch.zeros(batch, seq, dtype=torch.long),
            "drug_attention_mask": torch.ones(batch, seq, dtype=torch.long),
            "target_input_ids": torch.zeros(batch, seq, dtype=torch.long),
            "target_attention_mask": torch.ones(batch, seq, dtype=torch.long),
            "label": torch.full((batch,), 1.0),
        }


class _ConstModel(nn.Module):
    def __init__(self, value=1.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))
        self.value = value

    def forward(self, d, dm, t, tm):
        return torch.full((d.size(0),), self.value, dtype=self.w.dtype) + self.w


class _NanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))

    def forward(self, d, dm, t, tm):
        return torch.full((d.size(0),), float("nan"))


def _run(model, n_batches=4, accum=1, max_steps=None, on_step=None, hard_mining_start=999):
    cfg = Config(device="cpu", amp_dtype="fp32", grad_accum_steps=accum,
                 hard_mining_start_epoch=hard_mining_start, log_every_steps=0,
                 grad_clip_norm=1.0, learning_rate=1e-3)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    crit = nn.MSELoss(reduction="none")
    progress = {"global_step": 0, "micro_step": 0}
    return train_one_epoch(model, _fake_loader(n_batches), opt, None, None, crit,
                           cfg, "cpu", 1, progress, on_step=on_step, max_steps=max_steps), progress


def test_assert_finite_raises():
    with pytest.raises(FloatingPointError):
        assert_finite(torch.tensor(float("nan")), "x")
    with pytest.raises(FloatingPointError):
        assert_finite(torch.tensor(float("inf")), "x")
    assert_finite(torch.tensor(1.0), "x")  # no raise


def test_nonfinite_loss_fails_fast():
    with pytest.raises(FloatingPointError):
        _run(_NanModel())


def test_on_step_and_max_steps():
    steps = []
    (mean_loss, stopped), progress = _run(
        _ConstModel(), n_batches=5, max_steps=2, on_step=steps.append
    )
    assert stopped is True
    assert progress["global_step"] == 2
    assert steps == [1, 2]


def test_grad_accumulation_reduces_optimizer_steps():
    steps = []
    (_, stopped), progress = _run(
        _ConstModel(), n_batches=4, accum=2, on_step=steps.append
    )
    assert progress["global_step"] == 2  # 4 micro-batches / accum 2
    assert steps == [1, 2]


def test_returns_mean_loss_tuple():
    (mean_loss, stopped), progress = _run(_ConstModel(value=1.0), n_batches=3)
    assert stopped is False
    assert mean_loss == pytest.approx(0.0, abs=1e-6)  # label 1.0, pred 1.0
    assert progress["global_step"] == 3
