"""Checkpoint format, atomicity, compatibility and resume-resolution tests."""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from config import Config, load_config
from src import checkpoint as C


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def _make_bundle(tmp_path):
    model = ToyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    scaler = torch.amp.GradScaler("cpu")
    cfg = load_config(output_dir=str(tmp_path), experiment_name="unit")
    return model, opt, sched, scaler, cfg


def test_atomic_save_preserves_previous_as_bak(tmp_path):
    path = str(tmp_path / "ck.pt")
    C.atomic_torch_save({"v": 1}, path)
    assert os.path.exists(path)
    C.atomic_torch_save({"v": 2}, path)
    assert torch.load(path)["v"] == 2
    # Previous contents preserved.
    assert os.path.exists(path + ".bak")
    assert torch.load(path + ".bak")["v"] == 1
    # No leftover temp files.
    assert not any(n.startswith("ck.pt.tmp-") for n in os.listdir(tmp_path))


def test_full_checkpoint_roundtrip(tmp_path):
    model, opt, sched, scaler, cfg = _make_bundle(tmp_path)
    # Run one step so optimizer state is populated.
    loss = model(torch.randn(3, 4)).sum()
    loss.backward()
    opt.step()
    path = str(tmp_path / "full.pt")
    state = C.build_checkpoint(
        model=model, optimizer=opt, scheduler=sched, scaler=scaler,
        epoch=2, next_epoch=3, global_step=10, micro_step=0,
        best_val_loss=1.25, best_val_metrics={"rmse": 1.1},
        patience_counter=1, config=cfg, dataset_fp="deadbeef", sampler=None,
        history=[{"epoch": 1}],
    )
    C.save_checkpoint(path, state)

    loaded = C.load_checkpoint(path)
    assert C.is_full_checkpoint(loaded)
    assert loaded["epoch"] == 2 and loaded["next_epoch"] == 3
    assert loaded["global_step"] == 10
    assert loaded["best_val_loss"] == pytest.approx(1.25)
    assert loaded["dataset_fingerprint"] == "deadbeef"
    assert loaded["config"]["experiment_name"] == "unit"

    # Weights round-trip exactly.
    for k, v in model.state_dict().items():
        assert torch.equal(loaded["model_state"][k], v)


def test_weights_only_detection(tmp_path):
    model = ToyModel()
    path = str(tmp_path / "weights.pt")
    torch.save(model.state_dict(), path)
    loaded = C.load_checkpoint(path)
    assert loaded["checkpoint_type"] == "weights_only"
    assert not C.is_full_checkpoint(loaded)


def test_load_model_state_skips_incompatible(tmp_path):
    src = ToyModel()
    path = str(tmp_path / "weights.pt")
    torch.save(src.state_dict(), path)
    loaded = C.load_checkpoint(path)

    class Bigger(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 7)  # different out_features
            self.extra = nn.Linear(2, 2)

    bigger = Bigger()
    report = C.load_model_state(bigger, loaded, strict=False)
    assert "fc.weight" in report["mismatched"]
    assert "extra.weight" in report["missing"]
    # strict=True must raise.
    with pytest.raises(RuntimeError):
        C.load_model_state(Bigger(), loaded, strict=True)


def test_resolve_resume_modes(tmp_path):
    ckpt_dir = str(tmp_path / "checkpoints")
    os.makedirs(ckpt_dir)
    assert C.resolve_resume("none", ckpt_dir) is None
    assert C.resolve_resume("auto", ckpt_dir) is None

    latest = os.path.join(ckpt_dir, "latest.pt")
    torch.save({"checkpoint_type": "full"}, latest)
    C.write_manifest(ckpt_dir, {"latest": latest})
    assert C.resolve_resume("auto", ckpt_dir) == latest

    with pytest.raises(FileNotFoundError):
        C.resolve_resume(str(tmp_path / "nope.pt"), ckpt_dir)


def test_fallback_to_best_when_no_manifest(tmp_path):
    ckpt_dir = str(tmp_path / "c")
    os.makedirs(ckpt_dir)
    best = os.path.join(ckpt_dir, "best.pt")
    torch.save({"checkpoint_type": "full"}, best)
    assert C.resolve_resume("auto", ckpt_dir) == best


def test_compatibility_report_detects_arch_mismatch(tmp_path):
    _, _, _, _, cfg = _make_bundle(tmp_path)
    saved = cfg.to_dict()
    saved["hidden_dim"] = 512
    ckpt = {
        "checkpoint_type": "full", "format_version": C.FORMAT_VERSION,
        "model_state": {}, "optimizer_state": {}, "epoch": 1, "next_epoch": 2,
        "config": saved, "config_hash": "x",
    }
    rep = C.compatibility_report(ckpt, cfg)
    assert rep["compatible"] is False
    assert any("hidden_dim" in e for e in rep["errors"])


def test_compatibility_report_detects_pooling_mismatch(tmp_path):
    _, _, _, _, cfg = _make_bundle(tmp_path)
    saved = cfg.to_dict()
    saved["pooling"] = "mean" if cfg.pooling == "masked" else "masked"
    ckpt = {
        "checkpoint_type": "full", "format_version": C.FORMAT_VERSION,
        "model_state": {}, "optimizer_state": {}, "epoch": 1, "next_epoch": 2,
        "config": saved, "config_hash": "x",
    }
    rep = C.compatibility_report(ckpt, cfg)
    assert rep["compatible"] is False
    assert any("pooling" in e for e in rep["errors"])


def test_config_hash_stable_and_ignores_volatile():
    a = load_config(device="cpu", num_workers=0)
    b = load_config(device="cuda", num_workers=8)
    assert a.config_hash() == b.config_hash()
    c = load_config(hidden_dim=128)
    assert a.config_hash() != c.config_hash()


def test_apply_overrides_rejects_unknown():
    with pytest.raises(ValueError):
        from config import apply_overrides

        apply_overrides(Config(), {"not_a_field": 1})


def test_dataset_fingerprint_missing_and_stable(tiny_parquet):
    fp1 = C.dataset_fingerprint([tiny_parquet])
    fp2 = C.dataset_fingerprint([tiny_parquet])
    assert fp1 == fp2
    fp3 = C.dataset_fingerprint([tiny_parquet, "/definitely/missing.parquet"])
    assert fp3 != fp1
