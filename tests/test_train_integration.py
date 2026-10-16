"""Slow end-to-end training integration tests (real encoders, CPU).

These instantiate MoLFormer-XL and ESM-2, so they are marked ``slow``. Run with
``pytest -m "not slow"`` to skip.
"""

from __future__ import annotations

import json
import os

import pytest

from src.train import main

pytestmark = pytest.mark.slow


def _base_args(out_dir, split_paths, exp, epochs):
    return [
        "--experiment-name", exp,
        "--output-dir", str(out_dir),
        "--epochs", str(epochs),
        "--batch-size", "2",
        "--limit-train", "4",
        "--limit-val", "2",
        "--num-workers", "0",
        "--set", "device=cpu",
        "--set", "amp_dtype=fp32",
        "--set", f"train_path={split_paths['train']}",
        "--set", f"val_path={split_paths['val']}",
        "--set", f"test_path={split_paths['test']}",
    ]


def test_fresh_run_then_resume(tmp_path, tiny_parquets):
    out = tmp_path / "exp"
    exp = "t1"

    main(_base_args(out, tiny_parquets, exp, epochs=1) + ["--fresh-run"])
    ckdir = out / exp / "checkpoints"
    assert (ckdir / "best.pt").exists()
    assert (ckdir / "latest.pt").exists()
    assert (out / exp / "metrics.jsonl").exists()
    assert (out / exp / "final_report.json").exists()

    manifest = json.loads((ckdir / "manifest.json").read_text())
    assert manifest["global_step"] == 2  # 4 samples / batch 2

    # Resume: continue to epoch 2 -> 2 more optimizer steps.
    main(_base_args(out, tiny_parquets, exp, epochs=2) + ["--resume", "auto"])
    manifest = json.loads((ckdir / "manifest.json").read_text())
    assert manifest["global_step"] == 4
    assert manifest["epoch"] == 2

    # metrics.jsonl should have one record per epoch.
    lines = [json.loads(l) for l in (out / exp / "metrics.jsonl").read_text().splitlines() if l]
    assert len(lines) == 2
    assert {ln["epoch"] for ln in lines} == {1, 2}


def test_fresh_run_clears_previous(tmp_path, tiny_parquets):
    out = tmp_path / "exp"
    exp = "t2"
    main(_base_args(out, tiny_parquets, exp, epochs=1) + ["--fresh-run"])
    ckdir = out / exp / "checkpoints"
    stale = ckdir / "step_0000001.pt"
    stale.write_bytes(b"stale")

    main(_base_args(out, tiny_parquets, exp, epochs=1) + ["--fresh-run"])
    assert not stale.exists()  # cleared
    assert (ckdir / "latest.pt").exists()


def test_resume_none_refuses_to_overwrite(tmp_path, tiny_parquets):
    out = tmp_path / "exp"
    exp = "t3"
    ckdir = out / exp / "checkpoints"
    ckdir.mkdir(parents=True)
    (ckdir / "manifest.json").write_text("{}")

    with pytest.raises(SystemExit):
        main(_base_args(out, tiny_parquets, exp, epochs=1) + ["--resume", "none"])


def test_fresh_run_conflicts_with_explicit_resume(tmp_path, tiny_parquets):
    out = tmp_path / "exp"
    with pytest.raises(SystemExit):
        main(_base_args(out, tiny_parquets, "t4", epochs=1)
             + ["--fresh-run", "--resume", str(tmp_path / "x.pt")])
