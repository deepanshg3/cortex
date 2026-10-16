"""Model forward/masking tests plus slow GPU integration checks.

The pure-tensor tests run everywhere. The model-building tests are marked
``slow`` because they instantiate MoLFormer-XL and ESM-2 (weights are cached
after the first run) and ``gpu`` when they require CUDA.
"""

from __future__ import annotations

import pytest
import torch

from config import Config
from src.models import BindingModel


@pytest.mark.parametrize("L", list(range(4, 41)))
def test_compress_mask_matches_conv_length(L):
    mask = torch.ones(1, L, dtype=torch.long)
    comp = BindingModel.compress_mask(mask, 4)
    conv_out = (L - 4) // 4 + 1
    assert comp.shape[1] == conv_out


def test_compress_mask_preserves_ones():
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
    comp = BindingModel.compress_mask(mask, 4)
    assert comp.tolist() == [[1, 1]]


def test_compress_mask_identity_for_factor_one():
    mask = torch.tensor([[1, 0, 1]])
    assert torch.equal(BindingModel.compress_mask(mask, 1), mask)


def test_pooling_default_is_masked():
    assert Config().pooling == "masked"


@pytest.mark.slow
def test_forward_mean_pooling_legacy_is_finite(tiny_parquet):
    from src.tokenization import BindingDataset, dynamic_collate_fn

    cfg = Config(device="cpu", amp_dtype="fp32", num_workers=0, pooling="mean")
    ds = BindingDataset(tiny_parquet, cfg)
    batch = dynamic_collate_fn([ds[0], ds[1]])
    model = BindingModel(cfg)
    model.eval()
    with torch.no_grad():
        preds = model(
            batch["drug_input_ids"], batch["drug_attention_mask"],
            batch["target_input_ids"], batch["target_attention_mask"],
        )
    assert torch.isfinite(preds).all()


@pytest.mark.slow
def test_forward_is_finite_cpu(tiny_parquet, cfg):
    from src.tokenization import BindingDataset, dynamic_collate_fn

    ds = BindingDataset(tiny_parquet, cfg)
    batch = dynamic_collate_fn([ds[0], ds[1]])
    model = BindingModel(cfg)
    model.eval()
    with torch.no_grad():
        preds = model(
            batch["drug_input_ids"], batch["drug_attention_mask"],
            batch["target_input_ids"], batch["target_attention_mask"],
        )
    assert preds.shape == (2,)
    assert torch.isfinite(preds).all()


@pytest.mark.slow
def test_backward_produces_finite_grads_cpu(tiny_parquet, cfg):
    from src.tokenization import BindingDataset, dynamic_collate_fn

    ds = BindingDataset(tiny_parquet, cfg)
    batch = dynamic_collate_fn([ds[0], ds[1]])
    model = BindingModel(cfg)
    model.train()
    preds = model(
        batch["drug_input_ids"], batch["drug_attention_mask"],
        batch["target_input_ids"], batch["target_attention_mask"],
    )
    loss = torch.nn.functional.mse_loss(preds, batch["label"])
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None and p.requires_grad:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_amp_fp16_is_finite_on_gpu(tiny_parquet):
    """Regression test for the MoLFormer fp16 overflow (NaN) bug."""
    cfg = Config(device="cuda", amp_dtype="fp16", num_workers=0, batch_size=2,
                 amp_init_scale=4096.0, drug_tower_fp32=True)
    from src.tokenization import BindingDataset, dynamic_collate_fn

    ds = BindingDataset(tiny_parquet, cfg)
    batch = dynamic_collate_fn([ds[0], ds[1]])
    dev = "cuda"
    model = BindingModel(cfg).to(dev)
    crit = torch.nn.MSELoss(reduction="none")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=cfg.amp_init_scale)

    with torch.autocast("cuda", dtype=torch.float16, enabled=True):
        preds = model(
            batch["drug_input_ids"].to(dev), batch["drug_attention_mask"].to(dev),
            batch["target_input_ids"].to(dev), batch["target_attention_mask"].to(dev),
        )
        loss = crit(preds, batch["label"].to(dev)).mean()
    assert torch.isfinite(loss), "AMP loss is non-finite"

    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    bad = [n for n, p in model.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite grads under AMP in {bad[:5]}"
    scaler.step(opt)
    scaler.update()
