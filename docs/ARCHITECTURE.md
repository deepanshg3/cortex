# Architecture

Cortex is a dual-encoder drug–target binding-affinity (pKd) regressor. A SMILES
string and a protein FASTA are encoded independently, projected into a shared
space, and combined with cross-attention before a regression head.

```
SMILES  ──► MoLFormer-XL (+LoRA) ──► drug_proj (768→256) ──────────────┐
                                                                       │ key/value
FASTA   ──► ESM-2 8M      (+LoRA) ──► target_proj (320→256)            │
                        ──► Conv1d compressor (÷4) ─► protein queries ─┤
                                                                       ▼
                                            multi-head cross-attention (8 heads)
                                                                       │
                                            masked mean pool ──► MLP ──► pKd
```

Component summary (defaults from `config.Config`):

| Stage | Module | Shape |
|-------|--------|-------|
| Drug encoder | `ibm/MoLFormer-XL-both-10pct` (linear-attention transformer) | 768-d token states |
| Target encoder | `facebook/esm2_t6_8M_UR50D` | 320-d token states |
| Adapters | LoRA, `r=16`, `alpha=32`, `dropout=0.1`, targets `query,key,value,dense` | base frozen |
| Projections | `drug_proj` 768→256, `target_proj` 320→256 | 256-d |
| Compressor | `Conv1d(kernel=4, stride=4)` over the protein axis | seq/4 |
| Fusion | `nn.MultiheadAttention(embed_dim=256, heads=8)` — protein queries drug | 256-d |
| Pooling | masked mean over non-padded positions (`pooling="masked"`) | 256-d |
| Head | 3-layer GELU MLP: 256→256→128→1 | scalar pKd |

Parameter budget: **55,122,938 total / 3,236,097 trainable** (LoRA adapters +
projections + compressor + head). Only trainable parameters are optimised.

## Two correctness fixes relative to the original code

The original implementation lived in a single `models.py` and contained two
numerical/semantic problems. Both are addressed without adding parameters, so
existing weight files still load.

### 1. AMP fp16 overflow in MoLFormer (the "NaN" bug)

MoLFormer uses **Performer-style linear attention**: `MolformerFeatureMap` with
`num_random_features=32`, `feature_map_kernel="relu"` and
`linear_attention_eps=1e-06`. There is no softmax normaliser, so the attention
accumulation is an unnormalised sum. Under `torch.autocast(fp16)` the
`torch.matmul` calls are cast to fp16, whose maximum finite value is `65504`.
The unnormalised accumulation overflows to `inf` at encoder layer 0, and `inf`
propagates into the loss as `NaN`.

**Fix:** run the drug tower in fp32 while keeping AMP for the rest of the
network. `BindingModel._run_drug_tower` wraps the MoLFormer call in
`torch.autocast(device_type="cuda", enabled=False)` when `cfg.drug_tower_fp32`
is true (default). Verified: AMP-only produces NaN; the fp32 drug tower yields a
finite loss and finite gradients.

**Related AMP fix:** PyTorch's `GradScaler` default `init_scale=65536` exceeds
the fp16 max (`65504`), so the scale factor itself overflows when the scale
gradient first meets an fp16 autocast node — poisoning *every* parameter
gradient with NaN. `config.amp_init_scale = 4096.0` keeps the scale safely
inside fp16 range and the scaler adapts it during training.

### 2. Masked mean pooling

The compressor down-samples the protein axis by 4. The original code then
averaged the cross-attention output over **all** positions, including padding
introduced by the compressor, diluting the representation. The corrected
default (`pooling="masked"`) uses the compressed attention mask:

```python
mask   = compress_mask(target_attention_mask, factor).unsqueeze(-1)
pooled = (attn_output * mask).sum(1) / mask.sum(1).clamp(min=1.0)
```

Because this changes *forward semantics* (not parameters), a model trained under
the old pooling is best evaluated with `pooling="mean"`. Cortex exposes both
modes:

```bash
# corrected behaviour (default, for new training)
python -m src.evaluate --checkpoint experiments/run/checkpoints/best.pt --split test

# reproduce a legacy checkpoint that was trained with unmasked mean pooling
python -m src.evaluate --checkpoint checkpoints/best_model_2.pt --split test \
    --set pooling=mean
```

## Legacy checkpoints

Two weight files are stored under `checkpoints/` (DVC-tracked). They are plain
`state_dict`s with no optimizer/epoch metadata.

| File | Architecture | Loadability |
|------|--------------|-------------|
| `best_model.pt` | legacy: LoRA `r=8`, targets `query,key,value`, 2-layer MLP head, no `target_compressor` | partial — encoder backbones load, head/LoRA skipped |
| `best_model_2.pt` | current: LoRA `r=16`, targets `query,key,value,dense`, compressor, 3-layer head | **exact** — 0 missing / 0 unexpected / 0 mismatched |

`src.checkpoint.load_model_state(..., strict=False)` reports `missing`,
`unexpected` and `mismatched` keys and skips incompatible tensors, so both files
can be inspected or warm-started. See `docs/CHECKPOINTS_AND_RECOVERY.md`.
