# Cortex — Drug–Target Interaction Training

Cortex is a deep learning system for **Drug–Target Interaction (DTI) prediction**. It uses pretrained molecular and protein language models to learn interactions between chemical compounds and protein sequences and predicts their binding affinity as a continuous **pKd regression value**.

The training architecture combines **MoLFormer + ESM-2**, parameter-efficient **LoRA fine-tuning**, **cross-attention**, protein sequence compression, and a custom regression head. The training pipeline is optimized for limited GPU environments using mixed precision, gradient checkpointing, dynamic batching, hard-example mining, checkpoint resumption, and early stopping.

---

## Architecture

```text
                         Drug SMILES
                              │
                              ▼
                    ┌─────────────────┐
                    │    MoLFormer    │
                    │  Chemical Tower │
                    └────────┬────────┘
                             │
                        LoRA Adapters
                             │
                             ▼
                      Drug Projection
                             │
                             │
                             ▼
                       ┌─────────────┐
                       │             │
                       │    Cross    │
                       │  Attention  │
                       │             │
                       └──────▲──────┘
                              │
                         Key + Value
                              │
                      Drug Representations
                              

                       Protein Sequence
                              │
                              ▼
                    ┌─────────────────┐
                    │      ESM-2      │
                    │  Protein Tower  │
                    └────────┬────────┘
                             │
                        LoRA Adapters
                             │
                             ▼
                    Protein Projection
                             │
                             ▼
                    Conv1D Compression
                             │
                             │ Query
                             └──────────────►
                                           
                              Cross-Attention
                                    │
                                    ▼
                              Mean Pooling
                                    │
                                    ▼
                             3-Layer GELU MLP
                                    │
                                    ▼
                              Linear Output
                                    │
                                    ▼
                              Predicted pKd
```

The two pretrained Transformers independently produce contextual representations:

* **MoLFormer** learns molecular representations from SMILES.
* **ESM-2** learns protein representations from amino-acid sequences.

The representations are projected into a common 256-dimensional space and connected through cross-attention.

For cross-attention:

```text
Query → Protein representations
Key   → Drug representations
Value → Drug representations
```

This allows the protein representation to selectively attend to relevant molecular features.

---

# Dataset

Cortex initially downloads the `BindingDB_Kd` dataset through **Therapeutics Data Commons (TDC)**.

```python
from tdc.multi_pred import DTI

data_loader = DTI(name="BindingDB_Kd")
df = data_loader.get_data()
```

The raw dataset is converted to a Polars DataFrame and stored locally:

```text
data/
└── raw/
    └── raw_binding_data.csv
```

Polars is used for efficient dataframe processing and serialization.

---

# Binding Affinity

The raw Kd value is transformed into pKd before training.

```text
pKd = 9 - log10(Kd)
```

The transformation is implemented as:

```python
df_filtered = df_filtered.with_columns(
    (9.0 - (pl.col("Y") + 1e-10).log10())
    .cast(pl.Float32)
    .alias("pkd")
)
```

The resulting `pkd` value becomes the regression label for the neural network.

---

# Cold-Target Evaluation

Cortex uses a **cold-target split** rather than randomly splitting individual interactions.

First, unique protein targets are extracted:

```python
unique_targets = df_filtered.select("Target").unique()
```

10% of the unique targets are selected for testing:

```python
test_targets = unique_targets.sample(
    fraction=0.1,
    seed=42
)
```

Every interaction involving those targets is placed into the test set.

```text
                     All Targets
                          │
             ┌────────────┴────────────┐
             │                         │
        10% Targets               90% Targets
             │                         │
             ▼                         ▼
        Test Set                 Train + Validation
     Unseen Proteins
```

This means the final test set contains **protein targets that were not present during training**.

The objective is therefore to evaluate how well the model generalizes to previously unseen protein targets.

---

# Train / Validation Split

The remaining 90% of targets are used for training and validation.

The remaining interactions are shuffled:

```python
df_remaining_shuffled = df_remaining.sample(
    fraction=1.0,
    shuffle=True,
    seed=42
)
```

15% are used for validation:

```text
Remaining Data
      │
      ├── 85% → Training
      │
      └── 15% → Validation
```

The final datasets are stored as Parquet files:

```text
data/
└── processed/
    ├── train.parquet
    ├── val.parquet
    └── test.parquet
```

---

# Tokenization

Cortex uses Hugging Face tokenizers corresponding to each pretrained model.

### Drug

```text
SMILES
  │
  ▼
MoLFormer Tokenizer
  │
  ▼
Token IDs + Attention Mask
```

### Protein

```text
FASTA / Amino Acid Sequence
  │
  ▼
ESM-2 Tokenizer
  │
  ▼
Token IDs + Attention Mask
```

Each training sample contains:

```text
drug_input_ids
drug_attention_mask

target_input_ids
target_attention_mask

label
```

The token IDs are passed into the respective Transformer models, which generate contextual embeddings.

---

# Dynamic Batch Trimming

Samples are initially padded to fixed maximum lengths:

```text
Maximum drug length   = 200
Maximum protein length = 1000
```

However, processing every sequence at its maximum length wastes computation when the actual sequences are shorter.

Cortex dynamically trims each batch to the longest actual sequence in that batch.

```text
Fixed Padding

Batch:
200
200
200
200

        ↓

Dynamic Trimming

Actual maximum:
143

        ↓

143
143
143
143
```

This reduces unnecessary computation caused by padding.

---

# Model

## Chemical Tower — MoLFormer

```text
ibm/MoLFormer-XL-both-10pct
```

MoLFormer processes the SMILES representation and produces contextual molecular embeddings.

---

## Protein Tower — ESM-2

```text
facebook/esm2_t6_8M_UR50D
```

ESM-2 processes the amino-acid sequence and produces contextual protein embeddings.

---

# LoRA Fine-Tuning

Instead of fully fine-tuning both pretrained Transformers, Cortex uses **Low-Rank Adaptation (LoRA)**.

Configuration:

```python
lora_r = 16
lora_alpha = 32
lora_dropout = 0.1

target_modules = (
    "query",
    "key",
    "value",
    "dense"
)
```

LoRA adapters are injected into the relevant Transformer linear layers:

```text
Query
Key
Value
Dense / Attention Output Projection
```

This is applied independently to both:

```text
MoLFormer
ESM-2
```

The large pretrained weights remain frozen while the smaller LoRA parameters learn task-specific adaptations.

Conceptually:

```text
Pretrained Weight
       +
   LoRA Update
       ↓
Adapted Weight Behavior
```

This significantly reduces the number of parameters that need to be updated compared with full fine-tuning.

---

# Gradient Checkpointing

Both Transformer towers use gradient checkpointing.

```python
drug_tower.gradient_checkpointing_enable()
target_tower.gradient_checkpointing_enable()
```

Gradient checkpointing reduces GPU memory consumption by avoiding storage of every intermediate activation during the forward pass.

During backpropagation, selected activations are recomputed when required.

```text
Lower VRAM
   ↕
More recomputation
```

This makes the model more practical on constrained GPUs.

---

# Dimension Alignment

MoLFormer and ESM-2 do not necessarily produce representations with the same hidden dimension.

Cortex therefore projects both outputs into a common space:

```python
self.drug_proj = nn.Linear(
    drug_hidden_size,
    Config.hidden_dim
)

self.target_proj = nn.Linear(
    target_hidden_size,
    Config.hidden_dim
)
```

The shared dimension is:

```text
hidden_dim = 256
```

Therefore:

```text
MoLFormer
    ↓
Drug Projection
    ↓
256 dimensions


ESM-2
    ↓
Protein Projection
    ↓
256 dimensions
```

---

# Protein Sequence Compression

Protein sequences can be significantly longer than molecular sequences.

Cortex uses a `Conv1D` layer to reduce the protein sequence length:

```python
self.target_compressor = nn.Conv1d(
    in_channels=256,
    out_channels=256,
    kernel_size=4,
    stride=4
)
```

The feature dimension remains 256 while the sequence length is reduced.

Conceptually:

```text
Protein representation
        │
        ▼
Conv1D
kernel = 4
stride = 4
        │
        ▼
Shorter protein representation
```

This reduces the amount of sequence information that must be processed by subsequent interaction layers.

---

# Cross-Attention

Cortex uses multi-head cross-attention:

```python
nn.MultiheadAttention(
    embed_dim=256,
    num_heads=8,
    dropout=0.1,
    batch_first=True
)
```

The attention relationship is:

```text
Protein → Query

Drug → Key
Drug → Value
```

Conceptually:

```text
Protein Embeddings
       │
       ▼
     Query
       │
       │
       ▼
 Cross-Attention
       ▲
       │
   Key + Value
       │
       ▲
Drug Embeddings
```

Instead of simply concatenating the drug and protein representations, the model learns which molecular representations are relevant to each protein representation.

The output is an interaction-aware representation.

---

# Attention Masking

Padding tokens must not participate in attention.

The Hugging Face drug attention mask is therefore converted into a PyTorch padding mask:

```python
key_padding_mask = (drug_attention_mask == 0)
```

This prevents padded drug positions from contributing meaningful information to the cross-attention operation.

---

# Mean Pooling

Cross-attention produces a sequence of interaction representations.

Cortex converts these into one fixed-size representation:

```python
pooled_output = torch.mean(
    attn_output,
    dim=1
)
```

Conceptually:

```text
Interaction Token 1
Interaction Token 2
Interaction Token 3
       ...
Interaction Token N
       │
       ▼
  Mean Pooling
       │
       ▼
Single Interaction Vector
```

---

# Regression Head

The pooled interaction representation is passed through a three-layer MLP:

```text
256
 │
 ▼
Linear
 │
 ▼
GELU
 │
 ▼
Dropout
 │
 ▼
128
 │
 ▼
Linear
 │
 ▼
GELU
 │
 ▼
Dropout
 │
 ▼
1
 │
 ▼
Predicted pKd
```

Implementation:

```python
self.mlp = nn.Sequential(
    nn.Linear(256, 256),
    nn.GELU(),
    nn.Dropout(0.1),

    nn.Linear(256, 128),
    nn.GELU(),
    nn.Dropout(0.1),

    nn.Linear(128, 1)
)
```

The final layer is a single linear neuron because pKd is a continuous regression target.

There is no Softmax or Sigmoid after the final layer.

---

# Training Objective

Cortex trains the model using Mean Squared Error.

```python
nn.MSELoss()
```

Conceptually:

```text
Predicted pKd
      │
      ▼
Compare with
experimental pKd
      │
      ▼
Squared Error
      │
      ▼
Training Loss
```

During training, the unreduced loss is used:

```python
nn.MSELoss(reduction="none")
```

This produces a separate loss value for every sample in the batch.

That enables Online Hard Example Mining.

---

# Online Hard Example Mining

Cortex introduces hard-example mining after the model has completed its initial training period.

Configuration:

```python
hard_mining_start_epoch = 5
hard_mining_fraction = 0.5
```

Before epoch 5:

```text
All samples
    ↓
Average loss
    ↓
Backpropagation
```

From epoch 5:

```text
All samples
    ↓
Individual sample losses
    ↓
Select highest-loss 50%
    ↓
Average hard-example loss
    ↓
Backpropagation
```

The purpose is to focus learning on interactions the model currently finds difficult.

---

# Mixed Precision Training

Cortex uses CUDA automatic mixed precision:

```python
with torch.amp.autocast("cuda"):
```

Suitable operations can therefore run in lower precision instead of always using FP32.

Benefits include:

* Reduced GPU memory consumption
* Faster GPU computation
* More practical batch sizes

---

# Gradient Scaling

FP16 training can suffer from very small gradients.

Cortex uses:

```python
torch.amp.GradScaler("cuda")
```

The training process becomes:

```text
Loss
 ↓
Scale
 ↓
Backward
 ↓
Optimizer Step
 ↓
Update Scaler
```

This improves numerical stability during mixed-precision training.

---

# Optimizer

Cortex uses AdamW:

```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.05
)
```

### Learning Rate

```text
1e-4
```

controls the size of parameter updates.

### Weight Decay

```text
0.05
```

provides regularization to discourage excessive parameter growth and help reduce overfitting.

---

# Cosine Learning Rate Scheduling

Cortex uses:

```python
CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=1e-6
)
```

The learning rate gradually decreases throughout training.

```text
Learning Rate
     │
     │\
     │ \
     │  \
     │   \
     │    \____
     │
     └────────────── Epochs
```

This allows larger updates earlier in training and progressively smaller updates later.

---

# Early Stopping

Validation loss is monitored after every epoch.

Configuration:

```text
early_stopping_patience = 3
```

If the validation loss fails to improve for three consecutive epochs, training stops.

```text
Validation improves
        ↓
Continue training

Validation stops improving
        ↓
Patience counter
        ↓
3 epochs
        ↓
Stop
```

This prevents unnecessary training and helps limit overfitting.

---

# Checkpointing

Cortex saves the best model based on validation loss.

The checkpoint contains:

```python
{
    "epoch": epoch,
    "model_state": state_dict,
    "optimizer_state": optimizer.state_dict(),
    "scheduler_state": scheduler.state_dict(),
    "best_val_loss": best_val_loss
}
```

Therefore the checkpoint stores both model parameters and training state.

---

# Automatic Resume

When training starts, Cortex checks whether a previous checkpoint exists.

If found, it restores:

```text
Model weights
Optimizer state
Scheduler state
Previous epoch
Best validation loss
```

Training can then continue from the next epoch rather than starting from scratch.

This is useful for interrupted GPU sessions and limited-duration compute environments.

---

# Multi-GPU Training

When multiple CUDA devices are available, Cortex activates PyTorch `DataParallel`:

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

The batch is distributed across the available GPUs.

Before saving the checkpoint, the `DataParallel` wrapper is removed so the saved weights can be loaded into a normal single-device model.

---

# Configuration

Main training configuration:

| Parameter               | Value                         |
| ----------------------- | ----------------------------- |
| Drug model              | `ibm/MoLFormer-XL-both-10pct` |
| Protein model           | `facebook/esm2_t6_8M_UR50D`   |
| Maximum drug length     | 200                           |
| Maximum protein length  | 1000                          |
| Hidden dimension        | 256                           |
| Attention heads         | 8                             |
| Dropout                 | 0.1                           |
| LoRA rank               | 16                            |
| LoRA alpha              | 32                            |
| LoRA dropout            | 0.1                           |
| LoRA modules            | Query, Key, Value, Dense      |
| Batch size              | 32                            |
| Epochs                  | 20                            |
| Learning rate           | `1e-4`                        |
| Weight decay            | 0.05                          |
| Hard mining starts      | Epoch 5                       |
| Hard mining fraction    | 50%                           |
| Early stopping patience | 3                             |
| Scheduler               | Cosine Annealing              |
| Minimum LR              | `1e-6`                        |

---

# Training Pipeline

The complete training workflow:

```text
BindingDB_Kd
     │
     ▼
Raw Dataset
     │
     ▼
pKd Calculation
     │
     ▼
Cold-Target Split
     │
     ├──────────────► Unseen Test Targets
     │
     ▼
Train / Validation Split
     │
     ▼
Parquet Files
     │
     ▼
PyTorch Dataset
     │
     ▼
Tokenization
     │
     ▼
Dynamic Batch Trimming
     │
     ├───────────────────────┐
     ▼                       ▼
  MoLFormer                 ESM-2
     │                       │
   LoRA                    LoRA
     │                       │
     ▼                       ▼
Drug Projection       Protein Projection
                             │
                             ▼
                      Conv1D Compression
                             │
                             ▼
                      Cross Attention
                             │
                             ▼
                       Mean Pooling
                             │
                             ▼
                        GELU MLP
                             │
                             ▼
                       Predicted pKd
                             │
                             ▼
                         MSE Loss
                             │
                             ▼
                     Hard Example Mining
                             │
                             ▼
                     Mixed Precision
                             │
                             ▼
                           AdamW
                             │
                             ▼
                    Cosine LR Scheduler
                             │
                             ▼
                         Validation
                             │
                             ▼
                     Best Checkpoint
```

---

# Evaluation

The final evaluation is performed on the **cold-target test set**.

The evaluation script:

1. Loads `test.parquet`
2. Reconstructs the same Cortex architecture
3. Loads the trained checkpoint
4. Runs inference over the unseen targets
5. Collects predictions
6. Compares predictions with ground-truth pKd values

The model therefore evaluates on protein targets that were excluded from training.

---

# Evaluation Metrics

Cortex calculates:

### MSE

Mean Squared Error.

```text
Lower = Better
```

### RMSE

Root Mean Squared Error.

```text
RMSE = √MSE
```

The result is expressed on the pKd scale.

### R²

Measures the amount of variance explained by the model.

```text
Higher = Better
```

### Pearson Correlation

Measures the linear correlation between predicted and experimental pKd.

```text
Higher positive correlation = Better
```

---

# Project Structure

```text
cortex/
│
├── data/
│   ├── raw/
│   │   └── raw_binding_data.csv
│   │
│   └── processed/
│       ├── train.parquet
│       ├── val.parquet
│       └── test.parquet
│
├── checkpoints/
│   └── best_model.pt
│
├── src/
│   ├── tokenization.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── logger.py
│
├── config.py
│
└── ...
```

---

# Core Components

### `tokenization.py`

Responsible for:

* Loading processed Parquet datasets
* Loading Hugging Face tokenizers
* Tokenizing SMILES
* Tokenizing protein sequences
* Creating attention masks
* Returning PyTorch tensors

### `models.py`

Defines:

* MoLFormer chemical tower
* ESM-2 protein tower
* LoRA adapters
* Projection layers
* Protein Conv1D compressor
* Multi-head cross-attention
* Mean pooling
* GELU regression head

### `train.py`

Handles:

* Data loading
* Dynamic batching
* Forward pass
* MSE loss
* Hard-example mining
* Mixed precision
* Gradient scaling
* AdamW optimization
* Learning-rate scheduling
* Validation
* Checkpointing
* Early stopping
* Multi-GPU execution

### `evaluate.py`

Handles:

* Cold-target test loading
* Model reconstruction
* Checkpoint loading
* Test inference
* Regression metrics

### `config.py`

Centralizes:

* Model names
* Sequence limits
* Architecture parameters
* LoRA configuration
* Training hyperparameters
* Dataset paths
* Device configuration

---

# Technology Stack

### Deep Learning

* PyTorch
* Hugging Face Transformers
* PEFT
* LoRA

### Models

* IBM MoLFormer
* Meta ESM-2

### Data

* Therapeutics Data Commons
* BindingDB
* Polars
* Parquet

### Training Optimization

* Mixed Precision / FP16
* Gradient Scaling
* Gradient Checkpointing
* Dynamic Batch Trimming
* Online Hard Example Mining
* AdamW
* Cosine Annealing
* Early Stopping
* Checkpoint Resume
* Multi-GPU DataParallel

### Evaluation

* scikit-learn
* SciPy
* MSE
* RMSE
* R²
* Pearson Correlation

---

# Key Design Decisions

## Why two pretrained Transformers?

Molecules and proteins have fundamentally different sequential representations.

```text
SMILES → Molecular Language Model

Protein Sequence → Protein Language Model
```

Using specialized pretrained models allows each modality to be represented using a model trained specifically for that type of biological sequence.

---

## Why LoRA?

Full fine-tuning of both Transformer towers is computationally expensive.

LoRA allows Cortex to adapt the pretrained models while training a much smaller set of additional parameters.

---

## Why Cross-Attention?

Simple concatenation treats the two representations as independent feature vectors.

Cross-attention explicitly models their interaction:

```text
Protein → asks what molecular information is relevant
Drug    → provides the information
```

This creates an interaction-aware representation before regression.

---

## Why Cold-Target Testing?

Random interaction splits can allow the model to see the same targets during both training and testing.

The cold-target split instead evaluates:

```text
Training:
Known protein targets

Testing:
Previously unseen protein targets
```

This provides a stricter test of target-level generalization.

---

# End-to-End Model

At its core, Cortex learns:

```text
SMILES
  │
  ▼
MoLFormer
  │
  ▼
Molecular Context
  │
  │
  ├───────────────┐
  │               │
  │          Cross Attention
  │               ▲
  │               │
  │
Protein Sequence │
  │               │
  ▼               │
ESM-2             │
  │               │
  ▼               │
Protein Context ──┘
        │
        ▼
Interaction Representation
        │
        ▼
      MLP
        │
        ▼
     pKd
```

The final objective is to learn a mapping:

```text
f(SMILES, Protein Sequence) → pKd
```

while maintaining efficient training through parameter-efficient fine-tuning and GPU-aware optimization techniques.
