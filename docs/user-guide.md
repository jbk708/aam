# User Guide

Complete reference for AAM command-line interface and options.

## Commands Overview

| Command | Description |
|---------|-------------|
| `aam pretrain` | Self-supervised pre-training on UniFrac + nucleotide prediction |
| `aam train` | Supervised training/fine-tuning on target predictions |
| `aam predict` | Run inference with trained model |
| `aam rf-baseline` | Random Forest baseline comparison |

## Training Workflows

### Two-Stage Training (Recommended)

**Stage 1: Pre-training**

Train the encoder on self-supervised tasks (no labels required):

```bash
aam pretrain \
  --table <biom_file> \
  --unifrac-matrix <unifrac_matrix.npy> \
  --output-dir <output_dir> \
  --batch-size 8 \
  --epochs 100
```

**Stage 2: Fine-tuning**

Train the predictor with optional pre-trained encoder:

```bash
aam train \
  --table <biom_file> \
  --unifrac-matrix <unifrac_matrix.npy> \
  --metadata <metadata.tsv> \
  --metadata-column <target_column> \
  --output-dir <output_dir> \
  --pretrained-encoder <encoder_checkpoint.pt> \
  --freeze-base \
  --batch-size 8 \
  --epochs 100
```

### Single-Stage Training

Skip pre-training and train end-to-end:

```bash
aam train \
  --table <biom_file> \
  --unifrac-matrix <unifrac_matrix.npy> \
  --metadata <metadata.tsv> \
  --metadata-column <target_column> \
  --output-dir <output_dir> \
  --epochs 100
```

## Training Options

### Core Options

| Option | Description | Default |
|--------|-------------|---------|
| `--pretrained-encoder` | Load pre-trained SequenceEncoder checkpoint | - |
| `--freeze-base` | Freeze base model parameters | False |
| `--classifier` | Classification mode (requires `--out-dim > 1`) | False |
| `--target-transform` | Target normalization strategy | minmax |
| `--lr` | Learning rate | 1e-4 |
| `--optimizer` | 'adamw', 'adam', or 'sgd' | adamw |
| `--scheduler` | LR scheduler | warmup_cosine |
| `--best-metric` | Metric for best model selection | val_loss |

### Loss Weights

| Option | Description | Default |
|--------|-------------|---------|
| `--penalty` | Weight for UniFrac loss | 1.0 |
| `--nuc-penalty` | Weight for nucleotide loss | 1.0 |
| `--target-penalty` | Weight for target loss | 1.0 |
| `--count-penalty` | Weight for count loss | 1.0 |

## Target Normalization

Control how regression targets are normalized with `--target-transform`:

| Transform | Description | Use Case |
|-----------|-------------|----------|
| `none` | No normalization | Targets already in good range |
| `minmax` | Scale to [0, 1] range | **Default.** Most regression tasks |
| `zscore` | Global z-score (standardization) | Targets with outliers |
| `zscore-category` | Per-category z-score | Different target distributions per group |
| `log-minmax` | log(y+1) then minmax | Non-negative targets with wide range (0-600) |
| `log-zscore` | log(y+1) then z-score | Wide range with outliers |

**Examples:**

```bash
# Default min-max normalization
aam train --target-transform minmax ...

# Log transform for wide-range targets
aam train --target-transform log-minmax ...

# Per-category normalization
aam train --target-transform zscore-category --normalize-by "season" ...
```

## Regression Options

### Loss Functions

| Loss | Description | Use Case |
|------|-------------|----------|
| `huber` | Smooth L1 (MSE for small, MAE for large errors) | **Default.** Robust to outliers |
| `mse` | Mean squared error | Standard regression |
| `mae` | Mean absolute error | When outliers are important |
| `quantile` | Pinball loss for quantile regression | Uncertainty estimation |
| `asymmetric` | Different penalties for over/under prediction | Cost-sensitive regression |

### Quantile Regression

Predict multiple quantiles for uncertainty estimation:

```bash
aam train \
  --loss-type quantile \
  --quantiles 0.1,0.5,0.9 \
  ...
```

### Asymmetric Loss

Apply different penalties for over/under predictions:

```bash
aam train \
  --loss-type asymmetric \
  --over-penalty 1.0 \
  --under-penalty 2.0 \
  ...
```

### Per-Output Loss Configuration

For multi-output regression, configure loss per target:

```bash
aam train \
  --loss-config '{"0": "mse", "1": "huber"}' \
  ...
```

### Output Constraints

| Option | Description | Default |
|--------|-------------|---------|
| `--bounded-targets` | Sigmoid to bound output to [0, 1] | False |
| `--output-activation` | Non-negative: 'none', 'relu', 'softplus', 'exp' | none |
| `--learnable-output-scale` | Learnable scale and bias after target head | False |
| `--regressor-hidden-dims` | MLP hidden layers, e.g., '64,32' | None |
| `--regressor-dropout` | Dropout between MLP layers | 0.0 |

## Categorical Metadata

Condition predictions on categorical features like location or season.

### Basic Usage

```bash
aam train \
  --categorical-columns "location,season" \
  --categorical-embed-dim 16 \
  --categorical-fusion concat \
  ...
```

### Fusion Strategies

| Strategy | Description |
|----------|-------------|
| `concat` | Concatenate embeddings + project back |
| `add` | Project embeddings to embedding_dim + add |
| `gmu` | Gated Multimodal Unit - learned weighting |
| `cross-attention` | Position-specific modulation |

**Choosing a Strategy:**

| Use Case | Recommended |
|----------|-------------|
| Simple metadata conditioning | `concat` (default) |
| Per-category output adjustment | `concat` + `--conditional-output-scaling` |
| Adaptive weighting | `gmu` |
| Position-specific modulation | `cross-attention` |

Run `aam train --categorical-help` for detailed guidance.

### Conditional Output Scaling

Learn per-category scale and bias:

```bash
aam train \
  --categorical-columns "location,season" \
  --conditional-output-scaling "location" \
  ...
```

### Staged Training with Categoricals

```bash
# Stage 1: Pretrain encoder (no categoricals)
aam pretrain --table <biom> --unifrac-matrix <unifrac> --output-dir pretrain/

# Stage 2: Fine-tune with categoricals
aam train \
  --pretrained-encoder pretrain/checkpoints/best_model.pt \
  --freeze-base \
  --categorical-columns "location,season" \
  ...
```

## Model Architecture

| Option | Description | Default |
|--------|-------------|---------|
| `--embedding-dim` | Embedding dimension | 128 |
| `--attention-heads` | Number of attention heads | 4 |
| `--attention-layers` | Number of transformer layers | 4 |
| `--max-bp` | Maximum base pairs per sequence | 150 |
| `--token-limit` | Maximum ASVs per sample | 1024 |
| `--asv-sampling` | ASV selection: `first`, `abundance`, `random` | first |

## Memory Optimization

| Option | Description | Default |
|--------|-------------|---------|
| `--gradient-checkpointing` | Trade compute for memory | True |
| `--asv-chunk-size` | Process ASVs in chunks | 256 |
| `--gradient-accumulation-steps` | Accumulate gradients over N steps | 1 |
| `--token-limit` | Reduce for limited GPU memory | 1024 |

**For 24GB GPUs:**

```bash
aam pretrain \
  --batch-size 2 \
  --gradient-accumulation-steps 16 \
  --token-limit 256 \
  ...
```

**Medium model (fits 24GB):**

```bash
aam pretrain \
  --embedding-dim 768 \
  --attention-heads 6 \
  --attention-layers 4 \
  --batch-size 2 \
  --gradient-accumulation-steps 16 \
  --data-parallel \
  ...
```

**Large model (~380M params):**

```bash
aam pretrain \
  --embedding-dim 1152 \
  --attention-heads 12 \
  --attention-layers 8 \
  --batch-size 12 \
  --gradient-accumulation-steps 4 \
  --warmup-steps 5000 \
  --scheduler cosine_restarts \
  --patience 20 \
  --data-parallel \
  ...
```

## Multi-GPU Training

| Strategy | Flag | Use Case |
|----------|------|----------|
| **FSDP** | `--fsdp` | Pretraining (recommended) - best memory efficiency |
| **DataParallel** | `--data-parallel` | Pretraining (simpler) - higher GPU 0 memory |
| **DDP** | `--distributed` | Fine-tuning only |

```bash
# FSDP pretraining (recommended)
torchrun --nproc_per_node=4 -m aam.cli pretrain --fsdp ...

# DataParallel pretraining
aam pretrain --data-parallel ...

# DDP fine-tuning
torchrun --nproc_per_node=4 -m aam.cli train --distributed ...
```

**Note:** DDP computes UniFrac loss locally per GPU. FSDP/DataParallel gather embeddings across all GPUs first.

## Learning Rate Schedulers

| Scheduler | Use Case |
|-----------|----------|
| `warmup_cosine` | Default, stable training with gradual decay |
| `cosine_restarts` | Escape local minima with periodic warm restarts |
| `plateau` | Adaptive LR reduction when validation loss plateaus |
| `onecycle` | Fast experimentation |

Scheduler-specific options: `--scheduler-t0`, `--scheduler-t-mult`, `--scheduler-eta-min`, `--scheduler-patience`, `--scheduler-factor`

## Masked Autoencoder

| Option | Description | Default |
|--------|-------------|---------|
| `--nuc-mask-ratio` | Fraction of positions to mask | 0.15 |
| `--nuc-mask-strategy` | 'random' or 'span' masking | random |

## Monitoring Training

TensorBoard logs are saved to `{output_dir}/tensorboard/`:

```bash
tensorboard --logdir <output_dir>/tensorboard
```

### Progress Bar Labels

| Mode | Format |
|------|--------|
| Pretraining | `TL, LR, UL, NL, NA%` |
| Fine-tuning (regression) | `TL, LR, RL, UL` |
| Fine-tuning (classification) | `TL, LR, CL, UL` |

**Labels:** TL=Total Loss, LR=Learning Rate, UL=UniFrac Loss, NL=Nucleotide Loss, NA=Nucleotide Accuracy, RL=Regression Loss, CL=Classification Loss

### TensorBoard Metrics

- **Losses:** `train/total_loss`, `train/target_loss`, `train/unifrac_loss`, `train/nuc_loss`
- **Regression:** `val/mae`, `val/mse`, `val/r2`
- **Classification:** `val/accuracy`, `val/precision`, `val/recall`, `val/f1`
- **Fusion:** `gmu/gate_mean`, `cross_attn/weight_mean`

## Random Forest Baseline

Compare AAM against a Random Forest baseline:

```bash
aam rf-baseline \
  --table data/fall_train_only_all_outdoor.biom \
  --metadata data/fall_train_only_all_outdoor.tsv \
  --metadata-column add_0c \
  --train-samples output/train/train_samples.txt \
  --val-samples output/train/val_samples.txt \
  --output rf_predictions.tsv
```

| Option | Description | Default |
|--------|-------------|---------|
| `--n-estimators` | Number of trees | 500 |
| `--max-features` | Features per tree | sqrt |
| `--random-seed` | Random seed | 42 |
| `--n-jobs` | Parallel jobs (-1 for all) | -1 |

## HPC / SLURM Systems

When using `--compile-model` on SLURM systems:

```bash
module load gcc
```

## Cosmos (SDSC AMD MI300A)

**Environment setup:**

```bash
module load rocm/6.3.0
mamba create -n aam-rocm python=3.11 -y
mamba activate aam-rocm

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/rocm6.3

pip install -e ".[training]"

# Verify
python -c "import torch; print(f'ROCm: {torch.version.hip}')"
```

**Limitations:** `--compile-model` unsupported on ROCm. Flash Attention unavailable.

## Full Help

```bash
aam pretrain --help
aam train --help
aam predict --help
aam rf-baseline --help
```
