# Attention All Microbes (AAM)

Deep learning model for microbial sequencing data analysis using transformer-based attention mechanisms. Processes nucleotide sequences at multiple levels (nucleotide, ASV, sample) with self-supervised learning from phylogenetic information.

## Installation

### Prerequisites

- Python 3.9-3.12
- [Mamba](https://mamba.readthedocs.io/) or [Conda](https://docs.conda.io/) (Mamba recommended for faster resolution)
- PyTorch 2.3+ (with CUDA for GPU training, or ROCm for AMD GPUs)

### Setup

```bash
# Create conda environment
mamba create -n aam python=3.11 -y
mamba activate aam

# Install package
pip install -e .

# For development with all dependencies
pip install -e ".[dev,docs,training]"

# Verify installation
python -c "import aam; print('AAM installed successfully')"
```

### PyTorch Installation

If PyTorch is not already installed, install it first from [pytorch.org](https://pytorch.org/get-started/locally/):

```bash
# CUDA 12.x
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only (for development/testing)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# ROCm (AMD GPUs) - see Cosmos section below
pip install torch --index-url https://download.pytorch.org/whl/rocm6.3
```

## Usage

### Quick Start

Try AAM with the included test data (781 samples):

```bash
# Pre-training (self-supervised, ~2 min on GPU)
aam pretrain \
  --table data/fall_train_only_all_outdoor.biom \
  --unifrac-matrix data/fall_train_only_all_outdoor.h5 \
  --output-dir output/pretrain \
  --epochs 5 \
  --batch-size 8

# Fine-tuning on a regression target
aam train \
  --table data/fall_train_only_all_outdoor.biom \
  --unifrac-matrix data/fall_train_only_all_outdoor.h5 \
  --metadata data/fall_train_only_all_outdoor.tsv \
  --metadata-column add_0c \
  --output-dir output/train \
  --pretrained-encoder output/pretrain/checkpoints/best_model.pt \
  --epochs 10 \
  --batch-size 8

# Inference
aam predict \
  --model output/train/checkpoints/best_model.pt \
  --table data/fall_train_only_all_outdoor.biom \
  --output predictions.tsv
```

### Generating UniFrac Distance Matrices

**Important:** AAM requires pre-computed UniFrac distance matrices. Generate them before training using `ssu` from [unifrac-binaries](https://github.com/biocore/unifrac-binaries/tree/main):

```bash
ssu \
  -i <biom_file> \
  -t <tree_file> \
  -m unweighted_fp32 \
  -o <unifrac_matrix.h5> \
  --format hdf5_fp32
```

**Notes:**
- `unweighted_fp32` produces float32 precision (recommended for memory efficiency)
- Supported output formats: `.npy`, `.h5`, `.csv`
- Matrix should be symmetric `[N_samples, N_samples]` for pairwise UniFrac

### Pre-training (Stage 1: Self-supervised)

Pre-train SequenceEncoder on UniFrac + nucleotide prediction:

```bash
aam pretrain \
  --table <biom_file> \
  --unifrac-matrix <unifrac_matrix.npy> \
  --output-dir <output_dir> \
  --batch-size 8 \
  --epochs 100
```

### Training (Stage 2: Fine-tuning)

Train SequencePredictor with optional pre-trained encoder:

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

### Categorical Metadata

Condition target predictions on categorical features like location, season, or site type. Categorical embeddings are fused with sample embeddings before the target prediction head.

**CLI Usage:**

```bash
aam train \
  --table <biom_file> \
  --unifrac-matrix <unifrac_matrix.npy> \
  --metadata <metadata.tsv> \
  --metadata-column <target_column> \
  --categorical-columns "location,season" \
  --categorical-embed-dim 16 \
  --categorical-fusion concat \
  --output-dir <output_dir>
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--categorical-columns` | Comma-separated column names from metadata | - |
| `--categorical-embed-dim` | Embedding dimension per category | 16 |
| `--categorical-fusion` | Fusion strategy (see below) | concat |
| `--cross-attn-heads` | Attention heads for cross-attention fusion | 8 |
| `--conditional-output-scaling` | Columns for per-category output scale/bias | - |

**Fusion Strategies (`--categorical-fusion`):**

| Strategy | Description |
|----------|-------------|
| `concat` | Concatenate embeddings + project back to embedding_dim |
| `add` | Project embeddings to embedding_dim + add |
| `gmu` | Gated Multimodal Unit - learns to weight sequence vs categorical info |
| `cross-attention` | Each ASV position attends to categorical metadata independently |

**GMU (Gated Multimodal Unit):** Fuses after pooling using a learned gate that weights sequence vs categorical contributions. Good when you want the model to adaptively balance information sources.

**Cross-Attention:** Position-specific fusion where each ASV can attend differently to metadata. Use when different taxa should respond differently to the same categorical feature (e.g., some taxa respond strongly to "summer" while others don't).

**Conditional Output Scaling:**

Learn per-category scale and bias parameters applied after the base prediction: `output = prediction * scale[cat] + bias[cat]`. Useful when different categories have systematically different output ranges even after normalization.

```bash
aam train \
  --categorical-columns "location,season" \
  --conditional-output-scaling "location" \
  # ... other flags
```

**Staged Training with Categoricals:**

Categorical features work seamlessly with pretrained encoders:

```bash
# Stage 1: Pretrain encoder (no categoricals needed)
aam pretrain \
  --table <biom_file> \
  --unifrac-matrix <unifrac_matrix.npy> \
  --output-dir pretrain_output/

# Stage 2: Fine-tune with categoricals
aam train \
  --table <biom_file> \
  --unifrac-matrix <unifrac_matrix.npy> \
  --metadata <metadata.tsv> \
  --metadata-column <target_column> \
  --pretrained-encoder pretrain_output/checkpoints/best_model.pt \
  --freeze-base \
  --categorical-columns "location,season" \
  --output-dir finetune_output/
```

With `--freeze-base`, only the categorical embedder and target prediction head are trained while the pretrained encoder remains frozen.

**Notes:**
- Index 0 is reserved for unknown/missing categories
- Encoder is fitted on training data only to prevent leakage
- Unknown categories at inference time map to the unknown embedding (index 0)
- Categorical encoder state is saved in checkpoints for inference

**Best Practices:**

*Embedding dimension selection:*
- Start with `--categorical-embed-dim 16` (default) for most cases
- For high-cardinality columns (>50 categories), consider 32-64 to capture more nuance
- For low-cardinality columns (<10 categories), 8-16 is sufficient
- Rule of thumb: `embed_dim ≈ min(50, cardinality / 2)` for high-cardinality columns

*Handling rare categories:*
- Categories appearing <5 times in training may not learn meaningful embeddings
- Consider grouping rare categories into an "other" category before training
- The unknown embedding (index 0) provides a fallback for unseen categories at inference
- Monitor validation performance across category frequencies to detect underfitting

## Key Options

### Training

| Option | Description | Default |
|--------|-------------|---------|
| `--pretrained-encoder` | Load pre-trained SequenceEncoder checkpoint | - |
| `--freeze-base` | Freeze base model parameters | False |
| `--classifier` | Classification mode (requires `--out-dim > 1`) | False |
| `--target-transform` | Target normalization strategy (see below) | minmax |
| `--lr` | Learning rate | 1e-4 |
| `--optimizer` | 'adamw', 'adam', or 'sgd' | adamw |
| `--scheduler` | LR scheduler (see below) | warmup_cosine |
| `--best-metric` | Metric for best model selection: val_loss, r2, mae, accuracy, f1 | val_loss |

### Loss Weights

| Option | Description | Default |
|--------|-------------|---------|
| `--penalty` | Weight for UniFrac loss | 1.0 |
| `--nuc-penalty` | Weight for nucleotide loss | 1.0 |
| `--target-penalty` | Weight for target loss | 1.0 |
| `--count-penalty` | Weight for count loss | 1.0 |

### Target Normalization (`--target-transform`)

Controls how regression targets are normalized during training:

| Transform | Description | Use Case |
|-----------|-------------|----------|
| `none` | No normalization | Targets already in good range |
| `minmax` | Scale to [0, 1] range | **Default.** Most regression tasks |
| `zscore` | Global z-score (standardization) | Targets with outliers |
| `zscore-category` | Per-category z-score | Different target distributions per group |
| `log-minmax` | log(y+1) then minmax | Non-negative targets with wide range (0-600) |
| `log-zscore` | log(y+1) then z-score | Wide range with outliers |

**Example usage:**

```bash
# Default min-max normalization
aam train --target-transform minmax ...

# Log transform for wide-range targets (recommended for 0-600 range)
aam train --target-transform log-minmax ...

# Per-category normalization (e.g., different seasons have different baselines)
aam train --target-transform zscore-category --normalize-by "season" ...
```

**Note:** For `zscore-category`, use `--normalize-by` to specify categorical columns.

### Regression Options

| Option | Description | Default |
|--------|-------------|---------|
| `--target-transform` | Target normalization (see above) | minmax |
| `--normalize-by` | Columns for per-category normalization | - |
| `--loss-type` | Loss function: 'mse', 'mae', 'huber', 'quantile' | huber |
| `--quantiles` | Quantile levels for quantile regression (see below) | 0.1,0.5,0.9 |
| `--bounded-targets` | Sigmoid to bound output to [0, 1] | False |
| `--output-activation` | Non-negative constraint: 'none', 'relu', 'softplus', 'exp' | none |
| `--learnable-output-scale` | Learnable scale and bias after target head | False |
| `--regressor-hidden-dims` | MLP hidden layers, e.g., '64,32' | None (single linear) |
| `--regressor-dropout` | Dropout between MLP layers | 0.0 |

**Loss functions (`--loss-type`):**

| Loss | Description | Use Case |
|------|-------------|----------|
| `huber` | Smooth L1 (MSE for small errors, MAE for large) | **Default.** Robust to outliers |
| `mse` | Mean squared error | Standard regression |
| `mae` | Mean absolute error | When outliers are important |
| `quantile` | Pinball loss for quantile regression | Uncertainty estimation |

**Quantile Regression:**

Predict multiple quantiles (e.g., 10th, 50th, 90th percentiles) for uncertainty estimation:

```bash
aam train \
  --loss-type quantile \
  --quantiles 0.1,0.5,0.9 \
  ...
```

Output shape becomes `[batch, out_dim, num_quantiles]`. The pinball loss penalizes under/over-predictions asymmetrically based on quantile level.

**Choosing an output constraint:**
- **`--target-transform log-minmax`** (recommended for wide ranges): Use for non-negative targets with wide range (e.g., 0-600). Compresses range via log(y+1) to ~[0, 6.4], model predicts directly in log space, exp(x)-1 gives original scale.
- **`--bounded-targets`**: Use when targets are in [0, 1] range (e.g., normalized values, proportions)
- **`--output-activation softplus`**: Use for non-negative targets without normalization. Note: may cause flat predictions near 0 when combined with minmax normalization
- **`--output-activation exp`**: Use for strictly positive targets, but can cause numerical instability

**Note:** With `log-minmax` or `log-zscore`, `--bounded-targets` is auto-enabled to prevent exp() overflow.

**Deprecated flags:** `--normalize-targets`, `--no-normalize-targets`, `--normalize-targets-by`, and `--log-transform-targets` still work but emit deprecation warnings. Use `--target-transform` instead.

### Masked Autoencoder (Nucleotide Prediction)

| Option | Description | Default |
|--------|-------------|---------|
| `--nuc-mask-ratio` | Fraction of positions to mask | 0.15 |
| `--nuc-mask-strategy` | 'random' or 'span' masking | random |

### Model Architecture

| Option | Description | Default |
|--------|-------------|---------|
| `--embedding-dim` | Embedding dimension | 128 |
| `--attention-heads` | Number of attention heads | 4 |
| `--attention-layers` | Number of transformer layers | 4 |
| `--max-bp` | Maximum base pairs per sequence | 150 |
| `--token-limit` | Maximum ASVs per sample (**critical for memory**) | 1024 |

### Memory Optimization

| Option | Description | Default |
|--------|-------------|---------|
| `--gradient-checkpointing` | Trade compute for memory | **True** |
| `--asv-chunk-size` | Process ASVs in chunks | 256 |
| `--gradient-accumulation-steps` | Accumulate gradients over N steps | 1 |
| `--token-limit` | Reduce from 1024 to 256-512 for 24GB GPUs | 1024 |

**For limited GPU memory (24GB):**

```bash
aam pretrain \
  --table <biom_file> \
  --unifrac-matrix <unifrac_matrix.npy> \
  --output-dir <output_dir> \
  --batch-size 2 \
  --gradient-accumulation-steps 16 \
  --token-limit 256
```

**Medium model example (fits on 24GB GPU):**

```bash
aam pretrain \
  --table <biom_file> \
  --unifrac-matrix <unifrac_matrix.h5> \
  --output-dir <output_dir> \
  --embedding-dim 768 \
  --attention-heads 6 \
  --attention-layers 4 \
  --batch-size 2 \
  --gradient-accumulation-steps 16 \
  --data-parallel
```

**Large model example (~380M parameters):**

```bash
aam pretrain \
  --table <biom_file> \
  --unifrac-matrix <unifrac_matrix.h5> \
  --output-dir <output_dir> \
  --embedding-dim 1152 \
  --attention-heads 12 \
  --attention-layers 8 \
  --batch-size 12 \
  --gradient-accumulation-steps 4 \
  --warmup-steps 5000 \
  --scheduler cosine_restarts \
  --patience 20 \
  --data-parallel
```

### Multi-GPU Training

| Strategy | Flag | Use Case |
|----------|------|----------|
| **FSDP** | `--fsdp` | Pretraining (recommended) - best memory efficiency |
| **DataParallel** | `--data-parallel` | Pretraining (simpler) - higher GPU 0 memory |
| **DDP** | `--distributed` | Fine-tuning only |

```bash
# FSDP pretraining (recommended)
torchrun --nproc_per_node=4 -m aam.cli pretrain --fsdp ...

# DataParallel pretraining (simpler, single-node)
aam pretrain --data-parallel ...

# DDP fine-tuning
torchrun --nproc_per_node=4 -m aam.cli train --distributed ...
```

**Why FSDP/DataParallel for pretraining?** DDP computes UniFrac loss locally per GPU, missing cross-GPU pairwise comparisons. FSDP and DataParallel gather embeddings across all GPUs first.

**FSDP options:** Add `--fsdp-sharded-checkpoint` for faster large-model checkpoints (requires same world size to load).

### SLURM/HPC Systems

When using `--compile-model` on SLURM-based HPC systems, load a GCC module before running:

```bash
module load gcc
```

Without this, `torch.compile()` may fail with `fatal error: stdatomic.h: No such file or directory` because the system GCC lacks required headers for Triton compilation.

### Cosmos (SDSC AMD MI300A)

AAM supports ROCm for AMD GPUs. For [SDSC Cosmos](https://www.sdsc.edu/systems/cosmos/user_guide.html) (168 AMD Instinct MI300A APUs):

**Environment Setup (ROCm 6.3 + PyTorch 2.7.1):**

```bash
# Load ROCm module (6.3 is default)
module load rocm/6.3.0

# Create conda environment
mamba create -n aam-rocm python=3.11 -y
mamba activate aam-rocm

# Install PyTorch 2.7.1 with ROCm 6.3 support
# Note: Must specify exact versions to avoid dependency conflicts
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/rocm6.3

# Install AAM
cd /path/to/aam
pip install -e ".[training]"

# Verify ROCm detection
python -c "import torch; print(f'ROCm: {torch.version.hip}, PyTorch: {torch.__version__}')"
```

**ROCm Compatibility:**

| ROCm | PyTorch | Notes |
|------|---------|-------|
| **6.3** | **2.7.1** | Recommended. Defaults work correctly. |
| 6.2 | 2.5.1 | Requires `--attn-implementation math --no-gradient-checkpointing` |

**Limitations:** `--compile-model` unsupported on ROCm (auto-skipped). Flash Attention unavailable.

**MI300A:** 128GB unified memory per APU. Default `--token-limit 1024` works well.

### Learning Rate Schedulers

| Scheduler | Use Case |
|-----------|----------|
| `warmup_cosine` | Default, stable training with gradual decay |
| `cosine_restarts` | Escape local minima with periodic warm restarts |
| `plateau` | Adaptive LR reduction when validation loss plateaus |
| `onecycle` | Fast experimentation |

**Scheduler-specific options:** `--scheduler-t0`, `--scheduler-t-mult`, `--scheduler-eta-min`, `--scheduler-patience`, `--scheduler-factor`

See `aam <command> --help` for full options.

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

**Note:** NL/NA are hidden during fine-tuning when `--freeze-base` is set (nuc_penalty=0).

### TensorBoard Metrics

- **Losses:** `train/total_loss`, `train/target_loss`, `train/unifrac_loss`, `train/nuc_loss`
- **Regression:** `val/mae`, `val/mse`, `val/r2`
- **Classification:** `val/accuracy`, `val/precision`, `val/recall`, `val/f1`
- **Fusion:** `gmu/gate_mean` (GMU gate values), `cross_attn/weight_mean` (cross-attention weights)

## Baseline Comparison

### Random Forest Baseline

Compare AAM predictions against a Random Forest baseline using the same train/validation splits:

```bash
aam rf-baseline \
  --table data/fall_train_only_all_outdoor.biom \
  --metadata data/fall_train_only_all_outdoor.tsv \
  --metadata-column add_0c \
  --train-samples output/train/train_samples.txt \
  --val-samples output/train/val_samples.txt \
  --output rf_predictions.tsv
```

This uses the sample split files generated by `aam train` (via CLN-10) to ensure a fair comparison. Output includes:
- R², MAE, and RMSE metrics printed to console
- TSV file with predictions in the same format as AAM's `val_predictions.tsv`
- PNG scatter plot matching AAM's TensorBoard `validation/prediction_plot` style

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--n-estimators` | Number of trees | 500 |
| `--max-features` | Features per tree (sqrt, log2, or float) | sqrt |
| `--random-seed` | Random seed | 42 |
| `--n-jobs` | Parallel jobs (-1 for all cores) | -1 |

## Testing

```bash
pytest tests/ -v                           # Run all tests
pytest tests/ --cov=aam --cov-report=html  # With coverage
```

1173 tests covering data pipeline, models, training, and end-to-end workflows.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation.

## Documentation

Implementation details and design decisions are documented in `_design_plan/`.
