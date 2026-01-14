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
| `--categorical-fusion` | `concat` (concatenate + project) or `add` (project + add) | concat |

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
| `--normalize-targets` | Normalize regression targets to [0,1] | **True** |
| `--lr` | Learning rate | 1e-4 |
| `--optimizer` | 'adamw', 'adam', or 'sgd' | adamw |
| `--scheduler` | LR scheduler (see below) | warmup_cosine |

### Loss Weights

| Option | Description | Default |
|--------|-------------|---------|
| `--penalty` | Weight for UniFrac loss | 1.0 |
| `--nuc-penalty` | Weight for nucleotide loss | 1.0 |
| `--target-penalty` | Weight for target loss | 1.0 |

### Regression Options

| Option | Description | Default |
|--------|-------------|---------|
| `--log-transform-targets` | Apply log(y+1) transform to targets | False |
| `--bounded-targets` | Sigmoid to bound output to [0, 1] | False |
| `--output-activation` | Non-negative constraint: 'none', 'relu', 'softplus', 'exp' | none |
| `--learnable-output-scale` | Learnable scale and bias after target head | False |

**Choosing an output constraint:**
- **`--log-transform-targets --no-normalize-targets`** (recommended for wide ranges): Use for non-negative targets with wide range (e.g., 0-600). Compresses range via log(y+1) to ~[0, 6.4], model predicts directly in log space, exp(x)-1 gives original scale. No sigmoid needed since log range is small.
- **`--bounded-targets`**: Use when targets are in [0, 1] range (e.g., normalized values, proportions)
- **`--output-activation softplus`**: Use for non-negative targets without normalization. Note: may cause flat predictions near 0 when combined with `--normalize-targets`
- **`--output-activation exp`**: Use for strictly positive targets, but can cause numerical instability

**Note:** If using `--log-transform-targets` WITH `--normalize-targets` (not recommended), `--bounded-targets` is auto-enabled to prevent exp() overflow, but this can cause predictions to cluster at extremes due to sigmoid saturation.

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

## Testing

```bash
pytest tests/ -v                           # Run all tests
pytest tests/ --cov=aam --cov-report=html  # With coverage
```

919 tests covering data pipeline, models, training, and end-to-end workflows.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation.

## Documentation

Implementation details and design decisions are documented in `_design_plan/`.
