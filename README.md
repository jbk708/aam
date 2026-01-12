# Attention All Microbes (AAM)

Deep learning model for microbial sequencing data analysis using transformer-based attention mechanisms. Processes nucleotide sequences at multiple levels (nucleotide, ASV, sample) with self-supervised learning from phylogenetic information.

## Installation

### Prerequisites

- [Mamba](https://mamba.readthedocs.io/) or [Conda](https://docs.conda.io/) (Mamba is recommended for faster package resolution)

### Setup

```bash
# Create and activate environment
mamba env create -f environment.yml
mamba activate aam

# Install package
pip install -e .

# For development with all dependencies
pip install -e ".[dev,docs,training]"
```

## Usage

### Generating UniFrac Distance Matrices

**Important:** AAM requires pre-computed UniFrac distance matrices. Generate them before training using `ssu` from [unifrac-binaries](https://github.com/biocore/unifrac-binaries/tree/main) (included in environment):

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

### Categorical Metadata (Planned)

Condition target predictions on categorical features like location or season:

```python
from aam.data import CategoricalSchema, CategoricalColumnConfig

# Simple: auto-detect cardinality
schema = CategoricalSchema.from_column_names(["location", "season"])

# Advanced: explicit configuration
schema = CategoricalSchema(columns=[
    CategoricalColumnConfig(name="location", cardinality=10, embed_dim=8),
    CategoricalColumnConfig(name="season", cardinality=4, required=True),
], default_embed_dim=16)
```

Index 0 is reserved for unknown/missing categories.

### Inference

```bash
aam predict \
  --model <model_checkpoint.pt> \
  --table <biom_file> \
  --output <predictions.tsv>
```

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

### Multi-GPU Training

AAM supports two multi-GPU strategies with different trade-offs:

| Strategy | Flag | Use Case |
|----------|------|----------|
| **DataParallel** | `--data-parallel` | Single-node pretraining (recommended for UniFrac) |
| **DDP** | `--distributed` | Multi-node training, fine-tuning |

**For pretraining with UniFrac loss, use DataParallel:**

```bash
# DataParallel gathers outputs to GPU 0, preserving full pairwise comparisons
aam pretrain \
  --table <biom_file> \
  --unifrac-matrix <unifrac_matrix.npy> \
  --output-dir <output_dir> \
  --data-parallel \
  --batch-size 32
```

**Why DataParallel for pretraining?** DDP computes pairwise UniFrac loss locally per GPU, missing cross-GPU comparisons. This causes predictions to converge toward the mean (~0.5) instead of learning the full distance distribution. DataParallel gathers all outputs to GPU 0 before loss computation, preserving the correct pairwise behavior.

**Note:** GPU 0 has higher memory usage with DataParallel as it gathers all outputs. For fine-tuning (target prediction without pairwise loss), DDP is preferred for better scaling.

### SLURM/HPC Systems

When using `--compile-model` on SLURM-based HPC systems, load a GCC module before running:

```bash
module load gcc
```

Without this, `torch.compile()` may fail with `fatal error: stdatomic.h: No such file or directory` because the system GCC lacks required headers for Triton compilation.

### Cosmos (SDSC AMD MI300A)

AAM supports ROCm for AMD GPUs. For [SDSC Cosmos](https://www.sdsc.edu/systems/cosmos/user_guide.html) (168 AMD Instinct MI300A APUs):

**Environment Setup:**

```bash
# Create conda environment
mamba create -n aam-rocm python=3.11 -y
mamba activate aam-rocm

# Install PyTorch with ROCm support (no module available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Install AAM
cd /path/to/aam
pip install -e ".[training]"

# Verify ROCm detection
python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}')"
```

**Running Jobs:**

AAM automatically detects ROCm vs CUDA. Use standard commands:

```bash
# Single APU
aam pretrain --table data.biom --unifrac-matrix unifrac.npy --output-dir output/

# Multi-APU pretraining (uses all visible GPUs)
aam pretrain --data-parallel --batch-size 32 \
  --table data.biom --unifrac-matrix unifrac.npy --output-dir output/
```

**MI300A Memory:** Each APU has 128GB unified CPU/GPU memory. The default `--token-limit 1024` works well; increase for larger datasets.

**Required Attention Flags for ROCm:**

```bash
aam pretrain \
  --attn-implementation math \
  --no-gradient-checkpointing \
  --data-parallel \
  # ... other flags
```

| Flag | Why Required |
|------|--------------|
| `--attn-implementation math` | The `mem_efficient` SDPA backend produces incorrect results with attention masks on ROCm (numerical divergence in masked positions). The `math` backend is slower but numerically correct. |
| `--no-gradient-checkpointing` | Gradient checkpointing combined with ROCm attention can cause additional numerical issues. Disable for stability. |

**Known ROCm Limitations:**

| Feature | Status | Notes |
|---------|--------|-------|
| `--compile-model` | Not supported | Triton has type mismatch errors on ROCm. AAM automatically detects ROCm and skips compilation with a warning. |
| `--attn-implementation mem_efficient` | Broken with masks | Use `math` backend instead |
| Flash Attention | Build incompatible | ROCm 6.2+ not yet supported by ROCm Flash Attention fork |

**Performance Impact:** The `math` backend uses ~5x more compute and ~7x more memory than `mem_efficient`. With MI300A's 128GB memory, this is acceptable.

**Diagnostic Tool:** Run `python -m aam.tools.rocm_attention_diagnostic` to verify SDPA backend behavior on your system.

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

679 tests covering data pipeline, models, training, and end-to-end workflows.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation.

## Documentation

Implementation details and design decisions are documented in `_design_plan/`.
