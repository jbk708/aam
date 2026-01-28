# Getting Started

This guide walks you through installing AAM and running your first analysis.

## Prerequisites

- Python 3.9-3.12
- [Mamba](https://mamba.readthedocs.io/) or [Conda](https://docs.conda.io/) (Mamba recommended)
- PyTorch 2.3+ (CUDA for GPU, ROCm for AMD GPUs)

## Installation

### 1. Create Environment

```bash
mamba create -n aam python=3.11 -y
mamba activate aam
```

### 2. Install PyTorch

Choose the appropriate version for your hardware:

**CUDA 12.x (NVIDIA GPUs):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**CPU only (development/testing):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**ROCm 6.3 (AMD GPUs):**
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/rocm6.3
```

### 3. Install AAM

```bash
# Basic installation
pip install -e .

# With development dependencies
pip install -e ".[dev,docs,training]"
```

### 4. Verify Installation

```bash
python -c "import aam; print('AAM installed successfully')"
```

## Quick Start

Try AAM with the included test data (781 samples):

### Pre-training (Self-supervised)

```bash
aam pretrain \
  --table data/fall_train_only_all_outdoor.biom \
  --unifrac-matrix data/fall_train_only_all_outdoor.h5 \
  --output-dir output/pretrain \
  --epochs 5 \
  --batch-size 8
```

### Fine-tuning (Supervised)

```bash
aam train \
  --table data/fall_train_only_all_outdoor.biom \
  --unifrac-matrix data/fall_train_only_all_outdoor.h5 \
  --metadata data/fall_train_only_all_outdoor.tsv \
  --metadata-column add_0c \
  --output-dir output/train \
  --pretrained-encoder output/pretrain/checkpoints/best_model.pt \
  --epochs 10 \
  --batch-size 8
```

### Inference

```bash
aam predict \
  --model output/train/checkpoints/best_model.pt \
  --table data/fall_train_only_all_outdoor.biom \
  --output predictions.tsv
```

## Generating UniFrac Matrices

AAM requires pre-computed UniFrac distance matrices. Generate them using `ssu` from [unifrac-binaries](https://github.com/biocore/unifrac-binaries):

```bash
ssu \
  -i <biom_file> \
  -t <tree_file> \
  -m unweighted_fp32 \
  -o <unifrac_matrix.h5> \
  --format hdf5_fp32
```

Supported formats: `.npy`, `.h5`, `.csv`

## Troubleshooting

### PyTorch not detecting GPU

Verify CUDA/ROCm installation:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Out of memory errors

Reduce memory usage with:
```bash
aam train --token-limit 256 --batch-size 2 --gradient-accumulation-steps 16 ...
```

### ROCm-specific issues

For AMD GPUs, ensure ROCm module is loaded:
```bash
module load rocm/6.3.0
```

## Next Steps

- [User Guide](user-guide.md) - Full CLI reference and options
- [How It Works](how-it-works.md) - Understanding AAM's architecture
- **Architecture** - See `ARCHITECTURE.md` in repo root for technical design details
