# Attention All Microbes (AAM)

Deep learning model for microbial sequencing data analysis using transformer-based attention mechanisms. Processes nucleotide sequences at multiple levels (nucleotide, ASV, sample) with self-supervised learning from phylogenetic information.

## Features

- **Hierarchical attention**: Nucleotide → ASV → Sample level processing
- **Self-supervised pre-training**: Learn from UniFrac distances and masked nucleotide prediction
- **Transfer learning**: Pre-train on unlabeled data, fine-tune on your targets
- **Categorical conditioning**: Condition predictions on metadata (location, season, etc.)
- **Multi-GPU support**: FSDP, DataParallel, and DDP training strategies
- **ROCm support**: AMD GPU acceleration (tested on MI300A)

## Quick Start

```bash
# Install
mamba create -n aam python=3.11 -y && mamba activate aam
pip install torch --index-url https://download.pytorch.org/whl/cu121  # or cpu/rocm
pip install -e .

# Pre-train (self-supervised)
aam pretrain \
  --table data/fall_train_only_all_outdoor.biom \
  --unifrac-matrix data/fall_train_only_all_outdoor.h5 \
  --output-dir output/pretrain \
  --epochs 5

# Fine-tune (supervised)
aam train \
  --table data/fall_train_only_all_outdoor.biom \
  --unifrac-matrix data/fall_train_only_all_outdoor.h5 \
  --metadata data/fall_train_only_all_outdoor.tsv \
  --metadata-column add_0c \
  --pretrained-encoder output/pretrain/checkpoints/best_model.pt \
  --output-dir output/train

# Predict
aam predict \
  --model output/train/checkpoints/best_model.pt \
  --table data/fall_train_only_all_outdoor.biom \
  --output predictions.tsv
```

Test data (781 samples) is included in `data/`.

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | Installation, prerequisites, first run |
| [User Guide](docs/user-guide.md) | Full CLI reference, all options |
| [How It Works](docs/how-it-works.md) | Concepts (science/ML) + Implementation (code) |
| [Architecture](ARCHITECTURE.md) | Technical design and rationale |

## Requirements

- Python 3.9-3.12
- PyTorch 2.3+ (CUDA, ROCm, or CPU)
- Pre-computed UniFrac matrix (generate with [unifrac-binaries](https://github.com/biocore/unifrac-binaries))

## Installation

```bash
# Basic
pip install -e .

# With development dependencies
pip install -e ".[dev,docs,training]"
```

See [Getting Started](docs/getting-started.md) for detailed installation instructions including PyTorch setup for different hardware.

## Training Strategy

**Stage 1: Pre-training (self-supervised)**
- Train encoder on UniFrac prediction + masked nucleotide prediction
- No labels required

**Stage 2: Fine-tuning (supervised)**
- Load pre-trained encoder with `--pretrained-encoder`
- Train target prediction head
- Optional: `--freeze-base` to freeze encoder weights

See [User Guide](docs/user-guide.md) for complete training options.

## Key Options

| Option | Description |
|--------|-------------|
| `--token-limit` | Max ASVs per sample (reduce for memory) |
| `--categorical-columns` | Condition on metadata columns |
| `--freeze-base` | Freeze pre-trained encoder |
| `--loss-type` | huber, mse, mae, quantile, asymmetric |
| `--fsdp` / `--data-parallel` | Multi-GPU training |

Run `aam train --help` for all options.

## Monitoring

```bash
tensorboard --logdir output/train/tensorboard
```

## Testing

```bash
pytest tests/ -v                           # Run all tests
pytest tests/ --cov=aam --cov-report=html  # With coverage
```

1391 tests covering data pipeline, models, training, and end-to-end workflows.

## Project Structure

```
aam/
├── data/          # Data loading, tokenization, datasets
├── models/        # Model architectures (ASVEncoder, SequencePredictor, etc.)
├── training/      # Training loop, losses, metrics
└── cli/           # Command-line interface
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for future enhancements and ideas.

## License

[License information here]
