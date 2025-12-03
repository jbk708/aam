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

### Pre-training (Stage 1: Self-supervised)

Pre-train SequenceEncoder on UniFrac + nucleotide prediction:

```bash
python -m aam.cli pretrain \
  --table <biom_file> \
  --tree <tree_file> \
  --output-dir <output_dir> \
  --batch-size 8 \
  --epochs 100
```

### Training (Stage 2: Fine-tuning)

Train SequencePredictor with optional pre-trained encoder:

```bash
python -m aam.cli train \
  --table <biom_file> \
  --tree <tree_file> \
  --metadata <metadata.tsv> \
  --metadata-column <target_column> \
  --output-dir <output_dir> \
  --pretrained-encoder <encoder_checkpoint.pt> \
  --freeze-base \
  --batch-size 8 \
  --epochs 100
```

### Inference

Run predictions on new data:

```bash
python -m aam.cli predict \
  --model <model_checkpoint.pt> \
  --table <biom_file> \
  --tree <tree_file> \
  --output <predictions.tsv>
```

### Key Options

**Training:**
- `--pretrained-encoder`: Load pre-trained SequenceEncoder checkpoint
- `--freeze-base`: Freeze base model parameters (faster training)
- `--classifier`: Use classification mode (requires `--out-dim > 1`)
- `--batch-size`: Must be even for unweighted UniFrac

**Model:**
- `--embedding-dim`: Embedding dimension (default: 128)
- `--attention-heads`: Number of attention heads (default: 4)
- `--attention-layers`: Number of transformer layers (default: 4)
- `--max-bp`: Maximum base pairs per sequence (default: 150)
- `--token-limit`: Maximum ASVs per sample (default: 1024)

**Data:**
- `--unifrac-metric`: 'unifrac' or 'faith_pd' (default: 'unifrac')
- `--rarefy-depth`: Rarefaction depth (default: 5000)
- `--test-size`: Validation split size (default: 0.2)

See `python -m aam.cli <command> --help` for full options.

## Testing

Run the full test suite:

```bash
pytest tests/ -v --tb=short --no-header -rA
```

**Test Coverage:**
- 359+ tests covering all components
- Unit tests for data pipeline, models, training
- Integration tests for end-to-end workflows
- CUDA GPU support verified

**Quick Test Commands:**
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_biom_loader.py -v

# Run with coverage
pytest tests/ --cov=aam --cov-report=html

# Run only CUDA tests (if CUDA available)
pytest tests/ -v -k "cuda or device"
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## Documentation

Implementation details and design decisions are documented in `_design_plan/`.





