# CLI Interface

**Status:** Complete

## Overview
Modular command-line interface for training and inference. Implemented in `aam/cli/` package.

## Package Structure

```
aam/cli/
├── __init__.py      # Main CLI group, command registration
├── __main__.py      # Entry point for python -m aam.cli
├── utils.py         # Shared utilities (setup_logging, setup_device, etc.)
├── train.py         # Train command
├── pretrain.py      # Pretrain command
└── predict.py       # Predict command
```

## Commands

### train
Full training with metadata targets.
```bash
python -m aam.cli train \
  --table <biom_file> \
  --unifrac-matrix <matrix.npy> \
  --metadata <metadata.tsv> \
  --metadata-column <target_column> \
  --output-dir <output_dir>
```

### pretrain
Stage 1 self-supervised training (UniFrac + nucleotide prediction).
```bash
python -m aam.cli pretrain \
  --table <biom_file> \
  --unifrac-matrix <matrix.npy> \
  --output-dir <output_dir>
```

### predict
Generate predictions from trained model.
```bash
python -m aam.cli predict \
  --model <checkpoint.pt> \
  --table <biom_file> \
  --output <predictions.tsv>
```

## Key Features
- Comprehensive argument validation and error handling
- Transfer learning support (pretrained encoder, freeze base)
- Memory optimization options (gradient accumulation, chunked processing)
- TensorBoard logging integration
- Reproducibility support (random seed)
- Pre-computed UniFrac matrices via `--unifrac-matrix`

## Shared Utilities (utils.py)
- `setup_logging()` - Configure logging to console and file
- `setup_device()` - Setup CPU or CUDA device
- `setup_random_seed()` - Set seeds for reproducibility
- `validate_file_path()` - Validate file existence
- `validate_arguments()` - Validate CLI arguments

## Testing
Comprehensive unit tests in `tests/test_cli.py` (63 tests, all passing).
