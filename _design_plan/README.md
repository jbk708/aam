# PyTorch Implementation Plan

**Status:** ✅ Implementation Complete

This directory contains the design plan for the AAM (Attention All Microbes) project natively in PyTorch. All components have been implemented and tested.

## Overview

The implementation plan is organized into 13 documents, each covering a specific component. All components are **completed**:

1. **[00_overview.md](00_overview.md)** - Architecture diagram, design principles, training strategy ✅
2. **[01_biom_loader.md](01_biom_loader.md)** - BIOM table loading and rarefaction ✅
3. **[02_unifrac_computer.md](02_unifrac_computer.md)** - UniFrac distance computation ✅
4. **[03_dataset_tokenizer.md](03_dataset_tokenizer.md)** - Dataset and tokenization ✅
5. **[04_core_layers.md](04_core_layers.md)** - AttentionPooling and PositionEmbedding ✅
6. **[05_transformer.md](05_transformer.md)** - Transformer encoder ✅
7. **[06_asv_encoder.md](06_asv_encoder.md)** - ASV-level sequence processing ✅
8. **[07_base_sequence_encoder.md](07_base_sequence_encoder.md)** - Sample-level processing ✅
9. **[08_sequence_encoder.md](08_sequence_encoder.md)** - UniFrac prediction head ✅
10. **[09_sequence_predictor.md](09_sequence_predictor.md)** - Main prediction model ✅
11. **[10_training_losses.md](10_training_losses.md)** - Loss functions and metrics ✅
12. **[11_training_loop.md](11_training_loop.md)** - Training and validation loops ✅
13. **[12_cli_interface.md](12_cli_interface.md)** - Command-line interface ✅
14. **[13_testing.md](13_testing.md)** - Testing strategy ✅

## New Features

15. **[14_tensorboard_overlay.md](14_tensorboard_overlay.md)** - TensorBoard train/val overlay ⏳
16. **[15_save_best_model.md](15_save_best_model.md)** - Save single best model file ⏳
17. **[16_early_stopping_default.md](16_early_stopping_default.md)** - Early stopping default to 10 epochs ✅
18. **[17_validation_prediction_plots.md](17_validation_prediction_plots.md)** - Validation prediction plots ⏳

## Training Strategy

### Recommended: Staged Training

**Stage 1: Pre-train SequenceEncoder**
- Train on UniFrac + nucleotide prediction (self-supervised)
- No target labels required
- Save checkpoint

**Stage 2: Train SequencePredictor**
- Load pre-trained SequenceEncoder
- Option A: Freeze base (`freeze_base=True`) - faster
- Option B: Fine-tune jointly (`freeze_base=False`) - better performance

See **[00_overview.md](00_overview.md)** for detailed training strategy.

## Implementation Status

All components have been implemented and tested:

1. **Data Pipeline** (01-03) ✅
   - BIOM loading and rarefaction (`aam/data/biom_loader.py`)
   - UniFrac computation (`aam/data/unifrac.py`)
   - Tokenization and dataset creation (`aam/data/tokenizer.py`, `aam/data/dataset.py`)

2. **Core Components** (04-05) ✅
   - Attention pooling (`aam/models/attention_pooling.py`)
   - Position embeddings (`aam/models/position_embedding.py`)
   - Transformer encoder (`aam/models/transformer.py`)

3. **Model Architecture** (06-09) ✅
   - ASVEncoder (`aam/models/asv_encoder.py`)
   - SampleSequenceEncoder (`aam/models/sample_sequence_encoder.py`)
   - SequenceEncoder (`aam/models/sequence_encoder.py`)
   - SequencePredictor (`aam/models/sequence_predictor.py`)

4. **Training** (10-12) ✅
   - Loss functions (`aam/training/losses.py`)
   - Metrics (`aam/training/metrics.py`)
   - Training loop (`aam/training/trainer.py`)
   - CLI interface (`aam/cli.py`)

5. **Testing** (13) ✅
   - Unit tests: 333 tests passing (94% coverage)
   - Integration tests: 13 comprehensive tests
   - End-to-end tests: 3 slow tests with real data

## Key Design Principles

### 1. Native PyTorch
- Built using standard PyTorch patterns
- No references to TensorFlow implementation
- Follow PyTorch conventions

### 2. Composition Over Inheritance
- SequencePredictor composes SequenceEncoder
- Enables flexible base model swapping
- Supports freezing base model

### 3. Multi-Task Learning
- Parallel tasks share base embeddings
- Self-supervised learning (UniFrac, nucleotides)
- Auxiliary tasks (count prediction)
- Primary task (target prediction)

### 4. Staged Training
- Pre-train SequenceEncoder (self-supervised)
- Fine-tune SequencePredictor (with optional freezing)

## File Structure

```
aam/
├── data/
│   ├── __init__.py
│   ├── biom_loader.py
│   ├── unifrac.py
│   ├── tokenizer.py
│   └── dataset.py
├── models/
│   ├── __init__.py
│   ├── attention_pooling.py
│   ├── position_embedding.py
│   ├── transformer.py
│   ├── asv_encoder.py
│   ├── sample_sequence_encoder.py
│   ├── sequence_encoder.py
│   └── sequence_predictor.py
├── training/
│   ├── __init__.py
│   ├── losses.py
│   ├── metrics.py
│   └── trainer.py
└── cli.py
```

## Dependencies

### Required
- `torch` >= 2.0
- `numpy`, `pandas`
- `biom-format` - BIOM table I/O
- `scikit-bio` - Phylogenetic tree handling
- `unifrac` - UniFrac distance computation (package name is `unifrac`, install via `pip install unifrac` or `conda install -c biocore unifrac`)
- `scikit-learn` - Metrics
- `click` - CLI framework

### Optional
- `pytorch-lightning` - Training utilities
- `wandb` - Experiment tracking
- `tensorboard` - Visualization

## Quick Start

The implementation is complete. To use the project:

1. Read **[00_overview.md](00_overview.md)** for architecture and training strategy
2. Install dependencies: `pip install -e .` or `conda env create -f environment.yml`
3. Run tests: `pytest tests/ -v`
4. Train a model: `python -m aam.cli train --help` for usage
5. See individual component docs for implementation details

## Test Data

Test data files are available in the `./data/` folder for integration and end-to-end testing:
- `./data/fall_train_only_all_outdoor.biom` - BIOM table for testing
- `./data/fall_train_only_all_outdoor.tsv` - TSV version (alternative format)
- `./data/all-outdoors_sepp_tree.nwk` - Phylogenetic tree for UniFrac computation

Use these files for:
- Integration tests (data pipeline, model components)
- End-to-end tests (full training workflow)
- Validation tests (UniFrac computation accuracy)

For unit tests, generate small synthetic datasets as needed.

## Notes

- Each document provides clear implementation requirements
- Minimal reference code - focus on instructions
- Agent-friendly structure for automated implementation
- Test incrementally as you build
- Use test data from `./data/` folder for realistic validation
