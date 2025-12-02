# PyTorch Implementation Plan

This directory contains a comprehensive plan for building the AAM (Attention All Microbes) project natively in PyTorch.

## Overview

The implementation plan is organized into 13 documents, each covering a specific component:

1. **[00_overview.md](00_overview.md)** - Architecture diagram, design principles, training strategy
2. **[01_biom_loader.md](01_biom_loader.md)** - BIOM table loading and rarefaction
3. **[02_unifrac_computer.md](02_unifrac_computer.md)** - UniFrac distance computation
4. **[03_dataset_tokenizer.md](03_dataset_tokenizer.md)** - Dataset and tokenization
5. **[04_core_layers.md](04_core_layers.md)** - AttentionPooling and PositionEmbedding
6. **[05_transformer.md](05_transformer.md)** - Transformer encoder
7. **[06_asv_encoder.md](06_asv_encoder.md)** - ASV-level sequence processing
8. **[07_base_sequence_encoder.md](07_base_sequence_encoder.md)** - Sample-level processing
9. **[08_sequence_encoder.md](08_sequence_encoder.md)** - UniFrac prediction head
10. **[09_sequence_regressor.md](09_sequence_regressor.md)** - Main regression model
11. **[10_training_losses.md](10_training_losses.md)** - Loss functions and metrics
12. **[11_training_loop.md](11_training_loop.md)** - Training and validation loops
13. **[12_cli_interface.md](12_cli_interface.md)** - Command-line interface
14. **[13_testing.md](13_testing.md)** - Testing strategy

## Training Strategy

### Recommended: Staged Training

**Stage 1: Pre-train SequenceEncoder**
- Train on UniFrac + nucleotide prediction (self-supervised)
- No target labels required
- Save checkpoint

**Stage 2: Train SequenceRegressor**
- Load pre-trained SequenceEncoder
- Option A: Freeze base (`freeze_base=True`) - faster
- Option B: Fine-tune jointly (`freeze_base=False`) - better performance

See **[00_overview.md](00_overview.md)** for detailed training strategy.

## Implementation Order

1. **Data Pipeline** (01-03)
   - BIOM loading and rarefaction
   - UniFrac computation
   - Tokenization and dataset creation

2. **Core Components** (04-05)
   - Attention pooling
   - Position embeddings
   - Transformer encoder

3. **Model Architecture** (06-09)
   - ASVEncoder
   - BaseSequenceEncoder
   - SequenceEncoder (base model)
   - SequenceRegressor (composes encoder)

4. **Training** (10-12)
   - Loss functions
   - Training loop (with staged training support)
   - CLI interface

5. **Testing** (13)
   - Unit tests
   - Integration tests
   - Validation

## Key Design Principles

### 1. Native PyTorch
- Built using standard PyTorch patterns
- No references to TensorFlow implementation
- Follow PyTorch conventions

### 2. Composition Over Inheritance
- SequenceRegressor composes SequenceEncoder
- Enables flexible base model swapping
- Supports freezing base model

### 3. Multi-Task Learning
- Parallel tasks share base embeddings
- Self-supervised learning (UniFrac, nucleotides)
- Auxiliary tasks (count prediction)
- Primary task (target prediction)

### 4. Staged Training
- Pre-train SequenceEncoder (self-supervised)
- Fine-tune SequenceRegressor (with optional freezing)

## File Structure

```
aam/
├── data/
│   ├── __init__.py
│   ├── biom_loader.py
│   ├── unifrac_computer.py
│   ├── tokenizer.py
│   └── dataset.py
├── models/
│   ├── __init__.py
│   ├── attention_pooling.py
│   ├── position_embedding.py
│   ├── transformer.py
│   ├── asv_encoder.py
│   ├── base_sequence_encoder.py
│   ├── sequence_encoder.py
│   └── sequence_regressor.py
├── training/
│   ├── __init__.py
│   ├── losses.py
│   ├── metrics.py
│   └── trainer.py
└── utils.py
```

## Dependencies

### Required
- `torch` >= 2.0
- `numpy`, `pandas`
- `biom-format` - BIOM table I/O
- `scikit-bio` - Phylogenetic tree handling
- `unifrac-binaries` - UniFrac distance computation (https://github.com/biocore/unifrac-binaries)
- `scikit-learn` - Metrics

### Optional
- `pytorch-lightning` - Training utilities
- `wandb` - Experiment tracking
- `tensorboard` - Visualization

## Quick Start

1. Read **[00_overview.md](00_overview.md)** for architecture and training strategy
2. Start with **[01_biom_loader.md](01_biom_loader.md)** - Load BIOM tables
3. Implement **[02_unifrac_computer.md](02_unifrac_computer.md)** - Compute UniFrac
4. Build data pipeline: **[03_dataset_tokenizer.md](03_dataset_tokenizer.md)**
5. Implement model components incrementally
6. Wire up training: **[10_training_losses.md](10_training_losses.md)** → **[11_training_loop.md](11_training_loop.md)**
7. Test everything: **[13_testing.md](13_testing.md)**

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
