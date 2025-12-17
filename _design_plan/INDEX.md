# Design Plan Index

## Core Implementation (All Complete)

**Data Pipeline:**
- [01_biom_loader.md](01_biom_loader.md) - BIOM loading and rarefaction
- [02_unifrac_computer.md](02_unifrac_computer.md) - UniFrac (use `UniFracLoader` for pre-computed matrices)
- [03_dataset_tokenizer.md](03_dataset_tokenizer.md) - Dataset and tokenization

**Core Components:**
- [04_core_layers.md](04_core_layers.md) - AttentionPooling and PositionEmbedding
- [05_transformer.md](05_transformer.md) - Transformer encoder

**Model Architecture:**
- [06_asv_encoder.md](06_asv_encoder.md) - ASV-level processing
- [07_base_sequence_encoder.md](07_base_sequence_encoder.md) - Sample-level processing
- [08_sequence_encoder.md](08_sequence_encoder.md) - Encoder with UniFrac
- [09_sequence_predictor.md](09_sequence_predictor.md) - Main prediction model

**Training:**
- [10_training_losses.md](10_training_losses.md) - Loss functions
- [11_training_loop.md](11_training_loop.md) - Training loop
- [12_cli_interface.md](12_cli_interface.md) - CLI
- [13_testing.md](13_testing.md) - Testing strategy

## Analysis & Planning

- [14_training_features.md](14_training_features.md) - Phase 8 features (complete)
- [19_unifrac_underfitting_analysis.md](19_unifrac_underfitting_analysis.md) - UniFrac fixes (complete)
- [20_optimization_plan.md](20_optimization_plan.md) - Performance optimizations
- [22_sigmoid_saturation_fix.md](22_sigmoid_saturation_fix.md) - Distance normalization fix (complete)
- [FUTURE_WORK.md](FUTURE_WORK.md) - Outstanding enhancements

## Quick Start

1. Read [00_overview.md](00_overview.md) for architecture
2. See `.agents/PYTORCH_PORTING_TICKETS.md` for outstanding work
3. See `.agents/ARCHIVED_TICKETS.md` for completed work
