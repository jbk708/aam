# Design Plan Index

## Core Implementation (All Complete)

**Data Pipeline:**
- [01_biom_loader.md](01_biom_loader.md) - BIOM loading and rarefaction
- [02_unifrac_loader.md](02_unifrac_loader.md) - Pre-computed UniFrac matrix loading
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
- [12_cli_interface.md](12_cli_interface.md) - CLI package
- [13_testing.md](13_testing.md) - Testing strategy
- [14_training_features.md](14_training_features.md) - Advanced training features

## Planned Features

- [15_categorical_features.md](15_categorical_features.md) - Categorical metadata integration (CAT-1 through CAT-7)
- [16_regressor_optimization.md](16_regressor_optimization.md) - Regressor optimization for multi-environment data (REG-1 through REG-9)
- [17_attention_fusion.md](17_attention_fusion.md) - Attention-based categorical fusion and cleanup (FUS-1 through FUS-3, CLN-1 through CLN-6)

## Future Work

- [FUTURE_WORK.md](FUTURE_WORK.md) - Outstanding enhancements

## Historical Analysis (Archived)

See `archive/` for completed analysis documents.

## Quick Start

1. Read [00_overview.md](00_overview.md) for architecture
2. See `.agents/TICKET_OVERVIEW.md` for outstanding work
3. See `.agents/REGRESSOR_OPTIMIZATION_TICKETS.md` for regressor optimization tickets (REG-1 through REG-9)
