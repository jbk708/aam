# Design Plan Index

**Quick Navigation Guide**

## Core Implementation Documents (00-13) ✅

**Architecture & Overview:**
- **[00_overview.md](00_overview.md)** - Architecture, design principles, training strategy

**Data Pipeline:**
- **[01_biom_loader.md](01_biom_loader.md)** - BIOM table loading and rarefaction
- **[02_unifrac_computer.md](02_unifrac_computer.md)** - UniFrac distance computation
- **[03_dataset_tokenizer.md](03_dataset_tokenizer.md)** - Dataset and tokenization

**Core Components:**
- **[04_core_layers.md](04_core_layers.md)** - AttentionPooling and PositionEmbedding
- **[05_transformer.md](05_transformer.md)** - Transformer encoder

**Model Architecture:**
- **[06_asv_encoder.md](06_asv_encoder.md)** - ASV-level sequence processing
- **[07_base_sequence_encoder.md](07_base_sequence_encoder.md)** - Sample-level processing
- **[08_sequence_encoder.md](08_sequence_encoder.md)** - Encoder with UniFrac prediction
- **[09_sequence_predictor.md](09_sequence_predictor.md)** - Main prediction model

**Training Infrastructure:**
- **[10_training_losses.md](10_training_losses.md)** - Loss functions and metrics
- **[11_training_loop.md](11_training_loop.md)** - Training and validation loops
- **[12_cli_interface.md](12_cli_interface.md)** - Command-line interface
- **[13_testing.md](13_testing.md)** - Testing strategy

## Feature & Analysis Documents (14-20)

**Features:**
- **[14_training_features.md](14_training_features.md)** - Phase 8 training features and enhancements ✅

**Analysis & Planning:**
- **[19_unifrac_underfitting_analysis.md](19_unifrac_underfitting_analysis.md)** - UniFrac underfitting analysis and fixes ✅
- **[20_optimization_plan.md](20_optimization_plan.md)** - Performance optimization plan 📋
- **[22_sigmoid_saturation_fix.md](22_sigmoid_saturation_fix.md)** - Fix sigmoid saturation in distance normalization 🔴 HIGH PRIORITY
- **[FUTURE_WORK.md](FUTURE_WORK.md)** - Future enhancements and next steps 📋

## Quick Reference

**Getting Started:**
1. Read [00_overview.md](00_overview.md) for architecture overview
2. Review [README.md](README.md) for implementation status
3. Check [FUTURE_WORK.md](FUTURE_WORK.md) for next steps

**Implementation Status:**
- ✅ All core components (00-13) completed
- ✅ All Phase 8 features (14) completed
- ✅ All Phase 9 fixes (19) completed
- 📋 Phase 10 optimizations (20) planned

**Related Documents:**
- See `.agents/PYTORCH_PORTING_TICKETS.md` for ticket tracking
- See `.agents/WORKFLOW.md` for implementation workflow
