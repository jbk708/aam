# Design Plan Index

**Last Updated:** 2026-01-27

Design documents for AAM (Attention All Microbes) PyTorch implementation. These are internal implementation documents used during development.

## Quick Navigation

- For user documentation, see [docs/](../docs/)
- For architecture overview, see [ARCHITECTURE.md](../ARCHITECTURE.md)
- For contributing, see [CONTRIBUTING.md](../CONTRIBUTING.md)

---

## Core Implementation (Complete)

All core components are implemented and tested.

### Data Pipeline

| Document | Status | Description |
|----------|--------|-------------|
| [01_biom_loader.md](01_biom_loader.md) | Complete | BIOM loading and rarefaction |
| [02_unifrac_loader.md](02_unifrac_loader.md) | Complete | Pre-computed UniFrac matrix loading |
| [03_dataset_tokenizer.md](03_dataset_tokenizer.md) | Complete | Dataset and tokenization |

### Core Components

| Document | Status | Description |
|----------|--------|-------------|
| [04_core_layers.md](04_core_layers.md) | Complete | AttentionPooling and PositionEmbedding |
| [05_transformer.md](05_transformer.md) | Complete | Transformer encoder |

### Model Architecture

| Document | Status | Description |
|----------|--------|-------------|
| [06_asv_encoder.md](06_asv_encoder.md) | Complete | ASV-level processing |
| [07_base_sequence_encoder.md](07_base_sequence_encoder.md) | Complete | Sample-level processing |
| [08_sequence_encoder.md](08_sequence_encoder.md) | Complete | Encoder with UniFrac |
| [09_sequence_predictor.md](09_sequence_predictor.md) | Complete | Main prediction model |

### Training

| Document | Status | Description |
|----------|--------|-------------|
| [10_training_losses.md](10_training_losses.md) | Complete | Loss functions |
| [11_training_loop.md](11_training_loop.md) | Complete | Training loop |
| [12_cli_interface.md](12_cli_interface.md) | Complete | CLI package |
| [13_testing.md](13_testing.md) | Complete | Testing strategy |
| [14_training_features.md](14_training_features.md) | Complete | Advanced training features |

---

## Feature Extensions (Complete)

| Document | Status | Description |
|----------|--------|-------------|
| [15_categorical_features.md](15_categorical_features.md) | Complete | Categorical metadata integration (CAT-1 to CAT-7) |
| [16_regressor_optimization.md](16_regressor_optimization.md) | Complete | Regressor optimization (REG-1 to REG-8) |
| [17_attention_fusion.md](17_attention_fusion.md) | Complete | Attention-based fusion (FUS-1 to FUS-3) |

---

## Future Work

| Document | Description |
|----------|-------------|
| [FUTURE_WORK.md](FUTURE_WORK.md) | Outstanding enhancements and ideas |

---

## Archived Documents

Historical analysis documents moved to `archive/`:

| Document | Description |
|----------|-------------|
| [archive/19_unifrac_underfitting_analysis.md](archive/19_unifrac_underfitting_analysis.md) | UniFrac underfitting investigation |
| [archive/20_optimization_plan.md](archive/20_optimization_plan.md) | Performance optimization analysis |
| [archive/22_sigmoid_saturation_fix.md](archive/22_sigmoid_saturation_fix.md) | Sigmoid saturation issue fix |
| [archive/23_memory_optimization_plan.md](archive/23_memory_optimization_plan.md) | Memory optimization strategies |

---

## Related Files

| File | Purpose |
|------|---------|
| [00_overview.md](00_overview.md) | Architecture overview |
| [README.md](README.md) | Design plan overview |

---

## Ticket Tracking

See `.agents/` for ticket files:
- `TICKET_OVERVIEW.md` - Summary of outstanding work
- `PYTORCH_PORTING_TICKETS.md` - Core implementation tickets
- `REGRESSOR_OPTIMIZATION_TICKETS.md` - Regressor optimization tickets
- `DOCUMENTATION_TICKETS.md` - Documentation tickets
- `ARCHIVED_TICKETS.md` - Completed tickets
