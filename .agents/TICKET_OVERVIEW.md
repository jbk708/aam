# Ticket Overview

**Last Updated:** 2026-01-05
**Status:** 21 outstanding tickets (~88-129 hours)

## Quick Links
- **Categorical features:** `CATEGORICAL_FEATURE_TICKETS.md`
- **PyTorch work:** `PYTORCH_PORTING_TICKETS.md`
- **Cosmos onboarding:** `COSMOS_ONBOARDING_TICKETS.md`
- **Completed work:** `ARCHIVED_TICKETS.md`
- **Workflow:** `WORKFLOW.md`

---

## Outstanding Tickets by Priority

### HIGH (3 tickets, ~10-14 hours)

| Ticket | Description | Effort |
|--------|-------------|--------|
| **CAT-2** | Dataset Pipeline — Categorical Encoding | 3-4h |
| **CAT-3** | CategoricalEmbedder Module | 3-4h |
| **CAT-4** | SequencePredictor Integration | 4-6h |

### MEDIUM (9 tickets, ~30-46 hours)

| Ticket | Description | Effort |
|--------|-------------|--------|
| **CAT-5** | CLI and Configuration Updates | 3-4h |
| **CAT-6** | Checkpoint Compatibility and Transfer Learning | 3-4h |
| **CAT-7** | Documentation and Testing | 3-4h |
| **COS-2.2** | Unified Memory Optimization for MI300A | 4-6h |
| **COS-3.2** | Data Management Scripts | 2-3h |
| **COS-5.1** | ROCm CI/CD Pipeline | 4-6h |
| **COS-6.1** | MI300A Performance Profiling | 4-6h |
| **COS-6.2** | Flash Attention for ROCm | 4-6h |
| **COS-7.2** | Cosmos Best Practices Guide | 2-3h |

### LOW (9 tickets, ~55-80 hours)

| Ticket | Description | Effort |
|--------|-------------|--------|
| **COS-1.1** | ROCm Singularity Container Definition | 4-6h |
| **COS-3.1** | Create SLURM Job Scripts | 3-4h |
| **PYT-10.6** | Multi-GPU Training (DDP) - CUDA | 8-12h |
| **PYT-18.5** | Lazy Sample Embedding Computation | 4-6h |
| **PYT-18.6** | Memory-Aware Dynamic Batching | 4-6h |
| **PYT-12.1** | FSDP | 12-16h |
| **PYT-12.2** | Batch Size Optimization | 4-6h |
| **COS-4.2** | FSDP Support for Large Models | 8-12h |

### BACKLOG (Phases 13-17)

Future enhancement phases (~50+ hours):
- Phase 13: Attention Visualization, Feature Importance, Encoder Types
- Phase 14: Streaming Data, Augmentation
- Phase 15: Experiment Tracking, Hyperparameter Optimization
- Phase 16: Benchmarking, Error Analysis
- Phase 17: Docs, Tutorials, ONNX, Docker

---

## Recently Completed

**CAT-1: Categorical Metadata Schema Definition** (2026-01-05)
- `CategoricalColumnConfig`: per-column settings (name, cardinality, embed_dim, required)
- `CategoricalSchema`: collection with validation (duplicates, invalid values, empty names)
- Factory method `from_column_names()` for simple schemas
- 24 new tests in `tests/test_categorical.py`

**PYT-21.5: Skip Nucleotide Predictions During Fine-Tuning** (2026-01-05)
- Already implemented via existing logic chain
- `--freeze-base` auto-disables `nuc_penalty` → `return_nucleotides=False` → computation skipped
- No additional flags needed; behavior works automatically

**COS-5.2: Numerical Validation (CUDA vs ROCm)** (2025-12-19)
- Golden file validation infrastructure in `tests/validation/`
- `generate_golden_outputs()`: creates reference outputs on one platform
- `compare_golden_outputs()`: validates against golden on another platform
- CLI: `python -m tests.validation.numerical_validation generate|compare`

**COS-1.2 & COS-7.1: Merged to README**
- Cosmos setup instructions added to README (conda + pip install)
- Quick start guide merged into setup section

**PYT-21.3: Regressor Head Optimization**
- Unbounded regression by default (no sigmoid)
- LayerNorm before target projection (default: enabled)
- Learnable output scale/bias (opt-in)
- Xavier weight initialization

**Code Cleanup Phase (CLN) - All 6 tickets complete:**
- CLN-1: Remove Deprecated UniFrac Modules (~4165 lines deleted)
- CLN-2: Remove Dead Code (stripe_mode, lazy_unifrac)
- CLN-3: Add Package `__init__.py` Exports
- CLN-4: Fix Type Errors (ty)
- CLN-5: Extract CLI Helper Modules
- CLN-6: Extract Trainer Validation Logic

---

## Recommended Next Steps

### Categorical Features (In Progress)
1. ~~**CAT-1** - Schema definition~~ ✓ Complete
2. **CAT-2** - Dataset encoding (parallel with CAT-3)
3. **CAT-3** - Embedder module (parallel with CAT-2)
4. **CAT-4** - Model integration

### Cosmos Onboarding
1. **COS-5.1** - ROCm CI/CD pipeline
2. **COS-3.2** - Data management scripts

### PyTorch Improvements (As Needed)
1. **PYT-10.6** - Multi-GPU DDP testing
