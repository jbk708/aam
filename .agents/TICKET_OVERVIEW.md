# Ticket Overview

**Last Updated:** 2025-12-19
**Status:** 21 outstanding tickets (~55-84 hours)

## Quick Links
- **Code cleanup:** `CLEANUP_TICKETS.md` (NEW)
- **PyTorch work:** `PYTORCH_PORTING_TICKETS.md`
- **Cosmos onboarding:** `COSMOS_ONBOARDING_TICKETS.md`
- **Completed work:** `ARCHIVED_TICKETS.md`
- **Workflow:** `WORKFLOW.md`

---

## Outstanding Tickets by Priority

### HIGH (5 tickets, ~12-18 hours)

| Ticket | Description | Effort |
|--------|-------------|--------|
| **CLN-2** | Remove Dead Code (stripe_mode, lazy_unifrac) | 1-2h |
| **COS-1.2** | Cosmos Environment Setup Script | 2-3h |
| **COS-3.1** | Create SLURM Job Scripts | 3-4h |
| **COS-5.2** | Numerical Validation (CUDA vs ROCm) | 3-4h |
| **COS-7.1** | Cosmos Quick Start Guide | 2-3h |

### MEDIUM (9 tickets, ~28-44 hours)

| Ticket | Description | Effort |
|--------|-------------|--------|
| **CLN-3** | Add Package __init__.py Exports | 1h |
| **CLN-4** | Fix Mypy Configuration | 1h |
| **COS-1.1** | ROCm Singularity Container Definition | 4-6h |
| **COS-2.2** | Unified Memory Optimization for MI300A | 4-6h |
| **COS-3.2** | Data Management Scripts | 2-3h |
| **COS-5.1** | ROCm CI/CD Pipeline | 4-6h |
| **COS-6.1** | MI300A Performance Profiling | 4-6h |
| **COS-6.2** | Flash Attention for ROCm | 4-6h |
| **COS-7.2** | Cosmos Best Practices Guide | 2-3h |
| **PYT-21.3** | Regressor Head Optimization | 4-6h |

### LOW (7 tickets, ~34-52 hours)

| Ticket | Description | Effort |
|--------|-------------|--------|
| **CLN-5** | Extract CLI Helper Modules | 3-4h |
| **CLN-6** | Extract Trainer Validation Logic | 3-4h |
| **PYT-10.6** | Multi-GPU Training (DDP) - CUDA | 8-12h |
| **PYT-21.5** | Skip Nucleotide Predictions During Fine-Tuning | 2-3h |
| **PYT-18.5** | Lazy Sample Embedding Computation | 4-6h |
| **PYT-18.6** | Memory-Aware Dynamic Batching | 4-6h |
| **COS-4.2** | FSDP Support for Large Models | 8-12h |

### BACKLOG (Phases 12-17)

Future enhancement phases not yet broken into tickets:
- Phase 12: FSDP, Batch Size Optimization
- Phase 13: Attention Visualization, Feature Importance, Encoder Types
- Phase 14: Streaming Data, Augmentation
- Phase 15: Experiment Tracking, Hyperparameter Optimization
- Phase 16: Benchmarking, Error Analysis
- Phase 17: Docs, Tutorials, ONNX, Docker

---

## Recently Completed (2025-12-19)

- **CLN-1:** Remove Deprecated UniFrac Modules - Deleted 4165 lines (4 modules + 7 debug scripts)
- **COS-2.1:** ROCm Compatibility Audit - No changes needed, code already compatible

## Previously Archived

- **PYT-21.1/21.2/21.4:** Transfer Learning & Fine-Tuning fixes
- **PYT-20.1:** Masked Autoencoder for Nucleotide Prediction
- **PYT-19.1:** Test Fixes
- **PYT-18.1-18.3:** Memory Optimization
- **PYT-12.3:** Sequence Tokenization Caching
- **COS-1.0:** Native ROCm Environment
- **COS-4.1:** DDP for ROCm/RCCL

---

## Recommended Next Steps

### Code Cleanup (Priority)
1. ~~**CLN-1** - Remove deprecated UniFrac modules~~ ✅ COMPLETE
2. **CLN-2** - Remove stripe_mode/lazy_unifrac dead code

### Cosmos Onboarding (After Cleanup)
1. **COS-3.1** - SLURM job scripts (enables running jobs)
2. **COS-5.2** - Numerical validation (ensures correctness)
3. **COS-1.2** - Environment setup script
4. **COS-7.1** - Quick start guide

### PyTorch Improvements (As Needed)
1. **PYT-21.3** - Regressor head optimization
