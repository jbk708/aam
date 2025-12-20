# Ticket Overview

**Last Updated:** 2025-12-19
**Status:** 16 outstanding tickets (~72-106 hours)

## Quick Links
- **PyTorch work:** `PYTORCH_PORTING_TICKETS.md`
- **Cosmos onboarding:** `COSMOS_ONBOARDING_TICKETS.md`
- **Completed work:** `ARCHIVED_TICKETS.md`
- **Workflow:** `WORKFLOW.md`

---

## Outstanding Tickets by Priority

### HIGH (0 tickets)

No high priority tickets remaining.

### MEDIUM (7 tickets, ~25-40 hours)

| Ticket | Description | Effort |
|--------|-------------|--------|
| **COS-1.1** | ROCm Singularity Container Definition | 4-6h |
| **COS-2.2** | Unified Memory Optimization for MI300A | 4-6h |
| **COS-3.2** | Data Management Scripts | 2-3h |
| **COS-5.1** | ROCm CI/CD Pipeline | 4-6h |
| **COS-6.1** | MI300A Performance Profiling | 4-6h |
| **COS-6.2** | Flash Attention for ROCm | 4-6h |
| **COS-7.2** | Cosmos Best Practices Guide | 2-3h |

### LOW (9 tickets, ~53-77 hours)

| Ticket | Description | Effort |
|--------|-------------|--------|
| **COS-3.1** | Create SLURM Job Scripts | 3-4h |
| **PYT-10.6** | Multi-GPU Training (DDP) - CUDA | 8-12h |
| **PYT-21.5** | Skip Nucleotide Predictions During Fine-Tuning | 2-3h |
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

## Recently Completed (2025-12-19)

**COS-5.2: Numerical Validation (CUDA vs ROCm)**
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

### Cosmos Onboarding (Priority)
1. **COS-1.1** - Singularity container (reproducibility)
2. **COS-5.1** - ROCm CI/CD pipeline

### PyTorch Improvements (As Needed)
1. **PYT-21.5** - Skip nucleotide predictions during fine-tuning
2. **PYT-10.6** - Multi-GPU DDP testing
