# Outstanding Cosmos (AMD MI300A) Tickets

**Last Updated:** 2025-12-19
**Target System:** SDSC Cosmos - 168 AMD Instinct MI300A APUs (42 nodes × 4 APUs)
**Reference:** [Cosmos User Guide](https://www.sdsc.edu/systems/cosmos/user_guide.html)

**Completed:** COS-1.0 (Native ROCm), COS-1.2 (Setup - merged to README), COS-2.1 (Compatibility Audit), COS-4.1 (DDP), COS-5.2 (Numerical Validation), COS-7.1 (Quick Start - merged to README) - see `ARCHIVED_TICKETS.md`

---

## Phase 1: Environment Setup (1 remaining)

### COS-1.1: Create ROCm Singularity Container Definition
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Not Started

Create `singularity/aam-rocm.def` based on AMD Infinity Hub `rocm/pytorch` for reproducible production runs.

---

## Phase 2: ROCm Compatibility (1 remaining)

### COS-2.2: Unified Memory Optimization for MI300A
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Not Started

Research and optimize for MI300A's 128GB unified CPU/GPU memory. Potentially eliminate `.to(device)` overhead.

---

## Phase 3: SLURM Integration (2 remaining)

### COS-3.1: Create SLURM Job Scripts
**Priority:** LOW | **Effort:** 3-4 hours | **Status:** Not Started

Create template job scripts:
- `slurm/pretrain_single.sh` - Single APU
- `slurm/pretrain_node.sh` - Single node (4 APU)
- `slurm/pretrain_multi.sh` - Multi-node
- `slurm/train.sh`, `slurm/predict.sh`

### COS-3.2: Data Management Scripts
**Priority:** MEDIUM | **Effort:** 2-3 hours | **Status:** Not Started

Scripts for staging data to VAST and NVMe scratch.

---

## Phase 4: Multi-GPU (1 remaining)

### COS-4.2: FSDP Support for Large Models
**Priority:** LOW | **Effort:** 8-12 hours | **Status:** Not Started

FSDP for very large models (may not be needed with 128GB/APU).

---

## Phase 5: Testing & Validation (1 remaining)

### COS-5.1: ROCm CI/CD Pipeline
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Not Started

ROCm-specific tests and manual test procedure for Cosmos.

### COS-5.2: Numerical Validation (CUDA vs ROCm)
**Priority:** HIGH | **Effort:** 3-4 hours | **Status:** Complete

Golden file validation infrastructure in `tests/validation/`:
- `generate_golden_outputs()`: creates reference outputs on one platform
- `compare_golden_outputs()`: validates against golden on another platform
- CLI: `python -m tests.validation.numerical_validation generate|compare`
- Golden files generated on CPU (macOS arm64, torch 2.9.1)

---

## Phase 6: Performance (2 remaining)

### COS-6.1: MI300A Performance Profiling
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Not Started

Profile with `rocprof` and identify bottlenecks.

### COS-6.2: Flash Attention for ROCm
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Not Started

Verify/optimize attention implementation for ROCm.

---

## Phase 7: Documentation (1 remaining)

### COS-7.2: Cosmos Best Practices Guide
**Priority:** MEDIUM | **Effort:** 2-3 hours | **Status:** Not Started

Create `docs/cosmos_best_practices.md`.

---

## Summary

| Phase | Remaining | Est. Hours | Priority |
|-------|-----------|------------|----------|
| 1: Environment | 1 | 4-6 | MEDIUM |
| 2: ROCm | 1 | 4-6 | MEDIUM |
| 3: SLURM | 2 | 5-7 | LOW/MEDIUM |
| 4: Multi-GPU | 1 | 8-12 | LOW |
| 5: Testing | 1 | 4-6 | MEDIUM |
| 6: Performance | 2 | 8-12 | MEDIUM |
| 7: Documentation | 1 | 2-3 | MEDIUM |
| **Total** | **9** | **35-52** | |

## Recommended Order

1. **COS-1.1** - Singularity container (for reproducibility)
2. Remaining based on need
