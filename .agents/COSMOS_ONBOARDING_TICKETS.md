# Outstanding Cosmos (AMD MI300A) Tickets

**Last Updated:** 2026-01-07
**Target System:** SDSC Cosmos - 168 AMD Instinct MI300A APUs (42 nodes × 4 APUs)
**Reference:** [Cosmos User Guide](https://www.sdsc.edu/systems/cosmos/user_guide.html)

**Completed:** COS-1.0, COS-1.2, COS-2.1, COS-4.1, COS-5.2, COS-7.1 - see `ARCHIVED_TICKETS.md`
**Consolidated:** COS-4.2 merged into PYT-12.1 (FSDP)

---

## HIGH PRIORITY

### COS-8.1: Fix torch.compile() on ROCm/Triton
**Priority:** HIGH | **Effort:** 2-4 hours | **Status:** Not Started

`--compile-model` fails on ROCm with Triton compilation error in inductor backend.

**Error:**
```
triton.compiler.errors.CompilationError:
AssertionError('Loop-carried variable _tmp2 has initial type <[1, 2], int1>
but is re-assigned to <[1, 2], int8> in loop!')
```

**Root Cause:** Triton kernel generation has type mismatch in loop-carried variables. Occurs in `AttentionPooling.forward()` during mask processing.

**Scope:**
- Investigate if this is a known PyTorch/Triton issue on ROCm
- Test alternative backends: `torch.compile(backend="eager")`, `torch.compile(backend="aot_eager")`
- If upstream bug: document workaround, disable `--compile-model` on ROCm by default
- If code issue: fix type consistency in attention pooling mask operations
- Add runtime detection to warn/skip compilation on ROCm if unfixable

**Acceptance Criteria:**
- `aam pretrain --compile-model` either works on ROCm OR fails gracefully with clear message
- Document ROCm compilation status in README

**Files:** `aam/models/attention_pooling.py`, `aam/cli/pretrain.py`, `aam/cli/train.py`

**Workaround:** Remove `--compile-model` flag when running on ROCm until fixed.

---

### COS-8.2: Investigate ROCm Numerical Divergence
**Priority:** HIGH | **Effort:** 4-8 hours | **Status:** Not Started

Significant numerical divergence observed between CUDA (NVIDIA 3090) and ROCm (MI300A) with identical parameters.

**Observed Differences (same dataset, batch_size=2, single GPU):**

| Metric | NVIDIA 3090 | ROCm MI300A |
|--------|-------------|-------------|
| Speed | 1.26s/it | 1.66s/it |
| Total Loss | 0.76 | 1.35 |
| UniFrac Loss | 0.013 | 0.035 |
| Nuc Loss | 0.75 | 1.31 |
| Nuc Accuracy | **70%** | **42%** |

The 28% gap in nucleotide accuracy indicates a fundamental computation difference, not random variance.

**Potential Causes:**
1. Attention implementation differences (`--attn-implementation mem_efficient` default)
2. ROCm kernel numerical precision
3. Gradient checkpointing behavior differences
4. Mixed precision handling
5. Random number generation differences (even with same seed)

**Known Issue:** `--attn-implementation math` fails with gradient checkpointing on ROCm:
```
torch.utils.checkpoint: Recomputed values for the following tensors
have different metadata than during the forward pass.
```
Must use `--attn-implementation math --no-gradient-checkpointing` together.

**Investigation Steps:**
- [ ] Test with `--attn-implementation math --no-gradient-checkpointing`
- [ ] Test with `--mixed-precision none`
- [ ] Compare intermediate activations between platforms
- [ ] Profile with `rocprof` to identify divergent operations
- [ ] Check if specific layer types (attention, LayerNorm) show differences

**Acceptance Criteria:**
- Root cause identified and documented
- Either fix applied OR known limitation documented with workaround
- Training on ROCm produces comparable results to CUDA (within 5% metrics)

**Files:** Potentially `aam/models/`, attention implementations, training loop

**Related:** COS-6.1 (Performance Profiling), COS-6.2 (Flash Attention)

---

## Phase 1: Environment Setup (1 remaining)

### COS-1.1: Create ROCm Singularity Container Definition
**Priority:** LOW | **Effort:** 4-6 hours | **Status:** Not Started

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

## Phase 5: Testing & Validation (1 remaining)

### COS-5.1: ROCm CI/CD Pipeline
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Not Started

ROCm-specific tests and manual test procedure for Cosmos.

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
| **8: Blockers** | **2** | **6-12** | **HIGH** |
| 1: Environment | 1 | 4-6 | LOW |
| 2: ROCm | 1 | 4-6 | MEDIUM |
| 3: SLURM | 2 | 5-7 | LOW/MEDIUM |
| 5: Testing | 1 | 4-6 | MEDIUM |
| 6: Performance | 2 | 8-12 | MEDIUM |
| 7: Documentation | 1 | 2-3 | MEDIUM |
| **Total** | **10** | **33-52** | |

## Recommended Order

1. **COS-8.2** - Investigate ROCm numerical divergence (HIGH - training produces wrong results)
2. **COS-8.1** - Fix torch.compile() on ROCm (HIGH - blocks optimized training)
3. **COS-5.1** - ROCm CI/CD pipeline
4. **COS-3.2** - Data management scripts
5. **COS-2.2** - Unified memory optimization
6. **COS-6.1/6.2** - Performance profiling
7. Remaining based on need
