# ROCm Optimization Tickets (Cosmos MI300A)

**Last Updated:** 2026-01-12
**Target System:** SDSC Cosmos - 168 AMD Instinct MI300A APUs (42 nodes × 4 APUs)
**Reference:** [Cosmos User Guide](https://www.sdsc.edu/systems/cosmos/user_guide.html)

**Status:** Numerical accuracy parity achieved with CUDA. Focus now on performance optimization.

**Current Working Configuration:**
```bash
aam pretrain \
  --attn-implementation math \
  --no-gradient-checkpointing \
  --data-parallel \
  # ... other flags
```

**Known Trade-offs:**
- `--attn-implementation math` is required for correct results on ROCm
- Higher memory usage than `mem_efficient` attention
- Slower iteration rate than CUDA with optimized attention

---

## HIGH PRIORITY - Performance Optimization

### COS-9.1: ROCm-Optimized Attention Implementation
**Priority:** HIGH | **Effort:** 6-10 hours | **Status:** COMPLETE

The `math` attention implementation works correctly but has performance/memory penalties compared to optimized backends.

**Root Cause Identified (2026-01-12):**

The `mem_efficient` SDPA backend produces **catastrophically wrong results when using attention masks** on ROCm. Diagnostic results (PyTorch 2.5.1+rocm6.2, MI300A gfx942):

| Test | math vs mem_efficient | Status |
|------|----------------------|--------|
| fp32 no mask | max_diff=7.15e-07 | ✅ Fine |
| fp16 no mask | max_diff=4.88e-04 | ✅ Fine |
| **fp32 WITH mask** | **max_diff=1.73** | ❌ **BROKEN** |

Without attention masks, `mem_efficient` is numerically equivalent to `math`. With masks (which AAM uses for padding), results diverge catastrophically.

**Performance Impact:**
- `mem_efficient`: 4.85x faster, 7.5x less memory - but broken with masks
- `math`: Baseline performance, correct results
- `flash` (PyTorch native): "No available kernel" error on ROCm

**Flash Attention Investigation:**
- [ROCm/flash-attention](https://github.com/ROCm/flash-attention): Build fails on ROCm 6.2+ (missing `__builtin_amdgcn_mfma_f32_16x16x32_f16` intrinsic)
- [kailums/flash-attention-rocm](https://github.com/kailums/flash-attention-rocm): Supports gfx942, but build takes 2+ hours on NFS
- xFormers: Not tested (likely similar build issues)

**Resolution:** Document `math` backend as required for ROCm. The 128GB MI300A memory accommodates the higher usage.

**Completed:**
- [x] Created diagnostic tool: `python -m aam.tools.rocm_attention_diagnostic`
- [x] Identified root cause: `mem_efficient` SDPA mask handling bug on ROCm
- [x] Tested Flash Attention for ROCm (incompatible with ROCm 6.2+)
- [x] Updated README with required flags and explanation
- [x] Documented performance baseline

**Files:** `aam/tools/rocm_attention_diagnostic.py`, `README.md`

---

### COS-9.2: Fix torch.compile() on ROCm/Triton
**Priority:** HIGH | **Effort:** 2-4 hours | **Status:** COMPLETE

*(Renumbered from COS-8.1)*

`--compile-model` fails on ROCm with Triton type mismatch in inductor backend.

**Error:**
```
triton.compiler.errors.CompilationError:
AssertionError('Loop-carried variable _tmp2 has initial type <[1, 2], int1>
but is re-assigned to <[1, 2], int8> in loop!')
```

**Resolution:**
- Added ROCm detection via `torch.version.hip` in Trainer
- When ROCm is detected with `compile_model=True`, compilation is skipped gracefully with a warning
- Also catches Triton-related RuntimeErrors as fallback for edge cases
- Added `is_rocm()` utility function in `aam/cli/utils.py`
- Updated README with ROCm limitations table documenting `--compile-model` status

**Completed:**
- [x] Add runtime detection to skip/warn on ROCm
- [x] Document compilation status in README
- [x] Add unit tests for ROCm compile handling

**Acceptance Criteria:**
- [x] `--compile-model` fails gracefully with clear message on ROCm
- [x] Document compilation status in README

**Files:** `aam/training/trainer.py`, `aam/cli/utils.py`, `README.md`, `tests/test_trainer.py`, `tests/test_cli.py`

---

### COS-9.3: Memory Profiling and Optimization
**Priority:** HIGH | **Effort:** 4-6 hours | **Status:** COMPLETE ✅

Profile and optimize memory usage for ROCm `math` attention path.

**Completed:**
- [x] Added `--memory-profile` flag to pretrain command
- [x] Created `MemoryProfiler` class in `aam/training/memory_profiler.py`
- [x] Memory logging at key points: model creation, device transfer, training
- [x] Peak memory and utilization reporting
- [x] Unit tests for memory profiler
- [x] Hardware testing on MI300A completed

**Files:** `aam/training/memory_profiler.py`, `aam/cli/pretrain.py`

**Usage:**
```bash
aam pretrain --memory-profile --batch-size 8 ...
```

---

## MEDIUM PRIORITY - Architecture Optimization

### COS-9.4: MI300A Unified Memory Optimization
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** CANCELLED

*(Renumbered from COS-2.2)*

MI300A has 128GB unified CPU/GPU memory. This ticket was cancelled as the current memory configuration works well enough without specialized unified memory optimizations.

---

### COS-9.5: Kernel Profiling with rocprof
**Priority:** LOW | **Effort:** 4-6 hours | **Status:** Not Started

*(Renumbered from COS-6.1)*

Detailed performance profiling to identify ROCm-specific bottlenecks.

**Scope:**
- Profile training iteration with `rocprof --stats`
- Identify slowest kernels
- Compare kernel timing to CUDA equivalents (if baseline available)
- Profile attention operations specifically
- Document findings and optimization opportunities

**Acceptance Criteria:**
- Profiling report with top 10 bottleneck operations
- Identified optimization targets for future work

---

## LOW PRIORITY - Infrastructure

### COS-9.6: SLURM Job Templates
**Priority:** LOW | **Effort:** 3-4 hours | **Status:** Not Started

*(Consolidated from COS-3.1, COS-3.2)*

Create optimized SLURM scripts for Cosmos.

**Scope:**
- `slurm/pretrain_single.sh` - Single APU with optimal flags
- `slurm/pretrain_node.sh` - Single node (4 APU) with DataParallel
- `slurm/pretrain_multi.sh` - Multi-node (DDP, for fine-tuning only)
- Data staging scripts for VAST → NVMe scratch

---

### COS-9.7: ROCm Singularity Container
**Priority:** LOW | **Effort:** 4-6 hours | **Status:** Not Started

*(Renumbered from COS-1.1)*

Reproducible container for production runs.

---

### COS-9.8: Documentation and Best Practices
**Priority:** LOW | **Effort:** 2-3 hours | **Status:** Not Started

*(Consolidated from COS-7.2)*

Document ROCm-specific configuration and best practices.

**Scope:**
- Update README with ROCm section
- Document required flags (`--attn-implementation math --no-gradient-checkpointing`)
- Performance comparison table (CUDA vs ROCm)
- Troubleshooting guide

---

### COS-9.9: Investigate PyTorch 2.7 / aotriton 0.8.2 for SDPA Fix
**Priority:** MEDIUM | **Effort:** 2-4 hours | **Status:** COMPLETE ✅

The `mem_efficient` SDPA backend produces incorrect results with attention masks on ROCm (COS-9.1 root cause). This has been traced to a bug in **aotriton 0.8.0** which is fixed in **aotriton 0.8.2**.

**Background:**
- [Issue #147460](https://github.com/pytorch/pytorch/issues/147460): SDPA with custom attn_mask produces wrong outputs on ROCm 2.6.0
- [Issue #132004](https://github.com/pytorch/pytorch/issues/132004): Memory efficient attention causes image corruption on MI300X/MI250
- Fix merged via [PR #148433](https://github.com/pytorch/pytorch/pull/148433)

**Key Finding: Fix requires ROCm 6.3 ✅ Available on Cosmos**

| ROCm | PyTorch Available | aotriton | SDPA Status |
|------|-------------------|----------|-------------|
| 6.2 | 2.5.1 stable | 0.7.x | ❌ Broken with masks |
| **6.3 (Cosmos default)** | **2.7.0 - 2.9.1** | **0.8.2+** | ✅ **Fixed** |

Cosmos has ROCm 6.3 as default (`module load rocm/6.3.0`). PyTorch 2.7+ with the fix is available.

**Investigation Steps:**
- [x] Check if PyTorch 2.7+rocm is available for ROCm 6.2 → **No, requires ROCm 6.3**
- [x] Check Cosmos ROCm 6.3 availability → **Yes, rocm/6.3.0 is default**
- [x] Install PyTorch 2.7.1 and run diagnostic → **Completed**
- [x] Verify numerical comparison with mask passes → **max_diff: 1.10e-06 ✅**
- [x] Benchmark `mem_efficient` vs `math` → **3.76x faster, 4.4x less memory**
- [ ] Update `--attn-implementation` default for ROCm 6.3

**Results (PyTorch 2.7.1 + ROCm 6.3 on MI300A):**

| Metric | ROCm 6.2 (broken) | ROCm 6.3 (fixed) |
|--------|-------------------|------------------|
| Masked attention max_diff | ~1.73 | **1.10e-06** ✅ |
| mem_efficient speedup | N/A (broken) | **3.76x** vs math |
| mem_efficient memory | N/A (broken) | **0.23x** vs math |

**Conclusion:** The aotriton 0.8.2 fix in PyTorch 2.7.1 resolves the SDPA attention mask issue on ROCm.
`--attn-implementation math` is no longer required on ROCm 6.3 + PyTorch 2.7+.

**Remaining Work:**
- Update README to reflect ROCm 6.3 no longer needs `--attn-implementation math`
- Consider making `mem_efficient` the default on ROCm 6.3+

**Acceptance Criteria:**
- Determine if PyTorch 2.7+rocm6.3 fixes the attention mask issue on MI300A
- Document findings and update recommended configuration if applicable

**References:**
- [PyTorch ROCm SDPA Issue #147460](https://github.com/pytorch/pytorch/issues/147460)
- [Memory Efficient Attention Corruption #132004](https://github.com/pytorch/pytorch/issues/132004)
- [PyTorch ROCm 6.2 wheels](https://download.pytorch.org/whl/rocm6.2/torch/)
- [PyTorch ROCm 6.3 wheels](https://download.pytorch.org/whl/rocm6.3/torch/)

---

## Summary

| Ticket | Description | Effort | Priority |
|--------|-------------|--------|----------|
| **COS-9.1** | ROCm-optimized attention | 6-10h | COMPLETE |
| **COS-9.2** | Fix torch.compile() | 2-4h | COMPLETE |
| **COS-9.3** | Memory profiling | 4-6h | COMPLETE |
| **COS-9.4** | Unified memory optimization | 4-6h | CANCELLED |
| **COS-9.5** | Kernel profiling (rocprof) | 4-6h | LOW |
| **COS-9.6** | SLURM templates | 3-4h | LOW |
| **COS-9.7** | Singularity container | 4-6h | LOW |
| **COS-9.8** | Documentation | 2-3h | LOW |
| **COS-9.9** | PyTorch 2.7 SDPA fix investigation | 2-4h | COMPLETE |
| **Total remaining** | | **13-19h** | |

## Recommended Order

1. **COS-9.5** - Kernel profiling (detailed performance analysis)
2. **COS-9.6** - SLURM templates (infrastructure)
3. **COS-9.8** - Documentation
4. **COS-9.7** - Singularity container (as needed)

**Note:** COS-9.1, COS-9.2, COS-9.3, COS-9.9 complete. COS-9.4 cancelled.

---

## Resolved Issues

### COS-8.2: Numerical Divergence - RESOLVED
**Resolution:** Use `--attn-implementation math --no-gradient-checkpointing`

The `mem_efficient` SDPA backend produces incorrect results on ROCm (42% vs 70% nucleotide accuracy). Root cause appears to be numerical differences in the memory-efficient attention kernel on HIP/ROCm.

**Workaround:** The `math` implementation produces correct results matching CUDA. Trade-off is higher memory usage and slower iteration rate.

### COS-8.1: torch.compile() Triton Error - DOCUMENTED
**Status:** Workaround available (omit `--compile-model`)

Triton compilation fails with type mismatch error. Renumbered to COS-9.2 for proper fix.
