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
**Priority:** HIGH | **Effort:** 2-4 hours | **Status:** Not Started

*(Renumbered from COS-8.1)*

`--compile-model` fails on ROCm with Triton type mismatch in inductor backend.

**Error:**
```
triton.compiler.errors.CompilationError:
AssertionError('Loop-carried variable _tmp2 has initial type <[1, 2], int1>
but is re-assigned to <[1, 2], int8> in loop!')
```

**Investigation Steps:**
- [ ] Check if fixed in PyTorch 2.5+ / Triton updates
- [ ] Test alternative backends: `torch.compile(backend="eager")`, `aot_eager`
- [ ] Isolate which module triggers the error (likely `AttentionPooling`)
- [ ] If upstream bug: add runtime detection to skip/warn on ROCm

**Acceptance Criteria:**
- `--compile-model` either works on ROCm OR fails gracefully with clear message
- Document compilation status in README

**Files:** `aam/cli/pretrain.py`, `aam/cli/train.py`, potentially attention modules

---

### COS-9.3: Memory Profiling and Optimization
**Priority:** HIGH | **Effort:** 4-6 hours | **Status:** In Progress

Profile and optimize memory usage for ROCm `math` attention path.

**Completed:**
- [x] Added `--memory-profile` flag to pretrain command
- [x] Created `MemoryProfiler` class in `aam/training/memory_profiler.py`
- [x] Memory logging at key points: model creation, device transfer, training
- [x] Peak memory and utilization reporting
- [x] Unit tests for memory profiler

**Remaining (requires Cosmos hardware):**
- [ ] Profile peak memory usage with different batch sizes on MI300A
- [ ] Compare memory footprint: `math` vs `mem_efficient` (on CUDA baseline)
- [ ] Test `--use-expandable-segments` impact on ROCm
- [ ] Document optimal batch sizes for MI300A (128GB)

**Acceptance Criteria:**
- [x] `--memory-profile` flag implemented and tested
- [ ] Memory usage documented for various batch sizes (needs hardware)
- [ ] Recommendations for optimal batch size per GPU memory

**Files:** `aam/training/memory_profiler.py`, `aam/cli/pretrain.py`

**Usage:**
```bash
aam pretrain --memory-profile --batch-size 8 ...
```

---

## MEDIUM PRIORITY - Architecture Optimization

### COS-9.4: MI300A Unified Memory Optimization
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Not Started

*(Renumbered from COS-2.2)*

MI300A has 128GB unified CPU/GPU memory. Explore optimizations leveraging this architecture.

**Scope:**
- Research HIP unified memory APIs
- Test `torch.cuda.memory.set_per_process_memory_fraction()` tuning
- Evaluate if `.to(device)` overhead can be reduced
- Test larger batch sizes enabled by unified memory
- Profile CPU↔GPU transfer patterns

**Acceptance Criteria:**
- Document MI300A-specific memory optimizations
- Recommendations for leveraging unified memory

---

### COS-9.5: Kernel Profiling with rocprof
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Not Started

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

## Summary

| Ticket | Description | Effort | Priority |
|--------|-------------|--------|----------|
| **COS-9.1** | ROCm-optimized attention | 6-10h | COMPLETE |
| **COS-9.2** | Fix torch.compile() | 2-4h | HIGH |
| **COS-9.3** | Memory profiling | 4-6h | HIGH |
| **COS-9.4** | Unified memory optimization | 4-6h | MEDIUM |
| **COS-9.5** | Kernel profiling (rocprof) | 4-6h | MEDIUM |
| **COS-9.6** | SLURM templates | 3-4h | LOW |
| **COS-9.7** | Singularity container | 4-6h | LOW |
| **COS-9.8** | Documentation | 2-3h | LOW |
| **Total** | | **30-45h** | |

## Recommended Order

1. **COS-9.3** - Memory profiling (quick baseline, informs other work)
2. **COS-9.1** - Attention optimization (biggest potential gain)
3. **COS-9.2** - torch.compile() (may provide significant speedup)
4. **COS-9.5** - Kernel profiling (detailed analysis)
5. **COS-9.4** - Unified memory (architecture-specific optimization)
6. Remaining infrastructure as needed

---

## Resolved Issues

### COS-8.2: Numerical Divergence - RESOLVED
**Resolution:** Use `--attn-implementation math --no-gradient-checkpointing`

The `mem_efficient` SDPA backend produces incorrect results on ROCm (42% vs 70% nucleotide accuracy). Root cause appears to be numerical differences in the memory-efficient attention kernel on HIP/ROCm.

**Workaround:** The `math` implementation produces correct results matching CUDA. Trade-off is higher memory usage and slower iteration rate.

### COS-8.1: torch.compile() Triton Error - DOCUMENTED
**Status:** Workaround available (omit `--compile-model`)

Triton compilation fails with type mismatch error. Renumbered to COS-9.2 for proper fix.
