# Pretrain CLI Bugfix Tickets

**Last Updated:** 2026-01-28
**Status:** 4 tickets (~1.25 hours total) | **0 HIGH priority**

---

## Completed Tickets

### PRE-1: Fix Scheduler num_training_steps Ignoring Gradient Accumulation
**Priority:** HIGH | **Effort:** 0.5 hours | **Status:** ✅ Complete

Fixed scheduler's `num_training_steps` to divide by `gradient_accumulation_steps`, ensuring warmup and cosine decay behave correctly.

**Acceptance Criteria:**
- [x] `num_training_steps` is divided by `gradient_accumulation_steps`
- [x] Add test verifying scheduler steps match optimizer steps
- [x] Verify warmup completes at expected step count

---

### PRE-2: Fix Double Checkpoint Loading on Resume
**Priority:** HIGH | **Effort:** 0.5 hours | **Status:** ✅ Complete

When `--resume-from` is provided, the checkpoint was loaded twice.

**Implementation (2026-01-28):**
- Capture return value from `load_checkpoint()`
- Pass `start_epoch` and `initial_best_metric_value` to `train()` instead of `resume_from`
- Added test `test_pretrain_resume_loads_checkpoint_once` verifying single load

**Acceptance Criteria:**
- [x] Checkpoint is loaded exactly once when `--resume-from` is provided
- [x] Training resumes from correct epoch
- [x] Optimizer and scheduler state restored correctly

### PRE-3: Fix Batch Size Semantics for Distributed Training
**Priority:** MEDIUM | **Effort:** 1 hour | **Status:** ✅ Complete

CLI validation incorrectly treated `--batch-size` as total (dividing by world_size), but `create_distributed_dataloader` treats it as per-GPU.

**Implementation (2026-01-28):**
- Removed incorrect division by world_size in validation
- Updated CLI help to clarify batch_size is per-GPU
- Simplified error message

**Acceptance Criteria:**
- [x] Batch size semantics are consistent between CLI help, validation, and implementation
- [x] Documentation clearly states whether batch_size is total or per-GPU
- [x] Existing test `test_create_distributed_dataloader_batch_size_per_gpu` already covers this

---

## Outstanding Tickets

### PRE-4: Remove Unused MemoryProfiler Instance
**Priority:** LOW | **Effort:** 0.25 hours | **Status:** ✅ Complete

**Location:** `aam/cli/pretrain.py:692`

```python
profiler = MemoryProfiler(enabled=memory_profile)
```

This object is created but never used. Memory profiling is done via separate `log_gpu_memory_stats()` calls.

**Fix:** Remove the unused `profiler` variable.

**Implementation (2026-01-28):**
- Removed unused `profiler` variable instantiation
- Removed unused `MemoryProfiler` import (only `log_gpu_memory_stats` needed)

**Acceptance Criteria:**
- [x] Unused `profiler` variable removed
- [x] Memory profiling still works via `log_gpu_memory_stats()`

---

### PRE-5: Remove Unused val_sampler Variable
**Priority:** LOW | **Effort:** 0.25 hours | **Status:** Not Started

**Location:** `aam/cli/pretrain.py:478-486`

```python
val_loader, val_sampler = create_distributed_dataloader(...)
```

`val_sampler` is returned but never used. The `train_sampler` is correctly passed to Trainer, but `val_sampler` is unused.

**Fix:** Either:
1. Pass `val_sampler` to Trainer for proper `set_epoch()` calls, OR
2. Use `_` to indicate intentionally unused: `val_loader, _ = ...`

**Acceptance Criteria:**
- [ ] `val_sampler` is either used or explicitly ignored with `_`

---

### PRE-6: Track Actual Last Epoch in Final Checkpoint
**Priority:** LOW | **Effort:** 0.5 hours | **Status:** Not Started

**Location:** `aam/cli/pretrain.py:738`

```python
trainer.save_checkpoint(str(final_model_path), epoch=epochs - 1, ...)
```

**Problem:** If early stopping triggers at epoch 50 out of 100, checkpoint is saved with `epoch=99` instead of `epoch=50`.

**Fix:** Track actual last completed epoch from training history and use that value.

**Acceptance Criteria:**
- [ ] Final checkpoint metadata reflects actual last trained epoch
- [ ] Works correctly with and without early stopping

---

### PRE-7: Add Warning When auto_batch_size Skipped for CPU
**Priority:** LOW | **Effort:** 0.25 hours | **Status:** Not Started

**Location:** `aam/cli/pretrain.py:551`

```python
if auto_batch_size and device == "cuda" and not use_distributed and not data_parallel:
```

When `--device cpu` is used with `--auto-batch-size` (default), auto_batch_size is silently disabled. Line 616-617 logs a message for distributed training, but no message for CPU mode.

**Fix:** Add warning when auto_batch_size is skipped due to CPU mode.

**Acceptance Criteria:**
- [ ] Warning logged when `--auto-batch-size` is enabled but `--device cpu` is used
- [ ] Message clearly explains why auto_batch_size was skipped

---

### PRE-8: Improve Logger Existence Check Robustness
**Priority:** LOW | **Effort:** 0.25 hours | **Status:** Not Started

**Location:** `aam/cli/pretrain.py:746`

```python
if "logger" in locals():
```

This pattern is fragile. If code is refactored and logger moves to a different scope, this check could fail silently.

**Fix:** Initialize logger at module level or use a more robust pattern.

**Acceptance Criteria:**
- [ ] Logger availability check is robust to refactoring
- [ ] Error logging works in all failure scenarios

---

## Summary

| Ticket | Description | Effort | Priority | Status |
|--------|-------------|--------|----------|--------|
| **PRE-1** | Scheduler steps / gradient accumulation | 0.5h | HIGH | ✅ Complete |
| **PRE-2** | Double checkpoint loading | 0.5h | HIGH | ✅ Complete |
| **PRE-3** | Batch size semantics | 1h | MEDIUM | ✅ Complete |
| **PRE-4** | Remove unused profiler | 0.25h | LOW | ✅ Complete |
| **PRE-5** | Remove unused val_sampler | 0.25h | LOW | Not Started |
| **PRE-6** | Track actual last epoch | 0.5h | LOW | Not Started |
| **PRE-7** | CPU auto_batch_size warning | 0.25h | LOW | Not Started |
| **PRE-8** | Logger existence check | 0.25h | LOW | Not Started |
| **Total** | | **~2.5h** | | |

---

## Files Affected

- `aam/cli/pretrain.py` (all tickets)
- `aam/training/distributed.py` (PRE-3 documentation)
- `aam/training/trainer.py` (PRE-2 reference)
