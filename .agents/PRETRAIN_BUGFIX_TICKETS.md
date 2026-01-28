# Pretrain CLI Bugfix Tickets

**Last Updated:** 2026-01-28
**Status:** 1 ticket (~0.25 hours total) | **0 HIGH priority**

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
**Priority:** LOW | **Effort:** 0.25 hours | **Status:** ✅ Complete

**Location:** `aam/cli/pretrain.py:478-486`

**Implementation (2026-01-28):**
- Changed to `val_loader, _ = create_distributed_dataloader(...)` to indicate intentional non-use
- Removed unused `val_sampler = None` initialization
- val_sampler with `shuffle=False` doesn't need `set_epoch()` since validation data order doesn't matter

**Acceptance Criteria:**
- [x] `val_sampler` is either used or explicitly ignored with `_`

---

### PRE-6: Track Actual Last Epoch in Final Checkpoint
**Priority:** LOW | **Effort:** 0.5 hours | **Status:** ✅ Complete

**Location:** `aam/cli/pretrain.py:741`

**Problem:** If early stopping triggers at epoch 50 out of 100, checkpoint is saved with `epoch=99` instead of `epoch=50`.

**Implementation (2026-01-28):**
- Compute `actual_last_epoch = start_epoch + len(history["train_loss"]) - 1`
- Handles both fresh training and resume scenarios
- Added tests for early stopping with and without resume

**Acceptance Criteria:**
- [x] Final checkpoint metadata reflects actual last trained epoch
- [x] Works correctly with and without early stopping

---

### PRE-7: Add Warning When auto_batch_size Skipped for CPU
**Priority:** LOW | **Effort:** 0.25 hours | **Status:** ✅ Complete

**Location:** `aam/cli/pretrain.py:617`

**Implementation (2026-01-28):**
- Added `elif auto_batch_size and device != "cuda":` branch after distributed check
- Logs warning: "Auto batch size disabled for CPU training. Using --batch-size directly."

**Acceptance Criteria:**
- [x] Warning logged when `--auto-batch-size` is enabled but `--device cpu` is used
- [x] Message clearly explains why auto_batch_size was skipped

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
| **PRE-5** | Remove unused val_sampler | 0.25h | LOW | ✅ Complete |
| **PRE-6** | Track actual last epoch | 0.5h | LOW | ✅ Complete |
| **PRE-7** | CPU auto_batch_size warning | 0.25h | LOW | ✅ Complete |
| **PRE-8** | Logger existence check | 0.25h | LOW | Not Started |
| **Total** | | **~2.5h** | | |

---

## Files Affected

- `aam/cli/pretrain.py` (all tickets)
- `aam/training/distributed.py` (PRE-3 documentation)
- `aam/training/trainer.py` (PRE-2 reference)
