# Train CLI Bugfix Tickets

**Last Updated:** 2026-01-28
**Status:** 15 tickets | ~7 hours estimated
**Dev Branch:** `dev/train-bugfix`

All TRN ticket work should branch from and PR into `dev/train-bugfix`.

```bash
git checkout dev/train-bugfix
git checkout -b trn-{ticket}-{name}
# ... work ...
gh pr create --base dev/train-bugfix
```

---

## HIGH Priority Tickets

### TRN-1: Validate Metadata Contains All BIOM Samples
**Priority:** HIGH | **Effort:** 1 hour | **Status:** Not Started

**Location:** `aam/cli/train.py:835`

**Problem:** After filtering BIOM table to match matrix samples, there's no validation that the remaining samples exist in the metadata file. Missing samples will cause silent KeyError or unexpected behavior during metadata extraction.

**Current Code:**
```python
# Filter table to only include samples in matrix
table_obj = table_obj.filter(matrix_sample_ids, axis="sample", inplace=False)
sample_ids = matrix_sample_ids
# No validation that sample_ids exist in metadata_df
```

**Fix:** Add validation that all `sample_ids` are present in `metadata_df["sample_id"]`.

**Acceptance Criteria:**
- [ ] Clear error message when metadata is missing samples from BIOM table
- [ ] Error includes list of missing sample IDs (up to first 10)
- [ ] Validation happens before train/val split

---

### TRN-2: Add Empty Dataset Validation After Filtering
**Priority:** HIGH | **Effort:** 0.5 hours | **Status:** Not Started

**Location:** `aam/cli/train.py:879`

**Problem:** After filtering train/val metadata by sample IDs, there's no check for empty DataFrames. If all samples are filtered out (e.g., due to sample ID mismatches), training will fail with cryptic errors later.

**Current Code:**
```python
train_metadata = metadata_df[metadata_df["sample_id"].isin(train_ids)]
val_metadata = metadata_df[metadata_df["sample_id"].isin(val_ids)]
# No validation that train_metadata/val_metadata are non-empty
```

**Fix:** Add validation that both DataFrames have at least 1 row after filtering.

**Acceptance Criteria:**
- [ ] Clear error when train or val metadata is empty after filtering
- [ ] Error message explains possible causes (sample ID mismatch)
- [ ] Add test for this validation

---

### TRN-3: Add Target Column Type Validation
**Priority:** HIGH | **Effort:** 0.5 hours | **Status:** Not Started

**Location:** `aam/cli/train.py:945`

**Problem:** Target values are cast to float without checking if the column contains valid numeric data. Non-numeric values (e.g., "N/A", empty strings) will cause confusing NumPy errors.

**Current Code:**
```python
train_targets = train_metadata.set_index("sample_id")[metadata_column].values.astype(float)
```

**Fix:** Validate that target column values can be converted to float before the `.astype(float)` call. Provide clear error message listing problematic values.

**Acceptance Criteria:**
- [ ] Clear error when target column contains non-numeric values
- [ ] Error message lists first few invalid values and their sample IDs
- [ ] Skip validation when `--classifier` flag is set (categorical targets are expected)

---

### TRN-4: Validate Checkpoint Resume Fields
**Priority:** HIGH | **Effort:** 0.5 hours | **Status:** Not Started

**Location:** `aam/cli/train.py:1437`

**Problem:** When loading checkpoint for resume, required fields are accessed without validation. Corrupted or incompatible checkpoints will cause KeyError.

**Current Code:**
```python
checkpoint_info = trainer.load_checkpoint(resume_from, ...)
start_epoch = checkpoint_info["epoch"] + 1
initial_best_metric_value = checkpoint_info.get("best_metric_value", checkpoint_info["best_val_loss"])
```

**Fix:** Validate that `checkpoint_info` contains required keys (`epoch`, `best_val_loss`) before accessing them.

**Acceptance Criteria:**
- [ ] Clear error when checkpoint is missing required fields
- [ ] Error message lists expected vs actual checkpoint keys
- [ ] Checkpoint version/compatibility warning if format differs

---

### TRN-5: Fix drop_last=True for Validation DataLoader
**Priority:** HIGH | **Effort:** 0.5 hours | **Status:** Not Started

**Location:** `aam/cli/train.py:1131,1152`

**Problem:** Validation DataLoader uses `drop_last=True`, which drops the last incomplete batch. This means some validation samples are never evaluated, which:
1. Causes inconsistent validation metrics between runs (if batch size changes)
2. Can drop significant data if val set is small (e.g., 15 samples with batch_size=8 drops 7 samples)

**Current Code:**
```python
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    ...
    drop_last=True,  # Should be False for validation
)
```

**Fix:** Change `drop_last=True` to `drop_last=False` for validation DataLoader. Keep `drop_last=True` for training DataLoader (required for consistent batch sizes with BatchNorm and UniFrac pairwise losses).

**Note:** The model/loss should handle variable batch sizes gracefully. Verify UniFrac loss handles odd-sized batches.

**Acceptance Criteria:**
- [ ] Validation DataLoader uses `drop_last=False`
- [ ] Training DataLoader still uses `drop_last=True`
- [ ] Test verifying all validation samples are evaluated
- [ ] Verify UniFrac loss handles variable batch sizes

---

## MEDIUM Priority Tickets

### TRN-6: Fix Distributed Cleanup Race Condition
**Priority:** MEDIUM | **Effort:** 0.5 hours | **Status:** Not Started

**Location:** `aam/cli/train.py:1509`

**Problem:** In the exception handler, `distributed` and `fsdp` variables may not be defined if the error occurs before they are set. This causes NameError during cleanup.

**Current Code:**
```python
except Exception as e:
    logger.error(f"Training failed: {e}", exc_info=True)
    if distributed or fsdp:  # May be undefined
        cleanup_distributed()
    raise click.ClickException(str(e))
```

**Fix:** Use `locals().get()` or initialize `distributed`/`fsdp` to False at function start.

**Acceptance Criteria:**
- [ ] No NameError if exception occurs before distributed setup
- [ ] Distributed cleanup still works when variables are defined
- [ ] Test for early exception handling

---

### TRN-7: Validate Quantiles Are Sorted and Unique
**Priority:** MEDIUM | **Effort:** 0.25 hours | **Status:** Not Started

**Location:** `aam/cli/train.py:671`

**Problem:** Quantile values are validated for range (0, 1) but not checked for being sorted or unique. Duplicate or unsorted quantiles may cause undefined behavior in quantile loss.

**Current Code:**
```python
for q in quantiles_list:
    if not (0 < q < 1):
        raise click.ClickException(...)
# No check for sorted/unique
```

**Fix:** Add validation that quantiles are sorted ascending and unique.

**Acceptance Criteria:**
- [ ] Error if quantiles are not sorted ascending
- [ ] Error if quantiles contain duplicates
- [ ] Helpful error message with corrected example

---

### TRN-8: Strip Whitespace from metadata_column Argument
**Priority:** MEDIUM | **Effort:** 0.25 hours | **Status:** Not Started

**Location:** `aam/cli/train.py:776`

**Problem:** The `metadata_column` argument is not stripped of whitespace. Users may accidentally include leading/trailing spaces (e.g., copying from spreadsheet), causing column not found errors.

**Current Code:**
```python
if metadata_column not in metadata_df.columns:
    raise ValueError(f"Metadata column '{metadata_column}' not found...")
```

**Fix:** Strip whitespace from `metadata_column` early in validation.

**Acceptance Criteria:**
- [ ] `metadata_column.strip()` called before column lookup
- [ ] Warning logged if whitespace was stripped
- [ ] Same treatment for `--categorical-columns` argument

---

### TRN-9: Validate Sample Weights Shape and Positivity
**Priority:** MEDIUM | **Effort:** 0.5 hours | **Status:** Not Started

**Location:** `aam/training/trainer.py:274`

**Problem:** Sample weights from batch are moved to device without shape or value validation. Negative or zero weights, or shape mismatches, will cause silent incorrect loss computation.

**Current Code:**
```python
if "sample_weights" in batch:
    sample_weights = batch["sample_weights"].to(self.device)
# No validation
```

**Fix:** Add validation that sample_weights:
1. Has shape matching batch size
2. Contains only positive values
3. Is not all zeros

**Acceptance Criteria:**
- [ ] Error if sample_weights shape doesn't match batch size
- [ ] Error if sample_weights contains non-positive values
- [ ] Validation can be disabled via debug flag for performance

---

### TRN-10: Add Finally Block to Auto Batch Size Finder
**Priority:** MEDIUM | **Effort:** 0.5 hours | **Status:** Not Started

**Location:** `aam/cli/train.py:1309`

**Problem:** The auto batch size finder moves the model to GPU but only handles `RuntimeError`. Other exceptions (e.g., KeyboardInterrupt, CUDA OOM) will leave the model on GPU and CUDA cache dirty.

**Current Code:**
```python
try:
    result = finder.find_batch_size(...)
    ...
except RuntimeError as e:
    logger.warning(f"Auto batch size failed: {e}")
    model = model.cpu()
    torch.cuda.empty_cache()
```

**Fix:** Use `finally` block to ensure cleanup happens for any exception type.

**Acceptance Criteria:**
- [ ] Cleanup happens for all exception types (finally block)
- [ ] CUDA cache cleared regardless of exception type
- [ ] Model returned to CPU on any failure

---

### TRN-11: Validate Pretrained Encoder Weight Loading
**Priority:** MEDIUM | **Effort:** 0.5 hours | **Status:** Not Started

**Location:** `aam/cli/train.py:1238`

**Problem:** When loading pretrained encoder, only the count of loaded keys is checked. There's no validation that the loaded keys are actually encoder-related (could load unrelated weights).

**Current Code:**
```python
load_result = load_pretrained_encoder(pretrained_encoder, model, strict=False, logger=logger)
if load_result["loaded_keys"] == 0:
    raise click.ClickException("No keys were loaded from pretrained encoder.")
```

**Fix:** Add validation that loaded keys include expected encoder prefixes (e.g., `base_model.`, `encoder.`, `asv_encoder.`).

**Acceptance Criteria:**
- [ ] Warning if loaded keys don't match expected encoder patterns
- [ ] Log which key prefixes were loaded
- [ ] Error if no encoder-related keys found despite loaded_keys > 0

---

### TRN-12: Validate Distributed Broadcast Success
**Priority:** MEDIUM | **Effort:** 0.5 hours | **Status:** Not Started

**Location:** `aam/cli/train.py:852`

**Problem:** After broadcasting train/val splits across DDP processes, there's no validation that all processes received the same data.

**Current Code:**
```python
split_data = [train_ids, val_ids]
dist.broadcast_object_list(split_data, src=0)
train_ids, val_ids = split_data[0], split_data[1]
# No validation
```

**Fix:** Add barrier and optional hash check to verify broadcast consistency.

**Acceptance Criteria:**
- [ ] Barrier after broadcast to ensure all processes synchronized
- [ ] Debug mode: hash check to verify data consistency across ranks
- [ ] Warning log if broadcast takes unusually long

---

### TRN-13: Validate Categorical Encoder Handles Empty Data
**Priority:** MEDIUM | **Effort:** 0.25 hours | **Status:** Not Started

**Location:** `aam/cli/train.py:883`

**Problem:** If train_metadata is empty (due to filtering), CategoricalEncoder.fit() will receive empty data, potentially causing division by zero or empty cardinalities.

**Current Code:**
```python
if categorical_column_list:
    categorical_encoder = CategoricalEncoder()
    categorical_encoder.fit(train_metadata, columns=categorical_column_list)
    # What if train_metadata is empty?
```

**Fix:** Add check that train_metadata is non-empty before fitting categorical encoder.

**Acceptance Criteria:**
- [ ] Error if train_metadata is empty when categorical columns requested
- [ ] CategoricalEncoder.fit() raises clear error on empty input
- [ ] Test for empty metadata handling

---

## LOW Priority Tickets

### TRN-14: Suppress Duplicate Logging in Distributed Mode
**Priority:** LOW | **Effort:** 0.25 hours | **Status:** Not Started

**Location:** `aam/cli/train.py` (multiple)

**Problem:** In distributed training, some log messages are emitted from all processes, causing duplicate output. Only rank 0 should log non-critical messages.

**Current Code:**
```python
logger.info("Filtering tables for train/val splits...")  # All ranks log this
```

**Fix:** Wrap non-essential log messages with `is_main_process()` check, or configure logger to only emit on rank 0.

**Acceptance Criteria:**
- [ ] Non-critical info logs only appear once in distributed mode
- [ ] Error/warning logs still appear from all ranks (for debugging)
- [ ] Document which log levels are rank-0 only

---

### TRN-15: Fix best_metric_value Default in Checkpoint Loading
**Priority:** LOW | **Effort:** 0.25 hours | **Status:** Not Started

**Location:** `aam/training/trainer.py:1571`

**Problem:** When loading a checkpoint, `best_metric_value` defaults to `best_val_loss` if not present. However, if `best_metric` is accuracy or other higher-is-better metric, this default is incorrect.

**Current Code:**
```python
"best_metric_value": checkpoint.get("best_metric_value", checkpoint["best_val_loss"]),
```

**Fix:** Use metric-appropriate default (0.0 for higher-is-better, inf for lower-is-better).

**Acceptance Criteria:**
- [ ] Default value considers metric direction (higher/lower is better)
- [ ] Log warning if checkpoint lacks best_metric_value
- [ ] Test checkpoint loading with different metric types

---

## Summary

| Ticket | Description | Effort | Priority | Status |
|--------|-------------|--------|----------|--------|
| **TRN-1** | Validate metadata contains BIOM samples | 1h | HIGH | Not Started |
| **TRN-2** | Empty dataset validation after filtering | 0.5h | HIGH | Not Started |
| **TRN-3** | Target column type validation | 0.5h | HIGH | Not Started |
| **TRN-4** | Checkpoint resume field validation | 0.5h | HIGH | Not Started |
| **TRN-5** | Fix drop_last=True for validation DataLoader | 0.5h | HIGH | Not Started |
| **TRN-6** | Fix distributed cleanup race condition | 0.5h | MEDIUM | Not Started |
| **TRN-7** | Validate quantiles sorted and unique | 0.25h | MEDIUM | Not Started |
| **TRN-8** | Strip whitespace from metadata_column | 0.25h | MEDIUM | Not Started |
| **TRN-9** | Validate sample weights shape/positivity | 0.5h | MEDIUM | Not Started |
| **TRN-10** | Add finally block to auto batch size finder | 0.5h | MEDIUM | Not Started |
| **TRN-11** | Validate pretrained encoder weight loading | 0.5h | MEDIUM | Not Started |
| **TRN-12** | Validate distributed broadcast success | 0.5h | MEDIUM | Not Started |
| **TRN-13** | Validate categorical encoder handles empty data | 0.25h | MEDIUM | Not Started |
| **TRN-14** | Suppress duplicate logging in distributed mode | 0.25h | LOW | Not Started |
| **TRN-15** | Fix best_metric_value default in checkpoint | 0.25h | LOW | Not Started |
| **Total** | | **~7h** | | |

---

## Files Affected

- `aam/cli/train.py` (TRN-1 to TRN-8, TRN-10 to TRN-14)
- `aam/training/trainer.py` (TRN-9, TRN-15)
