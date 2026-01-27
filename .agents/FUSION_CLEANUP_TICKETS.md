# Attention Fusion & Code Cleanup Tickets

**Last Updated:** 2026-01-23
**Status:** 5 tickets remaining (~14-24 hours)
**Design Doc:** `_design_plan/17_attention_fusion.md`

---

## Overview

Two ticket series addressing:
1. **FUS-1 to FUS-3:** Attention-based categorical fusion (position-specific modulation)
2. **CLN-1 to CLN-6:** Code cleanup and consolidation

**MVP:** FUS-1 + FUS-2 (~9 hours) - enables position-specific categorical conditioning

---

## FUS: Attention Fusion Tickets

### FUS-1: Gated Multimodal Unit (GMU)
**Priority:** HIGH | **Effort:** 3-4 hours | **Status:** Complete

Learned gating between sequence and categorical modalities.

**Scope:**
- Create `aam/models/fusion.py` with `GMU` class
- Add `--categorical-fusion gmu` option
- Operates on pooled representations: `z * seq + (1-z) * cat`
- Log gate values to TensorBoard

**Implementation:**
```python
class GMU(nn.Module):
    def __init__(self, seq_dim: int, cat_dim: int):
        self.seq_transform = nn.Linear(seq_dim, seq_dim)
        self.cat_transform = nn.Linear(cat_dim, seq_dim)
        self.gate = nn.Linear(seq_dim + cat_dim, seq_dim)

    def forward(self, h_seq, h_cat):
        h_seq_t = torch.tanh(self.seq_transform(h_seq))
        h_cat_t = torch.tanh(self.cat_transform(h_cat))
        z = torch.sigmoid(self.gate(torch.cat([h_seq, h_cat], dim=-1)))
        return z * h_seq_t + (1 - z) * h_cat_t
```

**Acceptance Criteria:**
- [x] `--categorical-fusion gmu` works
- [x] Gate values logged to TensorBoard
- [x] 15+ unit tests (21 GMU tests + 7 integration tests)

**Files:** `aam/models/fusion.py`, `aam/models/sequence_predictor.py`, `aam/cli/train.py`, `tests/test_fusion.py`

---

### FUS-2: Cross-Attention Fusion
**Priority:** HIGH | **Effort:** 5-6 hours | **Status:** Complete

Position-specific metadata modulation via cross-attention.

**Scope:**
- Add `CrossAttentionFusion` to `aam/models/fusion.py`
- Sequence tokens attend to metadata tokens
- Add `--categorical-fusion cross-attention`
- Log attention weights to TensorBoard

**Key Difference from Current:**
- Current: Same categorical embedding broadcast to all ASV positions
- Cross-attention: Each ASV position attends differently to metadata

**Acceptance Criteria:**
- [x] `--categorical-fusion cross-attention` works
- [x] Per-position attention weights extractable
- [x] `--cross-attn-heads` configurable (default 8)
- [x] 20+ unit tests (26 unit tests + 7 integration tests)

**Files:** `aam/models/fusion.py`, `aam/models/sequence_predictor.py`, `aam/cli/train.py`, `tests/test_fusion.py`

---

### FUS-3: Perceiver-Style Latent Fusion
**Priority:** LOW | **Effort:** 6-8 hours | **Status:** Not Started

Learned latent bottleneck for linear complexity fusion.

**Scope:**
- Add `PerceiverFusion` to `aam/models/fusion.py`
- Latents attend to concatenated sequence + metadata
- Self-attention refinement layers
- Add `--categorical-fusion perceiver`

**Acceptance Criteria:**
- [ ] `--categorical-fusion perceiver` works
- [ ] `--perceiver-num-latents`, `--perceiver-num-layers` configurable
- [ ] 15+ unit tests

**Files:** `aam/models/fusion.py`, `aam/models/sequence_predictor.py`, `aam/cli/train.py`, `tests/test_fusion.py`

---

## CLN: Code Cleanup Tickets

### CLN-2: Unify Target Normalization
**Priority:** MEDIUM | **Effort:** 3-4 hours | **Status:** Complete

Replace fragmented normalization flags with single interface.

**Current (fragmented):**
```bash
--normalize-targets/--no-normalize-targets  # Global min-max
--normalize-targets-by <columns>            # Per-category z-score
--log-transform-targets                     # log(y+1)
```

**Proposed:**
```bash
--target-transform none|minmax|zscore|zscore-category|log-minmax|log-zscore
--normalize-by <columns>  # Only with zscore-category
```

**Acceptance Criteria:**
- [x] Single `--target-transform` flag
- [x] Old flags deprecated with warnings
- [x] Implicit behaviors made explicit
- [x] 27+ tests (GlobalNormalizer, parse_target_transform, dataset integration)

**Files:** `aam/cli/train.py`, `aam/data/dataset.py`, `aam/data/normalization.py`

---

### CLN-3: Remove Unused Parameters
**Priority:** LOW | **Effort:** 1-2 hours | **Status:** Complete

Remove dead code and unused function parameters.

**Items:**
1. `intermediate_size` params in model constructors (always 4×embedding_dim)
2. `is_rocm()` in `aam/cli/utils.py` (never called)
3. `unifrac_loader=None` in `inference_collate` (predict.py)
4. `CategoricalSchema` class (unused, CategoricalEncoder.from_dict used)

**Acceptance Criteria:**
- [ ] Identified parameters removed
- [ ] No test failures
- [ ] API surface simplified

**Files:** `aam/models/sequence_predictor.py`, `aam/models/sequence_encoder.py`, `aam/cli/utils.py`, `aam/cli/predict.py`, `aam/data/categorical.py`

---

### CLN-4: Extract Shared Training Utilities
**Priority:** LOW | **Effort:** 2-3 hours | **Status:** Complete

Reduce code duplication between `pretrain.py` and `train.py`.

**Duplicated (~135 lines):**
- Scheduler creation logic
- Distributed validation checks
- DataLoader creation patterns
- DataParallel wrapping (~20 lines each, nearly identical in both files)

**Scope:**
- Create `aam/cli/training_utils.py` with shared functions
- Refactor both CLI scripts to use shared code
- Extract `wrap_data_parallel()` helper function
- Align validation order (pretrain validates before setup, train validates after - should be consistent)
- Enhance `--data-parallel requires CUDA` error message to match FSDP's helpful format

**Acceptance Criteria:**
- [ ] Shared utilities extracted
- [ ] Both CLIs use shared code
- [ ] No behavior changes
- [ ] DataParallel wrapping extracted to shared function
- [ ] Validation order consistent between pretrain.py and train.py

**Files:** `aam/cli/training_utils.py` (new), `aam/cli/pretrain.py`, `aam/cli/train.py`

---

### CLN-5: Add DataParallel to train.py
**Priority:** MEDIUM | **Effort:** 2-3 hours | **Status:** Complete

Feature parity: DataParallel exists in pretrain.py but not train.py.

**Scope:**
- Add `--data-parallel` flag to train.py
- Copy DP setup from pretrain.py
- Mutually exclusive with `--distributed`/`--fsdp`

**Acceptance Criteria:**
- [x] `--data-parallel` available in train.py
- [x] Works with UniFrac auxiliary loss
- [x] 5+ tests (added 5 new tests)

**Files:** `aam/cli/train.py`, `tests/test_cli.py`

---

### CLN-6: Simplify Categorical Conditioning Docs
**Priority:** MEDIUM | **Effort:** 4-5 hours | **Status:** Complete

Document and validate the categorical conditioning systems.

**Current Systems:**
1. Base fusion (`--categorical-fusion concat|add`)
2. Advanced fusion (`--categorical-fusion gmu|cross-attention`)
3. Conditional scaling (`--conditional-output-scaling`)

**Scope:**
- Add `--categorical-help` showing decision tree
- Warn if redundant flags used together
- Document clear recommendations in README

| Use Case | Recommended |
|----------|-------------|
| Simple metadata | `--categorical-fusion concat` |
| Per-category shift | `concat` + `--conditional-output-scaling` |
| Adaptive weighting | `--categorical-fusion gmu` |
| Position-specific | `--categorical-fusion cross-attention` |

**Acceptance Criteria:**
- [x] Decision tree in `--categorical-help`
- [x] Validation warnings for redundant combos
- [x] README updated
- [x] 3 tests (1 help output + 2 parametrized warning tests)

**Files:** `aam/cli/train.py`, `README.md`, `tests/test_cli.py`

---

### CLN-7: Toggle Count Prediction
**Priority:** MEDIUM | **Effort:** 2-3 hours | **Status:** Complete

Allow disabling count prediction head entirely for simpler training.

**Current State:**
- Count prediction always enabled
- `--count-penalty 0.0` disables loss but head still computed
- Wastes compute when count prediction not needed

**Proposed CLI Flag:**
```bash
--no-count-prediction    # Disable count prediction head entirely
```

**Scope:**
- Add `--count-prediction/--no-count-prediction` flag (default: enabled)
- Skip count_encoder and count_head when disabled
- Remove count_prediction from output dict when disabled
- Validate `--count-penalty` warns if used with `--no-count-prediction`

**Acceptance Criteria:**
- [x] `--no-count-prediction` disables count head
- [x] Memory/compute savings when disabled
- [x] Backward compatible (default enabled)
- [x] 8 tests (6 model tests + 2 CLI tests)

**Files:** `aam/models/sequence_predictor.py`, `aam/cli/train.py`, `tests/test_sequence_predictor.py`, `tests/test_cli.py`

---

### CLN-8: Separate Learning Rate for Categorical Parameters
**Priority:** MEDIUM | **Effort:** 2-3 hours | **Status:** Complete

Allow different learning rate for categorical embeddings and fusion layers.

**Motivation:**
- Categorical embeddings may need higher LR to learn meaningful representations
- Or lower LR to prevent overfitting on small category counts
- GMU/fusion layers may benefit from different LR than base model

**Proposed CLI Flag:**
```bash
--categorical-lr 1e-3    # Learning rate for categorical parameters (default: same as --lr)
```

**Scope:**
- Add `--categorical-lr` flag
- Create parameter groups in optimizer: base params vs categorical params
- Categorical params include: `categorical_embedder`, `categorical_projection`, `gmu`, `output_scales`, `cross_attn_fusion`
- Support with AdamW optimizer

**Implementation:**
```python
param_groups = [
    {"params": base_params, "lr": lr},
    {"params": categorical_params, "lr": categorical_lr},
]
optimizer = AdamW(param_groups, weight_decay=weight_decay)
```

**Acceptance Criteria:**
- [x] `--categorical-lr` sets separate LR for categorical params
- [x] Default behavior unchanged (uses --lr for all)
- [x] Works with all optimizers
- [x] TensorBoard logs both learning rates
- [x] 5+ tests (5 tests)

**Files:** `aam/training/trainer.py`, `aam/cli/train.py`, `tests/test_trainer.py`

---

### CLN-9: Remove FiLM Conditioning
**Priority:** MEDIUM | **Effort:** 2-3 hours | **Status:** Complete

Remove FiLM (Feature-wise Linear Modulation) conditioning system entirely.

**Motivation:**
- Cross-attention fusion (FUS-2) provides superior position-specific modulation
- FiLM adds complexity without clear benefit over simpler alternatives
- Reduces codebase complexity and maintenance burden

**Scope:**
- Remove `aam/models/film.py`
- Remove `--film-conditioning` CLI flag
- Remove FiLM integration from `SequencePredictor`
- Remove FiLM from `predict.py` checkpoint loading
- Remove FiLM tests from `tests/test_film.py`
- Update README to remove FiLM documentation

**Files to Remove:**
- `aam/models/film.py`
- `tests/test_film.py`

**Files to Modify:**
- `aam/models/sequence_predictor.py` - remove FiLM imports and usage
- `aam/cli/train.py` - remove `--film-conditioning` flag
- `aam/cli/predict.py` - remove FiLM checkpoint handling
- `aam/training/trainer.py` - remove any FiLM logging
- `README.md` - remove FiLM documentation

**Acceptance Criteria:**
- [ ] `aam/models/film.py` deleted
- [ ] `tests/test_film.py` deleted
- [ ] `--film-conditioning` flag removed
- [ ] No FiLM references in codebase
- [ ] All tests pass
- [ ] README updated

---

### CLN-10: Training Output Artifacts
**Priority:** HIGH | **Effort:** 2-3 hours | **Status:** Complete

Save training/validation splits and best model predictions to output directory.

**Motivation:**
- Reproducibility: Know exactly which samples were used for training vs validation
- Analysis: Evaluate model predictions on validation set without re-running inference
- Debugging: Compare predictions across training runs with consistent splits

**Current State:**
- Train/val split performed via `train_test_split()` at train.py:606-617
- Sample IDs discarded after dataset creation
- Best model predictions computed during validation but not saved
- Users must re-run `aam predict` separately to get predictions

**Proposed Output Files:**
```
output_dir/
├── train_samples.txt       # One sample ID per line
├── val_samples.txt         # One sample ID per line
└── val_predictions.tsv     # sample_id, prediction, actual (from best epoch)
```

**Scope:**
1. Save `train_samples.txt` and `val_samples.txt` after split (train.py)
2. Capture validation predictions when best model is saved (trainer.py)
3. Write `val_predictions.tsv` with sample_id, prediction, actual columns
4. Include denormalized predictions (actual scale, not normalized)

**Implementation Notes:**
- Sample lists written immediately after `train_test_split()` in train.py
- Validation predictions already computed in `Evaluator.validate_epoch()` (trainer.py:914-942)
- Reservoir sampling captures up to 1000 samples for plots; need full predictions
- Store predictions in Trainer, write when `save_checkpoint()` saves best model

**Acceptance Criteria:**
- [x] `train_samples.txt` written with training sample IDs
- [x] `val_samples.txt` written with validation sample IDs
- [x] `val_predictions.tsv` written with best epoch predictions
- [x] Predictions denormalized to original scale
- [x] Works with DDP/FSDP (only rank 0 writes)
- [x] 8 tests (6 in test_trainer.py, 2 in test_cli.py)

**Files:** `aam/cli/train.py`, `aam/training/trainer.py`, `aam/training/evaluation.py`, `tests/test_trainer.py`, `tests/test_cli.py`

---

### CLN-BUG-1: Z-Score Denormalization Missing in TensorBoard
**Priority:** HIGH | **Effort:** 1-2 hours | **Status:** Complete

TensorBoard plots show z-score normalized values instead of original scale when using `--target-normalization zscore`.

**Current Behavior:**
- When `--target-normalization zscore` is used, target values are z-score normalized during training
- Validation metrics (R², MAE) are computed on normalized values
- TensorBoard prediction plots display normalized values (e.g., -2 to +2) instead of original scale
- `val_predictions.tsv` may also contain normalized values

**Expected Behavior:**
- TensorBoard plots should show predictions and actuals in original target scale
- Metrics displayed should reflect denormalized values for interpretability
- `val_predictions.tsv` should contain denormalized predictions

**Root Cause:**
- `Evaluator._denormalize_targets()` checks for `category_normalizer` but doesn't handle `global_normalizer` (z-score) denormalization
- The `GlobalNormalizer` class has `denormalize()` method but it's not being called in the evaluation path

**Scope:**
1. Update `Evaluator._denormalize_targets()` to handle `global_normalizer` case
2. Ensure `target_normalization_params` includes `global_normalizer` state when z-score is used
3. Add tests for z-score denormalization in TensorBoard/plots

**Acceptance Criteria:**
- [x] TensorBoard plots show original scale when using `--target-normalization zscore`
- [x] `val_predictions.tsv` contains denormalized values
- [x] Metrics logged to TensorBoard are on original scale
- [x] 6 tests (5 new z-score tests + 1 precedence test)

**Files:** `aam/training/evaluation.py`, `aam/training/trainer.py`, `tests/test_trainer.py`

---

### CLN-BUG-2: val_predictions.tsv Not Written When Resuming Training
**Priority:** HIGH | **Effort:** 1-2 hours | **Status:** Complete

`val_predictions.tsv` was not created when using `--resume-from` if validation never improved beyond the loaded checkpoint's best metric.

**Root Cause:**
- When resuming, `best_metric_value` is loaded from the checkpoint
- `_save_val_predictions()` was only called inside the "is better" block
- If validation never improves, predictions were never saved

**Fix:**
1. Track whether val_predictions.tsv was saved during training
2. Store the latest full_val_predictions from each validation epoch
3. At end of training, save predictions if not yet saved
4. Add warning when sample_ids are empty for debugging

**Acceptance Criteria:**
- [x] val_predictions.tsv created even when validation never improves
- [x] Works with `--resume-from` scenario
- [x] Warning logged when sample_ids empty
- [x] 1 new test for resume scenario
- [x] Shared MockBatchDataset class extracted

**Files:** `aam/training/trainer.py`, `tests/test_trainer.py`

---

### CLN-BUG-3: --resume-from Ignores New Learning Rate
**Priority:** HIGH | **Effort:** 1-2 hours | **Status:** Complete

When using `--resume-from` with a different `--lr` value, the checkpoint's learning rate overrides the command-line argument.

**Current Behavior:**
- User specifies `--resume-from checkpoint.pt --lr 1e-4`
- Training resumes with the learning rate from the checkpoint instead of 1e-4
- No warning that the specified LR was ignored

**Expected Behavior:**
- Command-line `--lr` should override the checkpoint's learning rate
- Or at minimum, warn the user that their specified LR is being ignored

**Root Cause:**
- Checkpoint loading restores optimizer state which includes learning rate
- The CLI-specified learning rate is set before checkpoint loading, then overwritten

**Scope:**
1. After loading optimizer state, update LR to match CLI argument if specified
2. Add logging to indicate when LR is being changed from checkpoint value
3. Consider adding `--reset-lr` flag for explicit control

**Acceptance Criteria:**
- [x] `--lr` overrides checkpoint learning rate when resuming
- [x] Log message indicates LR change from checkpoint value
- [x] 4 tests (LR override, scheduler base_lrs update, same LR no log, preserve checkpoint LR without target_lr)

**Files:** `aam/cli/train.py`, `aam/training/trainer.py`, `tests/test_trainer.py`

---

### CLN-BUG-4: --resume-from LR Override Undone by Double load_checkpoint Call
**Priority:** HIGH | **Effort:** 0.5-1 hour | **Status:** Complete

CLN-BUG-3 fix is bypassed because `load_checkpoint()` is called twice during resume.

**Current Behavior:**
```
2026-01-21 20:15:13,544 - INFO - Overriding checkpoint learning rate: 4.95e-04 -> 1.00e-04 (from CLI --lr)
...
Epoch 57/1000: ... LR=4.95e-04  <-- LR reverted back!
```

**Root Cause:**
1. `train.py:1151` calls `trainer.load_checkpoint(resume_from, target_lr=lr)` - LR override applied ✓
2. `train.py:1157` calls `trainer.train(..., resume_from=resume_from)`
3. `trainer.train()` at line 988 calls `self.load_checkpoint(resume_from)` **without** `target_lr`
4. Second call restores checkpoint LR, undoing the fix

**Fix Options:**
1. **Remove duplicate call** (recommended): Don't pass `resume_from` to `trainer.train()` since CLI already handles it
2. **OR** Add `target_lr` parameter to `Trainer.train()` and forward to `load_checkpoint()`
3. **OR** Remove the `load_checkpoint()` call in train.py and let `trainer.train()` handle it with new `target_lr` param

**Acceptance Criteria:**
- [x] LR override persists through entire training (visible in progress bar)
- [x] Only one "Overriding checkpoint learning rate" log message
- [x] Existing resume tests still pass (139 tests pass)
- [x] 1 new test verifying LR persists after train() starts

**Fix Applied:** Added `start_epoch` and `initial_best_metric_value` params to `Trainer.train()`. CLI now captures checkpoint_info and passes these params instead of resume_from, avoiding the double load_checkpoint call.

**Files:** `aam/cli/train.py`, `aam/training/trainer.py`, `tests/test_trainer.py`

---

### CLN-BUG-5: zscore-cat TensorBoard Output Not Denormalized
**Priority:** HIGH | **Effort:** 1-2 hours | **Status:** Complete

TensorBoard scatter plots for `--normalize zscore-cat` show normalized values (-1 to 2) instead of native scale (0 to max).

**Current Behavior:**
- `--normalize zscore` correctly denormalizes predictions in TensorBoard (shows native scale)
- `--normalize zscore-cat` shows normalized values in TensorBoard scatter plots
- Predictions appear as -1 to 2 instead of 0 to max_add_score

**Expected Behavior:**
- Both `zscore` and `zscore-cat` should denormalize predictions before TensorBoard logging
- Scatter plots should show values in native scale for interpretability

**Root Cause:**
The denormalization logic in trainer likely only handles `zscore` normalization, not `zscore-cat` which uses per-category statistics.

**Acceptance Criteria:**
- [x] `zscore-cat` predictions denormalized in TensorBoard scatter plots
- [x] Both train and validation scatter plots show native scale
- [x] Existing `zscore` denormalization continues to work
- [x] 2+ tests verifying denormalization for `zscore-cat`

**Fix Applied:** Added `get_reverse_mappings()` to CategoricalEncoder. Include categorical_encoder_mappings in target_normalization_params. Updated Evaluator._denormalize_targets() to handle per-sample category denormalization using the reverse mappings to reconstruct category keys from batch categorical_ids.

**Files:** `aam/data/categorical.py`, `aam/data/dataset.py`, `aam/training/evaluation.py`, `aam/training/trainer.py`, `tests/test_categorical.py`, `tests/test_trainer.py`

---

### CLN-BUG-6: Model Converging to Mean Prediction with freeze-base + cross-attention + random ASV sampling
**Priority:** HIGH | **Effort:** 2-4 hours | **Status:** Complete

Training with `--freeze-base --categorical-fusion cross-attention --asv-sampling random` causes model to converge to predicting the mean, with R² decreasing over epochs.

**Observed Behavior:**
```
Epoch 3: r2=0.0527, mae=0.8127
Epoch 4: r2=0.0430, mae=0.8168
Epoch 5: r2=0.0412, mae=0.8179
Epoch 6: r2=0.0319, mae=0.8213
```
- R² **decreasing** over training (getting worse)
- MAE **increasing** over training (getting worse)
- MAE ~0.8 with zscore normalization = predicting the mean

**Reproduction Command:**
```bash
torchrun --nproc_per_node=4 -m aam.cli train \
    --distributed --freeze-base \
    --categorical-columns facility,season \
    --categorical-fusion cross-attention --cross-attn-heads 8 \
    --conditional-output-scaling season \
    --target-transform zscore-category --normalize-by season \
    --asv-sampling random \
    # ... other flags
```

**Investigation Areas:**
1. Cross-attention fusion with frozen base embeddings - are gradients flowing correctly?
2. Random ASV sampling + frozen base - does varying ASV subsets confuse training?
3. Interaction between zscore-category normalization and conditional-output-scaling
4. Check if target_encoder/target_head are receiving useful gradients
5. Verify categorical embeddings are learning (not collapsed)

**Diagnostic Steps:**
- [ ] Log gradient norms for categorical embedder, cross-attention, target_head
- [ ] Test without `--freeze-base` to isolate the issue
- [ ] Test with `--asv-sampling abundance` instead of random
- [ ] Test without `--conditional-output-scaling`
- [ ] Check if issue reproduces without `--categorical-fusion cross-attention`

**Acceptance Criteria:**
- [ ] Root cause identified
- [ ] Fix implemented (or workaround documented if architectural limitation)
- [ ] Training with freeze-base + cross-attention shows improving R² over epochs

**Files:** TBD based on investigation

---

### CLN-BUG-7: Checkpoints Not Saved to New Output Directory on Resume
**Priority:** HIGH | **Effort:** 1-2 hours | **Status:** Complete

When using `--resume-from` with a different `--output-dir`, checkpoints are not saved to the new output directory.

**Reproduction:**
```bash
# Initial training saves to run1/
python -m aam.cli train --output-dir run1/ ...

# Resume with new output dir - checkpoints NOT saved to run2/
python -m aam.cli train --resume-from run1/checkpoints/best.pt --output-dir run2/ ...
```

**Current Behavior:**
- Training resumes correctly from checkpoint
- New `--output-dir` is created
- Checkpoints folder in new directory is empty or missing the latest model
- Best model checkpoint not written to new location

**Expected Behavior:**
- Checkpoints should be saved to the new `--output-dir/checkpoints/`
- Best model should be tracked and saved relative to new output directory
- `val_predictions.tsv` and other artifacts should go to new directory

**Root Cause:**
Checkpoints were only saved when validation metric improved over `best_metric_value`. When resuming with an already-good metric, validation might never beat it, so no checkpoint was saved to the new directory.

**Acceptance Criteria:**
- [x] Checkpoints saved to `--output-dir/checkpoints/` when resuming
- [x] Best model correctly tracked and saved to new location
- [x] All training artifacts (plots, predictions) saved to new output dir
- [x] Test verifying checkpoint saves to new directory on resume

**Fix Applied:** Save an initial checkpoint to the new checkpoint_dir immediately after resume, ensuring there's always a checkpoint in the output directory even if validation never beats the original best metric.

**Files:** `aam/training/trainer.py`, `tests/test_trainer.py`

---

### CLN-BUG-8: Multi-Pass Validation Fails in Distributed Training
**Priority:** HIGH | **Effort:** 1-2 hours | **Status:** Complete

Multi-pass validation (`--val-prediction-passes N`) crashes with `KeyError: 'total_loss'` when used with distributed training (`--distributed`).

**Reproduction:**
```bash
torchrun --nproc_per_node=4 -m aam.cli train \
    --distributed \
    --asv-sampling random \
    --val-prediction-passes 3 \
    # ... other flags
```

**Error:**
```
KeyError: 'total_loss'
  File "aam/training/trainer.py", line 1126, in train
    val_loss = val_results["total_loss"]
```

**Current Behavior:**
- Multi-pass validation runs on each rank
- Each rank logs "Multi-pass validation: 3 passes, 96 samples aggregated"
- `val_results` dictionary returned by multi-pass validation is missing `total_loss` key
- Training crashes on all ranks

**Expected Behavior:**
- Multi-pass validation should return the same dictionary structure as single-pass validation
- Must include `total_loss` key for training loop to continue
- Should work correctly in distributed mode

**Root Cause:**
The `_validate_epoch_multi_pass()` method in `Evaluator` likely returns a different result structure than `validate_epoch()`, missing required keys like `total_loss`.

**Investigation Areas:**
1. Check `_validate_epoch_multi_pass()` return value structure
2. Compare with `validate_epoch()` return value
3. Ensure all required keys are present in multi-pass results
4. Test distributed vs non-distributed behavior

**Acceptance Criteria:**
- [x] Multi-pass validation returns same dict structure as single-pass
- [x] `total_loss` key present in results (uses MAE for regression, 1-accuracy for classification)
- [x] Works with `--distributed` and `torchrun`
- [x] 1 test: test_evaluator_multi_pass_returns_total_loss

**Files:** `aam/training/evaluation.py`, `aam/training/trainer.py`, `tests/test_trainer.py`

---

### CLN-15: Multi-Pass Validation During Training
**Priority:** MEDIUM | **Effort:** 2-3 hours | **Status:** Complete

When training with `--asv-sampling random`, validation metrics can vary between epochs due to different random ASV subsets. Add multi-pass aggregation for more stable validation metrics.

**Current Behavior:**
- `--asv-sampling random` applies same random sampling to both train and validation
- Validation runs single pass per epoch
- Validation metrics may fluctuate due to random sampling variance

**Proposed Solution:**
Add `--val-prediction-passes N` option to `train` command:
1. Run N forward passes during validation with different random ASV subsets
2. Aggregate predictions (mean for regression) before computing metrics
3. Only applies when `--asv-sampling random` is used

**Implementation:**
- Added `_validate_epoch_multi_pass()` method to Evaluator
- Collects predictions per sample_id across all passes
- Aggregates using mean (regression) or mode (classification)

**Benefits:**
- More stable validation metrics when using random ASV sampling
- Better signal for early stopping and model selection
- Consistent with inference behavior when using `--prediction-passes`

**Acceptance Criteria:**
- [x] `--val-prediction-passes` CLI option for train command (default: 1)
- [x] Mean aggregation for regression during validation
- [x] Only applies when `--asv-sampling random` is used
- [x] Warning if used with non-random sampling
- [x] 7 tests (4 trainer, 3 CLI)

**Files:** `aam/cli/train.py`, `aam/training/trainer.py`, `aam/training/evaluation.py`, `tests/test_cli.py`, `tests/test_trainer.py`

---

### CLN-11: Consolidate Test Suite (Parent Ticket)
**Priority:** LOW | **Effort:** 8-12 hours | **Status:** Complete

Reduce test code duplication by extracting shared fixtures and utilities.

**Sub-tickets:**
- CLN-11.1: Consolidate duplicate fixtures to conftest.py
- CLN-11.2: Parametrize batch/sequence variation tests
- CLN-11.3: Extract shared test utilities

**Estimated Reduction:** 300-400 lines of duplicated code (~10-15% improvement)

---

### CLN-11.1: Consolidate Duplicate Fixtures to conftest.py
**Priority:** HIGH | **Effort:** 2-3 hours | **Status:** Complete

Move duplicate fixtures from individual test files to shared `tests/conftest.py`.

**Duplications Found:**

| Fixture | Files | Lines |
|---------|-------|-------|
| `sample_tokens` | test_sequence_encoder.py:71, test_sample_sequence_encoder.py:47, test_asv_encoder.py:39 | ~30 |
| `generate_150bp_sequence()` | test_biom_loader.py:13, test_dataset.py:20, test_tokenizer.py:9 | ~20 |
| `sample_embeddings` | test_transformer.py:23, test_attention_pooling.py:21 | ~15 |
| `sample_mask` | test_transformer.py:32, test_attention_pooling.py:30 | ~10 |
| `simple_table` | test_biom_loader.py:28, test_dataset.py:46 | ~30 |

**Scope:**
1. Add `generate_150bp_sequence()` helper to conftest.py
2. Add `sample_embeddings` fixture to conftest.py (batch=2, seq=10, dim=64)
3. Add `sample_mask` fixture to conftest.py
4. Add `simple_table` fixture to conftest.py
5. Remove duplicate definitions from individual test files

**Acceptance Criteria:**
- [x] All fixtures moved to conftest.py
- [x] Duplicate definitions removed from test files
- [x] All 1276 tests pass
- [x] ~100 lines removed (122 lines removed, 67 added = 40 net reduction)

**Files:**
- `tests/conftest.py` (expand)
- `tests/test_sample_sequence_encoder.py` (remove fixture)
- `tests/test_asv_encoder.py` (remove fixture)
- `tests/test_biom_loader.py` (remove helper + fixture)
- `tests/test_dataset.py` (remove helper + fixture)
- `tests/test_tokenizer.py` (remove helper)
- `tests/test_transformer.py` (remove fixtures)
- `tests/test_attention_pooling.py` (remove fixtures)

---

### CLN-11.2: Parametrize Batch/Sequence Variation Tests
**Priority:** MEDIUM | **Effort:** 3-4 hours | **Status:** Complete

Replace loop-based variation tests with `@pytest.mark.parametrize` for cleaner test output.

**Patterns Found:**

| Pattern | Files | Example |
|---------|-------|---------|
| Batch size loops | 5+ files | `for batch_size in [1, 4, 8]: ...` |
| Sequence length loops | 4+ files | `for seq_len in [10, 50, 100, 150]: ...` |
| Encoder type loops | 3+ files | `for encoder_type in ["faith_pd", "taxonomy"]: ...` |

**Scope:**
1. Convert batch size loops to `@pytest.mark.parametrize("batch_size", [1, 4, 8])`
2. Convert sequence length loops to `@pytest.mark.parametrize("seq_len", [10, 50, 100, 150])`
3. Convert encoder type loops to `@pytest.mark.parametrize("encoder_type", [...])`

**Benefits:**
- Better test output (each parameter combination shown separately)
- Easier to identify which specific parameter fails
- pytest parallelization works better with parametrized tests

**Acceptance Criteria:**
- [x] Batch size tests parametrized in test_sequence_encoder.py, test_sample_sequence_encoder.py, test_transformer.py
- [x] Sequence length tests parametrized
- [x] Encoder type tests parametrized (where not already done)
- [x] All tests pass (1301 tests)
- [x] Improved test output clarity (line reduction minimal, +4 net, but each param shown separately)

**Files:**
- `tests/test_sequence_encoder.py`
- `tests/test_sample_sequence_encoder.py`
- `tests/test_transformer.py`
- `tests/test_asv_encoder.py`
- `tests/test_attention_pooling.py`

---

### CLN-11.3: Extract Shared Test Utilities
**Priority:** LOW | **Effort:** 2-3 hours | **Status:** Complete

Extract shared test classes and utilities to conftest.py for reuse.

**Items to Extract:**

| Item | Location | Description |
|------|----------|-------------|
| `MockBatchDataset` | test_trainer.py:45-67 | Mock dataset for trainer tests |
| Device fixture | Multiple files | `torch.device("cuda" if torch.cuda.is_available() else "cpu")` |
| Loss function fixtures | test_losses.py, test_trainer.py | `MultiTaskLoss(...)` variations |

**Scope:**
1. Move `MockBatchDataset` to conftest.py
2. Create shared `device` fixture with CUDA/MPS/CPU detection
3. Create parameterized loss function fixture
4. Standardize tensor assertion helpers

**Acceptance Criteria:**
- [x] MockBatchDataset in conftest.py
- [x] Shared device fixture created (with CUDA cleanup)
- [x] Loss function fixture consolidated
- [x] small_model fixture consolidated
- [x] All tests pass (1301 tests)
- [x] Net 75 lines removed (-136, +61)

**Files:**
- `tests/conftest.py` (expanded with MockBatchDataset, device, small_model, loss_fn)
- `tests/test_trainer.py` (removed MockBatchDataset, device, small_model)
- `tests/test_lr_finder.py` (removed device, small_model, loss_fn)
- `tests/test_batch_size_finder.py` (removed device, small_model, loss_fn)
- `tests/test_integration.py` (removed device)

---

### CLN-17: Reduce Total Test Count Through Consolidation
**Priority:** LOW | **Effort:** 4-6 hours | **Status:** Complete

Analyze test suite (1301 tests) for opportunities to reduce test count while maintaining coverage.

**Motivation:**
- Large test counts can slow CI/CD pipelines
- Many tests may overlap in coverage
- Some parametrized tests may be over-testing with redundant parameter combinations

**Investigation Areas:**
1. Identify tests with overlapping coverage (e.g., multiple tests checking same code paths)
2. Review parametrize combinations - are all parameter values necessary?
3. Look for integration tests that could replace multiple unit tests
4. Check for tests that verify implementation details vs behavior

**Potential Consolidation Targets:**
- Model encoder tests (test_asv_encoder, test_sample_sequence_encoder, test_sequence_encoder, test_sequence_predictor) - lots of similar shape/gradient tests
- Transformer tests - many parameter variation tests
- CLI tests - may have redundant integration tests

**Acceptance Criteria:**
- [ ] Audit report identifying consolidation opportunities
- [ ] Reduce test count by 10-20% without losing coverage
- [ ] Maintain or improve test runtime
- [ ] No reduction in code coverage percentage

**Files:** All test files in `tests/`

---

### CLN-16: Consolidate Lazy Embedding Tests with Parametrize
**Priority:** HIGH | **Effort:** 1-2 hours | **Status:** Complete

Consolidate repetitive tests in `TestLazySampleEmbeddings` and `TestLazyBaseEmbeddings` using `pytest.mark.parametrize` and shared fixtures.

**Context:**
PYT-18.5 added 17 new tests for lazy embedding computation. Many tests follow similar patterns:
- Test for different encoder types (unifrac, faith_pd, combined)
- Test with/without various flags (nucleotides, categoricals, frozen_base)
- Test that embeddings are/aren't returned based on flag

**Scope:**
1. Extract common model creation into pytest fixtures in `conftest.py`
2. Use `@pytest.mark.parametrize` to test multiple encoder types in single test
3. Consolidate similar assertion patterns
4. Reduce test file line count while maintaining coverage

**Example Consolidation:**
```python
# Before: 3 separate tests
def test_sample_embeddings_not_returned_unifrac(...)
def test_sample_embeddings_not_returned_faith_pd(...)
def test_sample_embeddings_not_returned_combined(...)

# After: 1 parametrized test
@pytest.mark.parametrize("encoder_type", ["unifrac", "faith_pd", "combined"])
def test_sample_embeddings_not_returned_by_default(encoder_type, sample_tokens):
    encoder = SequenceEncoder(encoder_type=encoder_type, ...)
    result = encoder(sample_tokens)
    assert "sample_embeddings" not in result
```

**Target Files:**
- `tests/test_sequence_encoder.py` - `TestLazySampleEmbeddings` class
- `tests/test_sequence_predictor.py` - `TestLazyBaseEmbeddings` class
- `tests/conftest.py` - shared fixtures

**Acceptance Criteria:**
- [x] Use `@pytest.mark.parametrize` for encoder type variations
- [x] Extract shared model fixtures to `conftest.py`
- [x] Reduce total lines in lazy embedding test classes (32 lines removed)
- [x] All lazy embedding tests still pass (18 test cases, +1 from added coverage)
- [x] No loss of test coverage (added combined encoder "returned when requested" test)

**Files:** `tests/test_sequence_encoder.py`, `tests/test_sequence_predictor.py`, `tests/conftest.py`

---

### CLN-13: ASV Sampling Strategy When Exceeding Token Limit
**Priority:** MEDIUM | **Effort:** 2-3 hours | **Status:** Complete

Currently when a sample has more ASVs than `--token-limit`, the first N ASVs are kept based on their matrix order in the BIOM file. This is arbitrary and may drop high-abundance ASVs.

**Current Behavior:**
```python
# collate_fn in dataset.py
if num_asvs > token_limit:
    tokens = tokens[:token_limit]  # First N by matrix order
    counts = counts[:token_limit]
```

**Proposed Options (`--asv-sampling`):**
1. **`first`** (default, current behavior): Keep first N ASVs by matrix order
2. **`abundance`**: Sort by count descending, keep top N most abundant ASVs
3. **`random`**: Randomly sample N ASVs (different each batch = data augmentation)

**Implementation:**
1. Add `--asv-sampling` CLI option to train.py and pretrain.py
2. Pass `asv_sampling` to `collate_fn` via `partial()`
3. In `collate_fn`:
   ```python
   if num_asvs > token_limit:
       if asv_sampling == "abundance":
           # Sort by count descending
           sorted_idx = torch.argsort(counts.squeeze(), descending=True)[:token_limit]
           tokens = tokens[sorted_idx]
           counts = counts[sorted_idx]
       elif asv_sampling == "random":
           # Random sample (varies per batch = augmentation)
           perm = torch.randperm(num_asvs)[:token_limit]
           tokens = tokens[perm]
           counts = counts[perm]
       else:  # "first"
           tokens = tokens[:token_limit]
           counts = counts[:token_limit]
   ```

**Benefits:**
- `abundance`: Prioritizes most informative ASVs (higher counts = more signal)
- `random`: Acts as data augmentation, model sees different ASV subsets each epoch

**Acceptance Criteria:**
- [x] `--asv-sampling` CLI option with choices: first, abundance, random
- [x] Default behavior unchanged (`first`)
- [x] `abundance` sorts by count before truncating
- [x] `random` samples different ASVs each batch
- [x] 4 tests covering all sampling strategies + edge case
- [x] Documentation in CLI help text

**Files:** `aam/data/dataset.py`, `aam/cli/train.py`, `aam/cli/pretrain.py`, `tests/test_dataset.py`

---

### CLN-14: Multi-Pass Prediction Aggregation for Inference
**Priority:** LOW | **Effort:** 3-4 hours | **Status:** Complete

When using `--asv-sampling random` during training, predictions at inference time could benefit from running multiple forward passes with different random ASV subsets and aggregating the results.

**Problem:**
- Models trained with `--asv-sampling random` see different ASV subsets during training
- At inference, using a single random sample may give inconsistent predictions
- Using `abundance` is deterministic but may not match the distribution seen during training

**Proposed Solution:**
Add `--prediction-passes N` option to `predict` command:
1. Run N forward passes with different random ASV subsets
2. Aggregate predictions (mean for regression, vote/mean-logits for classification)
3. Optionally report prediction variance/confidence

**Implementation:**
```python
# In predict.py
for pass_idx in range(prediction_passes):
    batch = collate_fn(samples, token_limit, asv_sampling="random")
    pred = model(batch)
    predictions.append(pred)
# Aggregate
final_pred = torch.stack(predictions).mean(dim=0)
pred_std = torch.stack(predictions).std(dim=0)  # Optional confidence
```

**Benefits:**
- More robust predictions for high-ASV samples
- Provides uncertainty estimates via prediction variance
- Better matches training distribution when `random` sampling was used

**Acceptance Criteria:**
- [x] `--prediction-passes` CLI option for predict command (default: 1)
- [x] Mean aggregation for regression
- [x] Optional variance/std output for confidence
- [x] Only applies when `--asv-sampling random` is used
- [x] 5 tests (exceeds requirement)

**Files:** `aam/cli/predict.py`, `tests/test_cli.py`

---

## Summary

| Ticket | Description | Effort | Priority | Status |
|--------|-------------|--------|----------|--------|
| **FUS-1** | GMU baseline | 3-4h | HIGH | Complete |
| **FUS-2** | Cross-attention fusion | 5-6h | HIGH | Complete |
| **FUS-3** | Perceiver fusion | 6-8h | LOW | Not Started |
| **CLN-2** | Normalization unification | 3-4h | MEDIUM | Complete |
| **CLN-3** | Remove unused params | 1-2h | LOW | Complete |
| **CLN-4** | Extract shared utilities | 2-3h | LOW | Not Started |
| **CLN-5** | DataParallel in train.py | 2-3h | MEDIUM | Complete |
| **CLN-6** | Categorical docs/validation | 4-5h | MEDIUM | Complete |
| **CLN-7** | Toggle count prediction | 2-3h | MEDIUM | Complete |
| **CLN-8** | Categorical learning rate | 2-3h | MEDIUM | Complete |
| **CLN-9** | Remove FiLM conditioning | 2-3h | MEDIUM | Complete |
| **CLN-10** | Training output artifacts | 2-3h | HIGH | Complete |
| **CLN-BUG-1** | Z-score denorm in TensorBoard | 1-2h | HIGH | Complete |
| **CLN-BUG-2** | val_predictions.tsv not written on resume | 1-2h | HIGH | Complete |
| **CLN-BUG-3** | --resume-from ignores new learning rate | 1-2h | HIGH | Complete |
| **CLN-BUG-4** | LR override undone by double load_checkpoint | 0.5-1h | HIGH | Complete |
| **CLN-BUG-5** | zscore-cat TensorBoard not denormalized | 1-2h | HIGH | Complete |
| **CLN-BUG-6** | Model converging to mean with freeze-base+cross-attn | 2-4h | HIGH | Complete |
| **CLN-BUG-7** | Checkpoints not saved to new output dir on resume | 1-2h | HIGH | Complete |
| **CLN-BUG-8** | Multi-pass validation fails in distributed training | 1-2h | HIGH | Complete |
| **CLN-15** | Multi-pass validation during training | 2-3h | MEDIUM | Complete |
| **CLN-11** | Consolidate test suite (parent) | 8-12h | LOW | Complete |
| **CLN-11.1** | Consolidate duplicate fixtures | 2-3h | HIGH | Complete |
| **CLN-11.2** | Parametrize variation tests | 3-4h | MEDIUM | Complete |
| **CLN-11.3** | Extract shared utilities | 2-3h | LOW | Complete |
| **CLN-12** | Random Forest baseline script | 2-3h | LOW | Complete |
| **CLN-13** | ASV sampling strategy (abundance/random) | 2-3h | MEDIUM | Complete |
| **CLN-14** | Multi-pass prediction aggregation | 3-4h | LOW | Complete |
| **CLN-16** | Consolidate lazy embedding tests | 1-2h | HIGH | Complete |
| **CLN-17** | Reduce total test count | 4-6h | LOW | Not Started |
| **Total** | | **49-77h** | |

## Recommended Order

**Completed:**
- CLN-BUG-1 to CLN-BUG-8 (bug fixes)
- FUS-1, FUS-2 (fusion MVP)
- CLN-2, CLN-5, CLN-6, CLN-7, CLN-8, CLN-9, CLN-10, CLN-11.1, CLN-11.2, CLN-12, CLN-13, CLN-14, CLN-15, CLN-16

**Remaining - Low Priority:**
- CLN-3 (remove unused params)
- CLN-4 (shared utilities)
- CLN-11.3 (extract shared test utilities)
- FUS-3 (perceiver fusion, optional)

---

## UTIL: Utility Scripts

### CLN-12: Random Forest Baseline Script
**Priority:** LOW | **Effort:** 2-3 hours | **Status:** Complete

Create a helper script to run Random Forest regression as a comparison baseline against AAM model results.

**Motivation:**
- Provide a simple ML baseline for comparison with AAM predictions
- Use identical train/validation splits from AAM training (via CLN-10 output files)
- Enable fair comparison of AAM vs traditional ML on the same data splits

**Proposed CLI:**
```bash
python -m aam.cli.rf_baseline \
    --table <biom_file> \
    --metadata <metadata.tsv> \
    --metadata-column <target_column> \
    --train-samples <train_samples.txt> \
    --val-samples <val_samples.txt> \
    --output <predictions.tsv>
```

**Inputs:**
- `--table`: BIOM table file (same as AAM training)
- `--metadata`: Metadata TSV file with target column
- `--metadata-column`: Column name for regression target
- `--train-samples`: Text file with training sample IDs (one per line, from CLN-10)
- `--val-samples`: Text file with validation sample IDs (one per line, from CLN-10)
- `--output`: Output TSV file for predictions

**Optional Inputs:**
- `--n-estimators`: Number of trees (default: 500)
- `--max-features`: Max features per tree (default: "sqrt")
- `--random-seed`: Random seed for reproducibility (default: 42)
- `--n-jobs`: Number of parallel jobs (default: -1, all cores)

**Output Format (predictions.tsv):**
```
sample_id	actual	predicted
sample_001	12.5	11.8
sample_002	8.3	9.1
...
```

**Console Output:**
```
Loaded 500 training samples, 100 validation samples
Features: 1234 ASVs
Training Random Forest (n_estimators=500)...
Validation Metrics:
  R²:   0.752
  MAE:  1.23
  RMSE: 1.56
Predictions saved to: predictions.tsv
```

**Implementation Notes:**
- Use `biom.load_table()` for BIOM loading (consistent with AAM)
- Feature matrix: samples × ASVs (transposed from BIOM's ASVs × samples)
- Filter BIOM to only include samples in train + val lists
- Remove zero-sum features after filtering
- Use scikit-learn `RandomForestRegressor`

**Scope:**
1. Create `aam/cli/rf_baseline.py` with CLI interface
2. Load BIOM, metadata, and sample lists
3. Train RandomForestRegressor on training samples
4. Predict on validation samples
5. Compute and display R², MAE, RMSE metrics
6. Save predictions TSV

**Acceptance Criteria:**
- [x] Script runs with specified inputs
- [x] Uses identical samples to AAM training (from train_samples.txt/val_samples.txt)
- [x] Outputs metrics to console
- [x] Saves predictions in same format as AAM's val_predictions.tsv
- [x] Generates prediction plot matching AAM's TensorBoard style
- [x] Handles missing samples gracefully (warns if sample not in BIOM)
- [x] 20 tests (4 load_sample_ids, 2 load_biom, 3 train_rf, 3 metrics, 2 plot, 6 CLI)

**Dependencies:**
- CLN-10 (Training Output Artifacts) - provides train_samples.txt and val_samples.txt

**Files:** `aam/cli/rf_baseline.py` (new), `tests/test_rf_baseline.py` (new)
