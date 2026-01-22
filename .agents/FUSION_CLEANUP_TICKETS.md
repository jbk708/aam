# Attention Fusion & Code Cleanup Tickets

**Last Updated:** 2026-01-21
**Status:** 9 tickets (~23-38 hours)
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
**Priority:** LOW | **Effort:** 1-2 hours | **Status:** Not Started

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
**Priority:** LOW | **Effort:** 2-3 hours | **Status:** Not Started

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
**Priority:** MEDIUM | **Effort:** 4-5 hours | **Status:** Not Started

Document and validate the three parallel categorical systems.

**Current Systems:**
1. Base fusion (`--categorical-fusion concat|add`)
2. Conditional scaling (`--conditional-output-scaling`)
3. FiLM (`--film-conditioning`)

**Scope:**
- Add `--categorical-help` showing decision tree
- Warn if redundant flags used together
- Document clear recommendations in README

| Use Case | Recommended |
|----------|-------------|
| Simple metadata | `--categorical-fusion concat` |
| Per-category shift | `concat` + `--conditional-output-scaling` |
| Feature modulation | `--film-conditioning` |
| Position-specific | `--categorical-fusion cross-attention` (FUS-2) |

**Acceptance Criteria:**
- [ ] Decision tree in `--categorical-help`
- [ ] Validation warnings for redundant combos
- [ ] README updated

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
**Priority:** MEDIUM | **Effort:** 2-3 hours | **Status:** Not Started

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
- Categorical params include: `categorical_embedder`, `categorical_projection`, `gmu`, `conditional_scale`, `conditional_bias`
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
- [ ] `--categorical-lr` sets separate LR for categorical params
- [ ] Default behavior unchanged (uses --lr for all)
- [ ] Works with all optimizers
- [ ] TensorBoard logs both learning rates
- [ ] 5+ tests

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

### CLN-11: Consolidate Test Suite
**Priority:** LOW | **Effort:** 4-6 hours | **Status:** Not Started

Reduce test code duplication by extracting shared fixtures and utilities.

**Scope:**
1. Extract shared mock classes (e.g., `MockBatchDataset`) to `tests/conftest.py` or `tests/fixtures.py`
2. Consolidate similar test fixtures across test files
3. Extract repeated test setup patterns into shared utilities
4. Reduce inline class definitions in test methods
5. Review test organization and consider grouping related tests

**Acceptance Criteria:**
- [ ] Shared mock classes extracted to central location
- [ ] Duplicate fixtures consolidated
- [ ] No functionality changes to tests
- [ ] All tests pass
- [ ] Reduced lines of test code

**Files:** `tests/conftest.py`, `tests/test_trainer.py`, `tests/test_cli.py`

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
| **CLN-6** | Categorical docs/validation | 4-5h | MEDIUM | Not Started |
| **CLN-7** | Toggle count prediction | 2-3h | MEDIUM | Complete |
| **CLN-8** | Categorical learning rate | 2-3h | MEDIUM | Not Started |
| **CLN-9** | Remove FiLM conditioning | 2-3h | MEDIUM | Complete |
| **CLN-10** | Training output artifacts | 2-3h | HIGH | Complete |
| **CLN-BUG-1** | Z-score denorm in TensorBoard | 1-2h | HIGH | Complete |
| **CLN-BUG-2** | val_predictions.tsv not written on resume | 1-2h | HIGH | Complete |
| **CLN-BUG-3** | --resume-from ignores new learning rate | 1-2h | HIGH | Complete |
| **CLN-11** | Consolidate test suite | 4-6h | LOW | Not Started |
| **CLN-12** | Random Forest baseline script | 2-3h | LOW | Complete |
| **Total** | | **39-62h** | |

## Recommended Order

**Phase 0 - High Priority:**
1. CLN-BUG-1 (z-score denormalization bug)

**Phase 1 - Quick Wins:**
2. CLN-3 (remove dead code)
3. CLN-5 (DataParallel parity)

**Phase 2 - Fusion MVP:**
4. FUS-1 (GMU baseline)
5. FUS-2 (Cross-attention)

**Phase 3 - User Experience:**
6. CLN-6 (categorical docs)
7. CLN-7 (toggle count prediction)

**Phase 4 - Tech Debt:**
8. CLN-4 (shared utilities)
9. FUS-3 (perceiver, optional)

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
