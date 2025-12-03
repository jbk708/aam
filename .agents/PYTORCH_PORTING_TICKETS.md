# PyTorch Porting Tickets

**Priority**: HIGH - Core Implementation  
**Status**: Not Started

This document contains tickets for implementing the PyTorch port of AAM, organized by implementation order.

---

## Phase 1: Data Pipeline

### PYT-1.1: Implement BIOM Loader
**Priority:** HIGH | **Effort:** Medium | **Status:** ✅ Completed

**Description:**
Implement BIOM table loading and rarefaction as specified in `pytorch_porting_plan/01_biom_loader.md`.

**Files Created:**
- `aam/data/biom_loader.py`
- `tests/test_biom_loader.py`

**Acceptance Criteria:**
- [x] `BIOMLoader` class implemented
- [x] `load_table()` method loads BIOM file
- [x] `rarefy()` method subsamples to specified depth (uses biom-format's built-in `subsample()`)
- [x] `get_sequences()` extracts sequences from observation IDs (150bp sequences)
- [x] Handles empty samples/ASVs (drops samples below depth)
- [x] Supports reproducible rarefaction (random seed)
- [x] Unit tests pass (21 tests, 93% coverage)

**Implementation Notes:**
- Uses biom-format's built-in `subsample()` method for rarefaction
- `filter_and_sort()` removed - all ASVs are used, no filtering needed
- Sequences extracted directly from observation IDs (assumed to be 150bp DNA sequences)
- No metadata handling - BIOM tables assumed to have no metadata
- Samples with fewer reads than depth are dropped during rarefaction

**Dependencies:** None

**Actual Time:** ~4 hours

---

### PYT-1.2: Implement UniFrac Computer
**Priority:** HIGH | **Effort:** Medium | **Status:** ✅ Completed

**Description:**
Implement UniFrac distance computation using `biocore/unifrac-binaries` library (https://github.com/biocore/unifrac-binaries/tree/main) as specified in `pytorch_porting_plan/02_unifrac_computer.md`.

**Files Created:**
- `aam/data/unifrac.py`
- `tests/test_unifrac.py`

**Acceptance Criteria:**
- [x] `UniFracComputer` class implemented
- [x] Uses `unifrac` library for computation (package name is `unifrac`, not `unifrac-binaries`)
- [x] `compute_unweighted()` returns `DistanceMatrix`
- [x] `compute_faith_pd()` returns `pandas.Series` (as returned by library)
- [x] `extract_batch_distances()` extracts batch-level distances
- [x] Handles epoch regeneration (ready for integration)
- [x] Validates batch size is even
- [x] Unit tests pass (19 tests, all passing)

**Dependencies:** PYT-1.1

**Library:** `unifrac` (package name is `unifrac`, install via `pip install unifrac` or `conda install -c biocore unifrac`)

**Implementation Notes:**
- Uses `unifrac` package (not `unifrac-binaries` - package name differs from repo name)
- `compute_unweighted()` loads tree using `skbio.read()` and calls `unifrac.unweighted()`
- `compute_faith_pd()` returns `pandas.Series` (library default, not DistanceMatrix)
- `extract_batch_distances()` handles both `DistanceMatrix` (unweighted) and `Series` (faith_pd)
- Batch size validation integrated into `extract_batch_distances()` for unweighted metric
- Error handling for file not found, ASV ID mismatches, and invalid inputs
- Tests include unit tests with synthetic data and integration tests with real BIOM/tree files

**Estimated Time:** 4-6 hours
**Actual Time:** ~4 hours

---

### PYT-1.3: Implement Dataset and Tokenizer
**Priority:** HIGH | **Effort:** Medium | **Status:** ✅ Completed

**Description:**
Implement PyTorch Dataset and sequence tokenization as specified in `pytorch_porting_plan/03_dataset_tokenizer.md`.

**Files Created:**
- `aam/data/tokenizer.py`
- `aam/data/dataset.py`
- `tests/test_tokenizer.py`
- `tests/test_dataset.py`

**Acceptance Criteria:**
- [x] `SequenceTokenizer` class implemented
- [x] Nucleotide to token mapping (A/C/G/T → 1/2/3/4)
- [x] `ASVDataset` class inherits from `Dataset`
- [x] Custom `collate_fn` handles variable ASV counts
- [x] Handles epoch regeneration (ready for integration)
- [x] Unit tests pass (30 tests, all passing)

**Dependencies:** PYT-1.1, PYT-1.2

**Implementation Notes:**
- Class named `ASVDataset` (not `MicrobialDataset`)
- `SequenceTokenizer` maps A→1, C→2, G→3, T→4, invalid chars→0
- `ASVDataset` extracts sequences from BIOM table observation IDs
- Supports optional metadata with target column
- Supports optional UniFrac distances (unweighted or faith_pd)
- `collate_fn` pads samples to `token_limit` ASVs per sample
- Returns batched tensors: tokens `[B, S, L]`, counts `[B, S, 1]`
- Tests include unit tests and integration tests with DataLoader

**Estimated Time:** 6-8 hours
**Actual Time:** ~4 hours

---

## Phase 2: Core Components

### PYT-2.1: Implement Core Layers
**Priority:** HIGH | **Effort:** Low | **Status:** ✅ Completed

**Description:**
Implement AttentionPooling and PositionEmbedding as specified in `pytorch_porting_plan/04_core_layers.md`.

**Files Created:**
- `aam/models/attention_pooling.py`
- `aam/models/position_embedding.py`
- `tests/test_attention_pooling.py`
- `tests/test_position_embedding.py`

**Acceptance Criteria:**
- [x] `AttentionPooling` class implemented
- [x] `PositionEmbedding` class implemented
- [x] Masking utility functions implemented (`float_mask`, `create_mask_from_tokens`, `apply_mask`)
- [x] Unit tests pass (25 tests, 100% coverage)

**Implementation Notes:**
- `AttentionPooling` uses query projection, scales scores by `sqrt(hidden_dim)`, handles masking with `-inf` before softmax
- `PositionEmbedding` uses learned embeddings, adds (not concatenates) position information
- Utility functions handle mask conversion and application
- All components handle variable sequence lengths and device placement correctly

**Dependencies:** None

**Estimated Time:** 3-4 hours
**Actual Time:** ~3 hours

---

### PYT-2.2: Implement Transformer Encoder
**Priority:** HIGH | **Effort:** Medium | **Status:** ✅ Completed

**Description:**
Implement transformer encoder as specified in `pytorch_porting_plan/05_transformer.md`.

**Files Created:**
- `aam/models/transformer.py`
- `tests/test_transformer.py`

**Acceptance Criteria:**
- [x] `TransformerEncoder` class implemented
- [x] Uses PyTorch built-in or custom implementation
- [x] Handles masking correctly
- [x] Supports configurable layers, heads, dimensions
- [x] Unit tests pass (23 tests, all passing)

**Dependencies:** PYT-2.1

**Implementation Notes:**
- Uses PyTorch built-in `nn.TransformerEncoder` and `nn.TransformerEncoderLayer` (Option 1 from plan)
- Pre-norm architecture (`norm_first=True`) for training stability
- Batch-first format (`batch_first=True`) for consistency
- Mask conversion: converts input mask (1=valid, 0=padding) to PyTorch format (True=padding, False=valid)
- Output normalization with LayerNorm (eps=1e-6)
- Supports GELU and ReLU activations
- Default intermediate_size = 4 * hidden_dim
- All 23 unit tests pass, covering shapes, masking, gradients, dropout, and device handling

**Estimated Time:** 4-6 hours
**Actual Time:** ~3 hours

---

## Phase 3: Model Architecture

### PYT-3.1: Implement ASVEncoder
**Priority:** HIGH | **Effort:** Medium | **Status:** ✅ Completed

**Description:**
Implement ASV-level sequence encoder as specified in `pytorch_porting_plan/06_asv_encoder.md`.

**Files Created:**
- `aam/models/asv_encoder.py`
- `tests/test_asv_encoder.py`

**Acceptance Criteria:**
- [x] `ASVEncoder` class implemented
- [x] Processes nucleotide sequences
- [x] Returns ASV embeddings `[B, S, D]`
- [x] Optional nucleotide prediction head
- [x] Unit tests pass (28 tests, all passing)

**Dependencies:** PYT-2.1, PYT-2.2

**Implementation Notes:**
- Reshapes input from `[B, S, L]` to `[B*S, L]` for parallel processing of all ASVs
- Creates mask from tokens (1 for valid, 0 for padding) for transformer and attention pooling
- Token embedding layer maps vocab_size (default 5) to embedding_dim
- Position embedding uses max_bp + 1 to account for 0-indexed positions
- Transformer encoder processes nucleotide sequences with configurable layers, heads, and dimensions
- Attention pooling reduces sequence-level embeddings to single ASV embedding
- Optional nucleotide prediction head for self-supervised learning (only computed when requested)
- Handles variable-length sequences with padding correctly
- Converts tokens to long integers for embedding layer compatibility
- All 28 unit tests pass, covering shapes, masking, gradients, dropout, device handling, and various configurations
- Minimal inline comments following workflow principles

**Estimated Time:** 4-6 hours
**Actual Time:** ~4 hours

---

### PYT-3.2: Implement SampleSequenceEncoder
**Priority:** HIGH | **Effort:** Medium | **Status:** ✅ Completed

**Description:**
Implement sample-level encoder as specified in `pytorch_porting_plan/07_base_sequence_encoder.md`.

**Files Created:**
- `aam/models/sample_sequence_encoder.py`
- `tests/test_sample_sequence_encoder.py`

**Acceptance Criteria:**
- [x] `SampleSequenceEncoder` class implemented
- [x] Composes ASVEncoder
- [x] Processes ASV embeddings at sample level
- [x] Returns sample embeddings `[B, S, D]`
- [x] Unit tests pass (30 tests, all passing)

**Dependencies:** PYT-3.1

**Implementation Notes:**
- Class renamed from `BaseSequenceEncoder` to `SampleSequenceEncoder` for clarity
- Processes ASV embeddings through sample-level position embeddings and transformer
- Handles NaN values from fully padded ASVs
- Supports training mode with nucleotide predictions and inference mode

**Estimated Time:** 4-6 hours
**Actual Time:** ~4 hours

---

### PYT-3.3: Implement SequenceEncoder
**Priority:** HIGH | **Effort:** Medium | **Status:** ✅ Completed

**Description:**
Implement encoder with UniFrac prediction head as specified in `pytorch_porting_plan/08_sequence_encoder.md`.

**Files Created:**
- `aam/models/sequence_encoder.py`
- `tests/test_sequence_encoder.py`

**Acceptance Criteria:**
- [x] `SequenceEncoder` class implemented
- [x] Composes SampleSequenceEncoder
- [x] Predicts UniFrac distances
- [x] Returns sample embeddings and predictions
- [x] Supports different encoder types (unifrac, taxonomy, faith_pd, combined)
- [x] Unit tests pass (25 tests, all passing)

**Implementation Notes:**
- Composes `SampleSequenceEncoder` as base component
- Adds encoder transformer, attention pooling, and output head
- Supports four encoder types: `unifrac`, `taxonomy`, `faith_pd`, `combined`
- Combined type uses separate heads: `uni_ff`, `faith_ff`, `tax_ff`
- Returns dictionary with `base_prediction` and `sample_embeddings`
- Optional nucleotide predictions returned as side output (for loss only, not used as input)
- Handles mask creation from tokens for transformer and attention pooling
- Supports optional `base_output_dim` (defaults to `embedding_dim` if None)
- All 25 unit tests pass, covering shapes, masking, gradients, device handling, and all encoder types

**Dependencies:** PYT-3.2

**Estimated Time:** 6-8 hours
**Actual Time:** ~4 hours

---

### PYT-3.4: Implement SequencePredictor
**Priority:** HIGH | **Effort:** Medium | **Status:** ✅ Completed

**Description:**
Implement main prediction model as specified in `pytorch_porting_plan/09_sequence_predictor.md`.

**Files Created:**
- `aam/models/sequence_predictor.py`
- `tests/test_sequence_predictor.py`

**Acceptance Criteria:**
- [x] `SequencePredictor` class implemented
- [x] Composes SequenceEncoder as base model
- [x] Supports `freeze_base` parameter
- [x] Predicts target and counts
- [x] Returns dictionary of predictions
- [x] Unit tests pass (27 tests, all passing)

**Implementation Notes:**
- Composes SequenceEncoder as `base_model` (composition pattern, not inheritance)
- Extracts `embedding_dim` from base_model when provided to ensure dimension consistency
- Supports freezing base model parameters via `freeze_base=True`
- Count encoder: TransformerEncoder + Linear head → `[B, S, 1]` predictions
- Target encoder: TransformerEncoder + AttentionPooling + Linear head → `[B, out_dim]` predictions
- Uses base embeddings (`sample_embeddings`) from base_model, NOT base predictions (for loss only)
- Supports classification mode (log-softmax) and regression mode
- Returns dictionary with `target_prediction`, `count_prediction`, `base_embeddings`, and optionally `base_prediction`/`nuc_predictions`
- Handles all encoder types (unifrac, taxonomy, faith_pd, combined)
- All 27 unit tests pass, covering shapes, masking, gradients, device handling, frozen/unfrozen base, and all encoder types

**Dependencies:** PYT-3.3

**Estimated Time:** 6-8 hours
**Actual Time:** ~4 hours

---

## Phase 4: Training

### PYT-4.1: Implement Loss Functions
**Priority:** HIGH | **Effort:** Medium | **Status:** ✅ Completed

**Description:**
Implement multi-task loss functions as specified in `pytorch_porting_plan/10_training_losses.md`.

**Files Created:**
- `aam/training/losses.py`
- `aam/training/metrics.py`
- `tests/test_losses.py`
- `tests/test_metrics.py`

**Acceptance Criteria:**
- [x] Target loss (MSE/NLL) implemented
- [x] Count loss (masked MSE) implemented
- [x] Base loss (UniFrac MSE) implemented
- [x] Nucleotide loss (masked CrossEntropy) implemented
- [x] Total loss computation implemented
- [x] Metrics (MAE, accuracy, etc.) implemented
- [x] Unit tests pass (27 tests, all passing)

**Implementation Notes:**
- `MultiTaskLoss` class implements all loss functions with configurable penalty weights
- Target loss supports both regression (MSE) and classification (NLL with optional class weights)
- Count loss uses masked MSE to ignore padding ASVs
- Base loss handles all encoder types (unifrac, faith_pd, taxonomy, combined)
- Nucleotide loss uses masked CrossEntropy to ignore padding positions
- Total loss computes weighted sum of all component losses
- Metrics module provides regression (MAE, MSE, R2) and classification (accuracy, precision, recall, F1) metrics
- Count metrics support masked computation for valid ASVs only
- All losses handle missing outputs gracefully (inference mode)
- Mask creation from tokens or counts when not explicitly provided
- Device handling ensures all tensors are on the same device
- Metrics use `.tolist()` conversion to avoid numpy compatibility issues
- All 27 unit tests pass (18 for losses, 11 for metrics, 2 CUDA tests skipped)

**Dependencies:** PYT-3.4

**Estimated Time:** 4-6 hours
**Actual Time:** ~4 hours

---

### PYT-4.2: Implement Training Loop
**Priority:** HIGH | **Effort:** High | **Status:** ✅ Completed

**Description:**
Implement training and validation loops with staged training support as specified in `pytorch_porting_plan/11_training_loop.md`.

**Files Created:**
- `aam/training/trainer.py`
- `tests/test_trainer.py`

**Acceptance Criteria:**
- [x] Training epoch function implemented
- [x] Validation epoch function implemented
- [x] Main training function implemented
- [x] Supports loading pre-trained SequenceEncoder
- [x] Supports `freeze_base` parameter
- [x] Early stopping implemented
- [x] Checkpoint saving/loading implemented
- [x] Unit tests pass (21 tests, 81% coverage)

**Dependencies:** PYT-4.1

**Implementation Notes:**
- `Trainer` class implements full training and validation loops with progress bars (tqdm)
- `WarmupCosineScheduler` custom scheduler for warmup + cosine decay
- `create_optimizer()` creates AdamW optimizer, excludes frozen parameters when `freeze_base=True`
- `create_scheduler()` creates warmup + cosine decay scheduler
- `load_pretrained_encoder()` loads pre-trained SequenceEncoder into SequencePredictor
- Supports both dict and tuple batch formats from DataLoader
- Handles both SequenceEncoder and SequencePredictor models
- Early stopping with configurable patience (default: 50 epochs)
- Checkpoint saving includes model state, optimizer state, scheduler state, epoch, best loss, metrics
- Checkpoint loading supports resume training with optional optimizer/scheduler loading
- Metrics computation during validation (regression, classification, count metrics)
- All 21 unit tests pass, covering all acceptance criteria

**Estimated Time:** 8-10 hours
**Actual Time:** ~4 hours

---

### PYT-4.3: Implement CLI Interface
**Priority:** MEDIUM | **Effort:** Medium | **Status:** ✅ Completed

**Description:**
Implement command-line interface as specified in `pytorch_porting_plan/12_cli_interface.md`.

**Files Created:**
- `aam/cli.py`
- `tests/test_cli.py`

**Acceptance Criteria:**
- [x] Training command implemented
- [x] Inference command implemented (optional)
- [x] Argument validation implemented
- [x] Error handling implemented
- [x] Logging implemented
- [x] Integration tests pass (28 tests, all passing)

**Dependencies:** PYT-4.2

**Implementation Notes:**
- Uses `click` for argument parsing
- Training command includes full pipeline: data loading, model creation, training loop, checkpointing
- Predict command supports inference with model checkpoint loading
- Comprehensive argument validation (batch size must be even, classifier requires out_dim > 1, etc.)
- Logging to both console and file (`training.log`)
- Helper functions for device setup, random seed, file validation
- All 28 unit tests pass, covering setup functions, validation, CLI commands, and integration tests
- CLI can be invoked via `python -m aam.cli train` or `python -m aam.cli predict`

**Estimated Time:** 6-8 hours
**Actual Time:** ~4 hours

---

## Phase 5: Testing

### PYT-5.1: Write Unit Tests
**Priority:** HIGH | **Effort:** High | **Status:** ✅ Completed

**Description:**
Write comprehensive unit tests as specified in `pytorch_porting_plan/13_testing.md`.

**Files Created/Modified:**
- `tests/test_cli.py` (enhanced with integration tests)
- `tests/test_unifrac.py` (added error handling tests)
- `tests/test_dataset.py` (added edge case tests)
- `tests/test_trainer.py` (added edge case tests)
- Note: `test_models.py` not needed - all model components have dedicated test files

**Acceptance Criteria:**
- [x] All components have unit tests
- [x] Tests cover edge cases
- [x] All tests pass (333 tests passing, 4 skipped)
- [x] Test coverage > 80% (achieved 94% coverage)

**Implementation Notes:**
- Added CLI integration tests covering train() and predict() command flows
- Added error handling tests for unifrac.py (invalid tree files, ASV mismatches, missing sample IDs)
- Added edge case tests for dataset.py (token truncation, empty samples, string targets, Faith PD extraction)
- Added trainer edge case tests (scheduler warmup, checkpoint handling, early stopping, resume training)
- Coverage improved from 84% to 94% (well above 80% target)
- All 333 tests passing, 4 skipped (CUDA tests when CUDA not available)

**Dependencies:** All previous tickets

**Estimated Time:** 10-12 hours
**Actual Time:** ~6 hours

---

### PYT-5.2: Write Integration Tests
**Priority:** HIGH | **Effort:** Medium | **Status:** ✅ Completed

**Description:**
Write integration tests for data pipeline, model pipeline, and training pipeline.

**Files Created:**
- `tests/test_integration.py`

**Acceptance Criteria:**
- [x] Data pipeline integration test passes
- [x] Model pipeline integration test passes
- [x] Training pipeline integration test passes
- [x] End-to-end test passes

**Dependencies:** PYT-5.1

**Implementation Notes:**
- Created comprehensive integration test suite with 13 tests total
- Data Pipeline Integration (4 tests): Tests complete pipeline from BIOM loading → rarefaction → UniFrac computation → tokenization → dataset creation → DataLoader
- Model Pipeline Integration (3 tests): Tests model forward pass, output structure, and loss computation with all components
- Training Pipeline Integration (3 tests): Tests training step, validation step, and complete training loop
- End-to-End Tests (3 tests, marked `@pytest.mark.slow`): Tests full training workflow with real data files
- End-to-end tests use unweighted UniFrac with batch-level pairwise distances extracted via `extract_batch_distances()`
- Model configured with `base_output_dim = batch_size` for pairwise distance prediction
- All tests use correct method signatures and follow existing test patterns
- Tests handle missing data files gracefully with pytest.skip()
- 10 non-slow tests passing, 3 slow end-to-end tests passing (all 13 tests pass)

**Estimated Time:** 6-8 hours
**Actual Time:** ~4 hours

---

## Phase 6: Staged Training

### PYT-6.1: Implement SequenceEncoder Pre-training
**Priority:** MEDIUM | **Effort:** Medium | **Status:** ✅ Completed

**Description:**
Add support for pre-training SequenceEncoder separately (Stage 1 of training strategy).

**Files Created/Modified:**
- `aam/cli.py` (added `pretrain` command)
- `tests/test_cli.py` (added tests for pretrain command)

**Acceptance Criteria:**
- [x] Can train SequenceEncoder standalone
- [x] Saves checkpoint after pre-training
- [x] CLI supports pre-training mode
- [x] Tests pass

**Implementation Notes:**
- Added `pretrain` CLI command that trains SequenceEncoder on UniFrac + nucleotide prediction (self-supervised)
- No metadata/target labels required - only needs BIOM table and phylogenetic tree
- Creates SequenceEncoder model (not SequencePredictor)
- Loss function only uses base_loss (UniFrac) and nuc_loss (nucleotide prediction)
- Sets `base_output_dim` based on UniFrac metric: `batch_size` for unweighted UniFrac (pairwise distances), `1` for Faith PD
- Saves checkpoint as `pretrained_encoder.pt` in output directory
- Supports all standard training options: epochs, batch_size, learning rate, early stopping, etc.
- Tests include help command, missing args validation, file validation, batch size validation, and integration test
- Exception handler safely handles errors before logger initialization

**Dependencies:** PYT-4.2

**Estimated Time:** 4-6 hours
**Actual Time:** ~3 hours

---

### PYT-6.2: Implement SequencePredictor Fine-tuning
**Priority:** MEDIUM | **Effort:** Low | **Status:** ✅ Completed

**Description:**
Add support for loading pre-trained SequenceEncoder and fine-tuning SequencePredictor (Stage 2 of training strategy).

**Files Modified:**
- `aam/training/trainer.py` (has `load_pretrained_encoder` function)
- `aam/cli.py` (has `--freeze-base` option and `--pretrained-encoder` option)
- `tests/test_cli.py` (added tests for pretrained encoder loading)

**Acceptance Criteria:**
- [x] Supports `freeze_base=True` option (implemented in CLI train command)
- [x] Supports `freeze_base=False` (fine-tune jointly) (implemented in CLI train command)
- [x] Can load pre-trained SequenceEncoder checkpoint via CLI (implemented with `--pretrained-encoder` option)
- [x] Tests pass for loading pretrained encoder (5 tests, all passing)

**Implementation Notes:**
- `freeze_base` parameter is fully implemented and working in `train` command
- `load_pretrained_encoder()` function exists in `trainer.py` and is now called from CLI
- Added `--pretrained-encoder` CLI option to `train` command to load pretrained SequenceEncoder checkpoint
- Option validates file exists (via click's `exists=True`)
- Loading happens after model creation, before optimizer creation
- Uses `strict=False` for flexible loading
- Includes logging for loading actions
- Added 5 comprehensive tests covering help text, file validation, loading functionality, freeze_base integration, and error handling
- All 45 CLI tests pass (including 5 new tests)

**Dependencies:** PYT-6.1

**Estimated Time:** 2-4 hours
**Actual Time:** ~2 hours

---

## Summary

**Total Estimated Time:** 80-100 hours

**Critical Path:**
1. Data Pipeline (PYT-1.1 → PYT-1.2 → PYT-1.3)
2. Core Components (PYT-2.1 → PYT-2.2)
3. Model Architecture (PYT-3.1 → PYT-3.2 → PYT-3.3 → PYT-3.4)
4. Training (PYT-4.1 → PYT-4.2 → PYT-4.3)
5. Testing (PYT-5.1 → PYT-5.2)
6. Staged Training (PYT-6.1 → PYT-6.2)

**Notes:**
- Implement incrementally, test as you go
- Each ticket should be independently testable
- Follow the plan documents for detailed requirements
- Keep implementation simple and focused

---

## Phase 7: Bug Fixes

### PYT-7.1: Fix CUDA Device Test Failures
**Priority:** HIGH | **Effort:** Low | **Status:** ✅ Completed

**Description:**
Fix CUDA device comparison failures in tests. Tests compare `device(type='cuda', index=0)` with `device(type='cuda')` which fails because PyTorch assigns explicit indices to CUDA devices.

**Files Modified:**
- `tests/test_asv_encoder.py` - `test_forward_same_device`
- `tests/test_position_embedding.py` - `test_forward_same_device`
- `tests/test_transformer.py` - `test_forward_same_device`
- `tests/test_sample_sequence_encoder.py` - `test_forward_same_device`
- `tests/test_sequence_encoder.py` - `test_forward_same_device`
- `tests/test_sequence_predictor.py` - `test_forward_same_device`
- `tests/test_losses.py` - `test_losses_on_cuda`

**Acceptance Criteria:**
- [x] All CUDA device tests compare device types instead of exact device objects
- [x] Use `result.device.type == device.type` instead of `result.device == device`
- [x] All 7 failing tests pass on Linux with CUDA GPU
- [x] Tests still pass on CPU-only systems

**Implementation Notes:**
- Changed device comparison from `result.device == device` to `result.device.type == device.type`
- This handles both `cuda` and `cuda:0` correctly by comparing device types rather than exact device objects
- For dictionary outputs (sequence_encoder, sequence_predictor), updated all device assertions
- Fix ensures compatibility with both single-GPU and multi-GPU systems

**Dependencies:** None

**Estimated Time:** 1 hour
**Actual Time:** ~15 minutes
