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
**Priority:** HIGH | **Effort:** Medium | **Status:** Not Started

**Description:**
Implement encoder with UniFrac prediction head as specified in `pytorch_porting_plan/08_sequence_encoder.md`.

**Files to Create:**
- `aam/models/sequence_encoder.py`

**Acceptance Criteria:**
- [ ] `SequenceEncoder` class implemented
- [ ] Composes SampleSequenceEncoder
- [ ] Predicts UniFrac distances
- [ ] Returns sample embeddings and predictions
- [ ] Supports different encoder types (unifrac, taxonomy, faith_pd, combined)
- [ ] Unit tests pass

**Dependencies:** PYT-3.2

**Estimated Time:** 6-8 hours

---

### PYT-3.4: Implement SequenceRegressor
**Priority:** HIGH | **Effort:** Medium | **Status:** Not Started

**Description:**
Implement main regression model as specified in `pytorch_porting_plan/09_sequence_regressor.md`.

**Files to Create:**
- `aam/models/sequence_regressor.py`

**Acceptance Criteria:**
- [ ] `SequenceRegressor` class implemented
- [ ] Composes SequenceEncoder as base model
- [ ] Supports `freeze_base` parameter
- [ ] Predicts target and counts
- [ ] Returns dictionary of predictions
- [ ] Unit tests pass

**Dependencies:** PYT-3.3

**Estimated Time:** 6-8 hours

---

## Phase 4: Training

### PYT-4.1: Implement Loss Functions
**Priority:** HIGH | **Effort:** Medium | **Status:** Not Started

**Description:**
Implement multi-task loss functions as specified in `pytorch_porting_plan/10_training_losses.md`.

**Files to Create:**
- `aam/training/losses.py`
- `aam/training/metrics.py`

**Acceptance Criteria:**
- [ ] Target loss (MSE/NLL) implemented
- [ ] Count loss (masked MSE) implemented
- [ ] Base loss (UniFrac MSE) implemented
- [ ] Nucleotide loss (masked CrossEntropy) implemented
- [ ] Total loss computation implemented
- [ ] Metrics (MAE, accuracy, etc.) implemented
- [ ] Unit tests pass

**Dependencies:** PYT-3.4

**Estimated Time:** 4-6 hours

---

### PYT-4.2: Implement Training Loop
**Priority:** HIGH | **Effort:** High | **Status:** Not Started

**Description:**
Implement training and validation loops with staged training support as specified in `pytorch_porting_plan/11_training_loop.md`.

**Files to Create:**
- `aam/training/trainer.py`

**Acceptance Criteria:**
- [ ] Training epoch function implemented
- [ ] Validation epoch function implemented
- [ ] Main training function implemented
- [ ] Supports loading pre-trained SequenceEncoder
- [ ] Supports `freeze_base` parameter
- [ ] Early stopping implemented
- [ ] Checkpoint saving/loading implemented
- [ ] Unit tests pass

**Dependencies:** PYT-4.1

**Estimated Time:** 8-10 hours

---

### PYT-4.3: Implement CLI Interface
**Priority:** MEDIUM | **Effort:** Medium | **Status:** Not Started

**Description:**
Implement command-line interface as specified in `pytorch_porting_plan/12_cli_interface.md`.

**Files to Create:**
- `aam/cli.py` or `aam/__main__.py`

**Acceptance Criteria:**
- [ ] Training command implemented
- [ ] Inference command implemented (optional)
- [ ] Argument validation implemented
- [ ] Error handling implemented
- [ ] Logging implemented
- [ ] Integration tests pass

**Dependencies:** PYT-4.2

**Estimated Time:** 6-8 hours

---

## Phase 5: Testing

### PYT-5.1: Write Unit Tests
**Priority:** HIGH | **Effort:** High | **Status:** Not Started

**Description:**
Write comprehensive unit tests as specified in `pytorch_porting_plan/13_testing.md`.

**Files to Create:**
- `tests/test_biom_loader.py`
- `tests/test_unifrac_computer.py`
- `tests/test_tokenizer.py`
- `tests/test_dataset.py`
- `tests/test_models.py`
- `tests/test_losses.py`
- `tests/test_trainer.py`

**Acceptance Criteria:**
- [ ] All components have unit tests
- [ ] Tests cover edge cases
- [ ] All tests pass
- [ ] Test coverage > 80%

**Dependencies:** All previous tickets

**Estimated Time:** 10-12 hours

---

### PYT-5.2: Write Integration Tests
**Priority:** HIGH | **Effort:** Medium | **Status:** Not Started

**Description:**
Write integration tests for data pipeline, model pipeline, and training pipeline.

**Files to Create:**
- `tests/test_integration.py`

**Acceptance Criteria:**
- [ ] Data pipeline integration test passes
- [ ] Model pipeline integration test passes
- [ ] Training pipeline integration test passes
- [ ] End-to-end test passes

**Dependencies:** PYT-5.1

**Estimated Time:** 6-8 hours

---

## Phase 6: Staged Training

### PYT-6.1: Implement SequenceEncoder Pre-training
**Priority:** MEDIUM | **Effort:** Medium | **Status:** Not Started

**Description:**
Add support for pre-training SequenceEncoder separately (Stage 1 of training strategy).

**Files to Modify:**
- `aam/training/trainer.py`
- `aam/cli.py`

**Acceptance Criteria:**
- [ ] Can train SequenceEncoder standalone
- [ ] Saves checkpoint after pre-training
- [ ] CLI supports pre-training mode
- [ ] Tests pass

**Dependencies:** PYT-4.2

**Estimated Time:** 4-6 hours

---

### PYT-6.2: Implement SequenceRegressor Fine-tuning
**Priority:** MEDIUM | **Effort:** Low | **Status:** Not Started

**Description:**
Add support for loading pre-trained SequenceEncoder and fine-tuning SequenceRegressor (Stage 2 of training strategy).

**Files to Modify:**
- `aam/training/trainer.py`
- `aam/cli.py`

**Acceptance Criteria:**
- [ ] Can load pre-trained SequenceEncoder checkpoint
- [ ] Supports `freeze_base=True` option
- [ ] Supports `freeze_base=False` (fine-tune jointly)
- [ ] Tests pass

**Dependencies:** PYT-6.1

**Estimated Time:** 2-4 hours

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
