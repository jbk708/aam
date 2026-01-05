# Categorical Feature Integration Tickets

**Last Updated:** 2026-01-05
**Status:** 7 tickets (~24-36 hours)

---

## Phase 1: Schema & Encoding

### CAT-1: Categorical Metadata Schema Definition
**Priority:** HIGH | **Effort:** 2-3 hours | **Status:** Complete

Define a flexible schema for specifying categorical columns and their properties.

**Scope:**
- Design configuration structure (dataclass) declaring which metadata columns are categorical
- Fields: column name, cardinality (auto-detect or specified), embedding dimension, required/optional
- Reserve index 0 for unknown/missing categories
- Support sparse category combinations (not all locations have all seasons)

**Acceptance Criteria:**
- Schema supports arbitrary categorical columns without hardcoding
- Validation ensures categorical values match declared schema
- Clear error messages for undeclared categories at runtime

**Files:** `aam/data/categorical.py` (new)

**Dependencies:** None

---

### CAT-2: Dataset Pipeline — Categorical Encoding
**Priority:** HIGH | **Effort:** 3-4 hours | **Status:** Complete

Extend ASVDataset and collate function to encode and batch categorical metadata.

**Scope:**
- `CategoricalEncoder` class: fits on training data, transforms strings to integer indices
- Reserve index 0 for unknown/missing; indices 1–N for known categories
- Persist encoder state (category-to-index mappings) for inference
- Update `collate_fn` to include `categorical_ids: dict[str, Tensor]` in batch
- Handle missing categorical values gracefully

**Acceptance Criteria:**
- Encoder serializable/deserializable alongside model checkpoints
- Collate produces consistent tensor shapes regardless of categories present
- Unit tests for edge cases (unseen category at inference, missing values)

**Files:** `aam/data/categorical.py`, `aam/data/dataset.py`

**Dependencies:** CAT-1

---

## Phase 2: Model Components

### CAT-3: CategoricalEmbedder Module
**Priority:** HIGH | **Effort:** 3-4 hours | **Status:** Complete

Implement modular embedding layer for arbitrary categorical features.

**Scope:**
- `nn.Module` accepting dictionary of cardinalities, produces concatenated embeddings
- Support per-category embedding dimensions or shared dimension
- Dropout regularization on embeddings
- Output: `[B, total_embed_dim]` with method to broadcast to `[B, S, total_embed_dim]`
- Handle no categoricals configured (passthrough/no-op)

**Acceptance Criteria:**
- Module initializable from categorical schema (CAT-1)
- Forward handles missing categories (uses padding index 0)
- Documented input/output shapes

**Files:** `aam/models/categorical_embedder.py` (new)

**Dependencies:** CAT-1

---

### CAT-4: SequencePredictor Integration
**Priority:** HIGH | **Effort:** 4-6 hours | **Status:** Complete

Integrate categorical embeddings into SequencePredictor's target prediction pathway.

**Scope:**
- Add optional `CategoricalEmbedder` to `SequencePredictor`
- Inject after base model produces sample embeddings, before target encoder
- Two fusion strategies:
  - **Concatenate + project:** `[B, S, D + cat_dim]` → linear → `[B, S, D]`
  - **Additive conditioning:** Project cat embeddings to D, add to sample embeddings
- Backward compatible: model works identically with no categoricals
- Update forward signature: accept optional `categorical_ids` dict

**Acceptance Criteria:**
- Existing checkpoints without categoricals load without error
- Categorical embeddings don't affect base model or count encoder pathways
- Gradient flow verified through categorical embeddings

**Files:** `aam/models/sequence_predictor.py`

**Dependencies:** CAT-2, CAT-3

---

## Phase 3: CLI & Checkpointing

### CAT-5: CLI and Configuration Updates
**Priority:** MEDIUM | **Effort:** 3-4 hours | **Status:** Not Started

Expose categorical configuration through training CLI.

**Scope:**
- `--categorical-columns`: comma-separated column names from metadata
- `--categorical-embed-dim`: embedding dimension (default: 16)
- `--categorical-fusion`: fusion strategy (`concat` or `add`, default: `concat`)
- Config serialization: include categorical schema in saved checkpoints
- Prediction CLI auto-loads categorical encoder from checkpoint

**Acceptance Criteria:**
- `aam train --categorical-columns location,season` works end-to-end
- Saved checkpoint contains category mappings for reproducible inference
- Clear error if categorical column not found in metadata

**Files:** `aam/cli/train.py`, `aam/cli/predict.py`

**Dependencies:** CAT-1 through CAT-4

---

### CAT-6: Checkpoint Compatibility and Transfer Learning
**Priority:** MEDIUM | **Effort:** 3-4 hours | **Status:** Not Started

Ensure categorical features work with staged training workflow.

**Scope:**
- Pretrained `SequenceEncoder` checkpoints load into `SequencePredictor` with categoricals
- New categorical weights initialized appropriately
- `--freeze-base` with categorical fine-tuning: only categorical embedder + target encoder train
- Document weight initialization strategy

**Acceptance Criteria:**
- Stage 1 pretrain → Stage 2 with categoricals works without manual intervention
- `--freeze-base` excludes base model but includes categorical embedder in optimization

**Files:** `aam/training/trainer.py`, `aam/cli/train.py`

**Dependencies:** CAT-4, CAT-5

---

## Phase 4: Documentation & Testing

### CAT-7: Documentation and Testing
**Priority:** MEDIUM | **Effort:** 3-4 hours | **Status:** Not Started

Document usage and add integration tests.

**Scope:**
- Update README/architecture docs with categorical feature workflow
- Integration test: synthetic data with 2 categorical columns through full train loop
- Unit tests for `CategoricalEncoder` serialization round-trip
- Document best practices: embedding dim selection, handling rare categories

**Acceptance Criteria:**
- Example command in docs showing categorical training
- CI passes with new test coverage

**Files:** `README.md`, `tests/test_categorical.py` (new), `tests/test_integration.py`

**Dependencies:** CAT-1 through CAT-6

---

## Summary

| Phase | Tickets | Est. Hours | Priority |
|-------|---------|------------|----------|
| 1: Schema & Encoding | CAT-1, CAT-2 | 5-7 | HIGH |
| 2: Model Components | CAT-3, CAT-4 | 7-10 | HIGH |
| 3: CLI & Checkpointing | CAT-5, CAT-6 | 6-8 | MEDIUM |
| 4: Docs & Testing | CAT-7 | 3-4 | MEDIUM |
| **Total** | **7** | **21-29** | |

## Recommended Order

1. **CAT-1** - Schema definition (no dependencies)
2. **CAT-2** - Dataset encoding (parallel with CAT-3)
3. **CAT-3** - Embedder module (parallel with CAT-2)
4. **CAT-4** - Model integration
5. **CAT-5** - CLI updates
6. **CAT-6** - Checkpoint compatibility
7. **CAT-7** - Documentation and testing

---

## Future Work (Out of Scope)

- **Hierarchical embeddings:** Nested categories sharing embedding subspaces
- **Cross-dataset transfer:** Mapping categorical encoders between datasets with overlapping categories
- **Learned category interactions:** Attention between location and season embeddings before fusion
