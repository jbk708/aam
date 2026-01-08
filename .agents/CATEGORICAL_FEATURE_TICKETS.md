# Categorical Feature Integration Tickets

**Last Updated:** 2026-01-07
**Status:** CAT-1 through CAT-5 complete (see `ARCHIVED_TICKETS.md`), 2 remaining

---

## Remaining Tickets

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

**Dependencies:** CAT-4, CAT-5 (complete)

---

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

**Files:** `README.md`, `tests/test_categorical.py`, `tests/test_integration.py`

**Dependencies:** CAT-1 through CAT-6

---

## Summary

| Ticket | Description | Est. Hours | Status |
|--------|-------------|------------|--------|
| CAT-1 | Schema definition | 2-3 | Complete |
| CAT-2 | Dataset encoding | 3-4 | Complete |
| CAT-3 | Embedder module | 3-4 | Complete |
| CAT-4 | Model integration | 4-6 | Complete |
| CAT-5 | CLI updates | 3-4 | Complete |
| CAT-6 | Checkpoint compatibility | 3-4 | Not Started |
| CAT-7 | Documentation & testing | 3-4 | Not Started |
| **Remaining** | | **6-8** | |

## Recommended Order

1. **CAT-6** - Checkpoint compatibility (enables staged training with categoricals)
2. **CAT-7** - Documentation and testing

---

## Future Work (Out of Scope)

- **Hierarchical embeddings:** Nested categories sharing embedding subspaces
- **Cross-dataset transfer:** Mapping categorical encoders between datasets with overlapping categories
- **Learned category interactions:** Attention between location and season embeddings before fusion
