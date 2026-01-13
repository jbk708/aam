# Categorical Feature Integration Tickets

**Last Updated:** 2026-01-13
**Status:** All tickets complete (see `ARCHIVED_TICKETS.md`)

---

## Completed Tickets

### CAT-7: Documentation and Testing
**Priority:** HIGH | **Effort:** 3-4 hours | **Status:** COMPLETE

Document usage and add integration tests.

**Completed:**
- Added best practices to README for embedding dim selection and rare categories
- Added TestCategoricalIntegration with 3 integration tests:
  - `test_categorical_training_loop`: full train loop with 2 categorical columns
  - `test_categorical_checkpoint_roundtrip`: encoder state in checkpoint save/load
  - `test_categorical_unknown_categories_at_inference`: unknown cats map to index 0
- CategoricalEncoder serialization tests already existed in test_categorical.py

**Acceptance Criteria:**
- [x] Example command in docs showing categorical training
- [x] CI passes with new test coverage

**Files Modified:** `README.md`, `tests/test_integration.py`

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
| CAT-6 | Checkpoint compatibility | 3-4 | Complete |
| CAT-7 | Documentation & testing | 3-4 | Complete |
| **Remaining** | | **0** | |

## All categorical feature work is complete

---

## Future Work (Out of Scope)

- **Hierarchical embeddings:** Nested categories sharing embedding subspaces
- **Cross-dataset transfer:** Mapping categorical encoders between datasets with overlapping categories
- **Learned category interactions:** Attention between location and season embeddings before fusion
