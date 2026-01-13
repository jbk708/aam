# Categorical Feature Integration Tickets

**Last Updated:** 2026-01-12
**Status:** CAT-1 through CAT-6 complete (see `ARCHIVED_TICKETS.md`), 1 remaining

---

## Remaining Tickets

### CAT-7: Documentation and Testing
**Priority:** HIGH | **Effort:** 3-4 hours | **Status:** Not Started

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
| CAT-6 | Checkpoint compatibility | 3-4 | Complete |
| CAT-7 | Documentation & testing | 3-4 | Not Started |
| **Remaining** | | **3-4** | |

## Recommended Order

1. **CAT-7** - Documentation and testing

---

## Future Work (Out of Scope)

- **Hierarchical embeddings:** Nested categories sharing embedding subspaces
- **Cross-dataset transfer:** Mapping categorical encoders between datasets with overlapping categories
- **Learned category interactions:** Attention between location and season embeddings before fusion
