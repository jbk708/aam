# Ticket Overview

**Last Updated:** 2026-01-21
**Status:** ~18 outstanding tickets (~66-110 hours)

---

## Quick Links

| File | Status | Tickets |
|------|--------|---------|
| `FUSION_CLEANUP_TICKETS.md` | **NEW** | FUS-1 to FUS-3, CLN-1 to CLN-10 |
| `REGRESSOR_OPTIMIZATION_TICKETS.md` | 5 remaining | REG-5 to REG-9 |
| `PYTORCH_PORTING_TICKETS.md` | 6 remaining | PYT-12.2, PYT-18.5-18.6, PYT-19.3-19.4, PYT-MAINT-2 |
| `COSMOS_ONBOARDING_TICKETS.md` | 4 remaining | COS-9.5 to COS-9.8 |
| `CATEGORICAL_FEATURE_TICKETS.md` | **COMPLETE** | - |
| `DOCUMENTATION_TICKETS.md` | Backlog | DOC-2 to DOC-4 |
| `ARCHIVED_TICKETS.md` | Reference | All completed work |
| `WORKFLOW.md` | Reference | Branch naming, commit style |

---

## Priority Summary

### HIGH (0 tickets)

No high-priority tickets remaining.

### MEDIUM (5 tickets, ~16-21 hours)

| Ticket | Description | Effort | File |
|--------|-------------|--------|------|
| **PYT-12.2** | Batch size optimization | 4-6h | PYTORCH |
| **PYT-19.3** | Per-category loss weights | 3-4h | PYTORCH |
| **CLN-6** | Categorical docs/validation | 4-5h | FUSION_CLEANUP |
| **CLN-8** | Categorical learning rate | 2-3h | FUSION_CLEANUP |

### LOW (12 tickets, ~48-63 hours)

| Ticket | Description | Effort | File |
|--------|-------------|--------|------|
| **FUS-3** | Perceiver fusion | 6-8h | FUSION_CLEANUP |
| **REG-7** | Residual head | 2-3h | REGRESSOR |
| **REG-8** | Per-output loss | 3-4h | REGRESSOR |
| **REG-9** | Mixture of Experts | 6-8h | REGRESSOR |
| **PYT-18.5** | Lazy embeddings | 4-6h | PYTORCH |
| **PYT-18.6** | Memory-aware batching | 4-6h | PYTORCH |
| **PYT-19.4** | Hierarchical categories | 6-8h | PYTORCH |
| **PYT-MAINT-2** | TensorBoard logging | 2-4h | PYTORCH |
| **CLN-3** | Remove unused params | 1-2h | FUSION_CLEANUP |
| **CLN-4** | Extract shared utilities | 2-3h | FUSION_CLEANUP |
| **CLN-11** | Consolidate test suite | 4-6h | FUSION_CLEANUP |
| **COS-9.5-9.8** | ROCm infrastructure | 13-19h | COSMOS |

---

## Recommended Next Steps

### 1. User Experience (MEDIUM - ~7 hours)

Documentation and feature toggles:

```
CLN-6 (categorical docs) → CLN-8 (categorical learning rate)
```

---

## Recently Completed (2026-01-21)

| Ticket | Description |
|--------|-------------|
| CLN-7 | Toggle count prediction (8 tests) |
| CLN-12 | Random Forest baseline script (17 tests) |
| CLN-BUG-2 | val_predictions.tsv not written on resume (1 test) |
| CLN-BUG-1 | Z-score denormalization in TensorBoard (6 tests) |
| CLN-10 | Training output artifacts (8 tests) |
| CLN-9 | Remove FiLM conditioning |
| CLN-2 | Target normalization unification (27 tests) |
| FUS-2 | Cross-attention fusion (33 tests) |
| FUS-1 | GMU fusion (28 tests) |
| CLN-5 | DataParallel in train.py |
| CLN-3 | Remove unused params |
| REG-BUG-1 | FiLM identity initialization fix |
| REG-4 | FiLM layers (26 tests) |
| REG-3 | Conditional output scaling (22 tests) |
| REG-2 | Per-category normalization (26 tests) |
| REG-1 | MLP regression head (25 tests) |

See `ARCHIVED_TICKETS.md` for full history.

---

## Design Documents

| Document | Tickets |
|----------|---------|
| `_design_plan/17_attention_fusion.md` | FUS-1 to FUS-3, CLN-1 to CLN-6 |
| `_design_plan/16_regressor_optimization.md` | REG-1 to REG-9 |
| `_design_plan/15_categorical_features.md` | CAT-1 to CAT-7 (complete) |
