# Ticket Overview

**Last Updated:** 2026-01-27
**Status:** ~5 outstanding tickets (~20-41 hours)

---

## Quick Links

| File | Status | Tickets |
|------|--------|---------|
| `FUSION_CLEANUP_TICKETS.md` | 2 remaining | FUS-3, CLN-17 |
| `REGRESSOR_OPTIMIZATION_TICKETS.md` | 3 remaining | REG-7 to REG-9 |
| `PYTORCH_PORTING_TICKETS.md` | 2 remaining | PYT-19.4, PYT-MAINT-2 |
| `COSMOS_ONBOARDING_TICKETS.md` | 4 remaining | COS-9.5 to COS-9.8 |
| `DOCUMENTATION_TICKETS.md` | Backlog | DOC-2 to DOC-4 |
| `ARCHIVED_TICKETS.md` | Reference | All completed work |
| `WORKFLOW.md` | Reference | Branch naming, commit style |

---

## Priority Summary

### HIGH (0 tickets)

None remaining.

### MEDIUM (0 tickets)

None remaining.

### LOW (5 tickets, ~18-32 hours)

| Ticket | Description | Effort | File |
|--------|-------------|--------|------|
| ~~CLN-17~~ | ~~Reduce total test count~~ | ~~4-6h~~ | ~~FUSION_CLEANUP~~ |
| **FUS-3** | Perceiver fusion | 6-8h | FUSION_CLEANUP |
| ~~REG-7~~ | ~~Residual head~~ | ~~2-3h~~ | ~~REGRESSOR~~ |
| **REG-8** | Per-output loss | 3-4h | REGRESSOR |
| **REG-9** | Mixture of Experts | 6-8h | REGRESSOR |
| **PYT-19.4** | Hierarchical categories | 6-8h | PYTORCH |
| **PYT-MAINT-2** | TensorBoard logging | 2-4h | PYTORCH |

### Backlog (~13-19 hours)

| Ticket | Description | Effort | File |
|--------|-------------|--------|------|
| **COS-9.5-9.8** | ROCm infrastructure | 13-19h | COSMOS |
| **DOC-2 to DOC-4** | Documentation backlog | TBD | DOCUMENTATION |

---

## Recommended Next Steps

### 1. Regressor Improvements (LOW - ~3-4 hours)

```
REG-8 (Per-output loss) - Different loss per target column for multi-output regression
```

---

## Recently Completed (2026-01-27)

| Ticket | Description |
|--------|-------------|
| REG-7 | Residual regression head (15 tests) |
| CLN-17 | Consolidate test suite (22 tests removed, 521 lines reduced) |
| CLN-11.3 | Extract shared test utilities to conftest.py (75 lines removed) |
| CLN-11.2 | Parametrize batch/sequence variation tests (improved test output) |
| CLN-11.1 | Consolidate duplicate fixtures to conftest.py (122 lines removed) |
| CLN-16 | Consolidate lazy embedding tests with parametrize |
| PYT-18.5 | Lazy sample embedding computation (17 tests) |

| Ticket | Description |
|--------|-------------|
| CLN-BUG-6 | Model converging to mean fix |
| CLN-4 | Extract shared training utilities (~110 lines) |
| CLN-3 | Remove unused parameters (~550 lines) |
| PYT-19.3 | Per-category loss weights (28 tests) |
| PYT-12.2 | Auto batch size optimization (12 tests) |
| PYT-18.6 | Memory-aware batching (superseded by PYT-12.2) |
| CLN-BUG-8 | Multi-pass validation distributed fix (1 test) |
| CLN-8 | Categorical learning rate (5 tests) |
| CLN-15 | Multi-pass validation during training (7 tests) |
| CLN-BUG-7 | Checkpoints not saved to new dir on resume (1 test) |
| CLN-BUG-5 | zscore-cat TensorBoard denormalization (2 tests) |
| CLN-6 | Categorical docs/validation - --categorical-help (3 tests) |
| CLN-13 | ASV sampling strategy - abundance/random (4 tests) |
| CLN-BUG-4 | LR override undone by double load_checkpoint (1 test) |
| CLN-BUG-3 | --resume-from ignores new learning rate (4 tests) |
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
| REG-6 | Asymmetric loss (14 tests) |
| REG-5 | Quantile regression |

See `ARCHIVED_TICKETS.md` for full history.

---

## Design Documents

| Document | Tickets |
|----------|---------|
| `_design_plan/17_attention_fusion.md` | FUS-1 to FUS-3, CLN-1 to CLN-6 |
| `_design_plan/16_regressor_optimization.md` | REG-1 to REG-9 |
| `_design_plan/15_categorical_features.md` | CAT-1 to CAT-7 (complete) |
