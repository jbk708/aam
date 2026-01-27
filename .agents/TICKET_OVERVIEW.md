# Ticket Overview

**Last Updated:** 2026-01-27
**Status:** 2 outstanding tickets (~12-16 hours) + backlog

---

## Quick Links

| File | Status | Tickets |
|------|--------|---------|
| `DOCUMENTATION_TICKETS.md` | **COMPLETE** | DOC-5 to DOC-13 done |
| `FUSION_CLEANUP_TICKETS.md` | Complete | All done |
| `REGRESSOR_OPTIMIZATION_TICKETS.md` | 1 remaining | REG-9 |
| `PYTORCH_PORTING_TICKETS.md` | 1 remaining | PYT-19.4 |
| `COSMOS_ONBOARDING_TICKETS.md` | 2 remaining | COS-9.6, COS-9.7 |
| `ARCHIVED_TICKETS.md` | Reference | All completed work |
| `WORKFLOW.md` | Reference | Branch naming, commit style |

---

## Priority Summary

### HIGH (0 tickets)

Documentation overhaul complete.

### LOW (2 tickets, ~12-16 hours)

| Ticket | Description | Effort | File |
|--------|-------------|--------|------|
| **REG-9** | Mixture of Experts | 6-8h | REGRESSOR |
| **PYT-19.4** | Hierarchical categories | 6-8h | PYTORCH |

### Backlog (~19-28 hours)

| Ticket | Description | Effort | File |
|--------|-------------|--------|------|
| **COS-9.6-9.7** | ROCm infrastructure (SLURM, Singularity) | 7-10h | COSMOS |
| **DOC-3** | Tutorial Jupyter notebooks | 8-12h | DOCUMENTATION |
| **DOC-4** | Video walkthrough | 4-6h | DOCUMENTATION |

---

## Recently Completed (2026-01-27)

### Documentation Overhaul

| Ticket | Description |
|--------|-------------|
| **DOC-5** | Created docs/ folder structure, refactored README to ~130 lines |
| **DOC-6** | Wrote getting-started.md (installation, quickstart) |
| **DOC-7** | Wrote user-guide.md (full CLI reference, 400+ lines) |
| **DOC-8** | Wrote CONTRIBUTING.md (dev workflow) |
| **DOC-9** | Updated ARCHITECTURE.md with design rationale (~400 lines) |
| **DOC-10** | Wrote how-it-works.md Part 1: Concepts |
| **DOC-11** | Wrote how-it-works.md Part 2: Implementation |
| **DOC-2** | Set up Sphinx API reference generation |
| **DOC-12** | Consolidated _design_plan/ directory |
| **DOC-13** | Final review and polish |

### Previous Completions

| Ticket | Description |
|--------|-------------|
| FUS-3 | Perceiver-style latent fusion (36 tests) |
| REG-8 | Per-output loss config (19 tests) |
| COS-9.8 | ROCm documentation |
| PYT-MAINT-2 | TensorBoard logging improvements (6 tests) |
| REG-7 | Residual regression head (15 tests) |
| CLN-17 | Consolidate test suite |

See `ARCHIVED_TICKETS.md` for full history.

---

## Documentation Structure

New documentation structure created:

```
docs/
├── getting-started.md    # Installation + quickstart
├── user-guide.md         # Full CLI reference
├── how-it-works.md       # Concepts + implementation
├── api/                  # Sphinx API reference
│   ├── data.rst
│   ├── models.rst
│   ├── training.rst
│   └── cli.rst
├── conf.py               # Sphinx config
├── index.rst             # Sphinx index
└── Makefile              # Build commands

README.md                 # Overview (~130 lines)
ARCHITECTURE.md           # Design + rationale (~400 lines)
CONTRIBUTING.md           # Dev workflow
```

Build docs with: `cd docs && make html`

---

## Design Documents

| Document | Tickets |
|----------|---------|
| `_design_plan/17_attention_fusion.md` | FUS-1 to FUS-3 |
| `_design_plan/16_regressor_optimization.md` | REG-1 to REG-9 |
| `_design_plan/15_categorical_features.md` | CAT-1 to CAT-7 |
| `_design_plan/INDEX.md` | Full document listing |
