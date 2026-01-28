# Ticket Overview

**Last Updated:** 2026-01-28
**Status:** 18 outstanding tickets (~26-32 hours) | **4 HIGH priority**

---

## Quick Links

| File | Status | Tickets | Dev Branch |
|------|--------|---------|------------|
| `TRAIN_BUGFIX_TICKETS.md` | 14 remaining | **TRN-2 to TRN-5 (HIGH)**, TRN-6 to TRN-15 | `dev/train-bugfix` |
| `REGRESSOR_OPTIMIZATION_TICKETS.md` | 1 remaining | REG-9 | `main` |
| `PYTORCH_PORTING_TICKETS.md` | 1 remaining | PYT-19.4 | `main` |
| `COSMOS_ONBOARDING_TICKETS.md` | 2 remaining | COS-9.6, COS-9.7 | `main` |
| `ARCHIVED_TICKETS.md` | Reference | Completed work history | - |
| `WORKFLOW.md` | Reference | Branch naming, commit style, testing | - |

---

## Priority Summary

### HIGH (4 tickets, ~2 hours)

| Ticket | Description | Effort | File |
|--------|-------------|--------|------|
| **TRN-2** | Empty dataset validation after filtering | 0.5h | TRAIN_BUGFIX |
| **TRN-3** | Target column type validation | 0.5h | TRAIN_BUGFIX |
| **TRN-4** | Checkpoint resume field validation | 0.5h | TRAIN_BUGFIX |
| **TRN-5** | Fix drop_last=True for validation DataLoader | 0.5h | TRAIN_BUGFIX |

### MEDIUM (8 tickets, ~3.25 hours)

| Ticket | Description | Effort | File |
|--------|-------------|--------|------|
| **TRN-6** | Fix distributed cleanup race condition | 0.5h | TRAIN_BUGFIX |
| **TRN-7** | Validate quantiles sorted and unique | 0.25h | TRAIN_BUGFIX |
| **TRN-8** | Strip whitespace from metadata_column | 0.25h | TRAIN_BUGFIX |
| **TRN-9** | Validate sample weights shape/positivity | 0.5h | TRAIN_BUGFIX |
| **TRN-10** | Add finally block to auto batch size finder | 0.5h | TRAIN_BUGFIX |
| **TRN-11** | Validate pretrained encoder weight loading | 0.5h | TRAIN_BUGFIX |
| **TRN-12** | Validate distributed broadcast success | 0.5h | TRAIN_BUGFIX |
| **TRN-13** | Validate categorical encoder handles empty data | 0.25h | TRAIN_BUGFIX |

### LOW (6 tickets, ~20-24 hours)

| Ticket | Description | Effort | File |
|--------|-------------|--------|------|
| **TRN-14** | Suppress duplicate logging in distributed | 0.25h | TRAIN_BUGFIX |
| **TRN-15** | Fix best_metric_value default in checkpoint | 0.25h | TRAIN_BUGFIX |
| **REG-9** | Mixture of Experts | 6-8h | REGRESSOR |
| **PYT-19.4** | Hierarchical categories | 6-8h | PYTORCH |
| **COS-9.6** | SLURM templates | 3-4h | COSMOS |
| **COS-9.7** | Singularity container | 4-6h | COSMOS |

---

## Documentation

| Document | Description |
|----------|-------------|
| `docs/getting-started.md` | Installation + quickstart |
| `docs/user-guide.md` | Full CLI reference |
| `docs/how-it-works.md` | Concepts + implementation |
| `ARCHITECTURE.md` | Design decisions + rationale |
