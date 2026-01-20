# Regressor Optimization Tickets

**Last Updated:** 2026-01-20
**Status:** 5 tickets remaining (~17-25 hours)

**Completed:** REG-1 to REG-4, REG-BUG-1 (see `ARCHIVED_TICKETS.md`)

---

## Outstanding Tickets

### REG-5: Quantile Regression
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Not Started

Predict multiple quantiles for uncertainty estimation.

**Scope:**
- `--loss-type quantile` with pinball loss
- `--quantiles 0.1,0.5,0.9` (default)
- Output dim = original × num_quantiles

**Files:** `aam/training/losses.py`, `aam/models/sequence_predictor.py`, `aam/cli/train.py`

---

### REG-6: Asymmetric Loss
**Priority:** MEDIUM | **Effort:** 2-3 hours | **Status:** Not Started

Different penalties for over/under prediction.

**Scope:**
- `--loss-type asymmetric`
- `--over-penalty 2.0 --under-penalty 1.0`

**Files:** `aam/training/losses.py`, `aam/cli/train.py`

---

### REG-7: Residual Regression Head
**Priority:** LOW | **Effort:** 2-3 hours | **Status:** Not Started

Skip connection: `output = Linear(x) + MLP(x)`

**Scope:**
- `--residual-regression-head` flag
- Requires `--regressor-hidden-dims`

**Files:** `aam/models/sequence_predictor.py`, `aam/cli/train.py`

---

### REG-8: Per-Output Loss Config
**Priority:** LOW | **Effort:** 3-4 hours | **Status:** Not Started

Different loss per target column for multi-output regression.

**Scope:**
- `--loss-config '{"pH": "mse", "temp": "huber"}'`

**Files:** `aam/training/losses.py`, `aam/cli/train.py`

---

### REG-9: Mixture of Experts
**Priority:** LOW | **Effort:** 6-8 hours | **Status:** Not Started

Separate expert heads per category with learned routing.

**Scope:**
- `--moe-experts 4 --moe-routing location`
- Soft routing with load balancing

**Files:** `aam/models/moe.py` (new), `aam/models/sequence_predictor.py`, `aam/cli/train.py`

---

## Summary

| Ticket | Description | Effort | Priority |
|--------|-------------|--------|----------|
| **REG-5** | Quantile regression | 4-6h | MEDIUM |
| **REG-6** | Asymmetric loss | 2-3h | MEDIUM |
| **REG-7** | Residual head | 2-3h | LOW |
| **REG-8** | Per-output loss | 3-4h | LOW |
| **REG-9** | Mixture of Experts | 6-8h | LOW |
| **Total** | | **17-24h** | |
