# Code Cleanup Tickets

**Created:** 2025-12-18
**Status:** 5/6 complete (~3-4 hours remaining)

This file contains code cleanup and technical debt tickets.

---

## Phase CLN: Code Cleanup

### CLN-1: Remove Deprecated UniFrac Computation Modules
**Priority:** HIGH | **Effort:** 2-3 hours | **Status:** ✅ COMPLETE (2025-12-19)

**Problem:**
Several modules were deprecated in PYT-11.4 when we switched to pre-computed UniFrac matrices. These modules are no longer used and should be removed.

**Files to Delete (~1665 lines):**
- `aam/data/tree_pruner.py` (150 lines) - marked deprecated, only imported by unifrac.py
- `aam/data/unifrac.py` (1002 lines) - UniFracComputer deprecated
- `aam/data/unifrac_cache.py` (254 lines) - only used by deprecated modules
- `aam/scripts/compute_unifrac_parallel.py` (259 lines) - uses deprecated UniFracComputer

**Acceptance Criteria:**
- [x] Delete `aam/data/tree_pruner.py`
- [x] Delete `aam/data/unifrac.py`
- [x] Delete `aam/data/unifrac_cache.py`
- [x] Delete `aam/scripts/compute_unifrac_parallel.py`
- [x] Remove any test imports referencing deleted modules
- [x] Update documentation if referencing deprecated modules
- [x] All tests pass

**Notes:**
- Keep `aam/data/unifrac_loader.py` - this is the current, supported module
- The deprecated modules have deprecation warnings but are no longer imported anywhere in production code
- Also deleted 7 debug scripts that imported deprecated modules (~4165 lines total)

---

### CLN-2: Remove Dead Code Paths (stripe_mode, lazy_unifrac)
**Priority:** HIGH | **Effort:** 1-2 hours | **Status:** ✅ COMPLETE (2025-12-19)

**Problem:**
Legacy flags `stripe_mode` and `lazy_unifrac` are hardcoded to `False` in 16 places in cli.py. These were deprecated with PYT-11.4 but the dead code paths remain.

**Locations:**
- `aam/cli.py` - 16 occurrences of `lazy_unifrac=False`, `stripe_mode=False`
- `aam/training/losses.py` - stripe mode code paths (lines 320-365)
- `aam/data/dataset.py` - may have stripe_mode/lazy_unifrac parameters

**Acceptance Criteria:**
- [x] Remove `lazy_unifrac` parameter from ASVDataset calls in cli.py
- [x] Remove `stripe_mode` parameter from ASVDataset calls in cli.py
- [x] Remove stripe mode logic from `losses.py` (is_stripe_mode checks)
- [x] Remove parameters from ASVDataset if present
- [x] All tests pass

**Notes:**
- Removed `compute_stripe_distances` function from losses.py
- Removed `lazy_unifrac`, `stripe_mode`, `reference_sample_ids`, `all_sample_ids` parameters from collate_fn
- Removed `lazy_unifrac`, `stripe_mode`, `reference_sample_ids`, `unifrac_computer` parameters from ASVDataset
- Removed stripe mode tests from test_losses.py
- ~200 lines removed across codebase

---

### CLN-3: Add Package `__init__.py` Exports
**Priority:** MEDIUM | **Effort:** 1 hour | **Status:** ✅ COMPLETE (2025-12-19)

**Problem:**
`aam/data/__init__.py` and `aam/models/__init__.py` are empty, requiring verbose imports like `from aam.data.dataset import ASVDataset`.

**Solution:**
Add public API exports to enable cleaner imports like `from aam.data import ASVDataset`.

**Changes:**
```python
# aam/data/__init__.py
from aam.data.dataset import ASVDataset
from aam.data.biom_loader import BIOMLoader
from aam.data.unifrac_loader import UniFracLoader
from aam.data.tokenizer import SequenceTokenizer

__all__ = ["ASVDataset", "BIOMLoader", "UniFracLoader", "SequenceTokenizer"]

# aam/models/__init__.py
from aam.models.sequence_predictor import SequencePredictor
from aam.models.sequence_encoder import SequenceEncoder
from aam.models.sample_sequence_encoder import SampleSequenceEncoder
from aam.models.asv_encoder import ASVEncoder

__all__ = ["SequencePredictor", "SequenceEncoder", "SampleSequenceEncoder", "ASVEncoder"]
```

**Acceptance Criteria:**
- [x] Add exports to `aam/data/__init__.py`
- [x] Add exports to `aam/models/__init__.py`
- [x] Verify imports work: `from aam.data import ASVDataset`
- [x] All tests pass

---

### CLN-4: Fix Type Errors (ty)
**Priority:** MEDIUM | **Effort:** 1-2 hours | **Status:** ✅ COMPLETE (2025-12-19)

**Problem:**
Running `uvx ty check aam/` reports 19 type errors. Type checking is part of the CI workflow and these should be resolved.

**Solution:**
Fix type errors identified by `ty`. Most are likely:
- Missing type annotations
- Incorrect argument types (e.g., `float()` calls on tensors)
- Union type handling

**Acceptance Criteria:**
- [x] `uvx ty check aam/` passes with no errors
- [x] No behavior changes (type fixes only)

**Notes:**
- Fixed `validate_epoch` return type annotation (Dict instead of Optional[Tensor])
- Added explicit type annotations for val_results, val_predictions_dict, val_targets_dict
- Used `cast()` to narrow nn.Module attribute types (nuc_penalty, vocab_size)
- Used `getattr()` for dynamic attribute access on model objects

---

### CLN-5: Extract CLI Helper Modules
**Priority:** LOW | **Effort:** 3-4 hours | **Status:** ✅ COMPLETE (2025-12-19)

**Problem:**
`cli.py` is 1483 lines with mixed concerns (setup utilities, train command, pretrain command, predict command).

**Solution:**
Refactor into a cli package:
```
aam/cli/
├── __init__.py      # Main CLI group, imports commands
├── utils.py         # setup_logging, setup_device, setup_random_seed, validate_*
├── train.py         # train command
├── pretrain.py      # pretrain command
└── predict.py       # predict command
```

**Acceptance Criteria:**
- [x] Create `aam/cli/` package structure
- [x] Extract utility functions to `utils.py`
- [x] Extract train command to `train.py`
- [x] Extract pretrain command to `pretrain.py`
- [x] Extract predict command to `predict.py`
- [x] Update imports throughout codebase
- [x] CLI still works: `python -m aam.cli train --help`
- [x] All tests pass

**Notes:**
- Refactored 1448-line monolithic cli.py into modular package
- Updated test_cli.py patch paths to match new module structure
- Added __main__.py for python -m aam.cli support

---

### CLN-6: Extract Trainer Validation Logic
**Priority:** LOW | **Effort:** 3-4 hours | **Status:** Not Started

**Problem:**
`trainer.py` is 1896 lines. Validation and evaluation logic could be extracted.

**Solution:**
Extract to `aam/training/evaluation.py`:
- `validate_epoch()` method logic
- Streaming metrics computation
- Prediction collection and plotting

**Acceptance Criteria:**
- [ ] Create `aam/training/evaluation.py`
- [ ] Extract validation logic
- [ ] Trainer imports and uses evaluation module
- [ ] All tests pass
- [ ] No behavior changes

---

## Summary

| Ticket | Priority | Effort | Lines Affected |
|--------|----------|--------|----------------|
| CLN-1 | HIGH | 2-3h | -1665 (delete) |
| CLN-2 | HIGH | 1-2h | ~50 |
| CLN-3 | MEDIUM | 1h | ~20 |
| CLN-4 | MEDIUM | 1-2h | ~50 |
| CLN-5 | LOW | 3-4h | ~1500 (refactor) |
| CLN-6 | LOW | 3-4h | ~500 (refactor) |
| **Total** | | **12-16h** | |

## Recommended Order

1. **CLN-1** - Remove deprecated modules (biggest impact, cleanest codebase)
2. **CLN-2** - Remove dead code paths (depends on CLN-1 for unifrac.py removal)
3. **CLN-3** - Add __init__ exports (quick win)
4. **CLN-4** - Fix type errors (ty)
5. **CLN-5/6** - Refactoring (lower priority, larger effort)
