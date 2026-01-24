# Archived Tickets - Completed Work

**Last Updated:** 2026-01-23

Completed tickets for historical reference. For active work, see `TICKET_OVERVIEW.md`.

---

## Recent Completions (2026-01-23)

### PyTorch Porting
| Ticket | Description | Tests |
|--------|-------------|-------|
| **PYT-12.2** | Auto batch size optimization (`--auto-batch-size`, `--max-memory-fraction`) | 12 |
| **PYT-18.6** | Memory-aware batching (superseded by PYT-12.2) | - |
| **PYT-19.3** | Per-category loss weighting (`--categorical-loss-weights`) | 28 |

### Attention Fusion
| Ticket | Description | Tests |
|--------|-------------|-------|
| **FUS-1** | GMU baseline (`--categorical-fusion gmu`) | 28 |
| **FUS-2** | Cross-attention fusion (`--categorical-fusion cross-attention`) | 33 |

### Regressor Optimization
| Ticket | Description | Tests |
|--------|-------------|-------|
| **REG-5** | Quantile regression (`--loss-type quantile`) | 14+ |
| **REG-6** | Asymmetric loss (`--loss-type asymmetric`) | 14 |

### Code Cleanup
| Ticket | Description | Tests |
|--------|-------------|-------|
| **CLN-2** | Target normalization unification (`--target-transform`) | 27 |
| **CLN-5** | DataParallel in train.py | 5 |
| **CLN-6** | Categorical docs/validation (`--categorical-help`) | 3 |
| **CLN-7** | Toggle count prediction (`--no-count-prediction`) | 8 |
| **CLN-8** | Categorical learning rate (`--categorical-lr`) | 5 |
| **CLN-9** | Remove FiLM conditioning | - |
| **CLN-10** | Training output artifacts (train/val samples, predictions) | 8 |
| **CLN-12** | Random Forest baseline script | 17 |
| **CLN-13** | ASV sampling strategy (`--asv-sampling`) | 4 |
| **CLN-14** | Multi-pass prediction aggregation | 5+ |
| **CLN-15** | Multi-pass validation during training (`--val-prediction-passes`) | 7 |

### Bug Fixes
| Ticket | Description |
|--------|-------------|
| **CLN-BUG-1** | Z-score denormalization in TensorBoard (6 tests) |
| **CLN-BUG-2** | val_predictions.tsv not written on resume (1 test) |
| **CLN-BUG-3** | --resume-from ignores new learning rate (4 tests) |
| **CLN-BUG-4** | LR override undone by double load_checkpoint (1 test) |
| **CLN-BUG-5** | zscore-cat TensorBoard denormalization (2 tests) |
| **CLN-BUG-7** | Checkpoints not saved to new dir on resume (1 test) |
| **CLN-BUG-8** | Multi-pass validation fails in distributed training (1 test) |

---

## Regressor Optimization (REG-1 to REG-4, REG-BUG-1) - 2026-01-14

All high-priority categorical compensation tickets complete.

| Ticket | Description | Tests |
|--------|-------------|-------|
| **REG-1** | MLP regression head (`--regressor-hidden-dims`) | 25 |
| **REG-2** | Per-category target normalization (`--normalize-targets-by`) | 26 |
| **REG-3** | Conditional output scaling (`--conditional-output-scaling`) | 22 |
| **REG-4** | FiLM layers (`--film-conditioning`) | 26 |
| **REG-BUG-1** | FiLM identity init fix | 1 regression test |

**Files:** `aam/models/sequence_predictor.py`, `aam/models/film.py`, `aam/data/normalization.py`, `aam/cli/train.py`

---

## FSDP Implementation (PYT-12.1a/b/c) - 2026-01-13

| Ticket | Description | Tests |
|--------|-------------|-------|
| **PYT-12.1a** | FSDP infrastructure, `--fsdp` flag | 17 |
| **PYT-12.1b** | FSDP checkpoint save/load | 17 |
| **PYT-12.1c** | FSDP pretraining with embedding gathering | 16 |

**Files:** `aam/training/distributed.py`, `aam/training/losses.py`, `aam/cli/train.py`, `aam/cli/pretrain.py`

---

## Bug Fixes (PYT-BUG-1 to PYT-BUG-4) - 2026-01-14

| Ticket | Description |
|--------|-------------|
| **PYT-BUG-1** | Distributed validation metrics synchronized via all_reduce |
| **PYT-BUG-2** | Best model selection uses `--best-metric` (r2, mae, val_loss, etc.) |
| **PYT-BUG-3** | Count loss configurable via `--count-penalty` |
| **PYT-BUG-4** | Distributed validation plots show all GPUs via gather |

---

## Categorical Features (CAT-1 to CAT-7) - 2026-01-13

All categorical feature integration complete.

| Ticket | Description |
|--------|-------------|
| **CAT-1** | Schema definition (`CategoricalColumnConfig`, `CategoricalSchema`) |
| **CAT-2** | Dataset encoding (`CategoricalEncoder` with fit/transform) |
| **CAT-3** | Embedder module (`CategoricalEmbedder`) |
| **CAT-4** | SequencePredictor integration (concat/add fusion) |
| **CAT-5** | CLI flags (`--categorical-columns`, `--categorical-embed-dim`, `--categorical-fusion`) |
| **CAT-6** | Checkpoint compatibility, transfer learning |
| **CAT-7** | Documentation and integration tests |

**Files:** `aam/data/categorical.py`, `aam/models/categorical_embedder.py`, `aam/models/sequence_predictor.py`

---

## ROCm/Cosmos (COS-9.1 to COS-9.3, COS-9.9) - 2026-01-12

| Ticket | Description |
|--------|-------------|
| **COS-9.1** | ROCm attention investigation (mem_efficient broken with masks on ROCm 6.2) |
| **COS-9.2** | torch.compile() skip on ROCm with warning |
| **COS-9.3** | Memory profiling (`--memory-profile` flag) |
| **COS-9.9** | PyTorch 2.7 + ROCm 6.3 fixes SDPA (aotriton 0.8.2) |

**Resolution:** ROCm 6.3 + PyTorch 2.7+ works correctly. ROCm 6.2 requires `--attn-implementation math`.

---

## Documentation (DOC-1) - 2026-01-13

- README modernization (pip-only install, Quick Start, test data)
- Python 3.9-3.12 requirements documented
- Test count updated to 919

---

## Performance (PYT-10.6, PYT-10.7) - 2026-01-08

| Ticket | Description |
|--------|-------------|
| **PYT-10.6** | DDP validation (not suitable for pretraining due to local pairwise issue) |
| **PYT-10.7** | DataParallel for pretraining (`--data-parallel` flag in pretrain.py) |

---

## Output Constraints (PYT-19.1) - 2026-01-13

- `--output-activation` flag: none, relu, softplus, exp
- Validates mutual exclusion with `--bounded-targets`

---

## Earlier Work (2025)

### Phase 21: Transfer Learning
- PYT-21.1: Target loss improvements
- PYT-21.2: Pretrained encoder loading fix
- PYT-21.3: Regressor head optimization
- PYT-21.4: Training progress bar updates

### Phase CLN: Code Cleanup
- CLN-1 to CLN-6: Removed deprecated UniFrac modules, dead code, added exports, fixed types, extracted CLI/evaluation modules (~2000 lines removed)

### Phases 8-11: Core Features
- TensorBoard overlays, early stopping, prediction plots
- Mixed precision, torch.compile, gradient checkpointing
- UniFrac loss fixes (diagonal mask, bounded regression)
- Sigmoid saturation fix, LR scheduling
- Masked autoencoder for nucleotides

---

## Key Architecture Patterns

**CLI Flags Added (cumulative):**
```bash
# Performance
--mixed-precision fp16|bf16
--compile-model
--gradient-checkpointing
--attn-implementation sdpa|math|flash

# Training
--normalize-targets / --normalize-targets-by
--loss-type mse|mae|huber
--target-penalty / --nuc-penalty / --count-penalty
--best-metric r2|mae|val_loss|accuracy|f1

# Distributed
--distributed / --data-parallel / --fsdp
--fsdp-sharded-checkpoint

# Categorical
--categorical-columns / --categorical-embed-dim / --categorical-fusion
--conditional-output-scaling / --film-conditioning

# Regressor
--regressor-hidden-dims / --regressor-dropout
--output-activation / --bounded-targets
```

**Architecture:**
- UniFrac: Pairwise distances from embeddings (not direct prediction)
- Categorical: Concat/add fusion + FiLM + conditional scaling
- Distributed: DDP, DataParallel (pretrain), FSDP with embedding gathering
