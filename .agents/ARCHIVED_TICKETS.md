# Archived Tickets - Completed Work

**Last Updated:** 2025-12-18

This file contains completed tickets for historical reference. For active work, see `PYTORCH_PORTING_TICKETS.md` and `COSMOS_ONBOARDING_TICKETS.md`.

---

## Phase 21: Transfer Learning & Fine-Tuning (All Complete)

- **PYT-21.1:** Target Loss Improvements and Normalize-Targets Default - `--target-penalty`, normalize targets default
- **PYT-21.2:** Fix Pretrained Encoder Loading and Freeze-Base Verification - Strip `_orig_mod.` prefix, detailed logging
- **PYT-21.4:** Update Training Progress Bar for Fine-Tuning - Hide NL/NA when freeze-base, show RL/CL

## Phase 20: Self-Supervised Learning (All Complete)

- **PYT-20.1:** Masked Autoencoder for Nucleotide Prediction - `--nuc-mask-ratio`, `--nuc-mask-strategy`

## Phase 19: Maintenance (All Complete)

- **PYT-19.1:** Fix Failing Unit Tests - Mock fixes, flaky test threshold

## Phase 18: Memory Optimization (Partial - 4 Complete, 2 Remaining)

- **PYT-18.1:** Enable Memory-Efficient Defaults - `asv_chunk_size=256`, `gradient_checkpointing=True`
- **PYT-18.2:** Streaming Validation Metrics - O(batch) memory for validation
- **PYT-18.3:** Skip Nucleotide Predictions During Inference - Already implemented in architecture
- **PYT-18.4:** Configurable FFN Intermediate Size - CANCELLED (memory not a practical issue)

## Phase 12: Additional Performance (Partial - 1 Complete, 2 Remaining)

- **PYT-12.3:** Caching Mechanisms - Sequence tokenization cache

---

## Cosmos Onboarding (Partial - 3 Complete)

- **COS-1.0:** Native ROCm Environment - mamba + PyTorch ROCm wheels
- **COS-2.1:** ROCm Compatibility Audit - No changes needed, code already compatible via HIP backend
- **COS-4.1:** DDP for ROCm/RCCL - `--distributed`, `--sync-batchnorm` flags, `aam/training/distributed.py`

---

## Phase 8: Feature Enhancements (All Complete)

- PYT-8.1: TensorBoard Train/Val Overlay Verification
- PYT-8.2: Single Best Model File Saving
- PYT-8.3: Change Early Stopping Default to 10 Epochs
- PYT-8.4: Validation Prediction Plots
- PYT-8.5: Shuffled Batches for UniFrac Distance Extraction
- PYT-8.6: Fix Base Loss Shape Mismatch for Variable Batch Sizes
- PYT-8.7: Fix Model NaN Issue and Add Gradient Clipping
- PYT-8.8: Add Start Token to Prevent All-Padding Sequence NaN
- PYT-8.9: Fix NaN in Nucleotide Predictions During Pretraining
- PYT-8.10: Update Training Progress Bar, Rename base_loss to unifrac_loss
- PYT-8.11: Explore Learning Rate Optimizers and Schedulers

## Phase 9: UniFrac Underfitting Fixes (All Complete)

- PYT-8.12: Mask Diagonal in UniFrac Loss Computation
- PYT-8.13: Investigate Zero-Distance Samples (found to be extremely rare)
- PYT-8.14: Bounded Regression Loss for UniFrac Distances
- PYT-8.15: CANCELLED - Weighted Loss for Zero-Distance Pairs (not needed)
- PYT-8.16a: Investigate Boundary Prediction Clustering
- PYT-8.16b: Refactor UniFrac Distance Prediction to Match TensorFlow

## Phase 10: Performance Optimizations (All Complete except PYT-10.6)

- PYT-10.1: Mixed Precision Training (FP16/BF16) - `--mixed-precision` flag
- PYT-10.2: Model Compilation (`torch.compile()`) - `--compile-model` flag
- PYT-10.2.1: Fix Dependencies for Model Compilation - PyTorch >= 2.3.0
- PYT-10.3: Optimize Data Loading - num_workers=4, prefetch_factor=2
- PYT-10.3.1: Optimize Tree Loading with Pre-pruning - DEPRECATED with PYT-11.4
- PYT-10.4: Gradient Checkpointing - `--gradient-checkpointing` flag
- PYT-10.5: Optimize Attention Computation - `--attn-implementation` flag

## Phase 11: Critical Fixes (All Complete)

- PYT-11.1: Fix UniFrac Distance Predictions Exceeding 1.0 - sigmoid normalization
- PYT-11.2: CANCELLED - Reference Embedding for Stripe Mode (replaced by PYT-11.4)
- PYT-11.4: Refactor to Ingest Pre-Generated UniFrac Matrices - `UniFracLoader` class
- PYT-11.5: Fix Sigmoid Saturation - tanh normalization with fixed scale
- PYT-11.6: Optimize Learning Rate Scheduling - CosineAnnealingWarmRestarts, LR finder
- PYT-11.7: Fix Metadata Loading - column name whitespace/BOM handling
- PYT-11.8: Fix Regressor Output Bounds - sigmoid activation for target_head
- PYT-11.9: Target Normalization - `--normalize-targets`, `--loss-type` flags

---

## Key Implementation Patterns

### CLI Flags Added
```bash
# Performance
--mixed-precision fp16|bf16    # Mixed precision training
--compile-model                # torch.compile() optimization
--gradient-checkpointing       # Memory reduction
--attn-implementation sdpa     # Attention backend

# Training
--normalize-targets            # Normalize targets to [0,1] (default: True)
--loss-type mse|mae|huber      # Loss function selection
--target-penalty               # Weight for target loss
--nuc-mask-ratio               # MAE masking ratio (default: 0.15)

# Distributed
--distributed                  # Enable DDP
--sync-batchnorm              # SyncBatchNorm for multi-GPU

# Data
--unifrac-matrix <path>        # Pre-computed UniFrac matrix
```

### Architecture Changes
- UniFrac: Compute pairwise distances from embeddings (not direct prediction)
- Target prediction: Sigmoid-bounded output with normalization
- Attention: SDPA with configurable backends
- Nucleotide: Masked autoencoder (MAE) for continuous gradient signal
- Distributed: Full DDP support via `aam/training/distributed.py`

### Deprecated (with PYT-11.4)
- `--lazy-unifrac`, `--stripe-mode`, `--unifrac-threads`, `--prune-tree`
- `UniFracComputer.compute_*()` methods
- `aam/data/tree_pruner.py`
