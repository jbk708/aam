# Memory Optimization Plan

**Status:** Proposed
**Priority:** HIGH
**Goal:** Reduce GPU memory usage to enable larger batch sizes and longer sequences

## Current Memory Profile

| Component | Shape | Memory/Sample | Notes |
|-----------|-------|---------------|-------|
| Input tokens | [1024, 150] | 1.2 MB | Padded to token_limit |
| Token embeddings | [1024, 150, 128] | 24.6 MB | Flattened for transformer |
| Attention scores (math) | [1024, 150, 150] | 92.2 MB | **Quadratic in seq_len** |
| Attention scores (flash) | [1024, 150] | 614 KB | Linear in seq_len |
| Nucleotide logits | [1024, 150, 6] | 3.7 MB | Only during pretraining |
| UniFrac pairwise | [B, B] | B² × 4 bytes | **Quadratic in batch** |

**Estimated peak memory for batch_size=8, token_limit=1024:**
- Standard attention: ~800 MB activations + model weights
- With Flash Attention: ~200 MB activations + model weights

## Proposed Tickets

### Phase 1: Quick Wins (Estimated: 8-12 hours total)

#### PYT-18.1: Enable Memory-Efficient Defaults
**Priority:** HIGH | **Effort:** 2-3 hours

Update default values for better out-of-box memory efficiency:

```python
# Current defaults → Proposed defaults
asv_chunk_size: None → 256
gradient_checkpointing: False → True (training only)
attn_implementation: "sdpa" → "mem_efficient" (with flash fallback)
```

**Acceptance Criteria:**
- [ ] Update CLI defaults in `cli.py`
- [ ] Update Trainer defaults
- [ ] Add `--no-gradient-checkpointing` flag to opt out
- [ ] Document memory impact in help text
- [ ] Verify no accuracy regression with new defaults

**Files:** `aam/cli.py`, `aam/training/trainer.py`

---

#### PYT-18.2: Streaming Validation Metrics
**Priority:** HIGH | **Effort:** 3-4 hours

Current validation loop accumulates all predictions in memory:
```python
all_predictions["base_prediction"].append(base_pred)  # Grows to O(dataset_size)
```

Replace with streaming metric computation:

**Acceptance Criteria:**
- [ ] Compute metrics incrementally (running mean/variance)
- [ ] Only keep small buffer for final epoch plots
- [ ] Add `--validation-plot-samples` flag to limit plot data
- [ ] Reduce validation memory from O(dataset) to O(batch)
- [ ] Maintain metric accuracy

**Files:** `aam/training/trainer.py`, `aam/training/metrics.py`

---

#### PYT-18.3: Skip Nucleotide Predictions During Inference/Fine-tuning
**Priority:** MEDIUM | **Effort:** 2-3 hours

Nucleotide prediction head creates large tensors `[batch, 1024, 150, 6]` = 3.7 MB/sample.
Only needed during pretraining.

**Acceptance Criteria:**
- [ ] Add `predict_nucleotides` flag to ASVEncoder forward pass
- [ ] Auto-disable during `model.eval()` or fine-tuning
- [ ] Reduce memory by ~3.7 MB per sample during inference
- [ ] No impact on pretraining functionality

**Files:** `aam/models/asv_encoder.py`, `aam/models/sample_sequence_encoder.py`

---

### Phase 2: Architecture Optimizations (Estimated: 12-16 hours total)

#### PYT-18.4: Configurable FFN Intermediate Size
**Priority:** MEDIUM | **Effort:** 3-4 hours

Transformer FFN expands to 4× embedding_dim by default, creating large intermediate tensors.

**Acceptance Criteria:**
- [ ] Add `ffn_ratio` parameter (default: 4, can reduce to 2)
- [ ] Propagate through ASVEncoder, SampleSequenceEncoder, SequenceEncoder
- [ ] Add `--ffn-ratio` CLI flag
- [ ] Document memory vs accuracy trade-off
- [ ] Test with ffn_ratio=2 shows reduced memory

**Files:** `aam/models/transformer.py`, `aam/models/*.py`, `aam/cli.py`

---

#### PYT-18.5: Lazy Sample Embedding Computation
**Priority:** MEDIUM | **Effort:** 4-6 hours

Currently returns sample_embeddings even when not needed for loss computation.

**Acceptance Criteria:**
- [ ] Only compute/return sample_embeddings when needed
- [ ] Add `return_intermediates` flag to forward methods
- [ ] Reduce memory by ~512 KB × num_samples during training
- [ ] No impact on functionality when intermediates needed

**Files:** `aam/models/sequence_encoder.py`, `aam/models/sequence_predictor.py`

---

#### PYT-18.6: Memory-Aware Dynamic Batching
**Priority:** LOW | **Effort:** 4-6 hours

Add memory estimation to prevent OOM during data loading.

**Acceptance Criteria:**
- [ ] Estimate memory per sample based on actual ASV count (not token_limit)
- [ ] Dynamic batch size adjustment within DataLoader
- [ ] Add `--max-memory-gb` CLI flag
- [ ] Graceful handling when batch too large
- [ ] Log actual vs estimated memory usage

**Files:** `aam/data/dataset.py`, `aam/cli.py`

---

## Summary

| Ticket | Priority | Effort | Memory Reduction |
|--------|----------|--------|------------------|
| PYT-18.1 | HIGH | 2-3h | 30-40% (via defaults) |
| PYT-18.2 | HIGH | 3-4h | O(dataset) → O(batch) validation |
| PYT-18.3 | MEDIUM | 2-3h | ~3.7 MB/sample during inference |
| PYT-18.4 | MEDIUM | 3-4h | ~25% FFN memory with ratio=2 |
| PYT-18.5 | MEDIUM | 4-6h | ~512 KB × samples |
| PYT-18.6 | LOW | 4-6h | Prevents OOM |

**Total Estimated:** 18-26 hours

## Recommended Implementation Order

1. **PYT-18.1** (Quick win, changes defaults)
2. **PYT-18.2** (High impact on validation memory)
3. **PYT-18.3** (Easy win for inference)
4. **PYT-18.4** (Good trade-off for memory-constrained setups)
5. **PYT-18.5, PYT-18.6** (As needed)

## TensorFlow Comparison

| Feature | TensorFlow | PyTorch Current | PyTorch Proposed |
|---------|------------|-----------------|------------------|
| Mixed precision | Default | Flag | Flag (unchanged) |
| Gradient checkpointing | None | Flag | Default ON |
| Attention optimization | tf.function | SDPA backends | mem_efficient default |
| max_token_per_sample | 225 | 1024 | 1024 (configurable) |
| Validation accumulation | Full | Full | Streaming |
| FFN ratio | 4× | 4× | Configurable (2-4×) |

## Notes

- Flash Attention requires NVIDIA GPU with compute capability ≥ 8.0
- Memory-efficient attention works on broader hardware
- Gradient checkpointing trades ~30% compute for ~30% memory
- BF16 preferred over FP16 when available (better numerical stability)
