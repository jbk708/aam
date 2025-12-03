# Memory Optimization Guide

## Current Memory Bottlenecks

With `batch_size=2`, `token_limit=1024`, `max_bp=150`, and `embedding_dim=128`, the main memory consumers are:

1. **ASV-level attention matrices**: ~0.74 GB per transformer layer
   - Shape: `[batch_size * token_limit, max_bp, max_bp]` per head
   - With 4 heads: `[2048, 150, 150] * 4 = ~0.74 GB` per layer

2. **Sample-level attention matrices**: ~0.03 GB per transformer layer
   - Shape: `[batch_size, token_limit, token_limit]` per head
   - With 4 heads: `[2, 1024, 1024] * 4 = ~0.03 GB` per layer

3. **Embeddings**: ~0.16 GB
   - Shape: `[batch_size * token_limit, max_bp, embedding_dim]`
   - `[2048, 150, 128] * 4 bytes = ~0.16 GB`

4. **Multiple transformer layers**: ASV encoder (4 layers) + Sample encoder (4 layers) + Encoder transformer (4 layers) = 12 layers total

## Optimization Strategies

### 1. Chunked ASV Processing (Implemented)

**Option**: `--asv-chunk-size N`

Process ASVs in chunks of size N instead of all at once. This reduces peak memory usage by processing fewer ASVs simultaneously.

**Memory savings**: Reduces attention matrix size from `[batch_size * token_limit, max_bp, max_bp]` to `[batch_size * chunk_size, max_bp, max_bp]`

**Example**:
```bash
python -m aam.cli pretrain \
  --asv-chunk-size 256 \
  --batch-size 2 \
  --gradient-accumulation-steps 16
```

**Recommendation**: Start with `--asv-chunk-size 256` or `128` to reduce memory by 4x or 8x.

### 2. Reduce Token Limit

**Option**: `--token-limit N` (default: 1024)

Reduce the maximum number of ASVs per sample. This has quadratic impact on sample-level attention.

**Memory savings**: 
- Sample attention: `O(token_limit^2)` → reduces quadratically
- ASV attention: `O(token_limit)` → reduces linearly

**Example**:
```bash
python -m aam.cli pretrain \
  --token-limit 512 \
  --batch-size 2
```

**Recommendation**: Use `--token-limit 512` or `256` if your samples don't need all ASVs.

### 3. Gradient Accumulation (Already Implemented)

**Option**: `--gradient-accumulation-steps N`

Accumulate gradients over N steps before optimizer update. This allows using smaller batch sizes while maintaining effective batch size.

**Memory savings**: Reduces batch size per forward pass, but doesn't reduce peak memory per sample.

**Example**:
```bash
python -m aam.cli pretrain \
  --batch-size 1 \
  --gradient-accumulation-steps 32
```

### 4. Reduce Model Size

**Options**: `--embedding-dim`, `--attention-heads`, `--attention-layers`

Reduce model dimensions to decrease memory usage.

**Memory savings**: 
- Embeddings: `O(embedding_dim)`
- Attention: `O(embedding_dim)` per head
- FFN: `O(embedding_dim * intermediate_size)` where `intermediate_size = 4 * embedding_dim`

**Example**:
```bash
python -m aam.cli pretrain \
  --embedding-dim 64 \
  --attention-heads 2 \
  --attention-layers 2
```

### 5. Expandable Segments (Already Implemented)

**Option**: `--use-expandable-segments`

Enables PyTorch's expandable segments allocator to reduce memory fragmentation.

**Memory savings**: Reduces fragmentation, doesn't reduce peak memory but allows better utilization.

### 6. Mixed Precision Training (Future)

Use FP16/BF16 to halve memory usage. Requires PyTorch AMP support.

**Memory savings**: ~2x reduction in memory for activations and gradients.

## Recommended Settings for 24GB GPU

For a 24GB GPU with `batch_size=2` and `gradient_accumulation_steps=16`:

```bash
python -m aam.cli pretrain \
  --table data/fall_train_only_all_outdoor.biom \
  --tree data/all-outdoors_sepp_tree.nwk \
  --output-dir data/model-test \
  --batch-size 2 \
  --epochs 1000 \
  --gradient-accumulation-steps 16 \
  --asv-chunk-size 256 \
  --token-limit 512 \
  --use-expandable-segments
```

This should reduce memory usage significantly:
- ASV chunking: ~4x reduction in ASV-level attention
- Token limit reduction: ~4x reduction in sample-level attention
- Total estimated reduction: ~8-10x

## Monitoring Memory Usage

To monitor GPU memory usage:

```bash
watch -n 1 nvidia-smi
```

Or add this to your training script:
```python
import torch
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```
