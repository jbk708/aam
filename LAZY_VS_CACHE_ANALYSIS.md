# Lazy UniFrac vs Caching: When to Use What

## Current Implementation

### Upfront Computation (`--cache-unifrac` without `--lazy-unifrac`)
- **Computes**: Full N×N distance matrix upfront (~30 hours for 32K samples)
- **Caching**: ✅ **Highly Recommended**
  - Saves 30+ hours on subsequent runs
  - Cache file: ~7.7 GB (compressed)
  - Loads in seconds

### Lazy Computation (`--lazy-unifrac`)
- **Computes**: Distances per batch on-the-fly during training
- **In-memory caching**: ✅ Already implemented
  - Each `UniFracComputer` instance caches batch results
  - Cache key: `(frozenset(sample_ids), metric)`
  - Cache size: 128 batches (configurable)
- **Disk caching**: ❌ **Not currently implemented**

## Should You Cache with Lazy Loading?

### Short Answer: **No, not necessary**

### Why?

1. **In-memory cache already handles it**
   - Batch results are cached during training
   - Same batches in subsequent epochs use cached results
   - No disk I/O needed

2. **Lazy mode is designed for on-demand computation**
   - You don't wait 30 hours upfront
   - Computation happens during training
   - First epoch is slower, but training starts immediately

3. **Disk caching complexity**
   - Would need to cache many small batch results
   - Or implement a more complex structure
   - Adds overhead for minimal benefit

4. **Cache invalidation issues**
   - Batch composition may vary between runs
   - Different batch sizes, shuffling, etc.
   - Hard to maintain consistent cache keys

### When Disk Caching for Lazy Might Help

1. **Training interruption and resume**
   - If you resume training, batches might be recomputed
   - But in-memory cache handles this within a run

2. **Multiple training runs with same data**
   - If you run training multiple times with identical:
     - Table
     - Tree
     - Batch composition
     - Random seed
   - Could save some computation

3. **Very slow batch computation**
   - If each batch takes minutes (unlikely with pruned tree)
   - Disk cache could help across runs

## Recommendations

### Use Case 1: One-time training run
```
--lazy-unifrac --prune-tree
```
- No caching needed
- In-memory batch cache is sufficient
- Training starts immediately

### Use Case 2: Multiple training runs, same data
```
--cache-unifrac (without --lazy-unifrac)
```
- Compute full matrix once (~30 hours)
- Cache it (~7.7 GB)
- Subsequent runs load in seconds
- **Best for experimentation with hyperparameters**

### Use Case 3: Very large dataset, can't compute full matrix
```
--lazy-unifrac --prune-tree --num-workers 0
```
- No upfront computation
- Single worker (avoids multiple tree copies)
- In-memory batch cache handles repeats

## Current Limitations

1. **Batch cache not shared across workers**
   - Each DataLoader worker has its own `UniFracComputer`
   - Same batch might be computed multiple times by different workers
   - Could be optimized with shared memory cache

2. **No disk caching for lazy mode**
   - Would require implementing batch-level disk cache
   - Complex cache key management
   - Probably not worth the complexity

## Future Improvements

1. **Shared batch cache across workers**
   - Use `multiprocessing.Manager()` for shared cache
   - Avoid recomputing same batches across workers

2. **Optional disk caching for lazy mode**
   - If requested, cache batch results to disk
   - Useful for very slow batch computation
   - But adds complexity

## Bottom Line

**For your use case (32K samples, pruned tree):**

- **If you'll run training multiple times**: Use `--cache-unifrac` (upfront computation)
- **If this is a one-time run**: Use `--lazy-unifrac --prune-tree` (no disk caching needed)
- **If you're experimenting**: Use `--cache-unifrac` to avoid recomputing

The in-memory batch cache in lazy mode is usually sufficient. Disk caching for lazy mode would add complexity for minimal benefit in most cases.
