# Tree and Batch Loading Comparison: Main vs PYT-10.3.1

## Overview

This document compares how trees and batches are loaded in the main branch versus the current PYT-10.3.1 branch (with tree pruning optimizations).

## Key Finding

**The `aam/data/unifrac.py` file does not exist on main branch** - it's a new file added in the PyTorch porting effort. The main branch likely uses a different approach for UniFrac computation.

## Current Branch (PYT-10.3.1) Implementation

### Tree Loading Strategy

1. **Lazy Loading Per Worker Process**
   - Tree is NOT loaded during `setup_lazy_computation()`
   - Tree is loaded lazily in each worker process when `compute_batch_unweighted()` or `compute_batch_faith_pd()` is first called
   - Tree is cached per process (`self._tree`) to avoid reloading
   - Each DataLoader worker process loads its own copy of the tree

2. **Tree Pruning (New Feature)**
   - If `prune_tree=True`, tree is pre-pruned to only include ASVs in the BIOM table
   - Pruned tree is cached to disk (`{tree_path}.pruned.nwk`)
   - Pruning happens in `load_or_prune_tree()` which:
     - Checks if cached pruned tree exists
     - If not, loads full tree, prunes it, and saves to cache
     - Returns pruned tree for use
   - Pruning dramatically reduces tree size (21M tips → potentially <100K tips)

3. **Tree Filtering (Legacy)**
   - `filter_tree=True` option filters tree to ASVs in table during computation
   - This is done by unifrac library internally, not by pre-pruning

### Batch Loading Strategy

1. **Lazy Batch Computation**
   - Distances computed on-the-fly per batch via `compute_batch_unweighted()` or `compute_batch_faith_pd()`
   - Results are cached per `UniFracComputer` instance (not shared across workers)
   - Cache key: `(frozenset(sample_ids), metric)`

2. **Batch Processing Flow**
   ```
   DataLoader → collate_fn() → 
     - If lazy_unifrac: unifrac_computer.compute_batch_*()
     - Else: unifrac_computer.extract_batch_distances()
   ```

3. **Tree Loading in Batch Methods**
   - First call to `compute_batch_*()` loads tree if `self._tree is None`
   - Tree path resolution:
     - If pruning enabled: uses `self._pruned_tree_cache` path
     - Else: uses `self._tree_path`
   - Tree loaded once per worker process and cached

### Code Structure

**setup_lazy_computation():**
```python
def setup_lazy_computation(
    self,
    table: Table,
    tree_path: str,
    filter_tree: bool = True,
    prune_tree: bool = False,  # NEW
    pruned_tree_cache: Optional[str] = None,  # NEW
) -> None:
    # Store table and paths
    # If prune_tree: compute stats, set up pruning paths
    # DON'T load tree here - load lazily per worker
    self._tree = None  # Will be loaded per worker
```

**compute_batch_unweighted():**
```python
def compute_batch_unweighted(self, sample_ids, table=None, tree_path=None):
    # Check cache first
    # Resolve tree_path (pruned vs original)
    # Load tree lazily if self._tree is None:
    #   - If pruning: load_or_prune_tree()
    #   - Else: skbio.read(tree_path)
    # Filter table to batch samples
    # Compute distances for batch only
    # Cache result
```

## Main Branch Implementation

**Status:** `aam/data/unifrac.py` does not exist on main branch.

The main branch likely:
- Computes UniFrac distances upfront (not lazy)
- Loads tree once at startup
- Extracts batch distances from pre-computed matrix
- No tree pruning feature

## Key Differences

| Feature | Main Branch | PYT-10.3.1 Branch |
|---------|-------------|-------------------|
| Tree Loading | Upfront, single load | Lazy, per worker process |
| Tree Pruning | ❌ Not available | ✅ Pre-prune to ASVs in table |
| Batch Computation | Extract from full matrix | Compute on-the-fly per batch |
| Memory Usage | High (full tree + full matrix) | Lower (pruned tree, batch-only computation) |
| Startup Time | Slow (load tree + compute all distances) | Fast (no tree load, no upfront computation) |
| First Epoch | Fast (distances pre-computed) | Slower (compute per batch) |
| Subsequent Epochs | Fast | Fast (cached batches) |

## Performance Implications

### Current Branch (PYT-10.3.1) with Pruning

**Tree Loading:**
- Without pruning: 6+ minutes for 21M tip tree
- With pruning: Seconds (pruned tree ~100K tips)
- Per worker: Each DataLoader worker loads tree once (cached)

**UniFrac Computation:**
- Without pruning: ~3 min/batch for large tree
- With pruning: Seconds/batch (much smaller tree)
- Caching: Batch results cached to avoid recomputation

**Memory:**
- Without pruning: ~21M tips in memory per worker
- With pruning: ~100K tips in memory per worker
- Multiple workers: Each has its own tree copy

### Recommendations

1. **Use `--prune-tree` flag** for large trees (21M+ tips)
2. **Use `--num-workers 0`** if memory is constrained (single process, single tree copy)
3. **Use `--num-workers > 0`** if you have enough memory (parallel data loading, but each worker loads tree)

## Potential Issues

1. **Multiple Tree Copies in Memory**
   - Each DataLoader worker loads its own tree
   - With 4 workers + pruning: 4 copies of pruned tree
   - Without pruning: 4 copies of 21M tip tree (very high memory!)

2. **Tree Loading Overhead**
   - First batch in each worker triggers tree load
   - With pruning: Fast (seconds)
   - Without pruning: Slow (6+ minutes per worker!)

3. **Cache Not Shared Across Workers**
   - Each worker has its own `UniFracComputer` instance
   - Batch cache is per-instance, not shared
   - Same batch computed multiple times if processed by different workers

## Future Optimizations

1. **Shared Tree Cache Across Workers**
   - Use shared memory or file-based cache
   - Load tree once, share across all workers

2. **Shared Batch Cache**
   - Use multiprocessing.Manager() for shared cache
   - Avoid recomputing same batches across workers

3. **Pre-load Pruned Tree**
   - Load pruned tree during `setup_lazy_computation()` if single worker
   - Only lazy-load if multiple workers (to avoid memory issues)
