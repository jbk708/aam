# Parallel UniFrac Computation Guide

## Overview

For large datasets (like your 32K samples), computing the full UniFrac distance matrix can take 30+ hours with a single process. This guide shows how to use parallel computation to dramatically speed this up.

## Quick Start

### Step 1: Compute Distance Matrix in Parallel

```bash
python -m aam.scripts.compute_unifrac_parallel \
  --table /projects/deep-learning/aam-publication-data/agp-emp-large.biom \
  --tree /projects/deep-learning/aam-publication-data/2022.10.phylogeny.asv.pruned.nwk \
  --output data/unifrac_distances.npz \
  --metric unweighted \
  --num-workers 40 \
  --threads-per-worker 4 \
  --chunk-size 1000 \
  --rarefy-depth 5000 \
  --rarefy-seed 42
```

### Step 2: Use Pre-computed Matrix in Training

```bash
python -m aam.cli pretrain \
  --table /projects/deep-learning/aam-publication-data/agp-emp-large.biom \
  --tree /projects/deep-learning/aam-publication-data/2022.10.phylogeny.asv.pruned.nwk \
  --unifrac-cache-dir data/ \
  --output-dir data/agp-emp-run \
  ...other args...
```

The training script will automatically detect and use the cached distance matrix.

## Performance Comparison

### Single Process (Current)
- **Time**: ~30 hours for 32K samples
- **Workers**: 1 process, 20 threads
- **Memory**: ~7.7 GB for distance matrix

### Parallel (Recommended)
- **Time**: ~2-4 hours for 32K samples (with 40 workers)
- **Workers**: 40 processes, 4 threads each = 160 total threads
- **Memory**: ~7.7 GB for distance matrix (same)
- **Speedup**: ~7-15x faster

## Configuration

### Number of Workers

**Rule of thumb**: Use 2-4x your CPU core count
- 40-core machine → 40-80 workers
- 80-core machine → 80-160 workers

**Considerations**:
- More workers = faster computation (up to a point)
- Each worker loads the tree into memory
- With pruned tree (637K tips), memory per worker is manageable
- Too many workers can cause memory pressure

### Threads Per Worker

**Recommended**: 2-4 threads per worker
- Each worker uses OpenMP threads for UniFrac computation
- 4 threads per worker is usually optimal
- Total threads = num_workers × threads_per_worker

**Example**:
- 40 workers × 4 threads = 160 total threads
- Good for 80+ core machines

### Chunk Size

**Recommended**: 500-2000 samples per chunk
- Smaller chunks = more parallelization but more overhead
- Larger chunks = less overhead but less parallelization
- 1000 is usually a good balance

## Memory Considerations

### Per Worker Memory
- Tree: ~100-500 MB (pruned tree)
- Chunk table: ~50-200 MB (depends on chunk size)
- Chunk distances: ~10-50 MB
- **Total per worker**: ~200-800 MB

### Total Memory
- 40 workers × 800 MB = ~32 GB
- Plus distance matrix: ~8 GB
- **Total**: ~40 GB

**Recommendation**: Ensure you have enough RAM for all workers.

## Workflow

### Option 1: Pre-compute and Cache (Recommended)

1. **Compute once** (2-4 hours with parallel script):
   ```bash
   python -m aam.scripts.compute_unifrac_parallel \
     --table table.biom \
     --tree tree.pruned.nwk \
     --output unifrac_cache.npz \
     --num-workers 40
   ```

2. **Train multiple times** (loads in seconds):
   ```bash
   python -m aam.cli pretrain \
     --cache-unifrac \
     --unifrac-cache-dir . \
     ...
   ```

### Option 2: Use Lazy Computation

If you don't want to pre-compute:
```bash
python -m aam.cli pretrain \
  --lazy-unifrac \
  --prune-tree \
  ...
```

- No upfront computation
- Computes per batch during training
- First epoch slower, but training starts immediately

## Troubleshooting

### Out of Memory

**Symptoms**: Workers crash or system becomes unresponsive

**Solutions**:
1. Reduce `--num-workers`
2. Reduce `--chunk-size`
3. Use pruned tree (already doing this)
4. Use lazy computation instead

### Slow Performance

**Check**:
1. Are all workers busy? (check `htop` or `top`)
2. Is memory swapping? (check `free -h`)
3. Are threads actually parallelizing? (check CPU usage)

**Solutions**:
1. Increase `--num-workers` (if CPU cores available)
2. Increase `--threads-per-worker` (if memory available)
3. Use fewer workers but more threads per worker

### Cache Not Found

**Check**:
1. Is cache file in the right location?
2. Does cache file match table/tree/rarefaction params?

**Solution**:
- Recompute with `--no-cache` flag to force recomputation
- Or check cache key matches

## Example: Your Dataset

For your 32K sample dataset with pruned tree:

```bash
# Step 1: Compute in parallel (2-4 hours)
python -m aam.scripts.compute_unifrac_parallel \
  --table /projects/deep-learning/aam-publication-data/agp-emp-large.biom \
  --tree /projects/deep-learning/aam-publication-data/2022.10.phylogeny.asv.pruned.nwk \
  --output /projects/deep-learning/aam-publication-data/unifrac_distances.npz \
  --metric unweighted \
  --num-workers 40 \
  --threads-per-worker 4 \
  --chunk-size 1000 \
  --rarefy-depth 5000 \
  --rarefy-seed 42

# Step 2: Train (loads cache in seconds)
python -m aam.cli pretrain \
  --table /projects/deep-learning/aam-publication-data/agp-emp-large.biom \
  --tree /projects/deep-learning/aam-publication-data/2022.10.phylogeny.asv.pruned.nwk \
  --cache-unifrac \
  --unifrac-cache-dir /projects/deep-learning/aam-publication-data/ \
  --output-dir data/agp-emp-run \
  ...
```

## Benefits

1. **Much Faster**: 7-15x speedup vs single process
2. **Reusable**: Compute once, use many times
3. **Scalable**: Can use many workers on large machines
4. **Flexible**: Can still use lazy mode if preferred

## Limitations

1. **Memory**: Each worker loads tree (manageable with pruned tree)
2. **Complexity**: More moving parts than single-process
3. **Setup**: Need to run separate script first

But for large datasets, the speedup is worth it!
