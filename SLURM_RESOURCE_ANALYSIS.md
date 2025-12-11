# SLURM Resource Analysis for Parallel UniFrac

## Your Allocation

```bash
srun --time=96:00:00 -p short -N 1 -n 1 -c 40 --mem=128g --pty /bin/bash -l
```

- **CPUs**: 40 cores
- **Memory**: 128 GB
- **Time**: 96 hours (4 days)
- **Nodes**: 1

## Resource Requirements

### For Your Dataset
- **Samples**: 32,215
- **Tree tips**: 637,737 (pruned)
- **Estimated computation time**: 2-4 hours (with proper parallelization)
- **Distance matrix size**: ~7.7 GB

### Memory Breakdown

**Per Worker**:
- Tree: ~100-500 MB (pruned tree)
- Chunk table: ~50-200 MB (depends on chunk size)
- Chunk distances: ~10-50 MB
- **Total per worker**: ~200-800 MB

**Total Memory**:
- Workers: 40 × 800 MB = ~32 GB
- Distance matrix: ~8 GB
- System overhead: ~10 GB
- **Total needed**: ~50 GB

### CPU Usage

**Important**: With 40 cores, you have 40 CPUs available.

**Options**:

1. **40 workers, 1 thread each** (Recommended)
   - `--num-workers 40 --threads-per-worker 1`
   - Uses all 40 cores efficiently
   - No oversubscription

2. **20 workers, 2 threads each**
   - `--num-workers 20 --threads-per-worker 2`
   - Uses all 40 cores
   - Slightly less parallelization but more threads per worker

3. **10 workers, 4 threads each**
   - `--num-workers 10 --threads-per-worker 4`
   - Uses all 40 cores
   - More threads per worker (better for UniFrac internal parallelization)

## Verdict: ✅ **YES, More Than Enough!**

### CPU: ✅ Sufficient
- 40 cores is perfect for 40 workers
- Can use all cores efficiently

### Memory: ✅ More Than Enough
- Need ~50 GB, have 128 GB
- 2.5x headroom for safety

### Time: ✅ Plenty
- Need 2-4 hours, have 96 hours
- 24x headroom

## Recommended Command

```bash
# After getting your SLURM allocation, run:
python -m aam.scripts.compute_unifrac_parallel \
  --table /projects/deep-learning/aam-publication-data/agp-emp-large.biom \
  --tree /projects/deep-learning/aam-publication-data/2022.10.phylogeny.asv.pruned.nwk \
  --output /projects/deep-learning/aam-publication-data/unifrac_distances.npz \
  --metric unweighted \
  --num-workers 40 \
  --threads-per-worker 1 \
  --chunk-size 1000 \
  --rarefy-depth 5000 \
  --rarefy-seed 42
```

**Why `--threads-per-worker 1`?**
- You have 40 cores
- 40 workers × 1 thread = 40 threads (perfect match)
- No CPU oversubscription
- Each worker gets dedicated core

## Alternative: Fewer Workers, More Threads

If you want to leverage UniFrac's internal OpenMP parallelization:

```bash
python -m aam.scripts.compute_unifrac_parallel \
  --table ... \
  --tree ... \
  --output ... \
  --num-workers 10 \
  --threads-per-worker 4 \
  ...
```

This uses:
- 10 workers × 4 threads = 40 threads total
- Each worker uses 4 OpenMP threads internally
- May be slightly faster due to better cache locality

## Expected Performance

With 40 workers on 40 cores:
- **Computation time**: 2-4 hours (vs 30 hours single-process)
- **Speedup**: ~7-15x faster
- **Memory usage**: ~50 GB (well within 128 GB limit)

## Monitoring

While running, check resource usage:

```bash
# In another terminal (or use htop)
htop  # or top
```

You should see:
- ~40 processes (workers) using CPU
- ~50 GB memory usage
- All 40 cores at high utilization

## Potential Issues

### 1. CPU Oversubscription
**Problem**: Using more threads than cores (e.g., 40 workers × 4 threads = 160 threads on 40 cores)

**Solution**: Use `--threads-per-worker 1` or reduce `--num-workers`

### 2. Memory Pressure
**Problem**: Too many workers causing memory pressure

**Solution**: Reduce `--num-workers` or `--chunk-size`

### 3. I/O Bottleneck
**Problem**: All workers reading tree file simultaneously

**Solution**: Tree is already loaded per worker (this is fine)

## Summary

✅ **Your allocation is perfect!**

- 40 cores → 40 workers (1:1 ratio)
- 128 GB → Plenty of headroom
- 96 hours → Way more than needed

**Recommended settings**:
- `--num-workers 40`
- `--threads-per-worker 1`
- `--chunk-size 1000`

This will use all resources efficiently and complete in 2-4 hours.
