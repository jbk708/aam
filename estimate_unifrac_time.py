#!/usr/bin/env python
"""Estimate UniFrac computation time for a given dataset."""

import sys
from pathlib import Path

def estimate_unifrac_time(samples: int, tree_tips: int, threads: int = 20):
    """Estimate UniFrac computation time.
    
    Args:
        samples: Number of samples
        tree_tips: Number of tips in phylogenetic tree
        threads: Number of threads for computation
    
    Returns:
        Dictionary with time estimates
    """
    # Total pairwise distances (symmetric matrix, so N*(N-1)/2)
    total_distances = samples * (samples - 1) // 2
    
    # Time per distance calculation depends on tree size
    # Benchmarks suggest:
    # - Small trees (<10K tips): ~0.1-0.5ms per distance
    # - Medium trees (10K-100K tips): ~0.5-2ms per distance
    # - Large trees (100K-1M tips): ~2-10ms per distance
    # - Very large trees (>1M tips): ~10-50ms per distance
    
    if tree_tips < 10000:
        time_per_distance_ms = 0.3
    elif tree_tips < 100000:
        time_per_distance_ms = 1.0
    elif tree_tips < 1000000:
        time_per_distance_ms = 3.0  # Your case: 637K tips
    else:
        time_per_distance_ms = 10.0
    
    # With parallelization, effective throughput
    # Note: UniFrac parallelization isn't perfect, so we use a scaling factor
    parallel_efficiency = 0.7  # 70% efficiency with threading
    effective_threads = threads * parallel_efficiency
    
    # Compute time
    distances_per_second_per_thread = 1000 / time_per_distance_ms
    total_rate = distances_per_second_per_thread * effective_threads
    
    time_seconds = total_distances / total_rate
    time_minutes = time_seconds / 60
    time_hours = time_minutes / 60
    
    return {
        "samples": samples,
        "tree_tips": tree_tips,
        "threads": threads,
        "total_distances": total_distances,
        "time_per_distance_ms": time_per_distance_ms,
        "time_seconds": time_seconds,
        "time_minutes": time_minutes,
        "time_hours": time_hours,
        "estimated_memory_gb": (samples * samples * 8) / (1024**3),  # 8 bytes per float64
    }


if __name__ == "__main__":
    # Your specific case
    samples = 32215
    tree_tips = 637737  # Pruned tree
    threads = 20
    
    if len(sys.argv) > 1:
        samples = int(sys.argv[1])
    if len(sys.argv) > 2:
        tree_tips = int(sys.argv[2])
    if len(sys.argv) > 3:
        threads = int(sys.argv[3])
    
    result = estimate_unifrac_time(samples, tree_tips, threads)
    
    print("=" * 60)
    print("UniFrac Computation Time Estimate")
    print("=" * 60)
    print(f"Samples: {result['samples']:,}")
    print(f"Tree tips (pruned): {result['tree_tips']:,}")
    print(f"Threads: {result['threads']}")
    print(f"Total pairwise distances: {result['total_distances']:,}")
    print(f"Time per distance: ~{result['time_per_distance_ms']:.2f} ms")
    print()
    print("Estimated computation time:")
    print(f"  {result['time_seconds']:.0f} seconds")
    print(f"  {result['time_minutes']:.1f} minutes")
    print(f"  {result['time_hours']:.2f} hours")
    print()
    print(f"Estimated memory for distance matrix: {result['estimated_memory_gb']:.2f} GB")
    print()
    print("Note: This is a rough estimate. Actual time depends on:")
    print("  - Tree structure complexity")
    print("  - CPU performance and memory bandwidth")
    print("  - UniFrac library implementation and optimizations")
    print("  - System load and other processes")
    print()
    print("To get an accurate measurement, run a small test:")
    print("  python -c \"from aam.data.unifrac import UniFracComputer; import time; ...\"")
    print("=" * 60)
