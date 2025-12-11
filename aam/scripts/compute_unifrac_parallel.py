#!/usr/bin/env python
"""Parallel UniFrac distance matrix computation script.

This script computes UniFrac distances using multiple worker processes,
which is much faster than single-process computation for large datasets.
The computed distance matrix can then be loaded in training scripts.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Tuple, Union
import multiprocessing as mp

import numpy as np
import biom
from biom import Table
from skbio import DistanceMatrix
import pandas as pd

from aam.data.biom_loader import BIOMLoader
from aam.data.unifrac import UniFracComputer
from aam.data.unifrac_cache import get_cache_path, save_distance_matrix, load_distance_matrix


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_distance_chunk(
    args: Tuple[List[str], Table, str, int, str]
) -> Tuple[List[str], np.ndarray]:
    """Compute UniFrac distances for a chunk of sample pairs.
    
    Args:
        args: Tuple of (sample_ids, table, tree_path, num_threads, metric)
    
    Returns:
        Tuple of (sample_ids, distance_matrix_chunk)
    """
    sample_ids, table, tree_path, num_threads, metric = args
    
    # Create UniFracComputer for this worker
    unifrac_computer = UniFracComputer(num_threads=num_threads)
    
    # Filter table to these samples
    chunk_table = table.filter(sample_ids, axis="sample", inplace=False)
    
    # Compute distances
    if metric == "unweighted":
        distances = unifrac_computer.compute_unweighted(chunk_table, tree_path)
        return sample_ids, distances.data
    else:
        distances = unifrac_computer.compute_faith_pd(chunk_table, tree_path)
        return sample_ids, distances.values.reshape(-1, 1)


def compute_unifrac_parallel(
    table: Table,
    tree_path: str,
    metric: str = "unweighted",
    num_workers: int = None,
    threads_per_worker: int = 4,
    chunk_size: int = 1000,
    output_path: str = None,
    use_cache: bool = True,
) -> Union[DistanceMatrix, pd.Series]:
    """Compute UniFrac distances using multiple worker processes.
    
    Args:
        table: BIOM table
        tree_path: Path to phylogenetic tree
        metric: "unweighted" or "faith_pd"
        num_workers: Number of worker processes (default: CPU count)
        threads_per_worker: Threads per worker for UniFrac computation
        chunk_size: Number of samples per chunk
        output_path: Path to save distance matrix
        use_cache: Whether to use/save cache
    
    Returns:
        DistanceMatrix or Series
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    sample_ids = list(table.ids(axis="sample"))
    n_samples = len(sample_ids)
    
    logger.info(f"Computing {metric} UniFrac for {n_samples:,} samples")
    logger.info(f"Using {num_workers} workers with {threads_per_worker} threads each")
    logger.info(f"Chunk size: {chunk_size} samples per chunk")
    
    # Check cache first
    if use_cache and output_path:
        cache_path = Path(output_path)
        if cache_path.exists():
            logger.info(f"Loading cached distance matrix from {output_path}")
            try:
                from aam.data.unifrac_cache import load_distance_matrix
                cached = load_distance_matrix(cache_path, metric=metric)
                if cached is not None:
                    logger.info("Using cached distance matrix")
                    return cached
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
    
    start_time = time.time()
    
    if metric == "unweighted":
        # For unweighted, we compute full pairwise distances
        # Strategy: Compute in chunks, then assemble
        
        # Split samples into chunks
        chunks = []
        for i in range(0, n_samples, chunk_size):
            chunk_ids = sample_ids[i:i + chunk_size]
            chunks.append((chunk_ids, table, tree_path, threads_per_worker, metric))
        
        logger.info(f"Computing {len(chunks)} chunks in parallel...")
        
        # Compute chunks in parallel
        with mp.Pool(num_workers) as pool:
            results = pool.map(compute_distance_chunk, chunks)
        
        # Assemble full distance matrix
        logger.info("Assembling full distance matrix...")
        full_matrix = np.zeros((n_samples, n_samples))
        
        # Fill in diagonal blocks
        for idx, (chunk_ids, chunk_distances) in enumerate(results):
            start_idx = idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk_len = end_idx - start_idx
            
            if metric == "unweighted":
                # chunk_distances is already the full pairwise matrix for this chunk
                full_matrix[start_idx:end_idx, start_idx:end_idx] = chunk_distances
        
        # Compute cross-chunk distances
        logger.info("Computing cross-chunk distances...")
        cross_chunk_tasks = []
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                chunk_i_ids = chunks[i][0]
                chunk_j_ids = chunks[j][0]
                combined_ids = chunk_i_ids + chunk_j_ids
                cross_chunk_tasks.append((
                    combined_ids, table, tree_path, threads_per_worker, metric
                ))
        
        # Compute cross-chunk distances
        with mp.Pool(num_workers) as pool:
            cross_results = pool.map(compute_distance_chunk, cross_chunk_tasks)
        
        # Fill in cross-chunk blocks
        task_idx = 0
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                chunk_i_ids = chunks[i][0]
                chunk_j_ids = chunks[j][0]
                combined_ids, cross_distances = cross_results[task_idx]
                
                start_i = sample_ids.index(chunk_i_ids[0])
                start_j = sample_ids.index(chunk_j_ids[0])
                end_i = start_i + len(chunk_i_ids)
                end_j = start_j + len(chunk_j_ids)
                
                # Extract cross-chunk block
                cross_block = cross_distances[:len(chunk_i_ids), len(chunk_i_ids):]
                full_matrix[start_i:end_i, start_j:end_j] = cross_block
                full_matrix[start_j:end_j, start_i:end_i] = cross_block.T
                
                task_idx += 1
        
        # Make symmetric
        full_matrix = (full_matrix + full_matrix.T) / 2
        
        distance_matrix = DistanceMatrix(full_matrix, ids=sample_ids)
        
    else:  # faith_pd
        # For faith_pd, we compute per-sample values (much simpler)
        chunks = []
        for i in range(0, n_samples, chunk_size):
            chunk_ids = sample_ids[i:i + chunk_size]
            chunks.append((chunk_ids, table, tree_path, threads_per_worker, metric))
        
        logger.info(f"Computing {len(chunks)} chunks in parallel...")
        
        with mp.Pool(num_workers) as pool:
            results = pool.map(compute_distance_chunk, chunks)
        
        # Assemble series
        all_values = []
        all_ids = []
        for chunk_ids, chunk_values in results:
            all_ids.extend(chunk_ids)
            all_values.extend(chunk_values.flatten())
        
        distance_matrix = pd.Series(all_values, index=all_ids)
    
    elapsed = time.time() - start_time
    logger.info(f"Computation complete in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    # Save to cache if requested
    if use_cache and output_path:
        logger.info(f"Saving distance matrix to {output_path}")
        save_distance_matrix(distance_matrix, Path(output_path), metric=metric)
    
    return distance_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Compute UniFrac distances in parallel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--table", required=True, help="Path to BIOM table")
    parser.add_argument("--tree", required=True, help="Path to phylogenetic tree")
    parser.add_argument("--output", required=True, help="Output path for distance matrix (.npz)")
    parser.add_argument("--metric", choices=["unweighted", "faith_pd"], default="unweighted",
                       help="UniFrac metric type")
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of worker processes (default: CPU count)")
    parser.add_argument("--threads-per-worker", type=int, default=4,
                       help="Number of threads per worker for UniFrac computation")
    parser.add_argument("--chunk-size", type=int, default=1000,
                       help="Number of samples per chunk")
    parser.add_argument("--rarefy-depth", type=int, default=None,
                       help="Rarefaction depth")
    parser.add_argument("--rarefy-seed", type=int, default=None,
                       help="Rarefaction random seed")
    parser.add_argument("--no-cache", action="store_true",
                       help="Don't use/save cache")
    
    args = parser.parse_args()
    
    # Load table
    logger.info(f"Loading table from {args.table}")
    biom_loader = BIOMLoader()
    table = biom_loader.load_table(args.table)
    
    if args.rarefy_depth:
        logger.info(f"Rarefying to depth {args.rarefy_depth}")
        table = biom_loader.rarefy(table, depth=args.rarefy_depth, random_seed=args.rarefy_seed)
    
    # Compute distances
    distances = compute_unifrac_parallel(
        table=table,
        tree_path=args.tree,
        metric=args.metric,
        num_workers=args.num_workers,
        threads_per_worker=args.threads_per_worker,
        chunk_size=args.chunk_size,
        output_path=args.output,
        use_cache=not args.no_cache,
    )
    
    logger.info(f"Distance matrix computed and saved to {args.output}")
    logger.info(f"Shape: {distances.shape if hasattr(distances, 'shape') else len(distances)}")


if __name__ == "__main__":
    main()
