#!/usr/bin/env python
"""Pre-compute stripe-based UniFrac distances for training.

This script computes stripe-based UniFrac distances upfront, which can be
faster than lazy computation for large datasets. The computed stripe matrix
can then be loaded in training scripts using --no-lazy-unifrac.

Usage:
    python scripts/precompute_stripe_unifrac.py \
        --table data/table.biom \
        --tree data/tree.nwk \
        --output data/stripe_distances.npy \
        --reference-samples 100 \
        --prune-tree \
        --seed 42
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional
import random

import numpy as np
import biom
from biom import Table

from aam.data.biom_loader import BIOMLoader
from aam.data.unifrac import UniFracComputer
from aam.data.tree_pruner import load_or_prune_tree


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_reference_samples(
    reference_samples: str,
    sample_ids: List[str],
    seed: Optional[int] = None
) -> List[str]:
    """Parse reference samples from number or file path.
    
    Args:
        reference_samples: Number (e.g., '100') or file path
        sample_ids: List of all sample IDs
        seed: Random seed for reproducibility
    
    Returns:
        List of reference sample IDs
    """
    # Try to parse as number first
    try:
        num_ref = int(reference_samples)
        if num_ref <= 0:
            raise ValueError("Number of reference samples must be positive")
        if num_ref > len(sample_ids):
            logger.warning(
                f"Requested {num_ref} reference samples but only {len(sample_ids)} available. "
                f"Using all samples."
            )
            return sample_ids.copy()
        else:
            # Randomly select reference samples
            if seed is not None:
                random.seed(seed)
            return random.sample(sample_ids, num_ref)
    except ValueError:
        # Not a number, treat as file path
        ref_path = Path(reference_samples)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference samples file not found: {reference_samples}")
        with open(ref_path, 'r') as f:
            reference_sample_ids = [line.strip() for line in f if line.strip()]
        if not reference_sample_ids:
            raise ValueError(f"Reference samples file is empty: {reference_samples}")
        # Validate all reference samples exist
        missing_ref = set(reference_sample_ids) - set(sample_ids)
        if missing_ref:
            raise ValueError(f"Reference sample IDs not found in table: {sorted(missing_ref)}")
        return reference_sample_ids


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute stripe-based UniFrac distances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--table",
        required=True,
        type=str,
        help="Path to BIOM table file"
    )
    parser.add_argument(
        "--tree",
        required=True,
        type=str,
        help="Path to phylogenetic tree file (.nwk)"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output path for stripe distance matrix (.npy file)"
    )
    parser.add_argument(
        "--reference-samples",
        default=None,
        type=str,
        help="Reference samples: number (e.g., '100') or file path with sample IDs (one per line). "
             "If not specified, auto-selects 100 random samples or all if < 100."
    )
    parser.add_argument(
        "--reference-ids-output",
        default=None,
        type=str,
        help="Output path to save reference sample IDs (optional, for use in training)"
    )
    parser.add_argument(
        "--sample-ids-output",
        default=None,
        type=str,
        help="Output path to save all sample IDs in order (optional, for use in training)"
    )
    parser.add_argument(
        "--prune-tree",
        action="store_true",
        help="Pre-prune tree to only include ASVs in BIOM table"
    )
    parser.add_argument(
        "--unifrac-threads",
        default=None,
        type=int,
        help="Number of threads for UniFrac computation (default: all available CPU cores)"
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Random seed for reference sample selection"
    )
    parser.add_argument(
        "--rarefy-depth",
        default=None,
        type=int,
        help="Rarefaction depth (optional, will rarefy table if specified)"
    )
    parser.add_argument(
        "--rarefy-seed",
        default=None,
        type=int,
        help="Random seed for rarefaction"
    )
    
    args = parser.parse_args()
    
    logger.info("Loading BIOM table...")
    loader = BIOMLoader()
    table = loader.load_table(args.table)
    logger.info(f"Loaded table: {table.shape[0]} ASVs, {table.shape[1]} samples")
    
    # Rarefy if requested
    if args.rarefy_depth is not None:
        logger.info(f"Rarefying table to depth {args.rarefy_depth}...")
        table = loader.rarefy(table, depth=args.rarefy_depth, random_seed=args.rarefy_seed)
        logger.info(f"Rarefied table: {table.shape[0]} ASVs, {table.shape[1]} samples")
    
    # Get sample IDs
    sample_ids = list(table.ids(axis="sample"))
    logger.info(f"Total samples: {len(sample_ids)}")
    
    # Parse reference samples
    if args.reference_samples is not None:
        reference_sample_ids = parse_reference_samples(
            args.reference_samples, sample_ids, seed=args.seed
        )
        logger.info(f"Selected {len(reference_sample_ids)} reference samples")
    else:
        # Auto-select: randomly pick 100 or all if < 100
        if args.seed is not None:
            random.seed(args.seed)
        if len(sample_ids) <= 100:
            reference_sample_ids = sample_ids.copy()
        else:
            reference_sample_ids = random.sample(sample_ids, 100)
        logger.info(f"Auto-selected {len(reference_sample_ids)} random reference samples")
    
    # Save reference sample IDs if requested
    if args.reference_ids_output:
        ref_ids_path = Path(args.reference_ids_output)
        ref_ids_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ref_ids_path, 'w') as f:
            for ref_id in reference_sample_ids:
                f.write(f"{ref_id}\n")
        logger.info(f"Saved reference sample IDs to {ref_ids_path}")
    
    # Save all sample IDs if requested
    if args.sample_ids_output:
        sample_ids_path = Path(args.sample_ids_output)
        sample_ids_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sample_ids_path, 'w') as f:
            for sample_id in sample_ids:
                f.write(f"{sample_id}\n")
        logger.info(f"Saved all sample IDs to {sample_ids_path}")
    
    # Prune tree if requested
    tree_path = args.tree
    if args.prune_tree:
        logger.info("Pruning tree to only include ASVs in table...")
        pruned_tree_path = str(Path(args.tree).with_suffix('.pruned.nwk'))
        pruned_tree = load_or_prune_tree(args.tree, table, pruned_tree_path=pruned_tree_path)
        tree_path = pruned_tree_path
        logger.info(f"Using pruned tree: {tree_path}")
    
    # Compute stripe distances
    logger.info("Computing stripe-based UniFrac distances...")
    logger.info(f"Computing distances from {len(sample_ids)} samples to {len(reference_sample_ids)} reference samples...")
    
    computer = UniFracComputer(num_threads=args.unifrac_threads)
    
    start_time = time.time()
    stripe_distances = computer.compute_unweighted_stripe(
        table=table,
        tree_path=tree_path,
        reference_sample_ids=reference_sample_ids,
        test_sample_ids=None,  # Compute for all samples
        filter_table=True
    )
    elapsed_time = time.time() - start_time
    
    logger.info(f"Computed stripe distances in {elapsed_time:.2f} seconds")
    logger.info(f"Stripe matrix shape: {stripe_distances.shape}")
    logger.info(f"Memory usage: {stripe_distances.nbytes / 1024**2:.2f} MB")
    
    # Save stripe matrix
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, stripe_distances)
    logger.info(f"Saved stripe distance matrix to {output_path}")
    
    # Save metadata
    metadata_path = output_path.with_suffix('.metadata.npz')
    np.savez(
        metadata_path,
        sample_ids=sample_ids,
        reference_sample_ids=reference_sample_ids,
        shape=stripe_distances.shape
    )
    logger.info(f"Saved metadata to {metadata_path}")
    
    logger.info("Pre-computation complete!")


if __name__ == "__main__":
    main()
