"""UniFrac distance computation for microbial sequencing data."""

from typing import List, Optional, Union, Dict, Tuple
import tempfile
import os
from pathlib import Path
from functools import lru_cache

import biom
from biom import Table
import skbio
from skbio import DistanceMatrix, TreeNode
import pandas as pd
import numpy as np
import unifrac
import multiprocessing

# Import tree pruner
from aam.data.tree_pruner import load_or_prune_tree, get_pruning_stats


class UniFracComputer:
    """Compute phylogenetic distances (UniFrac) from BIOM tables and phylogenetic trees.

    This class provides functionality to:
    - Compute unweighted UniFrac distances between samples
    - Compute Faith's Phylogenetic Diversity (Faith PD) per sample
    - Extract batch-level distances from pre-computed distance matrices
    """

    def __init__(self, num_threads: Optional[int] = None, cache_size: int = 128):
        """Initialize UniFracComputer.
        
        Args:
            num_threads: Number of threads to use for UniFrac computation.
                        If None, uses all available CPU cores.
                        Sets OMP_NUM_THREADS environment variable for unifrac library.
            cache_size: Maximum number of batch distance computations to cache (for lazy computation).
                       Default: 128. Set to 0 to disable caching.
        """
        if num_threads is None:
            num_threads = multiprocessing.cpu_count()
        self.num_threads = num_threads
        self.cache_size = cache_size
        # Set OMP_NUM_THREADS for unifrac library (uses OpenMP internally)
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        
        # Cache for lazy batch computation: (frozenset(sample_ids), metric) -> distances
        self._batch_cache: Dict[Tuple[frozenset, str], np.ndarray] = {}
        self._table: Optional[Table] = None
        self._tree: Optional[TreeNode] = None
        self._tree_path: Optional[str] = None
        self._original_tree_path: Optional[str] = None
        self._prune_tree: bool = False
        self._table_for_pruning: Optional[Table] = None
        self._pruned_tree_cache: Optional[str] = None
        
        # Stripe-based computation support
        self._reference_sample_ids: Optional[List[str]] = None
        self._reference_counts_cache: Optional[Dict[str, np.ndarray]] = None

    def compute_unweighted(self, table: Table, tree_path: str, filter_table: bool = True) -> DistanceMatrix:
        """Compute unweighted UniFrac distances between samples.

        Args:
            table: Rarefied biom.Table object
            tree_path: Path to phylogenetic tree file (.nwk Newick format)
            filter_table: If True, filter table to only include ASVs present in tree

        Returns:
            skbio.DistanceMatrix containing pairwise unweighted UniFrac distances
            Shape: [N_samples, N_samples]

        Raises:
            FileNotFoundError: If tree file doesn't exist
            ValueError: If ASV IDs don't match between table and tree
        """
        tree_path_obj = Path(tree_path)
        if not tree_path_obj.exists():
            raise FileNotFoundError(f"Tree file not found: {tree_path}")

        try:
            tree = skbio.read(str(tree_path), format="newick", into=TreeNode)
        except Exception as e:
            raise ValueError(f"Error loading phylogenetic tree from {tree_path}: {e}")

        # Filter table to only include ASVs present in tree if requested
        if filter_table:
            tree_tips = {tip.name for tip in tree.tips() if tip.name is not None}
            table_asv_ids = set(table.ids(axis="observation"))
            asvs_to_keep = table_asv_ids.intersection(tree_tips)
            
            if len(asvs_to_keep) == 0:
                raise ValueError(
                    f"No ASVs from table found in tree. "
                    f"Table has {len(table_asv_ids)} ASVs, tree has {len(tree_tips)} tips, overlap: 0"
                )
            
            if len(asvs_to_keep) < len(table_asv_ids):
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Filtering table: {len(table_asv_ids)} ASVs -> {len(asvs_to_keep)} ASVs "
                    f"({len(table_asv_ids) - len(asvs_to_keep)} ASVs not in tree)"
                )
                table = table.filter(asvs_to_keep, axis="observation", inplace=False)

        try:
            distance_matrix = unifrac.unweighted(table, tree)
        except Exception as e:
            if "not found" in str(e).lower() or "mismatch" in str(e).lower() or "completely represented" in str(e).lower():
                raise ValueError(f"ASV IDs don't match between table and tree: {e}") from e
            raise ValueError(f"Error computing unweighted UniFrac: {e}") from e

        return distance_matrix

    def compute_faith_pd(self, table: Table, tree_path: str, filter_table: bool = True) -> Union[DistanceMatrix, pd.Series]:
        """Compute Faith's Phylogenetic Diversity (Faith PD) per sample.

        Args:
            table: Rarefied biom.Table object
            tree_path: Path to phylogenetic tree file (.nwk Newick format)
            filter_table: If True, filter table to only include ASVs present in tree

        Returns:
            pandas Series containing Faith PD values per sample
            Shape: [N_samples] (indexed by sample IDs)

        Raises:
            FileNotFoundError: If tree file doesn't exist
            ValueError: If ASV IDs don't match between table and tree
        """
        tree_path_obj = Path(tree_path)
        if not tree_path_obj.exists():
            raise FileNotFoundError(f"Tree file not found: {tree_path}")

        try:
            tree = skbio.read(str(tree_path), format="newick", into=TreeNode)
        except Exception as e:
            raise ValueError(f"Error loading phylogenetic tree from {tree_path}: {e}")

        # Filter table to only include ASVs present in tree if requested
        if filter_table:
            tree_tips = {tip.name for tip in tree.tips() if tip.name is not None}
            table_asv_ids = set(table.ids(axis="observation"))
            asvs_to_keep = table_asv_ids.intersection(tree_tips)
            
            if len(asvs_to_keep) == 0:
                raise ValueError(
                    f"No ASVs from table found in tree. "
                    f"Table has {len(table_asv_ids)} ASVs, tree has {len(tree_tips)} tips, overlap: 0"
                )
            
            if len(asvs_to_keep) < len(table_asv_ids):
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Filtering table: {len(table_asv_ids)} ASVs -> {len(asvs_to_keep)} ASVs "
                    f"({len(table_asv_ids) - len(asvs_to_keep)} ASVs not in tree)"
                )
                table = table.filter(asvs_to_keep, axis="observation", inplace=False)

        try:
            faith_pd_series = unifrac.faith_pd(table, tree)
        except Exception as e:
            if "not found" in str(e).lower() or "mismatch" in str(e).lower() or "completely represented" in str(e).lower():
                raise ValueError(f"ASV IDs don't match between table and tree: {e}") from e
            raise ValueError(f"Error computing Faith PD: {e}") from e

        return faith_pd_series

    def extract_batch_distances(
        self,
        distances: Union[DistanceMatrix, pd.Series],
        sample_ids: List[str],
        metric: str = "unweighted",
    ) -> np.ndarray:
        """Extract distances for a batch of samples from pre-computed distance matrix.

        Args:
            distances: Pre-computed DistanceMatrix (for unweighted) or Series (for faith_pd)
            sample_ids: List of sample IDs for the batch
            metric: Type of metric ("unweighted" or "faith_pd")

        Returns:
            numpy array containing batch-level distances
            - For "unweighted": Shape [batch_size, batch_size]
            - For "faith_pd": Shape [batch_size, 1]

        Raises:
            ValueError: If batch_size is not even (for unweighted UniFrac)
            ValueError: If sample_ids not found in distance matrix/series
            ValueError: If metric is invalid
            TypeError: If distances type doesn't match metric
        """
        if not sample_ids:
            if metric == "unweighted":
                return np.array([]).reshape(0, 0)
            else:
                return np.array([]).reshape(0, 1)

        if metric == "unweighted":
            self.validate_batch_size(len(sample_ids))

        if metric not in ("unweighted", "faith_pd"):
            raise ValueError(f"Invalid metric: {metric}. Must be 'unweighted' or 'faith_pd'")

        if metric == "unweighted":
            if not isinstance(distances, DistanceMatrix):
                raise TypeError(f"Expected DistanceMatrix for unweighted metric, got {type(distances)}")

            missing_ids = set(sample_ids) - set(distances.ids)
            if missing_ids:
                raise ValueError(f"Sample IDs not found in distance matrix: {sorted(missing_ids)}")

            try:
                # Filter the distance matrix to only include samples in this batch
                filtered_distances = distances.filter(sample_ids)
                
                # Reorder the filtered matrix to match the exact order of sample_ids in the batch
                # DistanceMatrix.filter() may not preserve the order of sample_ids, so we need to reorder
                id_to_idx = {id_: idx for idx, id_ in enumerate(filtered_distances.ids)}
                reorder_indices = [id_to_idx[id_] for id_ in sample_ids]
                reordered_data = filtered_distances.data[np.ix_(reorder_indices, reorder_indices)]
                
                return reordered_data
            except Exception as e:
                raise ValueError(f"Error filtering distance matrix for sample IDs: {e}") from e

        else:
            if not isinstance(distances, pd.Series):
                raise TypeError(f"Expected pandas Series for faith_pd metric, got {type(distances)}")

            missing_ids = set(sample_ids) - set(distances.index)
            if missing_ids:
                raise ValueError(f"Sample IDs not found in Faith PD series: {sorted(missing_ids)}")

            try:
                batch_values = distances.loc[sample_ids].to_numpy().reshape(-1, 1)
            except Exception as e:
                raise ValueError(f"Error extracting Faith PD values for sample IDs: {e}") from e

            return batch_values

    def validate_batch_size(self, batch_size: int) -> None:
        """Validate that batch size is even (required for UniFrac computation).

        Args:
            batch_size: Batch size to validate

        Raises:
            ValueError: If batch_size is not even
        """
        if batch_size % 2 != 0:
            raise ValueError(f"Batch size must be even (multiple of 2), got {batch_size}")

    def setup_lazy_computation(
        self,
        table: Table,
        tree_path: str,
        filter_tree: bool = True,
        prune_tree: bool = False,
        pruned_tree_cache: Optional[str] = None,
    ) -> None:
        """Setup for lazy batch-wise distance computation.
        
        This method stores the table and tree path for efficient batch computation.
        The tree is loaded lazily per worker process to avoid memory issues with multiprocessing.
        
        Args:
            table: Rarefied biom.Table object
            tree_path: Path to phylogenetic tree file (.nwk Newick format)
            filter_tree: If True, filter tree to only include ASVs present in the table (much faster)
            prune_tree: If True, pre-prune tree to only ASVs in table (dramatically reduces tree size)
            pruned_tree_cache: Optional path to cache pruned tree (default: {tree_path}.pruned.nwk)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        self._table = table
        self._tree_path = tree_path
        self._filter_tree = filter_tree
        
        tree_path_obj = Path(tree_path)
        if not tree_path_obj.exists():
            raise FileNotFoundError(f"Tree file not found: {tree_path}")
        
        # Get ASV IDs from table for tree filtering
        if filter_tree:
            self._table_asv_ids = set(table.ids(axis="observation"))
            logger.info(f"Will filter tree to {len(self._table_asv_ids)} ASVs present in table")
        else:
            self._table_asv_ids = None
        
        # Handle tree pruning if requested
        if prune_tree:
            logger.info("Tree will be pruned to only include ASVs in table...")
            if pruned_tree_cache is None:
                pruned_tree_cache = str(Path(tree_path).with_suffix('.pruned.nwk'))
            
            # Prune tree NOW (during setup) and save to cache
            # This avoids reloading the full tree in each worker process
            pruned_cache_path_obj = Path(pruned_tree_cache)
            if not pruned_cache_path_obj.exists():
                logger.info("Pruning tree now (this may take a few minutes for large trees)...")
                try:
                    from aam.data.tree_pruner import prune_tree_to_table, get_pruning_stats
                    
                    # Load full tree and get stats
                    full_tree = skbio.read(str(tree_path), format="newick", into=TreeNode)
                    stats = get_pruning_stats(full_tree, table)
                    logger.info(
                        f"Tree pruning: {stats['original_tree_tips']} tips -> {stats['final_tree_tips']} tips "
                        f"({stats['pruned_tips']} removed, {100 * stats['pruned_tips'] / stats['original_tree_tips']:.1f}% reduction)"
                    )
                    
                    # Prune and save
                    pruned_tree = prune_tree_to_table(full_tree, table, output_path=pruned_tree_cache)
                    logger.info(f"Pruned tree saved to: {pruned_tree_cache}")
                except Exception as e:
                    logger.error(f"Error pruning tree: {e}")
                    raise
            else:
                logger.info(f"Using existing pruned tree cache: {pruned_tree_cache}")
                # Get stats from existing pruned tree for logging
                try:
                    from aam.data.tree_pruner import get_pruning_stats
                    full_tree = skbio.read(str(tree_path), format="newick", into=TreeNode)
                    stats = get_pruning_stats(full_tree, table)
                    logger.info(
                        f"Pruned tree cache exists: {stats['original_tree_tips']} tips -> {stats['final_tree_tips']} tips "
                        f"({stats['pruned_tips']} removed, {100 * stats['pruned_tips'] / stats['original_tree_tips']:.1f}% reduction)"
                    )
                except Exception as e:
                    logger.warning(f"Could not compute pruning stats: {e}")
            
            # Store original tree path and pruned cache path
            self._original_tree_path = tree_path
            self._pruned_tree_cache = pruned_tree_cache
            self._table_for_pruning = table
            # Use pruned tree path for lazy loading
            self._tree_path = pruned_tree_cache
        else:
            self._tree_path = tree_path
            self._original_tree_path = None
            self._pruned_tree_cache = None
        
        # Don't load tree here - load it lazily in each worker process to avoid
        # memory issues when using DataLoader with multiple workers
        # Each worker will load the tree once and cache it
        self._tree = None
        self._prune_tree = prune_tree
        self._table_for_pruning = table if prune_tree else None
        logger.info(f"Lazy computation setup: tree will be loaded per worker process from {self._tree_path}")

    def compute_batch_unweighted(
        self,
        sample_ids: List[str],
        table: Optional[Table] = None,
        tree_path: Optional[str] = None,
    ) -> np.ndarray:
        """Compute unweighted UniFrac distances for a batch of samples.
        
        This method computes distances only for the specified samples, which is much
        more efficient than computing the full distance matrix for large datasets.
        Results are cached to avoid recomputation.
        
        Args:
            sample_ids: List of sample IDs to compute distances for
            table: Optional biom.Table (uses cached table if None and setup_lazy_computation was called)
            tree_path: Optional tree path (uses cached tree if None and setup_lazy_computation was called)
        
        Returns:
            numpy array containing pairwise distances [batch_size, batch_size]
        
        Raises:
            ValueError: If batch_size is not even or if table/tree not provided
        """
        if not sample_ids:
            return np.array([]).reshape(0, 0)
        
        self.validate_batch_size(len(sample_ids))
        
        # Check cache first
        cache_key = (frozenset(sample_ids), "unweighted")
        if cache_key in self._batch_cache:
            cached_distances = self._batch_cache[cache_key]
            # Reorder to match sample_ids order
            cached_ids = list(cache_key[0])
            id_to_idx = {id_: idx for idx, id_ in enumerate(cached_ids)}
            reorder_indices = [id_to_idx[id_] for id_ in sample_ids]
            return cached_distances[np.ix_(reorder_indices, reorder_indices)]
        
        # Use cached table/tree if available, otherwise use provided
        if table is None:
            table = self._table
        if tree_path is None:
            # Use pruned tree path if pruning is enabled, otherwise use original
            if self._prune_tree and self._pruned_tree_cache is not None:
                tree_path = self._pruned_tree_cache
            else:
                tree_path = self._tree_path
        
        if table is None or tree_path is None:
            raise ValueError("Must provide table and tree_path, or call setup_lazy_computation() first")
        
        # Load tree lazily (cache per process to avoid reloading)
        if self._tree is None:
            import logging
            logger = logging.getLogger(__name__)
            
            # Load tree (already pruned if pruning was enabled during setup)
            logger.info(f"Loading tree in worker process from {tree_path}...")
            tree_path_obj = Path(tree_path)
            if not tree_path_obj.exists():
                raise FileNotFoundError(f"Tree file not found: {tree_path}")
            self._tree = skbio.read(str(tree_path), format="newick", into=TreeNode)
            logger.info(f"Tree loaded in worker process ({len(list(self._tree.tips()))} tips)")
        
        tree = self._tree
        
        # Filter table to only include samples in batch
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Computing UniFrac distances for batch of {len(sample_ids)} samples...")
        batch_table = table.filter(sample_ids, axis="sample", inplace=False)
        
        try:
            # Compute distances for this batch only
            distance_matrix = unifrac.unweighted(batch_table, tree)
            logger.debug(f"UniFrac computation complete for batch")
            
            # Convert to numpy array and ensure correct order
            batch_ids = list(batch_table.ids(axis="sample"))
            id_to_idx = {id_: idx for idx, id_ in enumerate(batch_ids)}
            reorder_indices = [id_to_idx[id_] for id_ in sample_ids]
            distances = distance_matrix.data[np.ix_(reorder_indices, reorder_indices)]
            
            # Cache the result (with original order)
            if self.cache_size > 0:
                if len(self._batch_cache) >= self.cache_size:
                    # Remove oldest entry (simple FIFO, could use LRU but this is simpler)
                    oldest_key = next(iter(self._batch_cache))
                    del self._batch_cache[oldest_key]
                self._batch_cache[cache_key] = distances
            
            return distances
        except Exception as e:
            if "not found" in str(e).lower() or "mismatch" in str(e).lower():
                raise ValueError(f"ASV IDs don't match between table and tree: {e}") from e
            raise ValueError(f"Error computing unweighted UniFrac for batch: {e}") from e

    def compute_batch_faith_pd(
        self,
        sample_ids: List[str],
        table: Optional[Table] = None,
        tree_path: Optional[str] = None,
    ) -> np.ndarray:
        """Compute Faith PD for a batch of samples.
        
        Args:
            sample_ids: List of sample IDs to compute Faith PD for
            table: Optional biom.Table (uses cached table if None)
            tree_path: Optional tree path (uses cached tree if None)
        
        Returns:
            numpy array containing Faith PD values [batch_size, 1]
        """
        if not sample_ids:
            return np.array([]).reshape(0, 1)
        
        # Check cache first
        cache_key = (frozenset(sample_ids), "faith_pd")
        if cache_key in self._batch_cache:
            cached_values = self._batch_cache[cache_key]
            # Reorder to match sample_ids order
            cached_ids = list(cache_key[0])
            id_to_idx = {id_: idx for idx, id_ in enumerate(cached_ids)}
            reorder_indices = [id_to_idx[id_] for id_ in sample_ids]
            return cached_values[reorder_indices].reshape(-1, 1)
        
        # Use cached table/tree if available
        if table is None:
            table = self._table
        if tree_path is None:
            # Use pruned tree path if pruning is enabled, otherwise use original
            if self._prune_tree and self._pruned_tree_cache is not None:
                tree_path = self._pruned_tree_cache
            else:
                tree_path = self._tree_path
        
        if table is None or tree_path is None:
            raise ValueError("Must provide table and tree_path, or call setup_lazy_computation() first")
        
        # Load tree lazily (cache per process to avoid reloading)
        if self._tree is None:
            import logging
            logger = logging.getLogger(__name__)
            
            # Load tree (already pruned if pruning was enabled during setup)
            logger.info(f"Loading tree in worker process from {tree_path}...")
            tree_path_obj = Path(tree_path)
            if not tree_path_obj.exists():
                raise FileNotFoundError(f"Tree file not found: {tree_path}")
            self._tree = skbio.read(str(tree_path), format="newick", into=TreeNode)
            logger.info(f"Tree loaded in worker process ({len(list(self._tree.tips()))} tips)")
        
        tree = self._tree
        
        # Filter table to only include samples in batch
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Computing Faith PD for batch of {len(sample_ids)} samples...")
        batch_table = table.filter(sample_ids, axis="sample", inplace=False)
        
        try:
            # Compute Faith PD for this batch only
            faith_pd_series = unifrac.faith_pd(batch_table, tree)
            logger.debug(f"Faith PD computation complete for batch")
            
            # Convert to numpy array in correct order
            values = faith_pd_series.loc[sample_ids].to_numpy().reshape(-1, 1)
            
            # Cache the result
            if self.cache_size > 0:
                if len(self._batch_cache) >= self.cache_size:
                    oldest_key = next(iter(self._batch_cache))
                    del self._batch_cache[oldest_key]
                self._batch_cache[cache_key] = values.flatten()
            
            return values
        except Exception as e:
            if "not found" in str(e).lower() or "mismatch" in str(e).lower():
                raise ValueError(f"ASV IDs don't match between table and tree: {e}") from e
            raise ValueError(f"Error computing Faith PD for batch: {e}") from e

    def set_reference_samples(self, reference_sample_ids: List[str], table: Optional[Table] = None) -> None:
        """Set reference samples for stripe-based computation.
        
        Args:
            reference_sample_ids: List of reference sample IDs
            table: Optional biom.Table (uses cached table if None)
        
        Raises:
            ValueError: If reference samples not found in table
        """
        pass

    def compute_unweighted_stripe(
        self,
        table: Table,
        tree_path: str,
        reference_sample_ids: List[str],
        test_sample_ids: Optional[List[str]] = None,
        filter_table: bool = True,
    ) -> np.ndarray:
        """Compute unweighted UniFrac distances in stripe format.
        
        Args:
            table: Rarefied biom.Table object
            tree_path: Path to phylogenetic tree file (.nwk Newick format)
            reference_sample_ids: Reference sample IDs (columns of stripe matrix)
            test_sample_ids: Test sample IDs (rows of stripe matrix). If None, uses all samples.
            filter_table: If True, filter table to only include ASVs present in tree
        
        Returns:
            numpy array containing stripe distances [N_test_samples, N_reference_samples]
        
        Raises:
            FileNotFoundError: If tree file doesn't exist
            ValueError: If ASV IDs don't match between table and tree
            ValueError: If reference or test sample IDs not found in table
        """
        pass

    def compute_batch_unweighted_stripe(
        self,
        sample_ids: List[str],
        reference_sample_ids: Optional[List[str]] = None,
        table: Optional[Table] = None,
        tree_path: Optional[str] = None,
    ) -> np.ndarray:
        """Compute unweighted UniFrac distances in stripe format for a batch of samples.
        
        This method computes distances from batch samples to reference samples using
        unifrac.unweighted_dense_pair for efficiency.
        
        Args:
            sample_ids: List of test sample IDs to compute distances for (rows)
            reference_sample_ids: List of reference sample IDs (columns). 
                                 Uses cached reference set if None.
            table: Optional biom.Table (uses cached table if None)
            tree_path: Optional tree path (uses cached tree if None)
        
        Returns:
            numpy array containing stripe distances [batch_size, N_reference_samples]
        
        Raises:
            ValueError: If table/tree not provided or reference samples not set
        """
        pass

    def extract_batch_stripe_distances(
        self,
        stripe_distances: np.ndarray,
        sample_ids: List[str],
        reference_sample_ids: List[str],
        all_sample_ids: List[str],
    ) -> np.ndarray:
        """Extract stripe distances for a batch of samples from pre-computed stripe matrix.
        
        Args:
            stripe_distances: Pre-computed stripe matrix [N_all_samples, N_reference_samples]
            sample_ids: List of sample IDs for the batch (rows to extract)
            reference_sample_ids: List of reference sample IDs (columns, should match stripe_distances)
            all_sample_ids: List of all sample IDs (rows) matching stripe_distances
        
        Returns:
            numpy array containing batch stripe distances [batch_size, N_reference_samples]
        
        Raises:
            ValueError: If sample_ids or reference_sample_ids not found in all_sample_ids
            ValueError: If shapes don't match
        """
        pass
