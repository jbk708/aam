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

# Import tree pruner (avoid circular import)
try:
    from aam.data.tree_pruner import load_or_prune_tree, get_pruning_stats
except ImportError:
    # During initial stub phase, module may not exist yet
    load_or_prune_tree = None
    get_pruning_stats = None


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

    def compute_unweighted(self, table: Table, tree_path: str) -> DistanceMatrix:
        """Compute unweighted UniFrac distances between samples.

        Args:
            table: Rarefied biom.Table object
            tree_path: Path to phylogenetic tree file (.nwk Newick format)

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

        try:
            distance_matrix = unifrac.unweighted(table, tree)
        except Exception as e:
            if "not found" in str(e).lower() or "mismatch" in str(e).lower():
                raise ValueError(f"ASV IDs don't match between table and tree: {e}") from e
            raise ValueError(f"Error computing unweighted UniFrac: {e}") from e

        return distance_matrix

    def compute_faith_pd(self, table: Table, tree_path: str) -> Union[DistanceMatrix, pd.Series]:
        """Compute Faith's Phylogenetic Diversity (Faith PD) per sample.

        Args:
            table: Rarefied biom.Table object
            tree_path: Path to phylogenetic tree file (.nwk Newick format)

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

        try:
            faith_pd_series = unifrac.faith_pd(table, tree)
        except Exception as e:
            if "not found" in str(e).lower() or "mismatch" in str(e).lower():
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
            if load_or_prune_tree is None:
                raise ImportError("Tree pruning requires aam.data.tree_pruner module")
            logger.info("Tree will be pruned to only include ASVs in table...")
            if pruned_tree_cache is None:
                pruned_tree_cache = str(Path(tree_path).with_suffix('.pruned.nwk'))
            
            # Get pruning stats before pruning (load tree temporarily)
            if get_pruning_stats is not None:
                try:
                    full_tree = skbio.read(str(tree_path), format="newick", into=TreeNode)
                    stats = get_pruning_stats(full_tree, table)
                    logger.info(
                        f"Tree pruning: {stats['original_tree_tips']} tips -> {stats['final_tree_tips']} tips "
                        f"({stats['pruned_tips']} removed, {100 * stats['pruned_tips'] / stats['original_tree_tips']:.1f}% reduction)"
                    )
                except Exception as e:
                    logger.warning(f"Could not compute pruning stats: {e}")
            
            # Store original tree path and pruned cache path
            self._original_tree_path = tree_path
            self._pruned_tree_cache = pruned_tree_cache
            self._table_for_pruning = table
            logger.info(f"Pruned tree will be cached to: {pruned_tree_cache}")
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
            
            # Use pruned tree if available, otherwise load and prune if needed
            if self._prune_tree and load_or_prune_tree is not None and self._original_tree_path is not None:
                logger.info(f"Loading/pruning tree in worker process...")
                self._tree = load_or_prune_tree(
                    self._original_tree_path,
                    self._table_for_pruning if self._table_for_pruning is not None else table,
                    pruned_tree_path=self._pruned_tree_cache,
                )
            else:
                logger.info(f"Loading tree in worker process from {tree_path} (this may take a few minutes for large trees)...")
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
            
            # Use pruned tree if available, otherwise load and prune if needed
            if self._prune_tree and load_or_prune_tree is not None and self._original_tree_path is not None:
                logger.info(f"Loading/pruning tree in worker process...")
                self._tree = load_or_prune_tree(
                    self._original_tree_path,
                    self._table_for_pruning if self._table_for_pruning is not None else table,
                    pruned_tree_path=self._pruned_tree_cache,
                )
            else:
                logger.info(f"Loading tree in worker process from {tree_path} (this may take a few minutes for large trees)...")
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
