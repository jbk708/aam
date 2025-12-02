"""UniFrac distance computation for microbial sequencing data."""

from typing import List, Optional, Union
import tempfile
import os
from pathlib import Path

import biom
from biom import Table
import skbio
from skbio import DistanceMatrix, TreeNode
import pandas as pd
import numpy as np
import unifrac


class UniFracComputer:
    """Compute phylogenetic distances (UniFrac) from BIOM tables and phylogenetic trees.

    This class provides functionality to:
    - Compute unweighted UniFrac distances between samples
    - Compute Faith's Phylogenetic Diversity (Faith PD) per sample
    - Extract batch-level distances from pre-computed distance matrices
    """

    def __init__(self):
        """Initialize UniFracComputer."""
        pass

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
                filtered_distances = distances.filter(sample_ids)
            except Exception as e:
                raise ValueError(f"Error filtering distance matrix for sample IDs: {e}") from e

            return filtered_distances.data

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
