"""Load pre-computed UniFrac distance matrices for training.

This module replaces the computation logic with simple loading functionality.
Users should generate UniFrac matrices using unifrac-binaries or other tools,
then load them for training.
"""

from typing import List, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
from skbio import DistanceMatrix


class UniFracLoader:
    """Load and validate pre-computed UniFrac distance matrices.
    
    This class provides functionality to:
    - Load pre-computed distance matrices from disk (.npy, .h5, .csv formats)
    - Validate matrix dimensions match sample IDs
    - Extract batch-level distances from pre-computed matrices
    - Support both pairwise (N×N) and stripe (N×M) matrix formats
    """
    
    def __init__(self):
        """Initialize UniFracLoader."""
        pass
    
    def load_matrix(
        self,
        matrix_path: str,
        sample_ids: Optional[List[str]] = None,
        matrix_format: Optional[str] = None,
    ) -> Union[np.ndarray, DistanceMatrix, pd.Series]:
        """Load pre-computed UniFrac distance matrix from disk.
        
        Args:
            matrix_path: Path to pre-computed matrix file
            sample_ids: Optional list of sample IDs to validate against
            matrix_format: Optional format hint ('pairwise', 'stripe', 'faith_pd')
                          If None, inferred from file extension and shape
        
        Returns:
            Loaded matrix as numpy array, DistanceMatrix, or pandas Series
            - Pairwise: numpy array [N_samples, N_samples] or DistanceMatrix
            - Stripe: numpy array [N_test_samples, N_reference_samples]
            - Faith PD: pandas Series [N_samples] or numpy array [N_samples]
        
        Raises:
            FileNotFoundError: If matrix file doesn't exist
            ValueError: If matrix format is invalid or dimensions don't match
        """
        pass
    
    def extract_batch_distances(
        self,
        distances: Union[DistanceMatrix, pd.Series, np.ndarray],
        sample_ids: List[str],
        metric: str = "unweighted",
    ) -> np.ndarray:
        """Extract distances for a batch of samples from pre-computed distance matrix.
        
        Args:
            distances: Pre-computed DistanceMatrix (for unweighted) or Series (for faith_pd) or numpy array
            sample_ids: List of sample IDs for the batch
            metric: Type of metric ("unweighted" or "faith_pd")
        
        Returns:
            numpy array containing batch-level distances
            - For "unweighted": Shape [batch_size, batch_size]
            - For "faith_pd": Shape [batch_size, 1]
        
        Raises:
            ValueError: If sample_ids not found in distance matrix/series
            ValueError: If metric is invalid
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
    
    def validate_matrix_dimensions(
        self,
        matrix: Union[np.ndarray, DistanceMatrix, pd.Series],
        sample_ids: List[str],
        metric: str = "unweighted",
    ) -> None:
        """Validate that matrix dimensions match sample IDs.
        
        Args:
            matrix: Distance matrix to validate
            sample_ids: List of sample IDs to validate against
            metric: Type of metric ("unweighted", "faith_pd", or "stripe")
        
        Raises:
            ValueError: If dimensions don't match
        """
        pass
