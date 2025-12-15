"""Load pre-computed UniFrac distance matrices for training.

This module replaces the computation logic with simple loading functionality.
Users should generate UniFrac matrices using unifrac-binaries or other tools,
then load them for training.
"""

from typing import List, Optional, Union, Dict
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from skbio import DistanceMatrix

logger = logging.getLogger(__name__)


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
    ) -> Union[np.ndarray, DistanceMatrix, pd.Series, Tuple[Union[np.ndarray, DistanceMatrix, pd.Series], List[str]]]:
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
        matrix_path_obj = Path(matrix_path)
        if not matrix_path_obj.exists():
            raise FileNotFoundError(f"Matrix file not found: {matrix_path}")
        
        file_ext = matrix_path_obj.suffix.lower()
        
        if file_ext == '.npy' or file_ext == '.npz':
            return self._load_npy(matrix_path, sample_ids, matrix_format)
        elif file_ext == '.h5' or file_ext == '.hdf5':
            return self._load_h5(matrix_path, sample_ids, matrix_format)
        elif file_ext == '.csv':
            return self._load_csv(matrix_path, sample_ids, matrix_format)
        else:
            raise ValueError(f"Unsupported matrix file format: {file_ext}. Supported: .npy, .npz, .h5, .csv")
    
    def _load_npy(
        self,
        matrix_path: str,
        sample_ids: Optional[List[str]],
        matrix_format: Optional[str],
    ) -> Union[np.ndarray, DistanceMatrix, pd.Series]:
        """Load matrix from .npy file."""
        data = np.load(matrix_path, allow_pickle=True)
        
        if isinstance(data, np.ndarray):
            matrix = data
        elif isinstance(data, np.lib.npyio.NpzFile):
            if 'distances' in data:
                matrix = data['distances']
            elif 'stripe_distances' in data:
                matrix = data['stripe_distances']
            elif 'faith_pd' in data:
                matrix = data['faith_pd']
            else:
                keys = list(data.keys())
                if len(keys) == 1:
                    matrix = data[keys[0]]
                else:
                    raise ValueError(f"Multiple arrays in .npy file. Expected one of: distances, stripe_distances, faith_pd. Found: {keys}")
        else:
            raise ValueError(f"Unexpected data type in .npy file: {type(data)}")
        
        if sample_ids is not None:
            self.validate_matrix_dimensions(matrix, sample_ids, matrix_format or "unweighted")
        
        if matrix_format == "faith_pd" or (matrix.ndim == 1 and matrix_format is None):
            # Handle both 1D and 2D arrays (2D with shape [N, 1])
            if matrix.ndim == 2 and matrix.shape[1] == 1:
                matrix = matrix.flatten()
            return pd.Series(matrix, index=sample_ids) if sample_ids is not None else pd.Series(matrix)
        elif matrix_format == "pairwise" or (matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and matrix_format is None):
            if sample_ids is not None:
                return DistanceMatrix(matrix, ids=sample_ids)
            return matrix
        else:
            return matrix
    
    def _load_h5(
        self,
        matrix_path: str,
        sample_ids: Optional[List[str]],
        matrix_format: Optional[str],
    ) -> Union[np.ndarray, DistanceMatrix, pd.Series]:
        """Load matrix from HDF5 file.
        
        Supports multiple HDF5 formats:
        - AAM format: 'distances', 'stripe_distances', or 'faith_pd' keys
        - unifrac-binaries (ssu) format: 'matrix' key (distance matrix)
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required to load .h5 files. Install with: pip install h5py")
        
        with h5py.File(matrix_path, 'r') as f:
            # Try AAM format keys first
            if 'distances' in f:
                matrix = np.array(f['distances'])
            elif 'stripe_distances' in f:
                matrix = np.array(f['stripe_distances'])
            elif 'faith_pd' in f:
                matrix = np.array(f['faith_pd'])
            # Try unifrac-binaries (ssu) format
            elif 'matrix' in f:
                matrix = np.array(f['matrix'])
                # ssu stores sample IDs in 'order' key - use for validation/reordering if needed
                h5_sample_ids = None
                if 'order' in f:
                    order_data = f['order']
                    # Handle both string and bytes arrays
                    if order_data.dtype.kind == 'S':  # String/bytes
                        h5_sample_ids = [sid.decode('utf-8') for sid in order_data]
                    else:
                        h5_sample_ids = [str(sid) for sid in order_data]
                    
                    # If sample_ids provided, validate and potentially reorder matrix
                    if sample_ids is not None:
                        h5_set = set(h5_sample_ids)
                        sample_set = set(sample_ids)
                        
                        if h5_set != sample_set:
                            missing = sample_set - h5_set
                            extra = h5_set - sample_set
                            
                            if missing:
                                logger.warning(
                                    f"HDF5 file missing {len(missing)} samples from BIOM table. "
                                    f"Will use only samples present in both. "
                                    f"Missing: {sorted(list(missing))[:5]}{'...' if len(missing) > 5 else ''}"
                                )
                            
                            if extra:
                                logger.warning(
                                    f"HDF5 file has {len(extra)} extra samples not in BIOM table. "
                                    f"Will use only samples present in both."
                                )
                            
                            # Use intersection of sample IDs
                            common_ids = sorted(list(h5_set & sample_set))
                            if not common_ids:
                                raise ValueError(
                                    f"No common sample IDs between HDF5 file and BIOM table. "
                                    f"HDF5 has {len(h5_set)} samples, BIOM has {len(sample_set)} samples."
                                )
                            
                            logger.info(
                                f"Using {len(common_ids)} samples present in both HDF5 and BIOM table "
                                f"(out of {len(sample_set)} BIOM samples, {len(h5_set)} HDF5 samples)"
                            )
                            
                            # Filter and reorder matrix to match common_ids
                            h5_id_to_idx = {sid: idx for idx, sid in enumerate(h5_sample_ids)}
                            reorder_indices = [h5_id_to_idx[sid] for sid in common_ids]
                            matrix = matrix[np.ix_(reorder_indices, reorder_indices)]
                            
                            # Update h5_sample_ids to reflect filtered/reordered matrix
                            h5_sample_ids = common_ids
                        else:
                            # Reorder matrix to match sample_ids order if needed
                            if h5_sample_ids != sample_ids:
                                id_to_idx = {sid: idx for idx, sid in enumerate(h5_sample_ids)}
                                reorder_indices = [id_to_idx[sid] for sid in sample_ids]
                                matrix = matrix[np.ix_(reorder_indices, reorder_indices)]
                                logger.info(f"Reordered HDF5 matrix to match provided sample IDs")
                                h5_sample_ids = sample_ids
            
            # Store the actual sample IDs used (for later retrieval if needed)
            # This will be used when creating DistanceMatrix
            if h5_sample_ids is not None:
                # Store as attribute for later access
                matrix._aam_sample_ids = h5_sample_ids
            else:
                keys = list(f.keys())
                if len(keys) == 1:
                    matrix = np.array(f[keys[0]])
                else:
                    raise ValueError(
                        f"Multiple datasets in HDF5 file. Expected one of: distances, stripe_distances, "
                        f"faith_pd, or matrix (unifrac-binaries format). Found: {keys}"
                    )
            
        if sample_ids is not None:
            self.validate_matrix_dimensions(matrix, sample_ids, matrix_format or "unweighted")
        
        if matrix_format == "faith_pd" or (matrix.ndim == 1 and matrix_format is None):
            # Handle both 1D and 2D arrays (2D with shape [N, 1])
            if matrix.ndim == 2 and matrix.shape[1] == 1:
                matrix = matrix.flatten()
            return pd.Series(matrix, index=sample_ids) if sample_ids is not None else pd.Series(matrix)
        elif matrix_format == "pairwise" or (matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and matrix_format is None):
            if sample_ids is not None:
                return DistanceMatrix(matrix, ids=sample_ids)
            return matrix
        else:
            return matrix
    
    def _load_csv(
        self,
        matrix_path: str,
        sample_ids: Optional[List[str]],
        matrix_format: Optional[str],
    ) -> Union[np.ndarray, DistanceMatrix, pd.Series]:
        """Load matrix from CSV file."""
        df = pd.read_csv(matrix_path, index_col=0)
        matrix = df.values
        
        csv_sample_ids = list(df.index)
        
        if sample_ids is not None:
            if set(csv_sample_ids) != set(sample_ids):
                missing = set(sample_ids) - set(csv_sample_ids)
                extra = set(csv_sample_ids) - set(sample_ids)
                if missing:
                    logger.warning(f"CSV matrix missing {len(missing)} samples from expected IDs")
                if extra:
                    logger.warning(f"CSV matrix has {len(extra)} extra samples not in expected IDs")
            
            reorder_indices = [csv_sample_ids.index(sid) for sid in sample_ids if sid in csv_sample_ids]
            if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
                matrix = matrix[np.ix_(reorder_indices, reorder_indices)]
            elif matrix.ndim == 2 and matrix.shape[1] == 1:
                # Faith PD as column vector
                matrix = matrix[reorder_indices, 0]
            else:
                matrix = matrix[reorder_indices]
        
        if matrix_format == "faith_pd" or (matrix.ndim == 1 and matrix_format is None):
            # Handle both 1D and 2D arrays (2D with shape [N, 1])
            if matrix.ndim == 2 and matrix.shape[1] == 1:
                matrix = matrix.flatten()
            return pd.Series(matrix, index=sample_ids if sample_ids else csv_sample_ids)
        elif matrix_format == "pairwise" or (matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and matrix_format is None):
            ids = sample_ids if sample_ids else csv_sample_ids
            return DistanceMatrix(matrix, ids=ids)
        else:
            return matrix
    
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
        if not sample_ids:
            if metric == "unweighted":
                return np.array([]).reshape(0, 0)
            else:
                return np.array([]).reshape(0, 1)
        
        if metric not in ("unweighted", "faith_pd"):
            raise ValueError(f"Invalid metric: {metric}. Must be 'unweighted' or 'faith_pd'")
        
        if metric == "unweighted":
            if isinstance(distances, DistanceMatrix):
                missing_ids = set(sample_ids) - set(distances.ids)
                if missing_ids:
                    raise ValueError(f"Sample IDs not found in distance matrix: {sorted(missing_ids)}")
                
                filtered_distances = distances.filter(sample_ids)
                id_to_idx = {id_: idx for idx, id_ in enumerate(filtered_distances.ids)}
                reorder_indices = [id_to_idx[id_] for id_ in sample_ids]
                reordered_data = filtered_distances.data[np.ix_(reorder_indices, reorder_indices)]
                return reordered_data
            elif isinstance(distances, np.ndarray):
                if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
                    raise ValueError(f"Expected square matrix for unweighted metric, got shape {distances.shape}")
                return distances
            else:
                raise TypeError(f"Expected DistanceMatrix or numpy array for unweighted metric, got {type(distances)}")
        else:
            if isinstance(distances, pd.Series):
                missing_ids = set(sample_ids) - set(distances.index)
                if missing_ids:
                    raise ValueError(f"Sample IDs not found in Faith PD series: {sorted(missing_ids)}")
                batch_values = distances.loc[sample_ids].to_numpy().reshape(-1, 1)
                return batch_values
            elif isinstance(distances, np.ndarray):
                if distances.ndim == 1:
                    return distances.reshape(-1, 1)
                elif distances.ndim == 2 and distances.shape[1] == 1:
                    return distances
                else:
                    raise ValueError(f"Expected 1D array or column vector for faith_pd metric, got shape {distances.shape}")
            else:
                raise TypeError(f"Expected pandas Series or numpy array for faith_pd metric, got {type(distances)}")
    
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
        if not sample_ids:
            return np.array([]).reshape(0, len(reference_sample_ids) if reference_sample_ids else 0)
        
        expected_rows = len(all_sample_ids)
        expected_cols = len(reference_sample_ids)
        
        if stripe_distances.shape[0] != expected_rows:
            raise ValueError(
                f"Stripe matrix row count ({stripe_distances.shape[0]}) doesn't match "
                f"all_sample_ids length ({expected_rows})"
            )
        if stripe_distances.shape[1] != expected_cols:
            raise ValueError(
                f"Stripe matrix column count ({stripe_distances.shape[1]}) doesn't match "
                f"reference_sample_ids length ({expected_cols})"
            )
        
        all_sample_ids_set = set(all_sample_ids)
        missing_test = set(sample_ids) - all_sample_ids_set
        
        if missing_test:
            raise ValueError(f"Test sample IDs not found in all_sample_ids: {sorted(missing_test)}")
        # Note: reference_sample_ids don't need to be in all_sample_ids - they're separate
        
        try:
            batch_indices = [all_sample_ids.index(sid) for sid in sample_ids]
            extracted = stripe_distances[batch_indices, :]
            return extracted
        except Exception as e:
            raise ValueError(f"Error extracting batch stripe distances: {e}") from e
    
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
        n_samples = len(sample_ids)
        
        if isinstance(matrix, DistanceMatrix):
            if len(matrix.ids) != n_samples:
                raise ValueError(
                    f"DistanceMatrix has {len(matrix.ids)} samples, but expected {n_samples}"
                )
            if matrix.shape[0] != n_samples or matrix.shape[1] != n_samples:
                raise ValueError(
                    f"DistanceMatrix shape {matrix.shape} doesn't match expected ({n_samples}, {n_samples})"
                )
        elif isinstance(matrix, pd.Series):
            if len(matrix) != n_samples:
                raise ValueError(
                    f"Faith PD Series has {len(matrix)} samples, but expected {n_samples}"
                )
        elif isinstance(matrix, np.ndarray):
            if metric == "faith_pd":
                if matrix.ndim == 1:
                    if len(matrix) != n_samples:
                        raise ValueError(
                            f"Faith PD array has {len(matrix)} samples, but expected {n_samples}"
                        )
                elif matrix.ndim == 2 and matrix.shape[1] == 1:
                    if matrix.shape[0] != n_samples:
                        raise ValueError(
                            f"Faith PD array has {matrix.shape[0]} samples, but expected {n_samples}"
                        )
                else:
                    raise ValueError(
                        f"Faith PD array has unexpected shape {matrix.shape}, expected ({n_samples},) or ({n_samples}, 1)"
                    )
            elif metric == "unweighted":
                if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                    raise ValueError(
                        f"Unweighted UniFrac matrix must be square, got shape {matrix.shape}"
                    )
                if matrix.shape[0] != n_samples:
                    raise ValueError(
                        f"Unweighted UniFrac matrix has {matrix.shape[0]} samples, but expected {n_samples}"
                    )
            elif metric == "stripe":
                if matrix.ndim != 2:
                    raise ValueError(
                        f"Stripe matrix must be 2D, got shape {matrix.shape}"
                    )
                if matrix.shape[0] != n_samples:
                    raise ValueError(
                        f"Stripe matrix has {matrix.shape[0]} rows, but expected {n_samples}"
                    )
        else:
            raise TypeError(f"Unsupported matrix type: {type(matrix)}")
