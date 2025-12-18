"""PyTorch Dataset for microbial sequencing data."""

from typing import Dict, Optional, List, Union
import torch
from torch.utils.data import Dataset
import biom
from biom import Table
import pandas as pd
import numpy as np
from skbio import DistanceMatrix

from aam.data.tokenizer import SequenceTokenizer
from aam.data.biom_loader import BIOMLoader
from aam.data.unifrac_loader import UniFracLoader


def collate_fn(
    batch: List[Dict[str, Union[torch.Tensor, str]]],
    token_limit: int,
    unifrac_distances: Optional[Union[DistanceMatrix, pd.Series, np.ndarray]] = None,
    unifrac_metric: str = "unweighted",
    unifrac_loader: Optional["UniFracLoader"] = None,
    lazy_unifrac: bool = False,
    stripe_mode: bool = False,
    reference_sample_ids: Optional[List[str]] = None,
    all_sample_ids: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching samples with variable ASV counts.

    Args:
        batch: List of sample dictionaries
        token_limit: Maximum number of ASVs per sample
        unifrac_distances: Optional full distance matrix/series/stripe matrix for batch extraction
        unifrac_metric: Type of UniFrac metric ("unweighted" or "faith_pd")
        unifrac_loader: UniFracLoader instance for batch extraction (optional, created if None)
        lazy_unifrac: If True, compute distances on-the-fly (deprecated, should be False)
        stripe_mode: If True, use stripe-based distances instead of pairwise (deprecated, should be False)
        reference_sample_ids: List of reference sample IDs for stripe mode (deprecated)
        all_sample_ids: List of all sample IDs for stripe extraction (deprecated)

    Returns:
        Batched dictionary with padded tensors
    """
    batch_size = len(batch)
    max_bp = batch[0]["tokens"].shape[1]

    tokens_list = []
    counts_list = []
    sample_ids = []
    y_targets = []

    for sample in batch:
        tokens = sample["tokens"]
        counts = sample["counts"]
        num_asvs = tokens.shape[0]

        if num_asvs > token_limit:
            tokens = tokens[:token_limit]
            counts = counts[:token_limit]
            num_asvs = token_limit

        # Verify sample has at least one ASV with count > 0
        # This prevents all-padding samples that cause NaN in attention pooling
        if num_asvs == 0 or (counts.sum() == 0).all():
            import sys
            error_msg = (
                f"Sample {sample['sample_id']} has no ASVs with count > 0 "
                f"(num_asvs={num_asvs}, counts_sum={counts.sum().item()})"
            )
            print(f"ERROR: {error_msg}", file=sys.stderr, flush=True)
            raise ValueError(error_msg)

        padded_tokens = torch.zeros(token_limit, max_bp, dtype=torch.long)
        padded_counts = torch.zeros(token_limit, 1, dtype=torch.float)
        padded_tokens[:num_asvs] = tokens
        padded_counts[:num_asvs] = counts

        tokens_list.append(padded_tokens)
        counts_list.append(padded_counts)
        sample_ids.append(sample["sample_id"])

        if "y_target" in sample:
            y_targets.append(sample["y_target"])

    result = {
        "tokens": torch.stack(tokens_list),
        "counts": torch.stack(counts_list),
        "sample_ids": sample_ids,
    }

    if y_targets:
        result["y_target"] = torch.stack(y_targets)

    if unifrac_distances is not None:
        # Extract batch distances from pre-computed matrix
        if unifrac_loader is None:
            unifrac_loader = UniFracLoader()
        batch_distances = unifrac_loader.extract_batch_distances(unifrac_distances, sample_ids, metric=unifrac_metric)
        
        # Validate extracted distances
        if np.any(np.isnan(batch_distances)):
            import sys
            error_msg = f"NaN values found in extracted batch distances for sample_ids: {sample_ids}"
            print(f"ERROR: {error_msg}", file=sys.stderr, flush=True)
            print(f"batch_distances shape={batch_distances.shape}, min={np.nanmin(batch_distances)}, max={np.nanmax(batch_distances)}", file=sys.stderr, flush=True)
            raise ValueError(error_msg)
        if np.any(np.isinf(batch_distances)):
            import sys
            error_msg = f"Inf values found in extracted batch distances for sample_ids: {sample_ids}"
            print(f"ERROR: {error_msg}", file=sys.stderr, flush=True)
            raise ValueError(error_msg)
        
        result["unifrac_target"] = torch.FloatTensor(batch_distances)

    return result


class ASVDataset(Dataset):
    """PyTorch Dataset for microbial sequencing data."""

    def __init__(
        self,
        table: Table,
        metadata: Optional[pd.DataFrame] = None,
        unifrac_distances: Optional[Union[DistanceMatrix, pd.Series, np.ndarray]] = None,
        tokenizer: Optional[SequenceTokenizer] = None,
        max_bp: int = 150,
        token_limit: int = 1024,
        target_column: Optional[str] = None,
        unifrac_metric: str = "unweighted",
        lazy_unifrac: bool = True,
        unifrac_computer: Optional[UniFracLoader] = None,
        stripe_mode: bool = True,
        reference_sample_ids: Optional[List[str]] = None,
        normalize_targets: bool = False,
        target_min: Optional[float] = None,
        target_max: Optional[float] = None,
        normalize_counts: bool = False,
        count_min: Optional[float] = None,
        count_max: Optional[float] = None,
        cache_sequences: bool = True,
    ):
        """Initialize ASVDataset.

        Args:
            table: Rarefied biom.Table object
            metadata: Optional DataFrame with sample metadata and targets
            unifrac_distances: Optional pre-computed UniFrac distances/stripe matrix (ignored if lazy_unifrac=True)
            tokenizer: SequenceTokenizer instance (creates default if None)
            max_bp: Maximum base pairs per sequence
            token_limit: Maximum ASVs per sample
            target_column: Column name in metadata for target values
            unifrac_metric: Type of UniFrac metric ("unweighted" or "faith_pd")
            lazy_unifrac: If True, compute distances on-the-fly instead of using pre-computed distances (default: True)
            unifrac_computer: UniFracLoader instance (deprecated, not used)
            stripe_mode: If True, use stripe-based distances instead of pairwise (default: True)
            reference_sample_ids: List of reference sample IDs for stripe mode (required if stripe_mode=True)
            normalize_targets: If True, normalize target values to [0, 1] range (default: False)
            target_min: Minimum target value for normalization (computed from data if None)
            target_max: Maximum target value for normalization (computed from data if None)
            normalize_counts: If True, normalize count values to [0, 1] range (default: False)
            count_min: Minimum count value for normalization (computed from data if None)
            count_max: Maximum count value for normalization (computed from data if None)
            cache_sequences: If True, cache tokenized sequences at init for faster __getitem__ (default: True)
        """
        self.table = table
        self.metadata = metadata
        self.unifrac_distances = unifrac_distances
        self.tokenizer = tokenizer if tokenizer is not None else SequenceTokenizer()
        self.max_bp = max_bp
        self.token_limit = token_limit
        self.target_column = target_column
        self.unifrac_metric = unifrac_metric
        self.lazy_unifrac = lazy_unifrac
        # unifrac_computer is deprecated, kept for backward compatibility
        self.unifrac_computer = unifrac_computer
        self.stripe_mode = stripe_mode
        self.reference_sample_ids = reference_sample_ids

        self.sample_ids = list(table.ids(axis="sample"))
        self.observation_ids = list(table.ids(axis="observation"))
        
        # Validate stripe mode parameters
        if stripe_mode and reference_sample_ids is None:
            # Auto-select reference samples if not provided (randomly pick 100 or all if < 100)
            import random
            if len(self.sample_ids) <= 100:
                self.reference_sample_ids = self.sample_ids.copy()
            else:
                self.reference_sample_ids = random.sample(self.sample_ids, 100)
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Auto-selected {len(self.reference_sample_ids)} random reference samples for stripe mode")
        
        if stripe_mode and reference_sample_ids is not None:
            # Validate reference samples exist in table
            # Skip validation if using lazy_unifrac (deprecated, always validate now)
            if not lazy_unifrac:
                table_sample_ids = set(self.sample_ids)
                missing_ref = set(reference_sample_ids) - table_sample_ids
                if missing_ref:
                    raise ValueError(f"Reference sample IDs not found in table: {sorted(missing_ref)}")

        loader = BIOMLoader()
        self.sequences = loader.get_sequences(table)

        self.sequence_to_idx = {seq: idx for idx, seq in enumerate(self.observation_ids)}

        # Sequence cache: pre-tokenize and pad all sequences at init
        self.cache_sequences = cache_sequences
        self._sequence_cache: Optional[Dict[str, torch.Tensor]] = None
        if cache_sequences:
            self._sequence_cache = {}
            for seq in self.sequences:
                tokenized = self.tokenizer.tokenize(seq)
                padded = self.tokenizer.pad_sequences([tokenized], self.max_bp + 1)[0]
                self._sequence_cache[seq] = padded

        # Target normalization settings
        self.normalize_targets = normalize_targets
        self.target_min = target_min
        self.target_max = target_max
        self.target_scale = None  # Computed as (target_max - target_min)

        # Count normalization settings
        self.normalize_counts = normalize_counts
        self.count_min = count_min
        self.count_max = count_max
        self.count_scale = None  # Computed as (count_max - count_min)

        # Compute count normalization parameters from BIOM table if normalizing
        if normalize_counts:
            # Get all non-zero counts from the table
            matrix_data = table.matrix_data.toarray()
            all_counts = matrix_data[matrix_data > 0]
            if len(all_counts) > 0:
                if self.count_min is None:
                    self.count_min = float(all_counts.min())
                if self.count_max is None:
                    self.count_max = float(all_counts.max())
                self.count_scale = self.count_max - self.count_min
                # Avoid division by zero if all counts are the same
                if self.count_scale == 0:
                    self.count_scale = 1.0
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"All count values are identical ({self.count_min}). "
                        f"Count normalization will have no effect."
                    )

        if metadata is not None and target_column is not None:
            metadata.columns = metadata.columns.str.strip()
            if "sample_id" not in metadata.columns:
                found_columns = list(metadata.columns)
                raise ValueError(
                    f"metadata must have 'sample_id' column.\n"
                    f"Found columns: {found_columns}\n"
                    f"Expected: 'sample_id'\n"
                    f"Tip: Check for whitespace or encoding issues in column names."
                )
            self.metadata_dict = metadata.set_index("sample_id")[target_column].to_dict()

            # Compute normalization parameters if normalizing targets
            if normalize_targets:
                target_values = [float(v) for v in self.metadata_dict.values() if v is not None]
                if target_values:
                    if self.target_min is None:
                        self.target_min = float(min(target_values))
                    if self.target_max is None:
                        self.target_max = float(max(target_values))
                    self.target_scale = self.target_max - self.target_min
                    # Avoid division by zero if all targets are the same
                    if self.target_scale == 0:
                        self.target_scale = 1.0
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(
                            f"All target values are identical ({self.target_min}). "
                            f"Target normalization will have no effect."
                        )
        else:
            self.metadata_dict = None

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.sample_ids)

    def get_normalization_params(self) -> Optional[Dict[str, float]]:
        """Get target normalization parameters.

        Returns:
            Dictionary with 'target_min', 'target_max', 'target_scale' if normalization
            is enabled, None otherwise.
        """
        if self.normalize_targets and self.target_scale is not None:
            return {
                "target_min": self.target_min,
                "target_max": self.target_max,
                "target_scale": self.target_scale,
            }
        return None

    def denormalize_targets(self, normalized_values: Union[torch.Tensor, np.ndarray, float]) -> Union[torch.Tensor, np.ndarray, float]:
        """Denormalize target values back to original scale.

        Args:
            normalized_values: Normalized values in [0, 1] range

        Returns:
            Denormalized values in original target range
        """
        if not self.normalize_targets or self.target_scale is None:
            return normalized_values

        if isinstance(normalized_values, torch.Tensor):
            return normalized_values * self.target_scale + self.target_min
        elif isinstance(normalized_values, np.ndarray):
            return normalized_values * self.target_scale + self.target_min
        else:
            return float(normalized_values) * self.target_scale + self.target_min

    def get_count_normalization_params(self) -> Optional[Dict[str, float]]:
        """Get count normalization parameters.

        Returns:
            Dictionary with 'count_min', 'count_max', 'count_scale' if normalization
            is enabled, None otherwise.
        """
        if self.normalize_counts and self.count_scale is not None:
            return {
                "count_min": self.count_min,
                "count_max": self.count_max,
                "count_scale": self.count_scale,
            }
        return None

    def denormalize_counts(self, normalized_values: Union[torch.Tensor, np.ndarray, float]) -> Union[torch.Tensor, np.ndarray, float]:
        """Denormalize count values back to original scale.

        Args:
            normalized_values: Normalized values in [0, 1] range

        Returns:
            Denormalized values in original count range
        """
        if not self.normalize_counts or self.count_scale is None:
            return normalized_values

        if isinstance(normalized_values, torch.Tensor):
            return normalized_values * self.count_scale + self.count_min
        elif isinstance(normalized_values, np.ndarray):
            return normalized_values * self.count_scale + self.count_min
        else:
            return float(normalized_values) * self.count_scale + self.count_min

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
            - tokens: torch.LongTensor [num_asvs, max_bp]
            - counts: torch.FloatTensor [num_asvs, 1]
            - y_target: torch.FloatTensor [1] or [out_dim] (if metadata provided)
            - unifrac_target: torch.FloatTensor (if unifrac_distances provided)
            - sample_id: str
        """
        sample_id = self.sample_ids[idx]
        sample_ids_list = list(self.table.ids(axis="sample"))
        sample_idx = sample_ids_list.index(sample_id)

        data = self.table.matrix_data[:, sample_idx].toarray().flatten()
        asv_indices = np.where(data > 0)[0]

        tokens_list = []
        counts_list = []

        for asv_idx in asv_indices:
            sequence = self.sequences[asv_idx]
            if self._sequence_cache is not None:
                padded = self._sequence_cache[sequence]
            else:
                tokenized = self.tokenizer.tokenize(sequence)
                padded = self.tokenizer.pad_sequences([tokenized], self.max_bp + 1)[0]
            tokens_list.append(padded)
            counts_list.append(data[asv_idx])

        # Ensure sample has at least one ASV with count > 0
        # If not, this is a data quality issue that should be caught
        if not tokens_list:
            import sys
            error_msg = (
                f"Sample {sample_id} has no ASVs with count > 0. "
                f"This sample should be filtered out before creating the dataset."
            )
            print(f"ERROR: {error_msg}", file=sys.stderr, flush=True)
            raise ValueError(error_msg)
        
        tokens = torch.stack(tokens_list)
        counts = torch.FloatTensor(counts_list).unsqueeze(1)

        # Apply count normalization if enabled
        if self.normalize_counts and self.count_scale is not None:
            counts = (counts - self.count_min) / self.count_scale

        result = {
            "tokens": tokens,
            "counts": counts,
            "sample_id": sample_id,
        }

        if self.metadata_dict is not None and sample_id in self.metadata_dict:
            target_value = self.metadata_dict[sample_id]
            if isinstance(target_value, (int, float)):
                target_float = float(target_value)
            else:
                target_float = float(target_value)

            # Apply normalization if enabled
            if self.normalize_targets and self.target_scale is not None:
                target_float = (target_float - self.target_min) / self.target_scale

            result["y_target"] = torch.FloatTensor([target_float])

        return result
