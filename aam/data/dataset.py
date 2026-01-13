"""PyTorch Dataset for microbial sequencing data."""

from typing import Any, Dict, Optional, List, Union
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
from aam.data.categorical import CategoricalEncoder


def collate_fn(
    batch: List[Dict[str, Union[torch.Tensor, str]]],
    token_limit: int,
    unifrac_distances: Optional[Any] = None,  # DistanceMatrix, pd.Series, np.ndarray, or filtered variants
    unifrac_metric: str = "unweighted",
    unifrac_loader: Optional["UniFracLoader"] = None,
) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching samples with variable ASV counts.

    Args:
        batch: List of sample dictionaries
        token_limit: Maximum number of ASVs per sample
        unifrac_distances: Optional full distance matrix/series/stripe matrix for batch extraction
        unifrac_metric: Type of UniFrac metric ("unweighted" or "faith_pd")
        unifrac_loader: UniFracLoader instance for batch extraction (optional, created if None)

    Returns:
        Batched dictionary with padded tensors. Includes:
        - tokens: [B, token_limit, max_bp]
        - counts: [B, token_limit, 1]
        - sample_ids: List[str]
        - y_target: [B, 1] (if available)
        - unifrac_target: [B*(B-1)/2] (if unifrac_distances provided)
        - categorical_ids: dict[str, Tensor[B]] (if categorical data available)
    """
    batch_size = len(batch)
    max_bp = batch[0]["tokens"].shape[1]

    tokens_list = []
    counts_list = []
    sample_ids = []
    y_targets = []
    categorical_ids: Dict[str, List[int]] = {}

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

        # Collect categorical indices from each sample
        if "categorical_ids" in sample:
            for col_name, idx in sample["categorical_ids"].items():
                if col_name not in categorical_ids:
                    categorical_ids[col_name] = []
                categorical_ids[col_name].append(idx)

    result: Dict[str, Any] = {
        "tokens": torch.stack(tokens_list),
        "counts": torch.stack(counts_list),
        "sample_ids": sample_ids,
    }

    if y_targets:
        result["y_target"] = torch.stack(y_targets)

    # Convert categorical indices to tensors
    if categorical_ids:
        result["categorical_ids"] = {col: torch.tensor(indices, dtype=torch.long) for col, indices in categorical_ids.items()}

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
            print(
                f"batch_distances shape={batch_distances.shape}, min={np.nanmin(batch_distances)}, max={np.nanmax(batch_distances)}",
                file=sys.stderr,
                flush=True,
            )
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
        unifrac_distances: Optional[Any] = None,  # DistanceMatrix, pd.Series, np.ndarray, or filtered variants
        tokenizer: Optional[SequenceTokenizer] = None,
        max_bp: int = 150,
        token_limit: int = 1024,
        target_column: Optional[str] = None,
        unifrac_metric: str = "unweighted",
        normalize_targets: bool = False,
        target_min: Optional[float] = None,
        target_max: Optional[float] = None,
        log_transform_targets: bool = False,
        normalize_counts: bool = False,
        count_min: Optional[float] = None,
        count_max: Optional[float] = None,
        cache_sequences: bool = True,
        categorical_encoder: Optional[CategoricalEncoder] = None,
    ):
        """Initialize ASVDataset.

        Args:
            table: Rarefied biom.Table object
            metadata: Optional DataFrame with sample metadata and targets
            unifrac_distances: Optional pre-computed UniFrac distances
            tokenizer: SequenceTokenizer instance (creates default if None)
            max_bp: Maximum base pairs per sequence
            token_limit: Maximum ASVs per sample
            target_column: Column name in metadata for target values
            unifrac_metric: Type of UniFrac metric ("unweighted" or "faith_pd")
            normalize_targets: If True, normalize target values to [0, 1] range (default: False)
            target_min: Minimum target value for normalization (computed from data if None)
            target_max: Maximum target value for normalization (computed from data if None)
            log_transform_targets: If True, apply log(y + 1) transform to targets (default: False).
                Useful for non-negative targets with wide range. Inverse is exp(pred) - 1.
            normalize_counts: If True, normalize count values to [0, 1] range (default: False)
            count_min: Minimum count value for normalization (computed from data if None)
            count_max: Maximum count value for normalization (computed from data if None)
            cache_sequences: If True, cache tokenized sequences at init for faster __getitem__ (default: True)
            categorical_encoder: Optional fitted CategoricalEncoder for categorical metadata
        """
        self.table = table
        self.metadata = metadata
        self.unifrac_distances = unifrac_distances
        self.tokenizer = tokenizer if tokenizer is not None else SequenceTokenizer()
        self.max_bp = max_bp
        self.token_limit = token_limit
        self.target_column = target_column
        self.unifrac_metric = unifrac_metric

        self.sample_ids = list(table.ids(axis="sample"))
        self.observation_ids = list(table.ids(axis="observation"))

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
        self.log_transform_targets = log_transform_targets

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
                        f"All count values are identical ({self.count_min}). Count normalization will have no effect."
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
                # If log transform is enabled, compute min/max from log-transformed values
                if log_transform_targets and target_values:
                    import math

                    target_values = [math.log(v + 1) for v in target_values]
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
                            f"All target values are identical ({self.target_min}). Target normalization will have no effect."
                        )
        else:
            self.metadata_dict = None

        # Categorical encoding
        self.categorical_encoder = categorical_encoder
        self._categorical_cache: Optional[Dict[str, Dict[str, int]]] = None
        if categorical_encoder is not None and categorical_encoder.is_fitted:
            if metadata is None:
                raise ValueError("metadata is required when using categorical_encoder")
            # Pre-compute categorical indices for all samples
            cat_indices = categorical_encoder.transform(metadata, sample_ids=self.sample_ids)
            # Cache as sample_id -> {column -> index}
            self._categorical_cache = {}
            for i, sample_id in enumerate(self.sample_ids):
                self._categorical_cache[sample_id] = {col: int(cat_indices[col][i]) for col in categorical_encoder.column_names}

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.sample_ids)

    def get_normalization_params(self) -> Optional[Dict[str, Any]]:
        """Get target normalization parameters.

        Returns:
            Dictionary with normalization params if any transform is enabled, None otherwise.
            May include: 'target_min', 'target_max', 'target_scale', 'log_transform'.
        """
        params: Dict[str, Any] = {}
        if self.normalize_targets and self.target_scale is not None:
            params["target_min"] = self.target_min
            params["target_max"] = self.target_max
            params["target_scale"] = self.target_scale
        if self.log_transform_targets:
            params["log_transform"] = True
        return params if params else None

    def denormalize_targets(
        self, transformed_values: Union[torch.Tensor, np.ndarray, float]
    ) -> Union[torch.Tensor, np.ndarray, float]:
        """Denormalize/inverse-transform target values back to original scale.

        Applies inverse transforms in reverse order:
        1. If normalize_targets: denormalize from [0,1] to [target_min, target_max]
        2. If log_transform_targets: apply exp(x) - 1

        Args:
            transformed_values: Transformed values (normalized and/or log-transformed)

        Returns:
            Values in original target range
        """
        result = transformed_values

        # First, denormalize if normalization was applied
        if self.normalize_targets and self.target_scale is not None:
            if isinstance(result, torch.Tensor):
                result = result * self.target_scale + self.target_min
            elif isinstance(result, np.ndarray):
                result = result * self.target_scale + self.target_min
            else:
                result = float(result) * self.target_scale + self.target_min

        # Then, inverse log transform if it was applied
        if self.log_transform_targets:
            # Clamp to prevent exp() overflow (exp(88.7) overflows float32)
            MAX_EXP_INPUT = 88.0
            if isinstance(result, torch.Tensor):
                result = torch.exp(torch.clamp(result, max=MAX_EXP_INPUT)) - 1
            elif isinstance(result, np.ndarray):
                result = np.exp(np.clip(result, a_min=None, a_max=MAX_EXP_INPUT)) - 1
            else:
                import math

                result = math.exp(min(float(result), MAX_EXP_INPUT)) - 1

        return result

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

    def denormalize_counts(
        self, normalized_values: Union[torch.Tensor, np.ndarray, float]
    ) -> Union[torch.Tensor, np.ndarray, float]:
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
            - categorical_ids: dict[str, int] (if categorical_encoder provided)
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

            # Apply log transform if enabled (before normalization)
            if self.log_transform_targets:
                import math

                target_float = math.log(target_float + 1)

            # Apply normalization if enabled (after log transform)
            if self.normalize_targets and self.target_scale is not None:
                target_float = (target_float - self.target_min) / self.target_scale

            result["y_target"] = torch.FloatTensor([target_float])

        # Add categorical indices if available
        if self._categorical_cache is not None and sample_id in self._categorical_cache:
            result["categorical_ids"] = self._categorical_cache[sample_id]

        return result
