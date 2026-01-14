"""Per-category target normalization for regression tasks."""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# Minimum std to avoid division by zero
MIN_STD = 1e-8


class CategoryNormalizer:
    """Normalizes targets within each category to remove distributional shift.

    Computes per-category mean and std from training data. At training time,
    targets are normalized using the category-specific statistics. At inference,
    predictions are denormalized using the same statistics.

    For unseen categories, falls back to global statistics with a warning.
    """

    def __init__(self) -> None:
        """Initialize CategoryNormalizer."""
        self.stats: Dict[str, Dict[str, float]] = {}
        self.global_mean: float = 0.0
        self.global_std: float = 1.0
        self.columns: List[str] = []
        self._is_fitted: bool = False
        self._warned_categories: set = set()

    def fit(
        self,
        targets: np.ndarray,
        metadata: pd.DataFrame,
        columns: List[str],
        sample_ids: Optional[List[str]] = None,
    ) -> "CategoryNormalizer":
        """Fit normalizer on training data.

        Args:
            targets: Target values array [n_samples] or [n_samples, n_targets]
            metadata: DataFrame with sample metadata (must have 'sample_id' column)
            columns: List of categorical column names to group by
            sample_ids: Optional list of sample IDs to align targets with metadata.
                If provided, targets[i] corresponds to sample_ids[i].
                If None, assumes targets align with metadata row order.

        Returns:
            self for method chaining

        Raises:
            ValueError: If metadata is None, columns is empty, or columns don't exist
        """
        if metadata is None:
            raise ValueError("metadata is required for CategoryNormalizer.fit()")

        if not columns:
            raise ValueError("At least one column is required for CategoryNormalizer.fit()")

        # Validate columns exist
        for col in columns:
            if col not in metadata.columns:
                raise KeyError(f"Column '{col}' not found in metadata. Available: {list(metadata.columns)}")

        self.columns = list(columns)

        # Flatten targets if multi-dimensional (compute stats per sample, not per target dim)
        if targets.ndim > 1:
            # For multi-dimensional targets, we compute category stats using the mean across dims
            targets_flat = targets.mean(axis=1) if targets.ndim == 2 else targets.flatten()
        else:
            targets_flat = targets

        # Align targets with metadata using sample_ids
        if sample_ids is not None:
            # Create mapping from sample_id to target
            target_map = {sid: target for sid, target in zip(sample_ids, targets_flat)}

            # Align metadata rows with targets
            aligned_targets = []
            aligned_metadata_rows = []
            for _, row in metadata.iterrows():
                sid = row.get("sample_id")
                if sid in target_map:
                    aligned_targets.append(target_map[sid])
                    aligned_metadata_rows.append(row)

            targets_aligned = np.array(aligned_targets)
            metadata_aligned = pd.DataFrame(aligned_metadata_rows)
        else:
            targets_aligned = targets_flat
            metadata_aligned = metadata

        # Compute global statistics
        valid_mask = ~np.isnan(targets_aligned)
        if not valid_mask.any():
            raise ValueError("All targets are NaN, cannot compute normalization statistics")

        valid_targets = targets_aligned[valid_mask]
        self.global_mean = float(np.mean(valid_targets))
        self.global_std = float(np.std(valid_targets))
        if self.global_std < MIN_STD:
            self.global_std = MIN_STD

        # Compute per-category statistics
        self.stats = {}

        # Create category key for each sample
        category_keys = self._create_category_keys(metadata_aligned)

        # Group by category key
        unique_keys = category_keys.unique()
        for key in unique_keys:
            mask = (category_keys == key) & valid_mask
            if not mask.any():
                continue

            cat_targets = targets_aligned[mask]
            cat_mean = float(np.mean(cat_targets))
            cat_std = float(np.std(cat_targets))

            # Handle single-sample or zero-std categories
            if cat_std < MIN_STD:
                cat_std = MIN_STD

            self.stats[key] = {
                "mean": cat_mean,
                "std": cat_std,
            }

        self._is_fitted = True
        return self

    def _create_category_keys(self, metadata: pd.DataFrame) -> pd.Series:
        """Create category key strings for each row in metadata.

        Args:
            metadata: DataFrame with categorical columns

        Returns:
            Series of category key strings
        """
        key_parts = []
        for col in self.columns:
            key_parts.append(col + "=" + metadata[col].astype(str))

        if len(key_parts) == 1:
            return key_parts[0]
        else:
            return key_parts[0].str.cat([kp for kp in key_parts[1:]], sep=",")

    def normalize(
        self,
        target: Union[float, np.ndarray, torch.Tensor],
        category_key: str,
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """Normalize a target value using category-specific statistics.

        Args:
            target: Target value(s) to normalize
            category_key: Category key (e.g., "location=A,season=summer")

        Returns:
            Normalized target value(s)

        Raises:
            RuntimeError: If normalizer is not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("CategoryNormalizer must be fitted before calling normalize()")

        mean, std = self._get_stats(category_key)

        if isinstance(target, torch.Tensor):
            return (target - mean) / std
        elif isinstance(target, np.ndarray):
            return (target - mean) / std
        else:
            return (float(target) - mean) / std

    def denormalize(
        self,
        prediction: Union[float, np.ndarray, torch.Tensor],
        category_key: str,
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """Denormalize a prediction back to original scale.

        Args:
            prediction: Prediction value(s) to denormalize
            category_key: Category key (e.g., "location=A,season=summer")

        Returns:
            Denormalized prediction value(s)

        Raises:
            RuntimeError: If normalizer is not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("CategoryNormalizer must be fitted before calling denormalize()")

        mean, std = self._get_stats(category_key)

        if isinstance(prediction, torch.Tensor):
            return prediction * std + mean
        elif isinstance(prediction, np.ndarray):
            return prediction * std + mean
        else:
            return float(prediction) * std + mean

    def _get_stats(self, category_key: str) -> tuple:
        """Get mean and std for a category, falling back to global if unseen.

        Args:
            category_key: Category key string

        Returns:
            Tuple of (mean, std)
        """
        if category_key in self.stats:
            return self.stats[category_key]["mean"], self.stats[category_key]["std"]
        else:
            # Warn once per unseen category
            if category_key not in self._warned_categories:
                logger.warning(
                    f"Unseen category '{category_key}' - using global statistics "
                    f"(mean={self.global_mean:.4f}, std={self.global_std:.4f})"
                )
                self._warned_categories.add(category_key)
            return self.global_mean, self.global_std

    def get_category_key(
        self,
        metadata_row: Union[pd.Series, Dict[str, Any]],
    ) -> str:
        """Get the category key for a sample from its metadata.

        Args:
            metadata_row: Row of metadata as Series or dict

        Returns:
            Category key string (e.g., "location=A,season=summer")

        Raises:
            RuntimeError: If normalizer is not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("CategoryNormalizer must be fitted before calling get_category_key()")

        key_parts = []
        for col in self.columns:
            if isinstance(metadata_row, pd.Series):
                val = metadata_row[col]
            else:
                val = metadata_row[col]
            key_parts.append(f"{col}={val}")

        return ",".join(key_parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize normalizer state to dictionary for checkpointing.

        Returns:
            Dictionary with normalizer state
        """
        return {
            "stats": self.stats,
            "global_mean": self.global_mean,
            "global_std": self.global_std,
            "columns": self.columns,
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "CategoryNormalizer":
        """Reconstruct normalizer from serialized state.

        Args:
            state: Dictionary from to_dict()

        Returns:
            CategoryNormalizer instance
        """
        normalizer = cls()
        normalizer.stats = state["stats"]
        normalizer.global_mean = state["global_mean"]
        normalizer.global_std = state["global_std"]
        normalizer.columns = state["columns"]
        normalizer._is_fitted = True
        return normalizer

    @property
    def is_fitted(self) -> bool:
        """Return whether the normalizer has been fitted."""
        return self._is_fitted
