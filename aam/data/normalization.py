"""Target normalization for regression tasks."""

import logging
import warnings
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# Minimum std to avoid division by zero
MIN_STD = 1e-8

# Valid target transform types
TargetTransform = Literal["none", "minmax", "zscore", "zscore-category", "log-minmax", "log-zscore"]


class GlobalNormalizer:
    """Global target normalization using min-max or z-score.

    Computes global statistics from training data. At training time,
    targets are normalized using these statistics. At inference,
    predictions are denormalized using the same statistics.
    """

    def __init__(self, method: Literal["minmax", "zscore"] = "minmax") -> None:
        """Initialize GlobalNormalizer.

        Args:
            method: Normalization method - "minmax" for [0, 1] range, "zscore" for standardization
        """
        self.method = method
        self.min_val: float = 0.0
        self.max_val: float = 1.0
        self.scale: float = 1.0
        self.mean: float = 0.0
        self.std: float = 1.0
        self._is_fitted: bool = False

    def fit(self, targets: np.ndarray) -> "GlobalNormalizer":
        """Fit normalizer on training data.

        Args:
            targets: Target values array [n_samples] or [n_samples, n_targets]

        Returns:
            self for method chaining

        Raises:
            ValueError: If all targets are NaN
        """
        # Flatten if multi-dimensional
        if targets.ndim > 1:
            targets_flat = targets.flatten()
        else:
            targets_flat = targets

        # Filter NaN values
        valid_mask = ~np.isnan(targets_flat)
        if not valid_mask.any():
            raise ValueError("All targets are NaN, cannot compute normalization statistics")

        valid_targets = targets_flat[valid_mask]

        # Compute statistics
        self.min_val = float(np.min(valid_targets))
        self.max_val = float(np.max(valid_targets))
        self.scale = self.max_val - self.min_val
        if self.scale < MIN_STD:
            self.scale = 1.0
            logger.warning(f"All target values are identical ({self.min_val}). Normalization will have no effect.")

        self.mean = float(np.mean(valid_targets))
        self.std = float(np.std(valid_targets))
        if self.std < MIN_STD:
            self.std = 1.0

        self._is_fitted = True
        return self

    def normalize(
        self,
        target: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """Normalize a target value using global statistics.

        Args:
            target: Target value(s) to normalize

        Returns:
            Normalized target value(s)

        Raises:
            RuntimeError: If normalizer is not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("GlobalNormalizer must be fitted before calling normalize()")

        # Convert scalar to float for consistent arithmetic
        value = target if isinstance(target, (torch.Tensor, np.ndarray)) else float(target)

        if self.method == "minmax":
            return (value - self.min_val) / self.scale
        return (value - self.mean) / self.std

    def denormalize(
        self,
        prediction: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """Denormalize a prediction back to original scale.

        Args:
            prediction: Prediction value(s) to denormalize

        Returns:
            Denormalized prediction value(s)

        Raises:
            RuntimeError: If normalizer is not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("GlobalNormalizer must be fitted before calling denormalize()")

        # Convert scalar to float for consistent arithmetic
        value = prediction if isinstance(prediction, (torch.Tensor, np.ndarray)) else float(prediction)

        if self.method == "minmax":
            return value * self.scale + self.min_val
        return value * self.std + self.mean

    def to_dict(self) -> Dict[str, Any]:
        """Serialize normalizer state to dictionary for checkpointing.

        Returns:
            Dictionary with normalizer state
        """
        return {
            "method": self.method,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "scale": self.scale,
            "mean": self.mean,
            "std": self.std,
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "GlobalNormalizer":
        """Reconstruct normalizer from serialized state.

        Args:
            state: Dictionary from to_dict()

        Returns:
            GlobalNormalizer instance
        """
        normalizer = cls(method=state.get("method", "minmax"))
        normalizer.min_val = state["min_val"]
        normalizer.max_val = state["max_val"]
        normalizer.scale = state["scale"]
        normalizer.mean = state["mean"]
        normalizer.std = state["std"]
        normalizer._is_fitted = True
        return normalizer

    @property
    def is_fitted(self) -> bool:
        """Return whether the normalizer has been fitted."""
        return self._is_fitted


def parse_target_transform(
    target_transform: Optional[str],
    normalize_targets: bool,
    normalize_targets_by: Optional[str],
    log_transform_targets: bool,
) -> tuple[TargetTransform, bool]:
    """Parse and validate target transform configuration.

    Handles both new --target-transform flag and legacy flags with deprecation warnings.

    Args:
        target_transform: New unified flag value (none|minmax|zscore|zscore-category|log-minmax|log-zscore)
        normalize_targets: Legacy --normalize-targets flag
        normalize_targets_by: Legacy --normalize-targets-by columns
        log_transform_targets: Legacy --log-transform-targets flag

    Returns:
        Tuple of (resolved_transform, uses_log_transform)

    Raises:
        ValueError: If conflicting options are specified
    """
    # Build list of legacy flags that differ from defaults
    legacy_flags = []
    if not normalize_targets:
        legacy_flags.append("--no-normalize-targets")
    if normalize_targets_by:
        legacy_flags.append("--normalize-targets-by")
    if log_transform_targets:
        legacy_flags.append("--log-transform-targets")

    # New flag takes precedence
    if target_transform is not None:
        if legacy_flags:
            warnings.warn(
                f"--target-transform is set, ignoring legacy flags: {', '.join(legacy_flags)}. "
                "Remove legacy flags to suppress this warning.",
                DeprecationWarning,
                stacklevel=3,
            )
        uses_log = target_transform.startswith("log-")
        return target_transform, uses_log  # type: ignore[return-value]

    # Convert legacy flags to new transform
    resolved = _legacy_to_new_transform(normalize_targets, normalize_targets_by, log_transform_targets)

    if legacy_flags:
        warnings.warn(
            f"Legacy flags {', '.join(legacy_flags)} are deprecated. "
            f"Use --target-transform {resolved} instead.",
            DeprecationWarning,
            stacklevel=3,
        )

    return resolved, log_transform_targets


def _legacy_to_new_transform(
    normalize_targets: bool,
    normalize_targets_by: Optional[str],
    log_transform_targets: bool,
) -> TargetTransform:
    """Convert legacy flag combination to new transform type."""
    if normalize_targets_by:
        # Per-category z-score
        return "log-zscore" if log_transform_targets else "zscore-category"
    elif normalize_targets:
        # Global min-max (default)
        return "log-minmax" if log_transform_targets else "minmax"
    else:
        # No normalization
        return "none"


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
        key_parts = [col + "=" + metadata[col].astype(str) for col in self.columns]

        if len(key_parts) == 1:
            return key_parts[0]
        return key_parts[0].str.cat(key_parts[1:], sep=",")

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
        value = target if isinstance(target, (torch.Tensor, np.ndarray)) else float(target)
        return (value - mean) / std

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
        value = prediction if isinstance(prediction, (torch.Tensor, np.ndarray)) else float(prediction)
        return value * std + mean

    def _get_stats(self, category_key: str) -> tuple:
        """Get mean and std for a category, falling back to global if unseen.

        Args:
            category_key: Category key string

        Returns:
            Tuple of (mean, std)
        """
        if category_key in self.stats:
            return self.stats[category_key]["mean"], self.stats[category_key]["std"]

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

        return ",".join(f"{col}={metadata_row[col]}" for col in self.columns)

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
