"""Per-category target normalization for regression tasks."""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch


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
            metadata: DataFrame with sample metadata
            columns: List of categorical column names to group by
            sample_ids: Optional list of sample IDs to align targets with metadata

        Returns:
            self for method chaining
        """
        raise NotImplementedError("CategoryNormalizer.fit not yet implemented")

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
        """
        raise NotImplementedError("CategoryNormalizer.normalize not yet implemented")

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
        """
        raise NotImplementedError("CategoryNormalizer.denormalize not yet implemented")

    def get_category_key(
        self,
        metadata_row: Union[pd.Series, Dict[str, Any]],
    ) -> str:
        """Get the category key for a sample from its metadata.

        Args:
            metadata_row: Row of metadata as Series or dict

        Returns:
            Category key string (e.g., "location=A,season=summer")
        """
        raise NotImplementedError("CategoryNormalizer.get_category_key not yet implemented")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize normalizer state to dictionary for checkpointing.

        Returns:
            Dictionary with normalizer state
        """
        raise NotImplementedError("CategoryNormalizer.to_dict not yet implemented")

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "CategoryNormalizer":
        """Reconstruct normalizer from serialized state.

        Args:
            state: Dictionary from to_dict()

        Returns:
            CategoryNormalizer instance
        """
        raise NotImplementedError("CategoryNormalizer.from_dict not yet implemented")

    @property
    def is_fitted(self) -> bool:
        """Return whether the normalizer has been fitted."""
        return self._is_fitted
