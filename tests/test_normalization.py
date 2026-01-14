"""Tests for per-category target normalization."""

import numpy as np
import pandas as pd
import pytest
import torch

from aam.data.normalization import CategoryNormalizer


class TestCategoryNormalizer:
    """Tests for CategoryNormalizer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create targets with different distributions per category
        np.random.seed(42)
        n_samples = 100

        # Location A: mean=0.3, std~0.1
        # Location B: mean=0.7, std~0.2
        location_a_targets = np.random.normal(0.3, 0.1, n_samples // 2)
        location_b_targets = np.random.normal(0.7, 0.2, n_samples // 2)

        targets = np.concatenate([location_a_targets, location_b_targets])

        metadata = pd.DataFrame(
            {
                "sample_id": [f"sample_{i}" for i in range(n_samples)],
                "location": ["A"] * (n_samples // 2) + ["B"] * (n_samples // 2),
                "season": (["summer", "winter"] * (n_samples // 4)) * 2,
            }
        )

        sample_ids = list(metadata["sample_id"])

        return {
            "targets": targets,
            "metadata": metadata,
            "sample_ids": sample_ids,
        }

    def test_fit_single_column(self, sample_data):
        """Test fitting normalizer with a single categorical column."""
        normalizer = CategoryNormalizer()

        normalizer.fit(
            targets=sample_data["targets"],
            metadata=sample_data["metadata"],
            columns=["location"],
            sample_ids=sample_data["sample_ids"],
        )

        assert normalizer.is_fitted
        assert normalizer.columns == ["location"]
        assert "location=A" in normalizer.stats
        assert "location=B" in normalizer.stats
        assert "mean" in normalizer.stats["location=A"]
        assert "std" in normalizer.stats["location=A"]

        # Location A should have mean around 0.3
        assert abs(normalizer.stats["location=A"]["mean"] - 0.3) < 0.05
        # Location B should have mean around 0.7
        assert abs(normalizer.stats["location=B"]["mean"] - 0.7) < 0.05

    def test_fit_multiple_columns(self, sample_data):
        """Test fitting normalizer with multiple categorical columns."""
        normalizer = CategoryNormalizer()

        normalizer.fit(
            targets=sample_data["targets"],
            metadata=sample_data["metadata"],
            columns=["location", "season"],
            sample_ids=sample_data["sample_ids"],
        )

        assert normalizer.is_fitted
        assert normalizer.columns == ["location", "season"]
        assert "location=A,season=summer" in normalizer.stats
        assert "location=A,season=winter" in normalizer.stats
        assert "location=B,season=summer" in normalizer.stats
        assert "location=B,season=winter" in normalizer.stats

    def test_normalize_single_value(self, sample_data):
        """Test normalizing a single target value."""
        normalizer = CategoryNormalizer()
        normalizer.fit(
            targets=sample_data["targets"],
            metadata=sample_data["metadata"],
            columns=["location"],
            sample_ids=sample_data["sample_ids"],
        )

        # Normalize a value from location A
        target = 0.3  # Near the mean for location A
        normalized = normalizer.normalize(target, "location=A")

        # Should be close to 0 (z-score of mean is 0)
        assert abs(normalized) < 0.5

    def test_normalize_array(self, sample_data):
        """Test normalizing an array of target values."""
        normalizer = CategoryNormalizer()
        normalizer.fit(
            targets=sample_data["targets"],
            metadata=sample_data["metadata"],
            columns=["location"],
            sample_ids=sample_data["sample_ids"],
        )

        targets = np.array([0.3, 0.35, 0.25])
        normalized = normalizer.normalize(targets, "location=A")

        assert isinstance(normalized, np.ndarray)
        assert normalized.shape == targets.shape

    def test_normalize_tensor(self, sample_data):
        """Test normalizing a torch tensor of target values."""
        normalizer = CategoryNormalizer()
        normalizer.fit(
            targets=sample_data["targets"],
            metadata=sample_data["metadata"],
            columns=["location"],
            sample_ids=sample_data["sample_ids"],
        )

        targets = torch.tensor([0.3, 0.35, 0.25])
        normalized = normalizer.normalize(targets, "location=A")

        assert isinstance(normalized, torch.Tensor)
        assert normalized.shape == targets.shape

    def test_denormalize_single_value(self, sample_data):
        """Test denormalizing a single prediction value."""
        normalizer = CategoryNormalizer()
        normalizer.fit(
            targets=sample_data["targets"],
            metadata=sample_data["metadata"],
            columns=["location"],
            sample_ids=sample_data["sample_ids"],
        )

        # Denormalize z-score of 0 should give the mean
        denormalized = normalizer.denormalize(0.0, "location=A")

        assert abs(denormalized - normalizer.stats["location=A"]["mean"]) < 1e-6

    def test_denormalize_array(self, sample_data):
        """Test denormalizing an array of prediction values."""
        normalizer = CategoryNormalizer()
        normalizer.fit(
            targets=sample_data["targets"],
            metadata=sample_data["metadata"],
            columns=["location"],
            sample_ids=sample_data["sample_ids"],
        )

        predictions = np.array([0.0, 1.0, -1.0])
        denormalized = normalizer.denormalize(predictions, "location=A")

        assert isinstance(denormalized, np.ndarray)
        assert denormalized.shape == predictions.shape

    def test_denormalize_tensor(self, sample_data):
        """Test denormalizing a torch tensor of prediction values."""
        normalizer = CategoryNormalizer()
        normalizer.fit(
            targets=sample_data["targets"],
            metadata=sample_data["metadata"],
            columns=["location"],
            sample_ids=sample_data["sample_ids"],
        )

        predictions = torch.tensor([0.0, 1.0, -1.0])
        denormalized = normalizer.denormalize(predictions, "location=A")

        assert isinstance(denormalized, torch.Tensor)
        assert denormalized.shape == predictions.shape

    def test_normalize_denormalize_roundtrip(self, sample_data):
        """Test that normalize followed by denormalize returns original value."""
        normalizer = CategoryNormalizer()
        normalizer.fit(
            targets=sample_data["targets"],
            metadata=sample_data["metadata"],
            columns=["location"],
            sample_ids=sample_data["sample_ids"],
        )

        # Test with float
        original = 0.35
        normalized = normalizer.normalize(original, "location=A")
        recovered = normalizer.denormalize(normalized, "location=A")
        assert abs(recovered - original) < 1e-6

        # Test with array
        original_array = np.array([0.25, 0.3, 0.35])
        normalized_array = normalizer.normalize(original_array, "location=A")
        recovered_array = normalizer.denormalize(normalized_array, "location=A")
        np.testing.assert_array_almost_equal(recovered_array, original_array)

        # Test with tensor
        original_tensor = torch.tensor([0.25, 0.3, 0.35])
        normalized_tensor = normalizer.normalize(original_tensor, "location=A")
        recovered_tensor = normalizer.denormalize(normalized_tensor, "location=A")
        torch.testing.assert_close(recovered_tensor, original_tensor)

    def test_unseen_category_uses_global_stats(self, sample_data):
        """Test that unseen categories fall back to global statistics."""
        normalizer = CategoryNormalizer()
        normalizer.fit(
            targets=sample_data["targets"],
            metadata=sample_data["metadata"],
            columns=["location"],
            sample_ids=sample_data["sample_ids"],
        )

        # Use an unseen category
        unseen_key = "location=C"

        # Normalize using unseen category should use global stats
        target = 0.5
        normalized = normalizer.normalize(target, unseen_key)

        # Denormalize should return value near original
        recovered = normalizer.denormalize(normalized, unseen_key)
        assert abs(recovered - target) < 1e-6

        # Check that global stats were used
        global_normalized = (target - normalizer.global_mean) / normalizer.global_std
        assert abs(normalized - global_normalized) < 1e-6

    def test_get_category_key_series(self, sample_data):
        """Test category key generation from pandas Series."""
        normalizer = CategoryNormalizer()
        normalizer.fit(
            targets=sample_data["targets"],
            metadata=sample_data["metadata"],
            columns=["location", "season"],
            sample_ids=sample_data["sample_ids"],
        )

        row = sample_data["metadata"].iloc[0]
        key = normalizer.get_category_key(row)

        assert key == "location=A,season=summer"

    def test_get_category_key_dict(self, sample_data):
        """Test category key generation from dict."""
        normalizer = CategoryNormalizer()
        normalizer.fit(
            targets=sample_data["targets"],
            metadata=sample_data["metadata"],
            columns=["location", "season"],
            sample_ids=sample_data["sample_ids"],
        )

        row = {"location": "B", "season": "winter"}
        key = normalizer.get_category_key(row)

        assert key == "location=B,season=winter"

    def test_to_dict(self, sample_data):
        """Test serialization to dictionary."""
        normalizer = CategoryNormalizer()
        normalizer.fit(
            targets=sample_data["targets"],
            metadata=sample_data["metadata"],
            columns=["location"],
            sample_ids=sample_data["sample_ids"],
        )

        state = normalizer.to_dict()

        assert "stats" in state
        assert "global_mean" in state
        assert "global_std" in state
        assert "columns" in state
        assert state["columns"] == ["location"]
        assert "location=A" in state["stats"]
        assert "location=B" in state["stats"]

    def test_from_dict(self, sample_data):
        """Test deserialization from dictionary."""
        state = {
            "stats": {
                "location=A": {"mean": 0.3, "std": 0.1},
                "location=B": {"mean": 0.7, "std": 0.2},
            },
            "global_mean": 0.5,
            "global_std": 0.25,
            "columns": ["location"],
        }

        normalizer = CategoryNormalizer.from_dict(state)

        assert normalizer.is_fitted
        assert normalizer.columns == ["location"]
        assert normalizer.stats["location=A"]["mean"] == 0.3
        assert normalizer.global_mean == 0.5

    def test_to_dict_from_dict_roundtrip(self, sample_data):
        """Test that to_dict followed by from_dict preserves state."""
        normalizer = CategoryNormalizer()
        normalizer.fit(
            targets=sample_data["targets"],
            metadata=sample_data["metadata"],
            columns=["location"],
            sample_ids=sample_data["sample_ids"],
        )

        state = normalizer.to_dict()
        restored = CategoryNormalizer.from_dict(state)

        assert restored.is_fitted
        assert restored.columns == normalizer.columns
        assert restored.global_mean == normalizer.global_mean
        assert restored.global_std == normalizer.global_std

        for key in normalizer.stats:
            assert key in restored.stats
            assert restored.stats[key]["mean"] == normalizer.stats[key]["mean"]
            assert restored.stats[key]["std"] == normalizer.stats[key]["std"]

    def test_is_fitted_property(self):
        """Test is_fitted property before and after fitting."""
        normalizer = CategoryNormalizer()
        assert not normalizer.is_fitted

        # Create minimal data
        targets = np.array([0.1, 0.2, 0.3])
        metadata = pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3"],
                "category": ["A", "A", "B"],
            }
        )

        normalizer.fit(
            targets=targets,
            metadata=metadata,
            columns=["category"],
            sample_ids=["s1", "s2", "s3"],
        )

        assert normalizer.is_fitted

    def test_fit_with_sample_ids_alignment(self):
        """Test that sample_ids properly align targets with metadata."""
        # Metadata with different order than targets
        metadata = pd.DataFrame(
            {
                "sample_id": ["s3", "s1", "s2"],
                "category": ["B", "A", "A"],
            }
        )

        # Targets in order: s1=0.1, s2=0.2, s3=0.9
        targets = np.array([0.1, 0.2, 0.9])
        sample_ids = ["s1", "s2", "s3"]

        normalizer = CategoryNormalizer()
        normalizer.fit(
            targets=targets,
            metadata=metadata,
            columns=["category"],
            sample_ids=sample_ids,
        )

        # Category A should have mean of (0.1 + 0.2) / 2 = 0.15
        assert abs(normalizer.stats["category=A"]["mean"] - 0.15) < 1e-6
        # Category B should have mean of 0.9
        assert abs(normalizer.stats["category=B"]["mean"] - 0.9) < 1e-6

    def test_single_sample_category(self):
        """Test handling of categories with only one sample (std=0)."""
        targets = np.array([0.1, 0.2, 0.5])
        metadata = pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3"],
                "category": ["A", "A", "B"],  # B has only one sample
            }
        )

        normalizer = CategoryNormalizer()
        normalizer.fit(
            targets=targets,
            metadata=metadata,
            columns=["category"],
            sample_ids=["s1", "s2", "s3"],
        )

        # Category B has std=0, should use minimum std to avoid division by zero
        assert normalizer.stats["category=B"]["std"] > 0

        # Normalizing and denormalizing should still work
        original = 0.5
        normalized = normalizer.normalize(original, "category=B")
        recovered = normalizer.denormalize(normalized, "category=B")
        assert abs(recovered - original) < 1e-6


class TestCategoryNormalizerEdgeCases:
    """Edge case tests for CategoryNormalizer."""

    def test_fit_requires_metadata(self):
        """Test that fit requires metadata."""
        normalizer = CategoryNormalizer()
        targets = np.array([0.1, 0.2])

        with pytest.raises((ValueError, TypeError)):
            normalizer.fit(
                targets=targets,
                metadata=None,
                columns=["category"],
            )

    def test_fit_requires_columns(self):
        """Test that fit requires at least one column."""
        normalizer = CategoryNormalizer()
        targets = np.array([0.1, 0.2])
        metadata = pd.DataFrame(
            {
                "sample_id": ["s1", "s2"],
                "category": ["A", "B"],
            }
        )

        with pytest.raises((ValueError, TypeError)):
            normalizer.fit(
                targets=targets,
                metadata=metadata,
                columns=[],
            )

    def test_fit_validates_column_exists(self):
        """Test that fit validates columns exist in metadata."""
        normalizer = CategoryNormalizer()
        targets = np.array([0.1, 0.2])
        metadata = pd.DataFrame(
            {
                "sample_id": ["s1", "s2"],
                "category": ["A", "B"],
            }
        )

        with pytest.raises((ValueError, KeyError)):
            normalizer.fit(
                targets=targets,
                metadata=metadata,
                columns=["nonexistent_column"],
            )

    def test_normalize_before_fit_raises(self):
        """Test that normalize before fit raises error."""
        normalizer = CategoryNormalizer()

        with pytest.raises((ValueError, RuntimeError)):
            normalizer.normalize(0.5, "category=A")

    def test_denormalize_before_fit_raises(self):
        """Test that denormalize before fit raises error."""
        normalizer = CategoryNormalizer()

        with pytest.raises((ValueError, RuntimeError)):
            normalizer.denormalize(0.5, "category=A")

    def test_get_category_key_before_fit_raises(self):
        """Test that get_category_key before fit raises error."""
        normalizer = CategoryNormalizer()
        row = {"category": "A"}

        with pytest.raises((ValueError, RuntimeError)):
            normalizer.get_category_key(row)

    def test_multidimensional_targets(self):
        """Test with multi-dimensional target arrays."""
        # 2D targets: [n_samples, n_targets]
        targets = np.array([[0.1, 0.2], [0.2, 0.3], [0.8, 0.9], [0.9, 1.0]])
        metadata = pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3", "s4"],
                "category": ["A", "A", "B", "B"],
            }
        )

        normalizer = CategoryNormalizer()
        normalizer.fit(
            targets=targets,
            metadata=metadata,
            columns=["category"],
            sample_ids=["s1", "s2", "s3", "s4"],
        )

        # Should handle per-target normalization
        assert normalizer.is_fitted

    def test_nan_in_targets(self):
        """Test that NaN in targets raises or handles gracefully."""
        targets = np.array([0.1, np.nan, 0.3])
        metadata = pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3"],
                "category": ["A", "A", "A"],
            }
        )

        normalizer = CategoryNormalizer()

        # Should either raise or handle NaN gracefully
        # (implementation decides which approach)
        try:
            normalizer.fit(
                targets=targets,
                metadata=metadata,
                columns=["category"],
                sample_ids=["s1", "s2", "s3"],
            )
            # If it doesn't raise, verify NaN handling
            assert normalizer.is_fitted
        except ValueError:
            # Raising is also acceptable behavior
            pass
