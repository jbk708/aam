"""Tests for target normalization (global and per-category)."""

import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from aam.data.normalization import (
    CategoryNormalizer,
    CategoryWeighter,
    GlobalNormalizer,
    parse_target_transform,
)


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


class TestGlobalNormalizer:
    """Tests for GlobalNormalizer class."""

    def test_minmax_normalize(self):
        """Test min-max normalization to [0, 1] range."""
        normalizer = GlobalNormalizer(method="minmax")
        targets = np.array([0.0, 50.0, 100.0])
        normalizer.fit(targets)

        assert normalizer.is_fitted
        assert normalizer.min_val == 0.0
        assert normalizer.max_val == 100.0
        assert normalizer.scale == 100.0

        # Normalize
        normalized = normalizer.normalize(50.0)
        assert normalized == 0.5

        # Array normalization
        normalized_arr = normalizer.normalize(targets)
        np.testing.assert_array_almost_equal(normalized_arr, [0.0, 0.5, 1.0])

    def test_zscore_normalize(self):
        """Test z-score normalization (standardization)."""
        normalizer = GlobalNormalizer(method="zscore")
        targets = np.array([10.0, 20.0, 30.0])
        normalizer.fit(targets)

        assert normalizer.is_fitted
        assert normalizer.mean == pytest.approx(20.0)
        assert normalizer.std == pytest.approx(np.std(targets))

        # Mean should normalize to 0
        normalized = normalizer.normalize(20.0)
        assert normalized == pytest.approx(0.0)

    def test_minmax_denormalize(self):
        """Test min-max denormalization."""
        normalizer = GlobalNormalizer(method="minmax")
        targets = np.array([0.0, 50.0, 100.0])
        normalizer.fit(targets)

        # Denormalize
        denormalized = normalizer.denormalize(0.5)
        assert denormalized == 50.0

        # Array denormalization
        denormalized_arr = normalizer.denormalize(np.array([0.0, 0.5, 1.0]))
        np.testing.assert_array_almost_equal(denormalized_arr, targets)

    def test_zscore_denormalize(self):
        """Test z-score denormalization."""
        normalizer = GlobalNormalizer(method="zscore")
        targets = np.array([10.0, 20.0, 30.0])
        normalizer.fit(targets)

        # Denormalize z=0 should return mean
        denormalized = normalizer.denormalize(0.0)
        assert denormalized == pytest.approx(20.0)

    def test_roundtrip_minmax(self):
        """Test normalize followed by denormalize returns original (minmax)."""
        normalizer = GlobalNormalizer(method="minmax")
        targets = np.array([10.0, 50.0, 90.0])
        normalizer.fit(targets)

        original = 60.0
        normalized = normalizer.normalize(original)
        recovered = normalizer.denormalize(normalized)
        assert recovered == pytest.approx(original)

    def test_roundtrip_zscore(self):
        """Test normalize followed by denormalize returns original (zscore)."""
        normalizer = GlobalNormalizer(method="zscore")
        targets = np.array([10.0, 50.0, 90.0])
        normalizer.fit(targets)

        original = 60.0
        normalized = normalizer.normalize(original)
        recovered = normalizer.denormalize(normalized)
        assert recovered == pytest.approx(original)

    def test_tensor_support(self):
        """Test normalization with torch tensors."""
        normalizer = GlobalNormalizer(method="minmax")
        targets = np.array([0.0, 50.0, 100.0])
        normalizer.fit(targets)

        tensor = torch.tensor([0.0, 50.0, 100.0])
        normalized = normalizer.normalize(tensor)

        assert isinstance(normalized, torch.Tensor)
        torch.testing.assert_close(normalized, torch.tensor([0.0, 0.5, 1.0]))

    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization and deserialization."""
        normalizer = GlobalNormalizer(method="zscore")
        targets = np.array([10.0, 20.0, 30.0])
        normalizer.fit(targets)

        state = normalizer.to_dict()
        restored = GlobalNormalizer.from_dict(state)

        assert restored.is_fitted
        assert restored.method == "zscore"
        assert restored.mean == normalizer.mean
        assert restored.std == normalizer.std

    def test_identical_values_handling(self):
        """Test handling of identical values (std=0)."""
        normalizer = GlobalNormalizer(method="zscore")
        targets = np.array([5.0, 5.0, 5.0])
        normalizer.fit(targets)

        # std should be set to 1.0 to avoid division by zero
        assert normalizer.std == 1.0

        # Normalization should still work
        normalized = normalizer.normalize(5.0)
        denormalized = normalizer.denormalize(normalized)
        assert denormalized == pytest.approx(5.0)

    def test_normalize_before_fit_raises(self):
        """Test that normalize before fit raises error."""
        normalizer = GlobalNormalizer()

        with pytest.raises(RuntimeError):
            normalizer.normalize(0.5)

    def test_denormalize_before_fit_raises(self):
        """Test that denormalize before fit raises error."""
        normalizer = GlobalNormalizer()

        with pytest.raises(RuntimeError):
            normalizer.denormalize(0.5)


class TestParseTargetTransform:
    """Tests for parse_target_transform function."""

    def test_new_flag_none(self):
        """Test --target-transform none."""
        transform, uses_log = parse_target_transform(
            target_transform="none",
            normalize_targets=True,  # Should be ignored
            normalize_targets_by=None,
            log_transform_targets=False,
        )
        assert transform == "none"
        assert uses_log is False

    def test_new_flag_minmax(self):
        """Test --target-transform minmax."""
        transform, uses_log = parse_target_transform(
            target_transform="minmax",
            normalize_targets=True,
            normalize_targets_by=None,
            log_transform_targets=False,
        )
        assert transform == "minmax"
        assert uses_log is False

    def test_new_flag_zscore(self):
        """Test --target-transform zscore."""
        transform, uses_log = parse_target_transform(
            target_transform="zscore",
            normalize_targets=True,
            normalize_targets_by=None,
            log_transform_targets=False,
        )
        assert transform == "zscore"
        assert uses_log is False

    def test_new_flag_zscore_category(self):
        """Test --target-transform zscore-category."""
        transform, uses_log = parse_target_transform(
            target_transform="zscore-category",
            normalize_targets=True,
            normalize_targets_by=None,
            log_transform_targets=False,
        )
        assert transform == "zscore-category"
        assert uses_log is False

    def test_new_flag_log_minmax(self):
        """Test --target-transform log-minmax."""
        transform, uses_log = parse_target_transform(
            target_transform="log-minmax",
            normalize_targets=True,
            normalize_targets_by=None,
            log_transform_targets=False,
        )
        assert transform == "log-minmax"
        assert uses_log is True

    def test_new_flag_log_zscore(self):
        """Test --target-transform log-zscore."""
        transform, uses_log = parse_target_transform(
            target_transform="log-zscore",
            normalize_targets=True,
            normalize_targets_by=None,
            log_transform_targets=False,
        )
        assert transform == "log-zscore"
        assert uses_log is True

    def test_legacy_normalize_targets_default(self):
        """Test legacy --normalize-targets (default True) maps to minmax."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            transform, uses_log = parse_target_transform(
                target_transform=None,
                normalize_targets=True,
                normalize_targets_by=None,
                log_transform_targets=False,
            )
        assert transform == "minmax"
        assert uses_log is False

    def test_legacy_no_normalize_targets(self):
        """Test legacy --no-normalize-targets maps to none."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            transform, uses_log = parse_target_transform(
                target_transform=None,
                normalize_targets=False,
                normalize_targets_by=None,
                log_transform_targets=False,
            )
        assert transform == "none"
        assert uses_log is False
        # Should emit deprecation warning
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "--no-normalize-targets" in str(w[0].message)

    def test_legacy_normalize_targets_by(self):
        """Test legacy --normalize-targets-by maps to zscore-category."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            transform, uses_log = parse_target_transform(
                target_transform=None,
                normalize_targets=False,
                normalize_targets_by="location",
                log_transform_targets=False,
            )
        assert transform == "zscore-category"
        assert uses_log is False
        # Should emit deprecation warning
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)

    def test_legacy_log_transform_with_normalize(self):
        """Test legacy --log-transform-targets with --normalize-targets maps to log-minmax."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            transform, uses_log = parse_target_transform(
                target_transform=None,
                normalize_targets=True,
                normalize_targets_by=None,
                log_transform_targets=True,
            )
        assert transform == "log-minmax"
        assert uses_log is True
        # Should emit deprecation warning
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)

    def test_legacy_log_transform_with_category(self):
        """Test legacy --log-transform-targets with --normalize-targets-by maps to log-zscore."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            transform, uses_log = parse_target_transform(
                target_transform=None,
                normalize_targets=False,
                normalize_targets_by="location",
                log_transform_targets=True,
            )
        assert transform == "log-zscore"
        assert uses_log is True

    def test_new_flag_ignores_legacy_with_warning(self):
        """Test that new flag ignores legacy flags and emits warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            transform, uses_log = parse_target_transform(
                target_transform="zscore",
                normalize_targets=False,  # Legacy flag
                normalize_targets_by="location",  # Legacy flag
                log_transform_targets=True,  # Legacy flag
            )
        assert transform == "zscore"
        assert uses_log is False  # New flag value, not legacy
        # Should emit warning about ignoring legacy flags
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "ignoring legacy flags" in str(w[0].message)


class TestCategoryWeighter:
    """Tests for CategoryWeighter class (per-category loss weighting)."""

    @pytest.fixture
    def imbalanced_data(self):
        """Create imbalanced categorical data for testing."""
        # Location A: 80 samples (majority)
        # Location B: 20 samples (minority)
        n_a = 80
        n_b = 20
        n_total = n_a + n_b

        metadata = pd.DataFrame(
            {
                "sample_id": [f"sample_{i}" for i in range(n_total)],
                "location": ["A"] * n_a + ["B"] * n_b,
                "season": (["summer", "winter"] * (n_total // 2)),
            }
        )

        return {
            "metadata": metadata,
            "n_a": n_a,
            "n_b": n_b,
            "n_total": n_total,
        }

    def test_fit_auto_mode_inverse_frequency(self, imbalanced_data):
        """Test auto mode computes inverse frequency weights."""
        weighter = CategoryWeighter()

        weighter.fit(
            metadata=imbalanced_data["metadata"],
            columns=["location"],
            mode="auto",
        )

        assert weighter.is_fitted
        assert weighter.columns == ["location"]
        assert "location=A" in weighter.weights
        assert "location=B" in weighter.weights

        # Minority class (B) should have higher weight than majority (A)
        assert weighter.weights["location=B"] > weighter.weights["location=A"]

        # Verify inverse frequency formula: weight = N_total / (N_categories * N_category)
        n_total = imbalanced_data["n_total"]
        n_categories = 2
        expected_weight_a = n_total / (n_categories * imbalanced_data["n_a"])
        expected_weight_b = n_total / (n_categories * imbalanced_data["n_b"])

        assert abs(weighter.weights["location=A"] - expected_weight_a) < 1e-6
        assert abs(weighter.weights["location=B"] - expected_weight_b) < 1e-6

    def test_fit_auto_mode_balanced_weights_sum(self, imbalanced_data):
        """Test that auto weights are balanced (weighted sum = N_total)."""
        weighter = CategoryWeighter()

        weighter.fit(
            metadata=imbalanced_data["metadata"],
            columns=["location"],
            mode="auto",
        )

        # Weighted sum of samples should equal unweighted sum
        # sum(N_i * w_i) should equal N_total (or close to it, ensuring balance)
        n_a = imbalanced_data["n_a"]
        n_b = imbalanced_data["n_b"]
        weighted_sum = n_a * weighter.weights["location=A"] + n_b * weighter.weights["location=B"]

        # Should be equal to N_total (each category contributes equally after weighting)
        assert abs(weighted_sum - imbalanced_data["n_total"]) < 1e-6

    def test_fit_manual_mode(self, imbalanced_data):
        """Test manual mode uses user-specified weights."""
        weighter = CategoryWeighter()

        manual_weights = {"location=A": 1.0, "location=B": 3.0}

        weighter.fit(
            metadata=imbalanced_data["metadata"],
            columns=["location"],
            mode="manual",
            manual_weights=manual_weights,
        )

        assert weighter.is_fitted
        assert weighter.weights["location=A"] == 1.0
        assert weighter.weights["location=B"] == 3.0

    def test_fit_multiple_columns(self, imbalanced_data):
        """Test fitting with multiple categorical columns."""
        weighter = CategoryWeighter()

        weighter.fit(
            metadata=imbalanced_data["metadata"],
            columns=["location", "season"],
            mode="auto",
        )

        assert weighter.is_fitted
        assert weighter.columns == ["location", "season"]
        # Should have combination keys
        assert "location=A,season=summer" in weighter.weights
        assert "location=A,season=winter" in weighter.weights
        assert "location=B,season=summer" in weighter.weights
        assert "location=B,season=winter" in weighter.weights

    def test_get_weight(self, imbalanced_data):
        """Test get_weight returns correct weight for category."""
        weighter = CategoryWeighter()
        weighter.fit(
            metadata=imbalanced_data["metadata"],
            columns=["location"],
            mode="auto",
        )

        weight_a = weighter.get_weight("location=A")
        weight_b = weighter.get_weight("location=B")

        assert weight_a == weighter.weights["location=A"]
        assert weight_b == weighter.weights["location=B"]

    def test_get_weight_unseen_category(self, imbalanced_data):
        """Test get_weight for unseen category returns default weight."""
        weighter = CategoryWeighter()
        weighter.fit(
            metadata=imbalanced_data["metadata"],
            columns=["location"],
            mode="auto",
        )

        # Unseen category should return default weight
        weight_c = weighter.get_weight("location=C")
        assert weight_c == weighter.default_weight

    def test_get_weight_for_sample(self, imbalanced_data):
        """Test get_weight_for_sample from metadata row."""
        weighter = CategoryWeighter()
        weighter.fit(
            metadata=imbalanced_data["metadata"],
            columns=["location"],
            mode="auto",
        )

        # Test with dict
        row_dict = {"location": "A"}
        weight = weighter.get_weight_for_sample(row_dict)
        assert weight == weighter.weights["location=A"]

        # Test with Series
        row_series = pd.Series({"location": "B"})
        weight = weighter.get_weight_for_sample(row_series)
        assert weight == weighter.weights["location=B"]

    def test_to_dict(self, imbalanced_data):
        """Test serialization to dictionary."""
        weighter = CategoryWeighter()
        weighter.fit(
            metadata=imbalanced_data["metadata"],
            columns=["location"],
            mode="auto",
        )

        state = weighter.to_dict()

        assert "weights" in state
        assert "columns" in state
        assert "default_weight" in state
        assert state["columns"] == ["location"]
        assert "location=A" in state["weights"]
        assert "location=B" in state["weights"]

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        state = {
            "weights": {
                "location=A": 0.5,
                "location=B": 2.5,
            },
            "columns": ["location"],
            "default_weight": 1.0,
        }

        weighter = CategoryWeighter.from_dict(state)

        assert weighter.is_fitted
        assert weighter.columns == ["location"]
        assert weighter.weights["location=A"] == 0.5
        assert weighter.weights["location=B"] == 2.5
        assert weighter.default_weight == 1.0

    def test_to_dict_from_dict_roundtrip(self, imbalanced_data):
        """Test serialization roundtrip preserves state."""
        weighter = CategoryWeighter()
        weighter.fit(
            metadata=imbalanced_data["metadata"],
            columns=["location"],
            mode="auto",
        )

        state = weighter.to_dict()
        restored = CategoryWeighter.from_dict(state)

        assert restored.is_fitted
        assert restored.columns == weighter.columns
        assert restored.default_weight == weighter.default_weight
        for key in weighter.weights:
            assert key in restored.weights
            assert restored.weights[key] == weighter.weights[key]

    def test_is_fitted_property(self):
        """Test is_fitted property before and after fitting."""
        weighter = CategoryWeighter()
        assert not weighter.is_fitted

        metadata = pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3"],
                "category": ["A", "A", "B"],
            }
        )

        weighter.fit(metadata=metadata, columns=["category"], mode="auto")
        assert weighter.is_fitted


class TestCategoryWeighterEdgeCases:
    """Edge case tests for CategoryWeighter."""

    def test_fit_requires_metadata(self):
        """Test that fit requires metadata."""
        weighter = CategoryWeighter()

        with pytest.raises((ValueError, TypeError)):
            weighter.fit(metadata=None, columns=["category"], mode="auto")

    def test_fit_requires_columns(self):
        """Test that fit requires at least one column."""
        weighter = CategoryWeighter()
        metadata = pd.DataFrame(
            {
                "sample_id": ["s1", "s2"],
                "category": ["A", "B"],
            }
        )

        with pytest.raises((ValueError, TypeError)):
            weighter.fit(metadata=metadata, columns=[], mode="auto")

    def test_fit_validates_column_exists(self):
        """Test that fit validates columns exist in metadata."""
        weighter = CategoryWeighter()
        metadata = pd.DataFrame(
            {
                "sample_id": ["s1", "s2"],
                "category": ["A", "B"],
            }
        )

        with pytest.raises((ValueError, KeyError)):
            weighter.fit(metadata=metadata, columns=["nonexistent"], mode="auto")

    def test_fit_manual_requires_weights(self):
        """Test that manual mode requires manual_weights."""
        weighter = CategoryWeighter()
        metadata = pd.DataFrame(
            {
                "sample_id": ["s1", "s2"],
                "category": ["A", "B"],
            }
        )

        with pytest.raises((ValueError, TypeError)):
            weighter.fit(metadata=metadata, columns=["category"], mode="manual", manual_weights=None)

    def test_get_weight_before_fit_raises(self):
        """Test that get_weight before fit raises error."""
        weighter = CategoryWeighter()

        with pytest.raises((ValueError, RuntimeError)):
            weighter.get_weight("category=A")

    def test_get_weight_for_sample_before_fit_raises(self):
        """Test that get_weight_for_sample before fit raises error."""
        weighter = CategoryWeighter()
        row = {"category": "A"}

        with pytest.raises((ValueError, RuntimeError)):
            weighter.get_weight_for_sample(row)

    def test_single_category(self):
        """Test handling of single category (all samples same class)."""
        weighter = CategoryWeighter()
        metadata = pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3"],
                "category": ["A", "A", "A"],
            }
        )

        weighter.fit(metadata=metadata, columns=["category"], mode="auto")

        # Single category should have weight 1.0 (no imbalance)
        assert weighter.weights["category=A"] == pytest.approx(1.0)

    def test_invalid_mode(self):
        """Test that invalid mode raises error."""
        weighter = CategoryWeighter()
        metadata = pd.DataFrame(
            {
                "sample_id": ["s1", "s2"],
                "category": ["A", "B"],
            }
        )

        with pytest.raises(ValueError):
            weighter.fit(metadata=metadata, columns=["category"], mode="invalid_mode")
