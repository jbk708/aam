"""Unit tests for categorical metadata encoding."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aam.data.categorical import CategoricalEncoder


class TestCategoricalEncoder:
    """Test suite for CategoricalEncoder class."""

    @pytest.fixture
    def sample_metadata(self) -> pd.DataFrame:
        """Create sample metadata for testing."""
        return pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3", "s4"],
                "location": ["outdoor", "indoor", "outdoor", "mixed"],
                "season": ["spring", "summer", "fall", "winter"],
            }
        )

    @pytest.fixture
    def encoder(self) -> CategoricalEncoder:
        """Create unfitted encoder."""
        return CategoricalEncoder()

    def test_encoder_not_fitted_initially(self):
        """Test encoder is_fitted returns False before fit."""
        encoder = CategoricalEncoder()
        assert encoder.is_fitted is False

    def test_encoder_fitted_after_fit(self, sample_metadata):
        """Test encoder is_fitted returns True after fit."""
        encoder = CategoricalEncoder()
        encoder.fit(sample_metadata, columns=["location"])
        assert encoder.is_fitted is True

    def test_fit_returns_self(self, sample_metadata):
        """Test fit returns self for method chaining."""
        encoder = CategoricalEncoder()
        result = encoder.fit(sample_metadata, columns=["location"])
        assert result is encoder

    def test_fit_single_column(self, sample_metadata):
        """Test fitting encoder on single column."""
        encoder = CategoricalEncoder()
        encoder.fit(sample_metadata, columns=["location"])

        assert encoder.column_names == ["location"]
        cardinalities = encoder.get_cardinalities()
        # 3 unique values + 1 for unknown = 4
        assert cardinalities["location"] == 4

    def test_fit_multiple_columns(self, sample_metadata):
        """Test fitting encoder on multiple columns."""
        encoder = CategoricalEncoder()
        encoder.fit(sample_metadata, columns=["location", "season"])

        assert encoder.column_names == ["location", "season"]
        cardinalities = encoder.get_cardinalities()
        assert cardinalities["location"] == 4  # outdoor, indoor, mixed + unknown
        assert cardinalities["season"] == 5  # spring, summer, fall, winter + unknown

    def test_fit_missing_required_column_raises(self, sample_metadata):
        """Test fit raises when required column missing."""
        encoder = CategoricalEncoder()
        with pytest.raises(ValueError, match="missing_col"):
            encoder.fit(sample_metadata, columns=["missing_col"])

    def test_transform_basic(self, sample_metadata):
        """Test basic transformation to indices."""
        encoder = CategoricalEncoder()
        encoder.fit(sample_metadata, columns=["location"])

        result = encoder.transform(sample_metadata)

        assert "location" in result
        assert isinstance(result["location"], np.ndarray)
        assert result["location"].dtype == np.int64
        assert len(result["location"]) == 4

    def test_transform_indices_start_at_1(self, sample_metadata):
        """Test that known categories get indices >= 1."""
        encoder = CategoricalEncoder()
        encoder.fit(sample_metadata, columns=["location"])

        result = encoder.transform(sample_metadata)

        # All known values should have index >= 1
        assert all(result["location"] >= 1)
        # Index 0 is reserved for unknown/missing
        assert 0 not in result["location"]

    def test_transform_consistent_mapping(self, sample_metadata):
        """Test same category always maps to same index."""
        encoder = CategoricalEncoder()
        encoder.fit(sample_metadata, columns=["location"])

        result = encoder.transform(sample_metadata)

        # s1 and s3 both have "outdoor" - should have same index
        assert result["location"][0] == result["location"][2]

    def test_transform_unknown_category_maps_to_zero(self, sample_metadata):
        """Test unknown categories at inference map to index 0."""
        encoder = CategoricalEncoder()
        encoder.fit(sample_metadata, columns=["location"])

        # New data with unknown category
        new_metadata = pd.DataFrame(
            {
                "sample_id": ["s5"],
                "location": ["underwater"],  # not in training
            }
        )

        result = encoder.transform(new_metadata)

        assert result["location"][0] == 0

    def test_transform_missing_value_maps_to_zero(self, sample_metadata):
        """Test missing (NaN) values map to index 0."""
        encoder = CategoricalEncoder()
        encoder.fit(sample_metadata, columns=["location"])

        # New data with missing value
        new_metadata = pd.DataFrame(
            {
                "sample_id": ["s5"],
                "location": [None],
            }
        )

        result = encoder.transform(new_metadata)

        assert result["location"][0] == 0

    def test_transform_nan_maps_to_zero(self, sample_metadata):
        """Test NaN values map to index 0."""
        encoder = CategoricalEncoder()
        encoder.fit(sample_metadata, columns=["location"])

        # New data with NaN
        new_metadata = pd.DataFrame(
            {
                "sample_id": ["s5"],
                "location": [np.nan],
            }
        )

        result = encoder.transform(new_metadata)

        assert result["location"][0] == 0

    def test_transform_with_sample_ids(self, sample_metadata):
        """Test transform with specific sample IDs."""
        encoder = CategoricalEncoder()
        encoder.fit(sample_metadata, columns=["location"])

        # Request only specific samples in specific order
        result = encoder.transform(sample_metadata, sample_ids=["s3", "s1"])

        assert len(result["location"]) == 2
        # s3 has "outdoor", s1 has "outdoor" - should be same
        assert result["location"][0] == result["location"][1]

    def test_transform_before_fit_raises(self, sample_metadata):
        """Test transform raises error when not fitted."""
        encoder = CategoricalEncoder()
        with pytest.raises(RuntimeError, match="[Ff]it|[Nn]ot fitted"):
            encoder.transform(sample_metadata)

    def test_fit_transform(self, sample_metadata):
        """Test fit_transform combines fit and transform."""
        encoder = CategoricalEncoder()
        result = encoder.fit_transform(sample_metadata, columns=["location"])

        assert encoder.is_fitted
        assert "location" in result
        assert len(result["location"]) == 4

    def test_get_cardinalities_before_fit_raises(self):
        """Test get_cardinalities raises when not fitted."""
        encoder = CategoricalEncoder()
        with pytest.raises(RuntimeError, match="[Ff]it|[Nn]ot fitted"):
            encoder.get_cardinalities()

    def test_get_mappings(self, sample_metadata):
        """Test get_mappings returns category-to-index dicts."""
        encoder = CategoricalEncoder()
        encoder.fit(sample_metadata, columns=["location"])

        mappings = encoder.get_mappings()

        assert "location" in mappings
        assert isinstance(mappings["location"], dict)
        assert "outdoor" in mappings["location"]
        assert mappings["location"]["outdoor"] >= 1

    def test_get_mappings_before_fit_raises(self):
        """Test get_mappings raises when not fitted."""
        encoder = CategoricalEncoder()
        with pytest.raises(RuntimeError, match="[Ff]it|[Nn]ot fitted"):
            encoder.get_mappings()

    def test_column_names_before_fit_raises(self):
        """Test column_names raises when not fitted."""
        encoder = CategoricalEncoder()
        with pytest.raises(RuntimeError, match="[Ff]it|[Nn]ot fitted"):
            _ = encoder.column_names


class TestCategoricalEncoderSerialization:
    """Test suite for CategoricalEncoder serialization."""

    @pytest.fixture
    def fitted_encoder(self) -> CategoricalEncoder:
        """Create fitted encoder for serialization tests."""
        metadata = pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3"],
                "location": ["outdoor", "indoor", "outdoor"],
                "season": ["spring", "summer", "fall"],
            }
        )
        encoder = CategoricalEncoder()
        encoder.fit(metadata, columns=["location", "season"])
        return encoder

    def test_save_and_load_json(self, fitted_encoder):
        """Test saving and loading encoder from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "encoder.json"

            fitted_encoder.save(path)
            loaded = CategoricalEncoder.load(path)

            assert loaded.is_fitted
            assert loaded.column_names == fitted_encoder.column_names
            assert loaded.get_cardinalities() == fitted_encoder.get_cardinalities()
            assert loaded.get_mappings() == fitted_encoder.get_mappings()

    def test_save_before_fit_raises(self):
        """Test saving unfitted encoder raises error."""
        encoder = CategoricalEncoder()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "encoder.json"
            with pytest.raises(RuntimeError, match="[Ff]it|[Nn]ot fitted"):
                encoder.save(path)

    def test_to_dict_and_from_dict(self, fitted_encoder):
        """Test to_dict and from_dict for checkpoint serialization."""
        state = fitted_encoder.to_dict()

        assert isinstance(state, dict)
        assert "mappings" in state
        assert "column_names" in state

        restored = CategoricalEncoder.from_dict(state)

        assert restored.is_fitted
        assert restored.column_names == fitted_encoder.column_names
        assert restored.get_cardinalities() == fitted_encoder.get_cardinalities()

    def test_to_dict_before_fit_raises(self):
        """Test to_dict raises when not fitted."""
        encoder = CategoricalEncoder()
        with pytest.raises(RuntimeError, match="[Ff]it|[Nn]ot fitted"):
            encoder.to_dict()

    def test_loaded_encoder_transforms_correctly(self, fitted_encoder):
        """Test loaded encoder produces same transforms."""
        test_data = pd.DataFrame(
            {
                "sample_id": ["t1", "t2"],
                "location": ["outdoor", "indoor"],
                "season": ["spring", "unknown_season"],
            }
        )

        original_result = fitted_encoder.transform(test_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "encoder.json"
            fitted_encoder.save(path)
            loaded = CategoricalEncoder.load(path)

        loaded_result = loaded.transform(test_data)

        np.testing.assert_array_equal(
            original_result["location"],
            loaded_result["location"],
        )
        np.testing.assert_array_equal(
            original_result["season"],
            loaded_result["season"],
        )


class TestCategoricalEncoderEdgeCases:
    """Test suite for edge cases in CategoricalEncoder."""

    def test_empty_dataframe(self):
        """Test fitting on empty DataFrame."""
        metadata = pd.DataFrame(columns=["sample_id", "location"])
        encoder = CategoricalEncoder()
        encoder.fit(metadata, columns=["location"])

        # Should have cardinality 1 (just unknown)
        assert encoder.get_cardinalities()["location"] == 1

    def test_single_category(self):
        """Test column with single category."""
        metadata = pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3"],
                "location": ["outdoor", "outdoor", "outdoor"],
            }
        )
        encoder = CategoricalEncoder()
        encoder.fit(metadata, columns=["location"])

        # 1 unique + 1 unknown = 2
        assert encoder.get_cardinalities()["location"] == 2

    def test_all_missing_values(self):
        """Test column with all missing values."""
        metadata = pd.DataFrame(
            {
                "sample_id": ["s1", "s2"],
                "location": [None, np.nan],
            }
        )
        encoder = CategoricalEncoder()
        encoder.fit(metadata, columns=["location"])

        # Only unknown category
        assert encoder.get_cardinalities()["location"] == 1

        result = encoder.transform(metadata)
        assert all(result["location"] == 0)

    def test_numeric_categories_converted_to_string(self):
        """Test numeric categories are handled correctly."""
        metadata = pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3"],
                "year": [2020, 2021, 2020],
            }
        )
        encoder = CategoricalEncoder()
        encoder.fit(metadata, columns=["year"])

        # Should work with numeric values
        assert encoder.get_cardinalities()["year"] == 3  # 2020, 2021 + unknown

        result = encoder.transform(metadata)
        assert result["year"][0] == result["year"][2]  # same year

    def test_transform_reorders_by_sample_id(self):
        """Test transform with sample_ids reorders correctly."""
        metadata = pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3"],
                "location": ["A", "B", "C"],
            }
        )
        encoder = CategoricalEncoder()
        encoder.fit(metadata, columns=["location"])

        # Request reversed order
        result = encoder.transform(metadata, sample_ids=["s3", "s2", "s1"])

        # Get mappings to verify order
        mappings = encoder.get_mappings()
        expected = [mappings["location"]["C"], mappings["location"]["B"], mappings["location"]["A"]]
        np.testing.assert_array_equal(result["location"], expected)

    def test_sample_id_in_index(self):
        """Test transform works when sample_id is DataFrame index."""
        metadata = pd.DataFrame(
            {
                "location": ["A", "B", "C"],
            },
            index=["s1", "s2", "s3"],
        )
        metadata.index.name = "sample_id"

        encoder = CategoricalEncoder()
        encoder.fit(metadata, columns=["location"])

        result = encoder.transform(metadata, sample_ids=["s2", "s3"])
        assert len(result["location"]) == 2

    def test_whitespace_in_categories(self):
        """Test categories with whitespace are handled."""
        metadata = pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3"],
                "location": [" outdoor ", "indoor", " outdoor "],
            }
        )
        encoder = CategoricalEncoder()
        encoder.fit(metadata, columns=["location"])

        result = encoder.transform(metadata)
        # " outdoor " should map consistently
        assert result["location"][0] == result["location"][2]
