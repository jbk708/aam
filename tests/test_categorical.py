"""Unit tests for categorical metadata schema and encoding."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aam.data.categorical import (
    CategoricalColumnConfig,
    CategoricalEncoder,
    CategoricalSchema,
)


class TestCategoricalColumnConfig:
    """Test suite for CategoricalColumnConfig dataclass."""

    def test_basic_config(self):
        """Test creating a column config with just a name."""
        config = CategoricalColumnConfig(name="location")

        assert config.name == "location"
        assert config.cardinality is None
        assert config.embed_dim is None
        assert config.required is True

    def test_config_with_all_fields(self):
        """Test creating a column config with all fields specified."""
        config = CategoricalColumnConfig(
            name="season",
            cardinality=4,
            embed_dim=8,
            required=False,
        )

        assert config.name == "season"
        assert config.cardinality == 4
        assert config.embed_dim == 8
        assert config.required is False


class TestCategoricalSchema:
    """Test suite for CategoricalSchema dataclass."""

    def test_empty_schema(self):
        """Test creating an empty schema."""
        schema = CategoricalSchema()

        assert schema.columns == []
        assert schema.default_embed_dim == 16
        assert schema.column_names == []
        assert schema.total_embed_dim == 0

    def test_schema_with_single_column(self):
        """Test schema with one column."""
        config = CategoricalColumnConfig(name="location", cardinality=5)
        schema = CategoricalSchema(columns=[config])

        assert len(schema.columns) == 1
        assert schema.column_names == ["location"]

    def test_schema_with_multiple_columns(self):
        """Test schema with multiple columns."""
        configs = [
            CategoricalColumnConfig(name="location", cardinality=10),
            CategoricalColumnConfig(name="season", cardinality=4),
            CategoricalColumnConfig(name="site_type", cardinality=3),
        ]
        schema = CategoricalSchema(columns=configs)

        assert len(schema.columns) == 3
        assert schema.column_names == ["location", "season", "site_type"]

    def test_schema_custom_default_embed_dim(self):
        """Test schema with custom default embedding dimension."""
        schema = CategoricalSchema(default_embed_dim=32)

        assert schema.default_embed_dim == 32

    def test_get_column_exists(self):
        """Test getting an existing column configuration."""
        configs = [
            CategoricalColumnConfig(name="location", cardinality=10),
            CategoricalColumnConfig(name="season", cardinality=4),
        ]
        schema = CategoricalSchema(columns=configs)

        location = schema.get_column("location")
        assert location.name == "location"
        assert location.cardinality == 10

        season = schema.get_column("season")
        assert season.name == "season"
        assert season.cardinality == 4

    def test_get_column_not_found(self):
        """Test getting a column that doesn't exist raises KeyError."""
        config = CategoricalColumnConfig(name="location")
        schema = CategoricalSchema(columns=[config])

        with pytest.raises(KeyError, match="not_in_schema"):
            schema.get_column("not_in_schema")

    def test_get_embed_dim_uses_default(self):
        """Test embed_dim resolution falls back to schema default."""
        config = CategoricalColumnConfig(name="location")
        schema = CategoricalSchema(columns=[config], default_embed_dim=24)

        assert schema.get_embed_dim("location") == 24

    def test_get_embed_dim_uses_column_specific(self):
        """Test embed_dim resolution uses column-specific value when set."""
        config = CategoricalColumnConfig(name="location", embed_dim=8)
        schema = CategoricalSchema(columns=[config], default_embed_dim=24)

        assert schema.get_embed_dim("location") == 8

    def test_total_embed_dim_default_dims(self):
        """Test total embed dim with all columns using default."""
        configs = [
            CategoricalColumnConfig(name="location"),
            CategoricalColumnConfig(name="season"),
        ]
        schema = CategoricalSchema(columns=configs, default_embed_dim=16)

        assert schema.total_embed_dim == 32  # 16 + 16

    def test_total_embed_dim_mixed_dims(self):
        """Test total embed dim with mixed column-specific and default dims."""
        configs = [
            CategoricalColumnConfig(name="location", embed_dim=8),
            CategoricalColumnConfig(name="season"),  # uses default
            CategoricalColumnConfig(name="site_type", embed_dim=4),
        ]
        schema = CategoricalSchema(columns=configs, default_embed_dim=16)

        assert schema.total_embed_dim == 28  # 8 + 16 + 4

    def test_validate_metadata_columns_all_present(self):
        """Test validation passes when all required columns are present."""
        configs = [
            CategoricalColumnConfig(name="location", required=True),
            CategoricalColumnConfig(name="season", required=True),
        ]
        schema = CategoricalSchema(columns=configs)

        # Should not raise
        schema.validate_metadata_columns(["sample_id", "location", "season", "target"])

    def test_validate_metadata_columns_missing_required(self):
        """Test validation fails when required column is missing."""
        configs = [
            CategoricalColumnConfig(name="location", required=True),
            CategoricalColumnConfig(name="season", required=True),
        ]
        schema = CategoricalSchema(columns=configs)

        with pytest.raises(ValueError, match="season"):
            schema.validate_metadata_columns(["sample_id", "location", "target"])

    def test_validate_metadata_columns_missing_optional_ok(self):
        """Test validation passes when only optional column is missing."""
        configs = [
            CategoricalColumnConfig(name="location", required=True),
            CategoricalColumnConfig(name="season", required=False),
        ]
        schema = CategoricalSchema(columns=configs)

        # Should not raise
        schema.validate_metadata_columns(["sample_id", "location", "target"])

    def test_from_column_names_basic(self):
        """Test factory method creates schema from column names."""
        schema = CategoricalSchema.from_column_names(["location", "season"])

        assert len(schema.columns) == 2
        assert schema.column_names == ["location", "season"]
        assert all(c.cardinality is None for c in schema.columns)
        assert all(c.embed_dim is None for c in schema.columns)
        assert all(c.required is True for c in schema.columns)

    def test_from_column_names_custom_embed_dim(self):
        """Test factory method with custom default embed dim."""
        schema = CategoricalSchema.from_column_names(
            ["location", "season"],
            default_embed_dim=32,
        )

        assert schema.default_embed_dim == 32
        assert schema.get_embed_dim("location") == 32
        assert schema.get_embed_dim("season") == 32

    def test_from_column_names_empty_list(self):
        """Test factory method with empty column list."""
        schema = CategoricalSchema.from_column_names([])

        assert len(schema.columns) == 0
        assert schema.column_names == []


class TestCategoricalSchemaValidation:
    """Test suite for schema validation on creation."""

    def test_duplicate_column_names_rejected(self):
        """Test that duplicate column names are rejected."""
        configs = [
            CategoricalColumnConfig(name="location"),
            CategoricalColumnConfig(name="location"),  # duplicate
        ]

        with pytest.raises(ValueError, match="[Dd]uplicate"):
            CategoricalSchema(columns=configs)

    def test_negative_cardinality_rejected(self):
        """Test that negative cardinality is rejected."""
        config = CategoricalColumnConfig(name="location", cardinality=-1)

        with pytest.raises(ValueError, match="cardinality"):
            CategoricalSchema(columns=[config])

    def test_zero_cardinality_rejected(self):
        """Test that zero cardinality is rejected."""
        config = CategoricalColumnConfig(name="location", cardinality=0)

        with pytest.raises(ValueError, match="cardinality"):
            CategoricalSchema(columns=[config])

    def test_negative_embed_dim_rejected(self):
        """Test that negative embed_dim is rejected."""
        config = CategoricalColumnConfig(name="location", embed_dim=-1)

        with pytest.raises(ValueError, match="embed"):
            CategoricalSchema(columns=[config])

    def test_zero_embed_dim_rejected(self):
        """Test that zero embed_dim is rejected."""
        config = CategoricalColumnConfig(name="location", embed_dim=0)

        with pytest.raises(ValueError, match="embed"):
            CategoricalSchema(columns=[config])

    def test_empty_column_name_rejected(self):
        """Test that empty column name is rejected."""
        config = CategoricalColumnConfig(name="")

        with pytest.raises(ValueError, match="[Ee]mpty|[Nn]ame"):
            CategoricalSchema(columns=[config])


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

    def test_fit_with_schema(self, sample_metadata):
        """Test fitting encoder using CategoricalSchema."""
        schema = CategoricalSchema.from_column_names(["location", "season"])
        encoder = CategoricalEncoder(schema=schema)
        encoder.fit(sample_metadata)

        assert encoder.column_names == ["location", "season"]

    def test_fit_no_columns_no_schema_raises(self, sample_metadata):
        """Test fit raises error without columns or schema."""
        encoder = CategoricalEncoder()
        with pytest.raises(ValueError, match="column|schema"):
            encoder.fit(sample_metadata)

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
