"""Unit tests for categorical metadata schema and encoding."""

import pytest

from aam.data.categorical import CategoricalColumnConfig, CategoricalSchema


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
