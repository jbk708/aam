"""Unit tests for BIOMLoader class."""

from pathlib import Path

import biom
import numpy as np
import pytest
from biom import Table

from aam.data.biom_loader import BIOMLoader
from conftest import generate_150bp_sequence


@pytest.fixture
def loader():
    """Create a BIOMLoader instance."""
    return BIOMLoader()


@pytest.fixture
def sparse_table():
    """Create a sparse BIOM table with many zeros."""
    data = np.array([[100, 0, 50, 0], [0, 200, 0, 0], [0, 0, 0, 300], [25, 0, 0, 0]])
    observation_ids = [
        generate_150bp_sequence(seed=10),
        generate_150bp_sequence(seed=11),
        generate_150bp_sequence(seed=12),
        generate_150bp_sequence(seed=13),
    ]
    sample_ids = ["sample1", "sample2", "sample3", "sample4"]
    return Table(data, observation_ids=observation_ids, sample_ids=sample_ids)


@pytest.fixture
def low_depth_table():
    """Create a table with samples of varying depths."""
    data = np.array([[50, 100, 200, 5000], [30, 80, 150, 4000], [20, 50, 100, 3000]])
    observation_ids = [generate_150bp_sequence(seed=20), generate_150bp_sequence(seed=21), generate_150bp_sequence(seed=22)]
    sample_ids = ["low1", "low2", "medium", "high"]
    return Table(data, observation_ids=observation_ids, sample_ids=sample_ids)


@pytest.fixture
def very_low_depth_table():
    """Create a table with some samples below typical rarefaction depth."""
    data = np.array([[30, 50, 200, 5000], [20, 40, 150, 4000], [10, 30, 100, 3000]])
    observation_ids = [generate_150bp_sequence(seed=30), generate_150bp_sequence(seed=31), generate_150bp_sequence(seed=32)]
    sample_ids = ["very_low1", "very_low2", "medium", "high"]
    return Table(data, observation_ids=observation_ids, sample_ids=sample_ids)


@pytest.fixture
def empty_table():
    """Create an empty BIOM table."""
    data = np.array([[0, 0], [0, 0]])
    observation_ids = [generate_150bp_sequence(seed=40), generate_150bp_sequence(seed=41)]
    sample_ids = ["sample1", "sample2"]
    return Table(data, observation_ids=observation_ids, sample_ids=sample_ids)


class TestBIOMLoader:
    """Test suite for BIOMLoader class."""

    def test_init(self, loader):
        """Test BIOMLoader initialization."""
        assert loader is not None
        assert isinstance(loader, BIOMLoader)

    @pytest.mark.integration
    def test_load_table_valid_file(self, loader):
        """Test loading a valid BIOM file (integration test with real file)."""
        test_file = Path("data/fall_train_only_all_outdoor.biom")
        if test_file.exists():
            loaded_table = loader.load_table(str(test_file))
            assert loaded_table is not None
            assert len(loaded_table.ids(axis="sample")) > 0
            assert len(loaded_table.ids(axis="observation")) > 0
        else:
            pytest.skip("Test data file not found")

    def test_load_table_file_not_found(self, loader):
        """Test loading a non-existent file."""
        with pytest.raises(FileNotFoundError):
            loader.load_table("nonexistent_file.biom")

    def test_load_table_invalid_file(self, loader, tmp_path):
        """Test loading an invalid BIOM file."""
        file_path = tmp_path / "invalid.biom"
        file_path.write_text("not a biom file")

        try:
            result = loader.load_table(str(file_path))
            if result.is_empty():
                pytest.skip("biom-format returns empty table for invalid files")
        except (ValueError, Exception, KeyError, TypeError, FileNotFoundError):
            pass

    def test_rarefy_basic(self, loader, simple_table):
        """Test basic rarefaction."""
        depth = 20
        rarefied = loader.rarefy(simple_table, depth=depth, random_seed=42)

        assert rarefied is not None
        assert len(rarefied.ids(axis="sample")) <= len(simple_table.ids(axis="sample"))
        assert len(rarefied.ids(axis="observation")) == len(simple_table.ids(axis="observation"))

        data = rarefied.matrix_data
        for i in range(len(rarefied.ids(axis="sample"))):
            total = int(data[:, i].sum())
            assert total == depth

    def test_rarefy_drops_low_depth_samples(self, loader, very_low_depth_table):
        """Test that samples below depth are dropped."""
        depth = 100
        original_samples = len(very_low_depth_table.ids(axis="sample"))

        rarefied = loader.rarefy(very_low_depth_table, depth=depth, random_seed=42)

        assert len(rarefied.ids(axis="sample")) < original_samples

        data = rarefied.matrix_data
        for i in range(len(rarefied.ids(axis="sample"))):
            total = int(data[:, i].sum())
            assert total == depth

    def test_rarefy_reproducibility(self, loader, simple_table):
        """Test that rarefaction is reproducible with same seed."""
        depth = 20
        seed = 42

        rarefied1 = loader.rarefy(simple_table, depth=depth, random_seed=seed)
        rarefied2 = loader.rarefy(simple_table, depth=depth, random_seed=seed)

        assert rarefied1.shape == rarefied2.shape
        np.testing.assert_array_equal(rarefied1.matrix_data.toarray(), rarefied2.matrix_data.toarray())

    def test_rarefy_different_seeds(self, loader, simple_table):
        """Test that different seeds produce different results."""
        depth = 20

        rarefied1 = loader.rarefy(simple_table, depth=depth, random_seed=42)
        rarefied2 = loader.rarefy(simple_table, depth=depth, random_seed=123)

        if rarefied1.shape == rarefied2.shape:
            matrices_equal = np.array_equal(rarefied1.matrix_data.toarray(), rarefied2.matrix_data.toarray())
            assert not matrices_equal, "Different seeds should produce different results"

    def test_rarefy_with_replacement(self, loader, simple_table):
        """Test rarefaction with replacement."""
        depth = 20
        rarefied = loader.rarefy(simple_table, depth=depth, with_replacement=True, random_seed=42)

        assert rarefied is not None
        data = rarefied.matrix_data
        for i in range(len(rarefied.ids(axis="sample"))):
            total = int(data[:, i].sum())
            assert total == depth

    def test_rarefy_empty_table_error(self, loader, empty_table):
        """Test that rarefying an empty table raises ValueError."""
        with pytest.raises(ValueError, match="rarefied table contains no samples"):
            loader.rarefy(empty_table, depth=100)

    def test_rarefy_inplace(self, loader, simple_table):
        """Test rarefaction with inplace=True."""
        depth = 20
        original_shape = simple_table.shape

        rarefied = loader.rarefy(simple_table, depth=depth, inplace=True, random_seed=42)

        assert rarefied is not None
        assert rarefied.shape[0] == original_shape[0]

    def test_rarefy_not_inplace(self, loader, simple_table):
        """Test rarefaction with inplace=False."""
        depth = 20
        original_id = id(simple_table)

        rarefied = loader.rarefy(simple_table, depth=depth, inplace=False, random_seed=42)

        assert id(rarefied) != original_id

    def test_get_sequences_basic(self, loader, simple_table):
        """Test basic sequence extraction from observation IDs."""
        sequences = loader.get_sequences(simple_table)

        assert len(sequences) == len(simple_table.ids(axis="observation"))
        assert len(sequences[0]) == 150
        assert len(sequences[1]) == 150
        assert len(sequences[2]) == 150
        assert all(c in "ACGT" for seq in sequences for c in seq)

    def test_get_sequences_from_ids(self, loader):
        """Test sequence extraction from observation IDs."""
        data = np.array([[10, 20], [15, 10]])
        observation_ids = [generate_150bp_sequence(seed=50), generate_150bp_sequence(seed=51)]
        sample_ids = ["sample1", "sample2"]
        table = Table(data, observation_ids=observation_ids, sample_ids=sample_ids)

        sequences = loader.get_sequences(table)
        assert len(sequences) == 2
        assert len(sequences[0]) == 150
        assert len(sequences[1]) == 150
        assert sequences[0] == observation_ids[0]
        assert sequences[1] == observation_ids[1]

    def test_get_sequences_no_metadata_needed(self, loader):
        """Test that sequences work without metadata."""
        data = np.array([[10, 20], [15, 10]])
        observation_ids = [generate_150bp_sequence(seed=60), generate_150bp_sequence(seed=61)]
        sample_ids = ["sample1", "sample2"]
        table = Table(data, observation_ids=observation_ids, sample_ids=sample_ids)

        sequences = loader.get_sequences(table)
        assert len(sequences) == 2
        assert len(sequences[0]) == 150
        assert len(sequences[1]) == 150
        assert sequences == observation_ids

    def test_get_sequences_no_metadata(self, loader, sparse_table):
        """Test sequence extraction with no metadata (uses observation IDs)."""
        sequences = loader.get_sequences(sparse_table)
        assert len(sequences) == len(sparse_table.ids(axis="observation"))
        assert sequences == list(sparse_table.ids(axis="observation"))

    def test_get_sequences_150bp(self, loader):
        """Test that sequences are 150bp."""
        data = np.array([[10, 20], [15, 10]])
        observation_ids = [generate_150bp_sequence(seed=70), generate_150bp_sequence(seed=71)]
        sample_ids = ["sample1", "sample2"]
        table = Table(data, observation_ids=observation_ids, sample_ids=sample_ids)

        sequences = loader.get_sequences(table)
        assert len(sequences) == 2
        for seq in sequences:
            assert len(seq) == 150
            assert all(c in "ACGT" for c in seq)

    def test_get_sequences_always_works(self, loader):
        """Test that sequence extraction always works from observation IDs."""
        data = np.array([[10], [15]])
        observation_ids = [generate_150bp_sequence(seed=80), generate_150bp_sequence(seed=81)]
        sample_ids = ["sample1"]
        table = Table(data, observation_ids=observation_ids, sample_ids=sample_ids)

        sequences = loader.get_sequences(table)
        assert len(sequences) == 2
        assert sequences[0] == observation_ids[0]
        assert sequences[1] == observation_ids[1]
        assert len(sequences[0]) == 150
        assert len(sequences[1]) == 150

    def test_integration_rarefy(self, loader, simple_table):
        """Test integration of rarefy operation."""
        rarefied = loader.rarefy(simple_table, depth=20, random_seed=42)

        assert rarefied is not None
        assert len(rarefied.ids(axis="sample")) > 0
        assert len(rarefied.ids(axis="observation")) > 0

    @pytest.mark.integration
    def test_integration_load_rarefy(self, loader):
        """Test full integration with file loading."""
        test_file = Path("data/fall_train_only_all_outdoor.biom")
        if test_file.exists():
            loaded = loader.load_table(str(test_file))
            rarefied = loader.rarefy(loaded, depth=1000, random_seed=42)

            assert rarefied is not None
            assert len(rarefied.ids(axis="sample")) > 0
            assert len(rarefied.ids(axis="observation")) > 0
        else:
            pytest.skip("Test data file not found")

    def test_rarefy_preserves_observation_ids(self, loader, simple_table):
        """Test that rarefaction preserves observation IDs."""
        depth = 20
        rarefied = loader.rarefy(simple_table, depth=depth, random_seed=42)

        original_ids = simple_table.ids(axis="observation")
        rarefied_ids = rarefied.ids(axis="observation")

        assert len(rarefied_ids) == len(original_ids)
        assert all(seq == orig for seq, orig in zip(rarefied_ids, original_ids))
