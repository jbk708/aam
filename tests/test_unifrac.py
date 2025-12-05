"""Unit tests for UniFracComputer class."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch
from biom import Table
from skbio import DistanceMatrix
import pandas as pd

from aam.data.unifrac import UniFracComputer
from aam.data.biom_loader import BIOMLoader


def generate_150bp_sequence(seed=None):
    """Generate a random 150bp DNA sequence."""
    import random

    if seed is not None:
        random.seed(seed)
    bases = "ACGT"
    return "".join(random.choice(bases) for _ in range(150))


def create_simple_tree_file(tmp_path, observation_ids):
    """Create a simple Newick tree file for testing."""
    tree_path = tmp_path / "test_tree.nwk"

    if len(observation_ids) == 1:
        tree_str = f"{observation_ids[0]}:0.1;"
    elif len(observation_ids) == 2:
        tree_str = f"({observation_ids[0]}:0.1,{observation_ids[1]}:0.1);"
    else:
        tips = [f"{obs_id}:0.1" for obs_id in observation_ids]
        tree_str = "(" + ",".join(tips) + ");"

    tree_path.write_text(tree_str)
    return str(tree_path)


@pytest.fixture
def computer():
    """Create a UniFracComputer instance."""
    return UniFracComputer()


@pytest.fixture
def simple_table():
    """Create a simple BIOM table for testing."""
    data = np.array([[10, 20, 5], [15, 10, 25], [5, 30, 10]])
    observation_ids = [
        generate_150bp_sequence(seed=1),
        generate_150bp_sequence(seed=2),
        generate_150bp_sequence(seed=3),
    ]
    sample_ids = ["sample1", "sample2", "sample3"]
    return Table(data, observation_ids=observation_ids, sample_ids=sample_ids)


@pytest.fixture
def rarefied_table(simple_table):
    """Create a rarefied BIOM table for testing."""
    loader = BIOMLoader()
    return loader.rarefy(simple_table, depth=20, random_seed=42)


@pytest.fixture
def simple_tree_file(tmp_path, simple_table):
    """Create a simple tree file matching the simple_table."""
    observation_ids = list(simple_table.ids(axis="observation"))
    return create_simple_tree_file(tmp_path, observation_ids)


@pytest.fixture
def sample_distance_matrix():
    """Create a sample DistanceMatrix for testing batch extraction."""
    data = np.array(
        [
            [0.0, 0.5, 0.3, 0.7],
            [0.5, 0.0, 0.6, 0.4],
            [0.3, 0.6, 0.0, 0.8],
            [0.7, 0.4, 0.8, 0.0],
        ]
    )
    sample_ids = ["sample1", "sample2", "sample3", "sample4"]
    return DistanceMatrix(data, ids=sample_ids)


@pytest.fixture
def sample_faith_pd_series():
    """Create a sample Faith PD Series for testing."""
    values = np.array([2.5, 3.1, 2.8, 3.5])
    sample_ids = ["sample1", "sample2", "sample3", "sample4"]
    return pd.Series(values, index=sample_ids)


class TestUniFracComputer:
    """Test suite for UniFracComputer class."""

    def test_init(self, computer):
        """Test UniFracComputer initialization."""
        assert computer is not None
        assert isinstance(computer, UniFracComputer)

    def test_compute_unweighted_basic(self, computer, rarefied_table, simple_tree_file):
        """Test basic compute_unweighted functionality."""
        result = computer.compute_unweighted(rarefied_table, simple_tree_file)

        assert isinstance(result, DistanceMatrix)
        assert result.shape[0] == result.shape[1]
        assert result.shape[0] == len(rarefied_table.ids(axis="sample"))
        np.testing.assert_array_almost_equal(result.data, result.data.T)
        assert np.all(result.data >= 0)
        np.testing.assert_array_almost_equal(np.diag(result.data), 0)

    def test_compute_faith_pd_basic(self, computer, rarefied_table, simple_tree_file):
        """Test basic compute_faith_pd functionality."""
        result = computer.compute_faith_pd(rarefied_table, simple_tree_file)

        assert isinstance(result, pd.Series)
        assert len(result) == len(rarefied_table.ids(axis="sample"))
        assert list(result.index) == list(rarefied_table.ids(axis="sample"))
        assert np.all(result.values >= 0)

    def test_extract_batch_distances_basic(self, computer, sample_distance_matrix):
        """Test basic extract_batch_distances functionality."""
        sample_ids = ["sample1", "sample2"]
        result = computer.extract_batch_distances(sample_distance_matrix, sample_ids)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        np.testing.assert_array_almost_equal(result, result.T)
        np.testing.assert_array_almost_equal(np.diag(result), 0)
        expected = sample_distance_matrix.filter(sample_ids).data
        np.testing.assert_array_almost_equal(result, expected)

    def test_validate_batch_size_even(self, computer):
        """Test that validate_batch_size accepts even batch sizes."""
        computer.validate_batch_size(2)
        computer.validate_batch_size(4)
        computer.validate_batch_size(8)
        computer.validate_batch_size(16)
        computer.validate_batch_size(32)

    def test_validate_batch_size_odd(self, computer):
        """Test that validate_batch_size rejects odd batch sizes."""
        with pytest.raises(ValueError, match="Batch size must be even"):
            computer.validate_batch_size(1)

        with pytest.raises(ValueError, match="Batch size must be even"):
            computer.validate_batch_size(3)

        with pytest.raises(ValueError, match="Batch size must be even"):
            computer.validate_batch_size(5)

        with pytest.raises(ValueError, match="Batch size must be even"):
            computer.validate_batch_size(15)

    def test_validate_batch_size_zero(self, computer):
        """Test that validate_batch_size accepts zero (edge case)."""
        computer.validate_batch_size(0)

    def test_validate_batch_size_negative(self, computer):
        """Test that validate_batch_size handles negative numbers."""
        with pytest.raises(ValueError, match="Batch size must be even"):
            computer.validate_batch_size(-1)

        computer.validate_batch_size(-2)


class TestUniFracComputerIntegration:
    """Integration tests for UniFracComputer (require real data and library)."""

    @pytest.mark.integration
    def test_compute_unweighted_with_real_data(self, computer):
        """Test compute_unweighted with real BIOM table and tree."""
        biom_file = Path("data/fall_train_only_all_outdoor.biom")
        tree_file = Path("data/all-outdoors_sepp_tree.nwk")

        if not biom_file.exists() or not tree_file.exists():
            pytest.skip("Test data files not found")

        loader = BIOMLoader()
        table = loader.load_table(str(biom_file))
        rarefied = loader.rarefy(table, depth=1000, random_seed=42)

        result = computer.compute_unweighted(rarefied, str(tree_file))
        assert isinstance(result, DistanceMatrix)
        assert result.shape[0] == result.shape[1]
        assert result.shape[0] == len(rarefied.ids(axis="sample"))
        np.testing.assert_array_almost_equal(result.data, result.data.T)
        assert np.all(result.data >= 0)
        np.testing.assert_array_almost_equal(np.diag(result.data), 0)

    @pytest.mark.integration
    def test_compute_faith_pd_with_real_data(self, computer):
        """Test compute_faith_pd with real BIOM table and tree."""
        biom_file = Path("data/fall_train_only_all_outdoor.biom")
        tree_file = Path("data/all-outdoors_sepp_tree.nwk")

        if not biom_file.exists() or not tree_file.exists():
            pytest.skip("Test data files not found")

        loader = BIOMLoader()
        table = loader.load_table(str(biom_file))
        rarefied = loader.rarefy(table, depth=1000, random_seed=42)

        result = computer.compute_faith_pd(rarefied, str(tree_file))
        assert isinstance(result, pd.Series)
        assert len(result) == len(rarefied.ids(axis="sample"))
        assert np.all(result.values >= 0)

    @pytest.mark.integration
    def test_compute_unweighted_tree_file_not_found(self, computer, rarefied_table):
        """Test compute_unweighted with non-existent tree file."""
        with pytest.raises(FileNotFoundError):
            computer.compute_unweighted(rarefied_table, "nonexistent_tree.nwk")

    @pytest.mark.integration
    def test_compute_faith_pd_tree_file_not_found(self, computer, rarefied_table):
        """Test compute_faith_pd with non-existent tree file."""
        with pytest.raises(FileNotFoundError):
            computer.compute_faith_pd(rarefied_table, "nonexistent_tree.nwk")


class TestUniFracComputerErrorHandling:
    """Test error handling for UniFracComputer."""

    def test_compute_unweighted_file_not_found(self, computer, rarefied_table, tmp_path):
        """Test compute_unweighted with non-existent tree file."""
        non_existent = tmp_path / "nonexistent.nwk"
        with pytest.raises(FileNotFoundError, match="Tree file not found"):
            computer.compute_unweighted(rarefied_table, str(non_existent))

    def test_compute_unweighted_invalid_tree_format(self, computer, rarefied_table, tmp_path):
        """Test compute_unweighted with invalid tree file format."""
        invalid_tree = tmp_path / "invalid.nwk"
        invalid_tree.write_text("This is not a valid Newick tree")
        with pytest.raises(ValueError, match="Error loading phylogenetic tree"):
            computer.compute_unweighted(rarefied_table, str(invalid_tree))

    def test_compute_unweighted_asv_mismatch(self, computer, rarefied_table, tmp_path):
        """Test compute_unweighted with ASV ID mismatch."""
        mismatched_tree = tmp_path / "mismatched.nwk"
        mismatched_tree.write_text("(ASV999:0.1,ASV888:0.1);")
        with pytest.raises(ValueError, match="Error computing unweighted UniFrac"):
            computer.compute_unweighted(rarefied_table, str(mismatched_tree))

    def test_compute_unweighted_general_error(self, computer, rarefied_table, tmp_path):
        """Test compute_unweighted with general computation error."""
        tree_file = tmp_path / "tree.nwk"
        tree_file.write_text("(A:0.1,B:0.2);")
        with patch("aam.data.unifrac.unifrac.unweighted") as mock_unifrac:
            mock_unifrac.side_effect = Exception("General error")
            with pytest.raises(ValueError, match="Error computing unweighted UniFrac"):
                computer.compute_unweighted(rarefied_table, str(tree_file))

    def test_compute_faith_pd_file_not_found(self, computer, rarefied_table, tmp_path):
        """Test compute_faith_pd with non-existent tree file."""
        non_existent = tmp_path / "nonexistent.nwk"
        with pytest.raises(FileNotFoundError, match="Tree file not found"):
            computer.compute_faith_pd(rarefied_table, str(non_existent))

    def test_compute_faith_pd_invalid_tree_format(self, computer, rarefied_table, tmp_path):
        """Test compute_faith_pd with invalid tree file format."""
        invalid_tree = tmp_path / "invalid.nwk"
        invalid_tree.write_text("This is not a valid Newick tree")
        with pytest.raises(ValueError, match="Error loading phylogenetic tree"):
            computer.compute_faith_pd(rarefied_table, str(invalid_tree))

    def test_compute_faith_pd_asv_mismatch(self, computer, rarefied_table, tmp_path):
        """Test compute_faith_pd with ASV ID mismatch."""
        mismatched_tree = tmp_path / "mismatched.nwk"
        mismatched_tree.write_text("(ASV999:0.1,ASV888:0.1);")
        with pytest.raises(ValueError, match="Error computing Faith PD"):
            computer.compute_faith_pd(rarefied_table, str(mismatched_tree))

    def test_compute_faith_pd_general_error(self, computer, rarefied_table, tmp_path):
        """Test compute_faith_pd with general computation error."""
        tree_file = tmp_path / "tree.nwk"
        tree_file.write_text("(A:0.1,B:0.2);")
        with patch("aam.data.unifrac.unifrac.faith_pd") as mock_faith_pd:
            mock_faith_pd.side_effect = Exception("General error")
            with pytest.raises(ValueError, match="Error computing Faith PD"):
                computer.compute_faith_pd(rarefied_table, str(tree_file))

    def test_extract_batch_distances_missing_sample_id(self, computer, sample_distance_matrix):
        """Test extract_batch_distances with missing sample ID."""
        sample_ids = ["sample1", "nonexistent"]
        with pytest.raises(ValueError, match="not found in distance matrix"):
            computer.extract_batch_distances(sample_distance_matrix, sample_ids)

    def test_extract_batch_distances_faith_pd_missing_sample_id(self, computer, sample_faith_pd_series):
        """Test extract_batch_distances with missing sample ID for Faith PD."""
        sample_ids = ["sample1", "nonexistent"]
        with pytest.raises(ValueError, match="not found in Faith PD series"):
            computer.extract_batch_distances(sample_faith_pd_series, sample_ids, metric="faith_pd")

    def test_extract_batch_distances_invalid_batch_size(self, computer, sample_distance_matrix):
        """Test extract_batch_distances with odd batch size."""
        sample_ids = ["sample1", "sample2", "sample3"]
        with pytest.raises(ValueError, match="Batch size must be even"):
            computer.extract_batch_distances(sample_distance_matrix, sample_ids)

    def test_extract_batch_distances_invalid_metric(self, computer, sample_distance_matrix):
        """Test extract_batch_distances with invalid metric."""
        sample_ids = ["sample1", "sample2"]
        with pytest.raises(ValueError, match="Invalid metric"):
            computer.extract_batch_distances(sample_distance_matrix, sample_ids, metric="invalid")


class TestExtractBatchDistances:
    """Tests for extract_batch_distances method."""

    def test_extract_batch_distances_unweighted(self, computer, sample_distance_matrix):
        """Test extract_batch_distances for unweighted UniFrac."""
        sample_ids = ["sample1", "sample2"]
        result = computer.extract_batch_distances(sample_distance_matrix, sample_ids, metric="unweighted")

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        np.testing.assert_array_almost_equal(result, result.T)
        np.testing.assert_array_almost_equal(np.diag(result), 0)
        # Result should match the order of sample_ids (even if filtered matrix has different order)
        expected_filtered = sample_distance_matrix.filter(sample_ids)
        # Extract distances in the exact order of sample_ids
        id_to_idx = {id_: idx for idx, id_ in enumerate(expected_filtered.ids)}
        reorder_indices = [id_to_idx[id_] for id_ in sample_ids]
        expected = expected_filtered.data[np.ix_(reorder_indices, reorder_indices)]
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_extract_batch_distances_unweighted_shuffled_order(self, computer, sample_distance_matrix):
        """Test extract_batch_distances preserves shuffled batch order."""
        # Test with shuffled order: sample2, sample1 (instead of sample1, sample2)
        sample_ids = ["sample2", "sample1"]
        result = computer.extract_batch_distances(sample_distance_matrix, sample_ids, metric="unweighted")

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        np.testing.assert_array_almost_equal(result, result.T)
        np.testing.assert_array_almost_equal(np.diag(result), 0)
        
        # Verify the order matches sample_ids: [sample2, sample1]
        # Distance from sample2 to sample2 should be 0 (diagonal)
        # Distance from sample2 to sample1 should be at position [0, 1]
        expected_filtered = sample_distance_matrix.filter(sample_ids)
        id_to_idx = {id_: idx for idx, id_ in enumerate(expected_filtered.ids)}
        reorder_indices = [id_to_idx[id_] for id_ in sample_ids]
        expected = expected_filtered.data[np.ix_(reorder_indices, reorder_indices)]
        np.testing.assert_array_almost_equal(result, expected)
        
        # Verify that result[0, 1] is the distance from sample2 to sample1
        # In the original matrix: sample1=idx0, sample2=idx1, so distance is at [1, 0] = 0.5
        assert np.isclose(result[0, 1], 0.5), f"Expected distance from sample2 to sample1 to be 0.5, got {result[0, 1]}"

    def test_extract_batch_distances_faith_pd(self, computer, sample_faith_pd_series):
        """Test extract_batch_distances for Faith PD."""
        sample_ids = ["sample1", "sample2"]
        result = computer.extract_batch_distances(sample_faith_pd_series, sample_ids, metric="faith_pd")

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 1)
        expected = sample_faith_pd_series.loc[sample_ids].to_numpy().reshape(-1, 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_extract_batch_distances_odd_batch_size(self, computer, sample_distance_matrix):
        """Test extract_batch_distances validates batch size is even."""
        with pytest.raises(ValueError, match="even|multiple of 2"):
            computer.extract_batch_distances(sample_distance_matrix, ["sample1", "sample2", "sample3"], metric="unweighted")

    def test_extract_batch_distances_missing_sample_ids(self, computer, sample_distance_matrix):
        """Test extract_batch_distances with sample IDs not in distance matrix."""
        with pytest.raises(ValueError, match="not found"):
            computer.extract_batch_distances(sample_distance_matrix, ["nonexistent1", "nonexistent2"], metric="unweighted")

    def test_extract_batch_distances_empty_sample_ids(self, computer, sample_distance_matrix):
        """Test extract_batch_distances with empty sample IDs list."""
        result = computer.extract_batch_distances(sample_distance_matrix, [], metric="unweighted")
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 0)

    def test_extract_batch_distances_large_batch(self, computer, sample_distance_matrix):
        """Test extract_batch_distances with all samples (full batch)."""
        sample_ids = ["sample1", "sample2", "sample3", "sample4"]
        result = computer.extract_batch_distances(sample_distance_matrix, sample_ids, metric="unweighted")

        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 4)
        np.testing.assert_array_almost_equal(result, sample_distance_matrix.data)

    def test_extract_batch_distances_invalid_metric(self, computer, sample_distance_matrix):
        """Test extract_batch_distances with invalid metric."""
        with pytest.raises(ValueError, match="Invalid metric"):
            computer.extract_batch_distances(sample_distance_matrix, ["sample1", "sample2"], metric="invalid_metric")
