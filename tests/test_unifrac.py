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
    return UniFracComputer(num_threads=1)  # Use 1 thread for tests to avoid environment variable conflicts


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

    @pytest.mark.skip(reason="Computation functionality deprecated in PYT-11.4. Use pre-computed matrices instead.")
    def test_compute_unweighted_basic(self, computer, rarefied_table, simple_tree_file):
        """Test basic compute_unweighted functionality.
        
        DEPRECATED: Computation functionality removed in PYT-11.4.
        Users should generate UniFrac matrices using unifrac-binaries.
        """
        pytest.skip("Computation functionality deprecated. Use pre-computed matrices.")

    @pytest.mark.skip(reason="Computation functionality deprecated in PYT-11.4. Use pre-computed matrices instead.")
    def test_compute_faith_pd_basic(self, computer, rarefied_table, simple_tree_file):
        """Test basic compute_faith_pd functionality.
        
        DEPRECATED: Computation functionality removed in PYT-11.4.
        Users should generate UniFrac matrices using unifrac-binaries.
        """
        pytest.skip("Computation functionality deprecated. Use pre-computed matrices.")

    def test_extract_batch_distances_basic(self, sample_distance_matrix):
        """Test basic extract_batch_distances functionality using UniFracLoader."""
        from aam.data.unifrac_loader import UniFracLoader
        
        loader = UniFracLoader()
        sample_ids = ["sample1", "sample2"]
        result = loader.extract_batch_distances(sample_distance_matrix, sample_ids, metric="unweighted")

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        np.testing.assert_array_almost_equal(result, result.T)
        np.testing.assert_array_almost_equal(np.diag(result), 0)
        expected = sample_distance_matrix.filter(sample_ids).data
        np.testing.assert_array_almost_equal(result, expected)

    @pytest.mark.skip(reason="validate_batch_size deprecated in PYT-11.4. Batch size validation no longer needed.")
    def test_validate_batch_size_even(self, computer):
        """Test that validate_batch_size accepts even batch sizes.
        
        DEPRECATED: Batch size validation removed in PYT-11.4.
        """
        pytest.skip("validate_batch_size deprecated. Batch size validation no longer needed.")

    @pytest.mark.skip(reason="validate_batch_size deprecated in PYT-11.4. Batch size validation no longer needed.")
    def test_validate_batch_size_odd(self, computer):
        """Test that validate_batch_size rejects odd batch sizes.
        
        DEPRECATED: Batch size validation removed in PYT-11.4.
        """
        pytest.skip("validate_batch_size deprecated. Batch size validation no longer needed.")

    @pytest.mark.skip(reason="validate_batch_size deprecated in PYT-11.4. Batch size validation no longer needed.")
    def test_validate_batch_size_zero(self, computer):
        """Test that validate_batch_size accepts zero (edge case).
        
        DEPRECATED: Batch size validation removed in PYT-11.4.
        """
        pytest.skip("validate_batch_size deprecated. Batch size validation no longer needed.")

    @pytest.mark.skip(reason="validate_batch_size deprecated in PYT-11.4. Batch size validation no longer needed.")
    def test_validate_batch_size_negative(self, computer):
        """Test that validate_batch_size handles negative numbers.
        
        DEPRECATED: Batch size validation removed in PYT-11.4.
        """
        pytest.skip("validate_batch_size deprecated. Batch size validation no longer needed.")


class TestUniFracComputerIntegration:
    """Integration tests for UniFracComputer (require real data and library).
    
    DEPRECATED: Computation functionality removed in PYT-11.4.
    These tests are kept for reference but are skipped.
    """

    @pytest.mark.integration
    @pytest.mark.skip(reason="Computation functionality deprecated in PYT-11.4. Use pre-computed matrices instead.")
    def test_compute_unweighted_with_real_data(self, computer):
        """Test compute_unweighted with real BIOM table and tree.
        
        DEPRECATED: Computation functionality removed in PYT-11.4.
        Users should generate UniFrac matrices using unifrac-binaries.
        """
        pytest.skip("Computation functionality deprecated. Use pre-computed matrices.")

    @pytest.mark.integration
    @pytest.mark.skip(reason="Computation functionality deprecated in PYT-11.4. Use pre-computed matrices instead.")
    def test_compute_faith_pd_with_real_data(self, computer):
        """Test compute_faith_pd with real BIOM table and tree.
        
        DEPRECATED: Computation functionality removed in PYT-11.4.
        Users should generate UniFrac matrices using unifrac-binaries.
        """
        pytest.skip("Computation functionality deprecated. Use pre-computed matrices.")

    @pytest.mark.integration
    @pytest.mark.skip(reason="Computation functionality deprecated in PYT-11.4. Use pre-computed matrices instead.")
    def test_compute_unweighted_tree_file_not_found(self, computer, rarefied_table):
        """Test compute_unweighted with non-existent tree file.
        
        DEPRECATED: Computation functionality removed in PYT-11.4.
        """
        pytest.skip("Computation functionality deprecated. Use pre-computed matrices.")

    @pytest.mark.integration
    @pytest.mark.skip(reason="Computation functionality deprecated in PYT-11.4. Use pre-computed matrices instead.")
    def test_compute_faith_pd_tree_file_not_found(self, computer, rarefied_table):
        """Test compute_faith_pd with non-existent tree file.
        
        DEPRECATED: Computation functionality removed in PYT-11.4.
        """
        pytest.skip("Computation functionality deprecated. Use pre-computed matrices.")


class TestUniFracComputerErrorHandling:
    """Test error handling for UniFracComputer.
    
    DEPRECATED: Computation functionality removed in PYT-11.4.
    These tests are kept for reference but are skipped.
    """

    @pytest.mark.skip(reason="Computation functionality deprecated in PYT-11.4. Use pre-computed matrices instead.")
    def test_compute_unweighted_file_not_found(self, computer, rarefied_table, tmp_path):
        """Test compute_unweighted with non-existent tree file.
        
        DEPRECATED: Computation functionality removed in PYT-11.4.
        """
        pytest.skip("Computation functionality deprecated. Use pre-computed matrices.")

    @pytest.mark.skip(reason="Computation functionality deprecated in PYT-11.4. Use pre-computed matrices instead.")
    def test_compute_unweighted_invalid_tree_format(self, computer, rarefied_table, tmp_path):
        """Test compute_unweighted with invalid tree file format.
        
        DEPRECATED: Computation functionality removed in PYT-11.4.
        """
        pytest.skip("Computation functionality deprecated. Use pre-computed matrices.")

    @pytest.mark.skip(reason="Computation functionality deprecated in PYT-11.4. Use pre-computed matrices instead.")
    def test_compute_unweighted_asv_mismatch(self, computer, rarefied_table, tmp_path):
        """Test compute_unweighted with ASV ID mismatch.
        
        DEPRECATED: Computation functionality removed in PYT-11.4.
        """
        pytest.skip("Computation functionality deprecated. Use pre-computed matrices.")

    @pytest.mark.skip(reason="Computation functionality deprecated in PYT-11.4. Use pre-computed matrices instead.")
    def test_compute_unweighted_general_error(self, computer, rarefied_table, tmp_path):
        """Test compute_unweighted with general computation error.
        
        DEPRECATED: Computation functionality removed in PYT-11.4.
        """
        pytest.skip("Computation functionality deprecated. Use pre-computed matrices.")

    @pytest.mark.skip(reason="Computation functionality deprecated in PYT-11.4. Use pre-computed matrices instead.")
    def test_compute_faith_pd_file_not_found(self, computer, rarefied_table, tmp_path):
        """Test compute_faith_pd with non-existent tree file.
        
        DEPRECATED: Computation functionality removed in PYT-11.4.
        """
        pytest.skip("Computation functionality deprecated. Use pre-computed matrices.")

    @pytest.mark.skip(reason="Computation functionality deprecated in PYT-11.4. Use pre-computed matrices instead.")
    def test_compute_faith_pd_invalid_tree_format(self, computer, rarefied_table, tmp_path):
        """Test compute_faith_pd with invalid tree file format.
        
        DEPRECATED: Computation functionality removed in PYT-11.4.
        """
        pytest.skip("Computation functionality deprecated. Use pre-computed matrices.")

    @pytest.mark.skip(reason="Computation functionality deprecated in PYT-11.4. Use pre-computed matrices instead.")
    def test_compute_faith_pd_asv_mismatch(self, computer, rarefied_table, tmp_path):
        """Test compute_faith_pd with ASV ID mismatch.
        
        DEPRECATED: Computation functionality removed in PYT-11.4.
        """
        pytest.skip("Computation functionality deprecated. Use pre-computed matrices.")

    @pytest.mark.skip(reason="Computation functionality deprecated in PYT-11.4. Use pre-computed matrices instead.")
    def test_compute_faith_pd_general_error(self, computer, rarefied_table, tmp_path):
        """Test compute_faith_pd with general computation error.
        
        DEPRECATED: Computation functionality removed in PYT-11.4.
        """
        pytest.skip("Computation functionality deprecated. Use pre-computed matrices.")

    def test_extract_batch_distances_missing_sample_id(self, sample_distance_matrix):
        """Test extract_batch_distances with missing sample ID using UniFracLoader."""
        from aam.data.unifrac_loader import UniFracLoader
        
        loader = UniFracLoader()
        sample_ids = ["sample1", "nonexistent"]
        with pytest.raises(ValueError, match="not found in distance matrix"):
            loader.extract_batch_distances(sample_distance_matrix, sample_ids, metric="unweighted")

    def test_extract_batch_distances_faith_pd_missing_sample_id(self, sample_faith_pd_series):
        """Test extract_batch_distances with missing sample ID for Faith PD using UniFracLoader."""
        from aam.data.unifrac_loader import UniFracLoader
        
        loader = UniFracLoader()
        sample_ids = ["sample1", "nonexistent"]
        with pytest.raises(ValueError, match="not found in Faith PD series"):
            loader.extract_batch_distances(sample_faith_pd_series, sample_ids, metric="faith_pd")

    def test_extract_batch_distances_invalid_metric(self, sample_distance_matrix):
        """Test extract_batch_distances with invalid metric using UniFracLoader."""
        from aam.data.unifrac_loader import UniFracLoader
        
        loader = UniFracLoader()
        sample_ids = ["sample1", "sample2"]
        with pytest.raises(ValueError, match="Invalid metric"):
            loader.extract_batch_distances(sample_distance_matrix, sample_ids, metric="invalid")


class TestExtractBatchDistances:
    """Tests for extract_batch_distances method using UniFracLoader."""

    def test_extract_batch_distances_unweighted(self, sample_distance_matrix):
        """Test extract_batch_distances for unweighted UniFrac using UniFracLoader."""
        from aam.data.unifrac_loader import UniFracLoader
        
        loader = UniFracLoader()
        sample_ids = ["sample1", "sample2"]
        result = loader.extract_batch_distances(sample_distance_matrix, sample_ids, metric="unweighted")

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

    def test_extract_batch_distances_unweighted_shuffled_order(self, sample_distance_matrix):
        """Test extract_batch_distances preserves shuffled batch order using UniFracLoader."""
        from aam.data.unifrac_loader import UniFracLoader
        
        loader = UniFracLoader()
        # Test with shuffled order: sample2, sample1 (instead of sample1, sample2)
        sample_ids = ["sample2", "sample1"]
        result = loader.extract_batch_distances(sample_distance_matrix, sample_ids, metric="unweighted")

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

    def test_extract_batch_distances_faith_pd(self, sample_faith_pd_series):
        """Test extract_batch_distances for Faith PD using UniFracLoader."""
        from aam.data.unifrac_loader import UniFracLoader
        
        loader = UniFracLoader()
        sample_ids = ["sample1", "sample2"]
        result = loader.extract_batch_distances(sample_faith_pd_series, sample_ids, metric="faith_pd")

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 1)
        expected = sample_faith_pd_series.loc[sample_ids].to_numpy().reshape(-1, 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_extract_batch_distances_missing_sample_ids(self, sample_distance_matrix):
        """Test extract_batch_distances with sample IDs not in distance matrix using UniFracLoader."""
        from aam.data.unifrac_loader import UniFracLoader
        
        loader = UniFracLoader()
        with pytest.raises(ValueError, match="not found"):
            loader.extract_batch_distances(sample_distance_matrix, ["nonexistent1", "nonexistent2"], metric="unweighted")

    def test_extract_batch_distances_empty_sample_ids(self, sample_distance_matrix):
        """Test extract_batch_distances with empty sample IDs list using UniFracLoader."""
        from aam.data.unifrac_loader import UniFracLoader
        
        loader = UniFracLoader()
        result = loader.extract_batch_distances(sample_distance_matrix, [], metric="unweighted")
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 0)

    def test_extract_batch_distances_large_batch(self, sample_distance_matrix):
        """Test extract_batch_distances with all samples (full batch) using UniFracLoader."""
        from aam.data.unifrac_loader import UniFracLoader
        
        loader = UniFracLoader()
        sample_ids = ["sample1", "sample2", "sample3", "sample4"]
        result = loader.extract_batch_distances(sample_distance_matrix, sample_ids, metric="unweighted")

        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 4)
        np.testing.assert_array_almost_equal(result, sample_distance_matrix.data)

    def test_extract_batch_distances_invalid_metric(self, sample_distance_matrix):
        """Test extract_batch_distances with invalid metric using UniFracLoader."""
        from aam.data.unifrac_loader import UniFracLoader
        
        loader = UniFracLoader()
        with pytest.raises(ValueError, match="Invalid metric"):
            loader.extract_batch_distances(sample_distance_matrix, ["sample1", "sample2"], metric="invalid_metric")
