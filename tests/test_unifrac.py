"""Unit tests for UniFrac distance extraction (via UniFracLoader)."""

import pytest
import numpy as np
import pandas as pd
from skbio import DistanceMatrix

from aam.data.unifrac_loader import UniFracLoader


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


class TestExtractBatchDistancesErrors:
    """Test error handling for extract_batch_distances."""

    def test_extract_batch_distances_missing_sample_id(self, sample_distance_matrix):
        """Test extract_batch_distances with missing sample ID."""
        loader = UniFracLoader()
        sample_ids = ["sample1", "nonexistent"]
        with pytest.raises(ValueError, match="not found in distance matrix"):
            loader.extract_batch_distances(sample_distance_matrix, sample_ids, metric="unweighted")

    def test_extract_batch_distances_faith_pd_missing_sample_id(self, sample_faith_pd_series):
        """Test extract_batch_distances with missing sample ID for Faith PD."""
        loader = UniFracLoader()
        sample_ids = ["sample1", "nonexistent"]
        with pytest.raises(ValueError, match="not found in Faith PD series"):
            loader.extract_batch_distances(sample_faith_pd_series, sample_ids, metric="faith_pd")

    def test_extract_batch_distances_invalid_metric(self, sample_distance_matrix):
        """Test extract_batch_distances with invalid metric."""
        loader = UniFracLoader()
        sample_ids = ["sample1", "sample2"]
        with pytest.raises(ValueError, match="Invalid metric"):
            loader.extract_batch_distances(sample_distance_matrix, sample_ids, metric="invalid")


class TestExtractBatchDistances:
    """Tests for extract_batch_distances method using UniFracLoader."""

    def test_extract_batch_distances_unweighted(self, sample_distance_matrix):
        """Test extract_batch_distances for unweighted UniFrac."""
        loader = UniFracLoader()
        sample_ids = ["sample1", "sample2"]
        result = loader.extract_batch_distances(sample_distance_matrix, sample_ids, metric="unweighted")

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        np.testing.assert_array_almost_equal(result, result.T)
        np.testing.assert_array_almost_equal(np.diag(result), 0)
        expected_filtered = sample_distance_matrix.filter(sample_ids)
        id_to_idx = {id_: idx for idx, id_ in enumerate(expected_filtered.ids)}
        reorder_indices = [id_to_idx[id_] for id_ in sample_ids]
        expected = expected_filtered.data[np.ix_(reorder_indices, reorder_indices)]
        np.testing.assert_array_almost_equal(result, expected)

    def test_extract_batch_distances_unweighted_shuffled_order(self, sample_distance_matrix):
        """Test extract_batch_distances preserves shuffled batch order."""
        loader = UniFracLoader()
        sample_ids = ["sample2", "sample1"]
        result = loader.extract_batch_distances(sample_distance_matrix, sample_ids, metric="unweighted")

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        np.testing.assert_array_almost_equal(result, result.T)
        np.testing.assert_array_almost_equal(np.diag(result), 0)

        expected_filtered = sample_distance_matrix.filter(sample_ids)
        id_to_idx = {id_: idx for idx, id_ in enumerate(expected_filtered.ids)}
        reorder_indices = [id_to_idx[id_] for id_ in sample_ids]
        expected = expected_filtered.data[np.ix_(reorder_indices, reorder_indices)]
        np.testing.assert_array_almost_equal(result, expected)

        assert np.isclose(result[0, 1], 0.5), f"Expected distance from sample2 to sample1 to be 0.5, got {result[0, 1]}"

    def test_extract_batch_distances_faith_pd(self, sample_faith_pd_series):
        """Test extract_batch_distances for Faith PD."""
        loader = UniFracLoader()
        sample_ids = ["sample1", "sample2"]
        result = loader.extract_batch_distances(sample_faith_pd_series, sample_ids, metric="faith_pd")

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 1)
        expected = sample_faith_pd_series.loc[sample_ids].to_numpy().reshape(-1, 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_extract_batch_distances_missing_sample_ids(self, sample_distance_matrix):
        """Test extract_batch_distances with sample IDs not in distance matrix."""
        loader = UniFracLoader()
        with pytest.raises(ValueError, match="not found"):
            loader.extract_batch_distances(sample_distance_matrix, ["nonexistent1", "nonexistent2"], metric="unweighted")

    def test_extract_batch_distances_empty_sample_ids(self, sample_distance_matrix):
        """Test extract_batch_distances with empty sample IDs list."""
        loader = UniFracLoader()
        result = loader.extract_batch_distances(sample_distance_matrix, [], metric="unweighted")
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 0)

    def test_extract_batch_distances_large_batch(self, sample_distance_matrix):
        """Test extract_batch_distances with all samples (full batch)."""
        loader = UniFracLoader()
        sample_ids = ["sample1", "sample2", "sample3", "sample4"]
        result = loader.extract_batch_distances(sample_distance_matrix, sample_ids, metric="unweighted")

        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 4)
        np.testing.assert_array_almost_equal(result, sample_distance_matrix.data)

    def test_extract_batch_distances_invalid_metric(self, sample_distance_matrix):
        """Test extract_batch_distances with invalid metric."""
        loader = UniFracLoader()
        with pytest.raises(ValueError, match="Invalid metric"):
            loader.extract_batch_distances(sample_distance_matrix, ["sample1", "sample2"], metric="invalid_metric")
