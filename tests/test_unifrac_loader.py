"""Tests for UniFracLoader class."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from skbio import DistanceMatrix
from aam.data.unifrac_loader import UniFracLoader


class TestUniFracLoader:
    """Test suite for UniFracLoader."""

    @pytest.fixture
    def loader(self):
        """Create UniFracLoader instance."""
        return UniFracLoader()

    @pytest.fixture
    def sample_ids(self):
        """Sample IDs for testing."""
        return ["sample1", "sample2", "sample3", "sample4"]

    @pytest.fixture
    def pairwise_matrix(self, sample_ids):
        """Create pairwise distance matrix."""
        n = len(sample_ids)
        matrix = np.random.rand(n, n)
        matrix = (matrix + matrix.T) / 2  # Make symmetric
        np.fill_diagonal(matrix, 0.0)  # Diagonal is 0
        return matrix

    @pytest.fixture
    def faith_pd_values(self, sample_ids):
        """Create Faith PD values."""
        return np.random.rand(len(sample_ids))

    @pytest.fixture
    def stripe_matrix(self, sample_ids):
        """Create stripe distance matrix."""
        n_test = len(sample_ids)
        n_ref = 3
        return np.random.rand(n_test, n_ref)

    def test_load_npy_pairwise(self, loader, sample_ids, pairwise_matrix, tmp_path):
        """Test loading pairwise matrix from .npy file."""
        matrix_path = tmp_path / "distances.npy"
        np.save(matrix_path, pairwise_matrix)

        loaded = loader.load_matrix(str(matrix_path), sample_ids=sample_ids, matrix_format="pairwise")

        assert isinstance(loaded, (np.ndarray, DistanceMatrix))
        if isinstance(loaded, DistanceMatrix):
            assert loaded.shape == (len(sample_ids), len(sample_ids))
            assert list(loaded.ids) == sample_ids
        else:
            assert loaded.shape == pairwise_matrix.shape
            np.testing.assert_array_almost_equal(loaded, pairwise_matrix)

    def test_load_npy_faith_pd(self, loader, sample_ids, faith_pd_values, tmp_path):
        """Test loading Faith PD from .npy file."""
        matrix_path = tmp_path / "faith_pd.npy"
        np.save(matrix_path, faith_pd_values)

        loaded = loader.load_matrix(str(matrix_path), sample_ids=sample_ids, matrix_format="faith_pd")

        assert isinstance(loaded, pd.Series)
        assert len(loaded) == len(sample_ids)
        assert list(loaded.index) == sample_ids
        np.testing.assert_array_almost_equal(loaded.values, faith_pd_values)

    def test_load_npy_npz_file(self, loader, sample_ids, pairwise_matrix, tmp_path):
        """Test loading from .npz file with multiple arrays."""
        matrix_path = tmp_path / "distances.npz"
        np.savez(matrix_path, distances=pairwise_matrix)

        loaded = loader.load_matrix(str(matrix_path), sample_ids=sample_ids, matrix_format="pairwise")

        assert isinstance(loaded, (np.ndarray, DistanceMatrix))
        if isinstance(loaded, DistanceMatrix):
            assert loaded.shape == (len(sample_ids), len(sample_ids))
        else:
            assert loaded.shape == pairwise_matrix.shape

    def test_load_csv_pairwise(self, loader, sample_ids, pairwise_matrix, tmp_path):
        """Test loading pairwise matrix from CSV file."""
        matrix_path = tmp_path / "distances.csv"
        df = pd.DataFrame(pairwise_matrix, index=sample_ids, columns=sample_ids)
        df.to_csv(matrix_path)

        loaded = loader.load_matrix(str(matrix_path), sample_ids=sample_ids, matrix_format="pairwise")

        assert isinstance(loaded, DistanceMatrix)
        assert loaded.shape == (len(sample_ids), len(sample_ids))
        assert list(loaded.ids) == sample_ids

    def test_load_csv_faith_pd(self, loader, sample_ids, faith_pd_values, tmp_path):
        """Test loading Faith PD from CSV file."""
        matrix_path = tmp_path / "faith_pd.csv"
        df = pd.DataFrame(faith_pd_values, index=sample_ids, columns=["faith_pd"])
        df.to_csv(matrix_path)

        loaded = loader.load_matrix(str(matrix_path), sample_ids=sample_ids, matrix_format="faith_pd")

        assert isinstance(loaded, pd.Series)
        assert len(loaded) == len(sample_ids)

    def test_load_h5_pairwise(self, loader, sample_ids, pairwise_matrix, tmp_path):
        """Test loading pairwise matrix from HDF5 file."""
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not available")

        matrix_path = tmp_path / "distances.h5"
        with h5py.File(matrix_path, "w") as f:
            f.create_dataset("distances", data=pairwise_matrix)

        loaded = loader.load_matrix(str(matrix_path), sample_ids=sample_ids, matrix_format="pairwise")

        assert isinstance(loaded, (np.ndarray, DistanceMatrix))
        if isinstance(loaded, DistanceMatrix):
            assert loaded.shape == (len(sample_ids), len(sample_ids))
        else:
            assert loaded.shape == pairwise_matrix.shape

    def test_load_matrix_validation_mismatch(self, loader, sample_ids, pairwise_matrix, tmp_path):
        """Test that loading fails when matrix dimensions don't match sample IDs."""
        matrix_path = tmp_path / "distances.npy"
        # Create matrix with wrong size
        wrong_matrix = np.random.rand(len(sample_ids) + 1, len(sample_ids) + 1)
        np.save(matrix_path, wrong_matrix)

        with pytest.raises((ValueError, Exception), match="(doesn't match|must match)"):
            loader.load_matrix(str(matrix_path), sample_ids=sample_ids, matrix_format="pairwise")

    def test_load_matrix_file_not_found(self, loader):
        """Test that loading fails when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            loader.load_matrix("nonexistent.npy")

    def test_load_matrix_unsupported_format(self, loader, tmp_path):
        """Test that loading fails for unsupported formats."""
        matrix_path = tmp_path / "distances.txt"
        matrix_path.write_text("test")

        with pytest.raises(ValueError, match="Unsupported matrix file format"):
            loader.load_matrix(str(matrix_path))

    def test_extract_batch_distances_pairwise(self, loader, sample_ids, pairwise_matrix):
        """Test extracting batch distances from pairwise matrix."""
        distance_matrix = DistanceMatrix(pairwise_matrix, ids=sample_ids)

        batch_ids = ["sample1", "sample2"]
        batch_distances = loader.extract_batch_distances(distance_matrix, batch_ids, metric="unweighted")

        assert batch_distances.shape == (2, 2)
        assert batch_distances[0, 0] == 0.0  # Diagonal
        assert batch_distances[1, 1] == 0.0  # Diagonal

    def test_extract_batch_distances_faith_pd(self, loader, sample_ids, faith_pd_values):
        """Test extracting batch distances from Faith PD series."""
        faith_pd_series = pd.Series(faith_pd_values, index=sample_ids)

        batch_ids = ["sample1", "sample2"]
        batch_distances = loader.extract_batch_distances(faith_pd_series, batch_ids, metric="faith_pd")

        assert batch_distances.shape == (2, 1)
        np.testing.assert_array_almost_equal(batch_distances.flatten(), [faith_pd_values[0], faith_pd_values[1]])

    def test_extract_batch_distances_numpy_array(self, loader, sample_ids, pairwise_matrix):
        """Test extracting batch distances from numpy array."""
        batch_ids = ["sample1", "sample2"]
        # Create indices
        indices = [sample_ids.index(sid) for sid in batch_ids]
        expected = pairwise_matrix[np.ix_(indices, indices)]

        batch_distances = loader.extract_batch_distances(pairwise_matrix, batch_ids, metric="unweighted")

        # For numpy arrays, we need to handle differently since there's no ID mapping
        # This test verifies the function handles numpy arrays
        assert isinstance(batch_distances, np.ndarray)

    def test_extract_batch_distances_missing_ids(self, loader, sample_ids, pairwise_matrix):
        """Test that extraction fails when sample IDs are missing."""
        distance_matrix = DistanceMatrix(pairwise_matrix, ids=sample_ids)

        batch_ids = ["sample1", "nonexistent"]

        with pytest.raises(ValueError, match="not found in distance matrix"):
            loader.extract_batch_distances(distance_matrix, batch_ids, metric="unweighted")

    def test_extract_batch_distances_empty_batch(self, loader, sample_ids, pairwise_matrix):
        """Test extracting batch distances for empty batch."""
        distance_matrix = DistanceMatrix(pairwise_matrix, ids=sample_ids)

        batch_distances = loader.extract_batch_distances(distance_matrix, [], metric="unweighted")

        assert batch_distances.shape == (0, 0)

    def test_extract_batch_stripe_distances(self, loader, sample_ids, stripe_matrix):
        """Test extracting stripe distances for a batch."""
        all_sample_ids = sample_ids
        reference_sample_ids = ["ref1", "ref2", "ref3"]
        batch_ids = ["sample1", "sample2"]

        batch_distances = loader.extract_batch_stripe_distances(stripe_matrix, batch_ids, reference_sample_ids, all_sample_ids)

        assert batch_distances.shape == (2, 3)
        np.testing.assert_array_almost_equal(batch_distances[0, :], stripe_matrix[0, :])
        np.testing.assert_array_almost_equal(batch_distances[1, :], stripe_matrix[1, :])

    def test_extract_batch_stripe_distances_shape_mismatch(self, loader, sample_ids):
        """Test that stripe extraction fails when shapes don't match."""
        stripe_matrix = np.random.rand(len(sample_ids), 2)  # Wrong number of columns
        reference_sample_ids = ["ref1", "ref2", "ref3"]  # Expects 3 columns

        with pytest.raises(ValueError, match="doesn't match"):
            loader.extract_batch_stripe_distances(stripe_matrix, ["sample1"], reference_sample_ids, sample_ids)

    def test_extract_batch_stripe_distances_missing_ids(self, loader, sample_ids, stripe_matrix):
        """Test that stripe extraction fails when sample IDs are missing."""
        reference_sample_ids = ["ref1", "ref2", "ref3"]

        with pytest.raises(ValueError, match="not found in all_sample_ids"):
            loader.extract_batch_stripe_distances(stripe_matrix, ["nonexistent"], reference_sample_ids, sample_ids)

    def test_validate_matrix_dimensions_pairwise(self, loader, sample_ids, pairwise_matrix):
        """Test validation of pairwise matrix dimensions."""
        distance_matrix = DistanceMatrix(pairwise_matrix, ids=sample_ids)

        # Should not raise
        loader.validate_matrix_dimensions(distance_matrix, sample_ids, metric="unweighted")

    def test_validate_matrix_dimensions_faith_pd(self, loader, sample_ids, faith_pd_values):
        """Test validation of Faith PD dimensions."""
        faith_pd_series = pd.Series(faith_pd_values, index=sample_ids)

        # Should not raise
        loader.validate_matrix_dimensions(faith_pd_series, sample_ids, metric="faith_pd")

    def test_validate_matrix_dimensions_mismatch(self, loader, sample_ids, pairwise_matrix):
        """Test that validation fails when dimensions don't match."""
        distance_matrix = DistanceMatrix(pairwise_matrix, ids=sample_ids)
        wrong_sample_ids = sample_ids + ["extra"]

        with pytest.raises(ValueError, match="(doesn't match|but expected)"):
            loader.validate_matrix_dimensions(distance_matrix, wrong_sample_ids, metric="unweighted")

    def test_validate_matrix_dimensions_stripe(self, loader, sample_ids, stripe_matrix):
        """Test validation of stripe matrix dimensions."""
        reference_sample_ids = ["ref1", "ref2", "ref3"]

        # Should not raise
        loader.validate_matrix_dimensions(stripe_matrix, sample_ids, metric="stripe")

        # Should raise if wrong number of rows
        with pytest.raises(ValueError, match="(doesn't match|but expected)"):
            loader.validate_matrix_dimensions(stripe_matrix, sample_ids + ["extra"], metric="stripe")

    # REG-10a: Weighted UniFrac tests
    def test_extract_batch_distances_weighted_pairwise(self, loader, sample_ids, pairwise_matrix):
        """Test extracting batch distances from pairwise matrix with weighted metric."""
        distance_matrix = DistanceMatrix(pairwise_matrix, ids=sample_ids)

        batch_ids = ["sample1", "sample2"]
        batch_distances = loader.extract_batch_distances(distance_matrix, batch_ids, metric="weighted")

        assert batch_distances.shape == (2, 2)
        assert batch_distances[0, 0] == 0.0  # Diagonal
        assert batch_distances[1, 1] == 0.0  # Diagonal

    def test_extract_batch_distances_weighted_numpy_array(self, loader, sample_ids, pairwise_matrix):
        """Test extracting weighted distances from numpy array."""
        batch_ids = ["sample1", "sample2"]

        batch_distances = loader.extract_batch_distances(pairwise_matrix, batch_ids, metric="weighted")

        assert isinstance(batch_distances, np.ndarray)
        assert batch_distances.shape == pairwise_matrix.shape

    def test_extract_batch_distances_weighted_empty_batch(self, loader, sample_ids, pairwise_matrix):
        """Test extracting weighted batch distances for empty batch."""
        distance_matrix = DistanceMatrix(pairwise_matrix, ids=sample_ids)

        batch_distances = loader.extract_batch_distances(distance_matrix, [], metric="weighted")

        assert batch_distances.shape == (0, 0)

    def test_extract_batch_distances_weighted_missing_ids(self, loader, sample_ids, pairwise_matrix):
        """Test that weighted extraction fails when sample IDs are missing."""
        distance_matrix = DistanceMatrix(pairwise_matrix, ids=sample_ids)

        batch_ids = ["sample1", "nonexistent"]

        with pytest.raises(ValueError, match="not found in distance matrix"):
            loader.extract_batch_distances(distance_matrix, batch_ids, metric="weighted")

    def test_extract_batch_distances_invalid_metric(self, loader, sample_ids, pairwise_matrix):
        """Test that extraction fails for invalid metric."""
        distance_matrix = DistanceMatrix(pairwise_matrix, ids=sample_ids)

        with pytest.raises(ValueError, match="Invalid metric"):
            loader.extract_batch_distances(distance_matrix, ["sample1"], metric="invalid")

    def test_extract_batch_distances_weighted_wrong_shape(self, loader):
        """Test that weighted extraction fails for non-square matrix."""
        wrong_matrix = np.random.rand(4, 3)  # Non-square

        with pytest.raises(ValueError, match="Expected square matrix"):
            loader.extract_batch_distances(wrong_matrix, ["s1"], metric="weighted")
