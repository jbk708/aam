"""Unit tests for stripe-based UniFrac computation methods.

DEPRECATED: Stripe mode and computation functionality is deprecated as of PYT-11.4.
Users should generate UniFrac matrices using unifrac-binaries or other external tools.
These tests are kept for reference but are skipped.
"""

import pytest

# Mark all tests in this file as deprecated
pytestmark = pytest.mark.skip(
    reason="Stripe mode and computation functionality deprecated in PYT-11.4. Use pre-computed matrices instead."
)
import numpy as np
from pathlib import Path
from biom import Table
from skbio import TreeNode
import skbio

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
    return UniFracComputer(num_threads=1)


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
def larger_table():
    """Create a larger BIOM table for stripe testing."""
    np.random.seed(42)
    n_observations = 10
    n_samples = 20

    observation_ids = [generate_150bp_sequence(seed=i) for i in range(n_observations)]
    sample_ids = [f"sample_{i:02d}" for i in range(n_samples)]

    data = np.random.randint(0, 100, size=(n_observations, n_samples))
    return Table(data, observation_ids=observation_ids, sample_ids=sample_ids)


@pytest.fixture
def rarefied_table(simple_table):
    """Create a rarefied BIOM table for testing."""
    loader = BIOMLoader()
    return loader.rarefy(simple_table, depth=20, random_seed=42)


@pytest.fixture
def rarefied_larger_table(larger_table):
    """Create a rarefied larger BIOM table for testing."""
    loader = BIOMLoader()
    return loader.rarefy(larger_table, depth=50, random_seed=42)


@pytest.fixture
def simple_tree_file(tmp_path, simple_table):
    """Create a simple tree file matching the simple_table."""
    observation_ids = list(simple_table.ids(axis="observation"))
    return create_simple_tree_file(tmp_path, observation_ids)


@pytest.fixture
def larger_tree_file(tmp_path, larger_table):
    """Create a tree file matching the larger_table."""
    observation_ids = list(larger_table.ids(axis="observation"))
    return create_simple_tree_file(tmp_path, observation_ids)


class TestStripeUniFracComputer:
    """Test suite for stripe-based UniFrac computation."""

    def test_set_reference_samples(self, computer, simple_table):
        """Test setting reference samples."""
        reference_sample_ids = ["sample1", "sample2"]
        computer.set_reference_samples(reference_sample_ids, table=simple_table)

        assert computer._reference_sample_ids == reference_sample_ids
        assert computer._reference_counts_cache is not None
        assert len(computer._reference_counts_cache) == len(reference_sample_ids)
        assert "sample1" in computer._reference_counts_cache
        assert "sample2" in computer._reference_counts_cache

    def test_set_reference_samples_missing_sample(self, computer, simple_table):
        """Test setting reference samples with missing sample ID."""
        reference_sample_ids = ["sample1", "nonexistent"]

        with pytest.raises(ValueError, match="not found"):
            computer.set_reference_samples(reference_sample_ids, table=simple_table)

    def test_set_reference_samples_empty(self, computer, simple_table):
        """Test setting empty reference samples."""
        with pytest.raises(ValueError, match="empty"):
            computer.set_reference_samples([], table=simple_table)

    def test_compute_unweighted_stripe(self, computer, rarefied_larger_table, larger_tree_file):
        """Test computing stripe distances."""
        all_sample_ids = list(rarefied_larger_table.ids(axis="sample"))
        reference_sample_ids = all_sample_ids[:5]
        test_sample_ids = all_sample_ids[5:10]

        stripe = computer.compute_unweighted_stripe(
            table=rarefied_larger_table,
            tree_path=larger_tree_file,
            reference_sample_ids=reference_sample_ids,
            test_sample_ids=test_sample_ids,
        )

        assert stripe.shape == (len(test_sample_ids), len(reference_sample_ids))
        assert np.all(stripe >= 0.0)
        assert np.all(stripe <= 1.0)
        assert not np.any(np.isnan(stripe))
        assert not np.any(np.isinf(stripe))

    def test_compute_unweighted_stripe_all_samples(self, computer, rarefied_larger_table, larger_tree_file):
        """Test computing stripe distances with all samples as test samples."""
        all_sample_ids = list(rarefied_larger_table.ids(axis="sample"))
        reference_sample_ids = all_sample_ids[:5]

        stripe = computer.compute_unweighted_stripe(
            table=rarefied_larger_table,
            tree_path=larger_tree_file,
            reference_sample_ids=reference_sample_ids,
            test_sample_ids=None,  # Should use all samples
        )

        assert stripe.shape == (len(all_sample_ids), len(reference_sample_ids))
        assert np.all(stripe >= 0.0)
        assert np.all(stripe <= 1.0)

    def test_compute_unweighted_stripe_validation(self, computer, rarefied_larger_table, larger_tree_file):
        """Test stripe computation with invalid sample IDs."""
        all_sample_ids = list(rarefied_larger_table.ids(axis="sample"))
        reference_sample_ids = ["nonexistent1", "nonexistent2"]
        test_sample_ids = all_sample_ids[:5]

        with pytest.raises(ValueError, match="not found"):
            computer.compute_unweighted_stripe(
                table=rarefied_larger_table,
                tree_path=larger_tree_file,
                reference_sample_ids=reference_sample_ids,
                test_sample_ids=test_sample_ids,
            )

    def test_compute_unweighted_stripe_matches_pairwise(self, computer, rarefied_larger_table, larger_tree_file):
        """Test that stripe computation matches pairwise matrix extraction."""
        all_sample_ids = list(rarefied_larger_table.ids(axis="sample"))
        reference_sample_ids = all_sample_ids[:5]
        test_sample_ids = all_sample_ids[5:10]

        # Compute stripe
        stripe = computer.compute_unweighted_stripe(
            table=rarefied_larger_table,
            tree_path=larger_tree_file,
            reference_sample_ids=reference_sample_ids,
            test_sample_ids=test_sample_ids,
        )

        # Compute full pairwise matrix and extract stripe
        full_matrix = computer.compute_unweighted(rarefied_larger_table, larger_tree_file)
        test_indices = [full_matrix.ids.index(sid) for sid in test_sample_ids]
        ref_indices = [full_matrix.ids.index(sid) for sid in reference_sample_ids]
        expected_stripe = full_matrix.data[np.ix_(test_indices, ref_indices)]

        # Compare (allow small numerical differences)
        max_diff = np.max(np.abs(stripe - expected_stripe))
        assert max_diff < 1e-6, f"Max difference: {max_diff}"

    def test_compute_batch_unweighted_stripe(self, computer, rarefied_larger_table, larger_tree_file):
        """Test computing batch stripe distances."""
        all_sample_ids = list(rarefied_larger_table.ids(axis="sample"))
        reference_sample_ids = all_sample_ids[:5]
        test_sample_ids = all_sample_ids[5:8]

        # Set up lazy computation
        computer.setup_lazy_computation(rarefied_larger_table, larger_tree_file)
        computer.set_reference_samples(reference_sample_ids, table=rarefied_larger_table)

        stripe = computer.compute_batch_unweighted_stripe(
            sample_ids=test_sample_ids,
            reference_sample_ids=reference_sample_ids,
        )

        assert stripe.shape == (len(test_sample_ids), len(reference_sample_ids))
        assert np.all(stripe >= 0.0)
        assert np.all(stripe <= 1.0)
        assert not np.any(np.isnan(stripe))
        assert not np.any(np.isinf(stripe))

    def test_compute_batch_unweighted_stripe_uses_cached_reference(self, computer, rarefied_larger_table, larger_tree_file):
        """Test that batch stripe uses cached reference samples."""
        all_sample_ids = list(rarefied_larger_table.ids(axis="sample"))
        reference_sample_ids = all_sample_ids[:5]
        test_sample_ids = all_sample_ids[5:8]

        computer.setup_lazy_computation(rarefied_larger_table, larger_tree_file)
        computer.set_reference_samples(reference_sample_ids, table=rarefied_larger_table)

        # Don't pass reference_sample_ids - should use cached
        stripe = computer.compute_batch_unweighted_stripe(sample_ids=test_sample_ids)

        assert stripe.shape == (len(test_sample_ids), len(reference_sample_ids))

    def test_compute_batch_unweighted_stripe_no_reference(self, computer, rarefied_larger_table, larger_tree_file):
        """Test batch stripe computation without reference samples set."""
        all_sample_ids = list(rarefied_larger_table.ids(axis="sample"))
        test_sample_ids = all_sample_ids[5:8]

        computer.setup_lazy_computation(rarefied_larger_table, larger_tree_file)

        with pytest.raises(ValueError, match="reference"):
            computer.compute_batch_unweighted_stripe(sample_ids=test_sample_ids)

    def test_extract_batch_stripe_distances(self, computer):
        """Test extracting batch stripe distances from pre-computed stripe matrix."""
        n_all_samples = 20
        n_reference_samples = 5
        n_batch_samples = 3

        all_sample_ids = [f"sample_{i:02d}" for i in range(n_all_samples)]
        reference_sample_ids = all_sample_ids[:n_reference_samples]
        batch_sample_ids = all_sample_ids[5:8]

        # Create mock stripe matrix
        stripe_matrix = np.random.rand(n_all_samples, n_reference_samples)

        extracted = computer.extract_batch_stripe_distances(
            stripe_distances=stripe_matrix,
            sample_ids=batch_sample_ids,
            reference_sample_ids=reference_sample_ids,
            all_sample_ids=all_sample_ids,
        )

        assert extracted.shape == (n_batch_samples, n_reference_samples)

        # Verify correct extraction
        batch_indices = [all_sample_ids.index(sid) for sid in batch_sample_ids]
        expected = stripe_matrix[np.ix_(batch_indices, list(range(n_reference_samples)))]
        np.testing.assert_array_equal(extracted, expected)

    def test_extract_batch_stripe_distances_validation(self, computer):
        """Test extract_batch_stripe_distances with invalid inputs."""
        n_all_samples = 20
        n_reference_samples = 5

        all_sample_ids = [f"sample_{i:02d}" for i in range(n_all_samples)]
        reference_sample_ids = all_sample_ids[:n_reference_samples]
        stripe_matrix = np.random.rand(n_all_samples, n_reference_samples)

        # Test with missing sample ID
        with pytest.raises(ValueError, match="not found"):
            computer.extract_batch_stripe_distances(
                stripe_distances=stripe_matrix,
                sample_ids=["nonexistent"],
                reference_sample_ids=reference_sample_ids,
                all_sample_ids=all_sample_ids,
            )

        # Test with shape mismatch
        with pytest.raises(ValueError, match="row count|column count|shape"):
            computer.extract_batch_stripe_distances(
                stripe_distances=np.random.rand(10, 3),  # Wrong shape
                sample_ids=all_sample_ids[:3],
                reference_sample_ids=reference_sample_ids,
                all_sample_ids=all_sample_ids,
            )

    def test_stripe_computation_caching(self, computer, rarefied_larger_table, larger_tree_file):
        """Test that stripe computation results are cached."""
        all_sample_ids = list(rarefied_larger_table.ids(axis="sample"))
        reference_sample_ids = all_sample_ids[:5]
        test_sample_ids = all_sample_ids[5:8]

        computer.setup_lazy_computation(rarefied_larger_table, larger_tree_file)
        computer.set_reference_samples(reference_sample_ids, table=rarefied_larger_table)

        # First computation
        stripe1 = computer.compute_batch_unweighted_stripe(sample_ids=test_sample_ids)

        # Second computation (should use cache)
        stripe2 = computer.compute_batch_unweighted_stripe(sample_ids=test_sample_ids)

        np.testing.assert_array_equal(stripe1, stripe2)

    def test_stripe_no_batch_size_restriction(self, computer, rarefied_larger_table, larger_tree_file):
        """Test that stripe computation doesn't require even batch size."""
        all_sample_ids = list(rarefied_larger_table.ids(axis="sample"))
        reference_sample_ids = all_sample_ids[:5]
        test_sample_ids = all_sample_ids[5:9]  # 4 samples (odd number)

        computer.setup_lazy_computation(rarefied_larger_table, larger_tree_file)
        computer.set_reference_samples(reference_sample_ids, table=rarefied_larger_table)

        # Should not raise ValueError about batch size
        stripe = computer.compute_batch_unweighted_stripe(sample_ids=test_sample_ids)
        assert stripe.shape == (4, 5)  # 4 test samples, 5 reference samples
