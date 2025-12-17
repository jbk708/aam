"""Unit tests for tree pruning utilities.

DEPRECATED: Tree pruning functionality is deprecated as of PYT-11.4.
Tree pruning should be handled by unifrac-binaries or other external tools
when generating UniFrac matrices. These tests are kept for reference but are skipped.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import biom
from biom import Table
from skbio import TreeNode

from aam.data.tree_pruner import prune_tree_to_table, get_pruning_stats, load_or_prune_tree

# Mark all tests in this file as deprecated
pytestmark = pytest.mark.skip(reason="Tree pruning functionality deprecated in PYT-11.4. Use unifrac-binaries instead.")


def create_simple_tree() -> TreeNode:
    """Create a simple test tree."""
    # Tree: ((A:0.1,B:0.1):0.2,(C:0.1,D:0.1):0.2);
    tree_str = "((A:0.1,B:0.1):0.2,(C:0.1,D:0.1):0.2);"
    return TreeNode.read([tree_str], format="newick")


def create_simple_table(observation_ids: list) -> Table:
    """Create a simple BIOM table with specified observation IDs."""
    num_samples = 3
    data = np.array([[10, 20, 5] for _ in observation_ids])
    sample_ids = [f"sample{i}" for i in range(num_samples)]
    return Table(data, observation_ids=observation_ids, sample_ids=sample_ids)


@pytest.fixture
def simple_tree():
    """Create a simple test tree."""
    return create_simple_tree()


@pytest.fixture
def simple_table_all_asvs():
    """Create a table with all ASVs from simple tree."""
    return create_simple_table(["A", "B", "C", "D"])


@pytest.fixture
def simple_table_subset_asvs():
    """Create a table with subset of ASVs from simple tree."""
    return create_simple_table(["A", "B"])


class TestPruneTreeToTable:
    """Test tree pruning functionality."""

    def test_prune_tree_all_asvs_present(self, simple_tree, simple_table_all_asvs):
        """Test pruning when all ASVs in tree are present in table."""
        pruned = prune_tree_to_table(simple_tree, simple_table_all_asvs)

        assert pruned is not None
        tips = [tip.name for tip in pruned.tips()]
        assert set(tips) == {"A", "B", "C", "D"}
        assert len(tips) == 4

    def test_prune_tree_subset_asvs(self, simple_tree, simple_table_subset_asvs):
        """Test pruning to subset of ASVs."""
        pruned = prune_tree_to_table(simple_tree, simple_table_subset_asvs)

        assert pruned is not None
        tips = [tip.name for tip in pruned.tips()]
        assert set(tips) == {"A", "B"}
        assert len(tips) == 2

    def test_prune_tree_single_asv(self, simple_tree):
        """Test pruning to single ASV."""
        table = create_simple_table(["A"])
        pruned = prune_tree_to_table(simple_tree, table)

        assert pruned is not None
        tips = [tip.name for tip in pruned.tips() if tip.name is not None]
        # When pruning to a single tip, the tree structure may change
        # but the tip should still be present (possibly as root)
        assert "A" in [tip.name for tip in pruned.tips() if tip.name is not None] or any(
            node.name == "A" for node in pruned.traverse() if node.name is not None
        )

    def test_prune_tree_no_overlap(self, simple_tree):
        """Test pruning when no ASVs overlap (should raise ValueError)."""
        table = create_simple_table(["X", "Y", "Z"])

        with pytest.raises(ValueError, match="No ASVs from table found in tree"):
            prune_tree_to_table(simple_tree, table)

    def test_prune_tree_preserves_structure(self, simple_tree, simple_table_subset_asvs):
        """Test that pruning preserves tree structure (branch lengths, topology)."""
        pruned = prune_tree_to_table(simple_tree, simple_table_subset_asvs)

        # Tree should still be valid
        assert pruned is not None
        # Should have at least one tip
        assert len(list(pruned.tips())) > 0
        # Tree should be connected
        assert pruned.root() is not None

    def test_prune_tree_saves_to_file(self, simple_tree, simple_table_subset_asvs, tmp_path):
        """Test that pruned tree can be saved to file."""
        output_path = str(tmp_path / "pruned_tree.nwk")
        pruned = prune_tree_to_table(simple_tree, simple_table_subset_asvs, output_path=output_path)

        assert Path(output_path).exists()
        # Verify saved tree can be loaded
        loaded = TreeNode.read(output_path, format="newick")
        tips = [tip.name for tip in loaded.tips()]
        assert set(tips) == {"A", "B"}


class TestGetPruningStats:
    """Test pruning statistics functionality."""

    def test_get_pruning_stats_all_overlap(self, simple_tree, simple_table_all_asvs):
        """Test stats when all ASVs overlap."""
        stats = get_pruning_stats(simple_tree, simple_table_all_asvs)

        assert stats["original_tree_tips"] == 4
        assert stats["biom_observations"] == 4
        assert stats["overlap_count"] == 4
        assert stats["final_tree_tips"] == 4
        assert stats["pruned_tips"] == 0

    def test_get_pruning_stats_partial_overlap(self, simple_tree, simple_table_subset_asvs):
        """Test stats when only some ASVs overlap."""
        stats = get_pruning_stats(simple_tree, simple_table_subset_asvs)

        assert stats["original_tree_tips"] == 4
        assert stats["biom_observations"] == 2
        assert stats["overlap_count"] == 2
        assert stats["final_tree_tips"] == 2
        assert stats["pruned_tips"] == 2

    def test_get_pruning_stats_no_overlap(self, simple_tree):
        """Test stats when no ASVs overlap."""
        table = create_simple_table(["X", "Y"])
        stats = get_pruning_stats(simple_tree, table)

        assert stats["original_tree_tips"] == 4
        assert stats["biom_observations"] == 2
        assert stats["overlap_count"] == 0
        assert stats["final_tree_tips"] == 0
        assert stats["pruned_tips"] == 4


class TestLoadOrPruneTree:
    """Test load_or_prune_tree functionality."""

    def test_load_existing_pruned_tree(self, simple_tree, simple_table_subset_asvs, tmp_path):
        """Test loading existing pruned tree cache."""
        # Create and save pruned tree
        pruned_tree_path = tmp_path / "tree.pruned.nwk"
        full_tree_path = tmp_path / "tree.nwk"

        # Save full tree
        simple_tree.write(str(full_tree_path), format="newick")

        # Prune and save
        pruned = prune_tree_to_table(simple_tree, simple_table_subset_asvs, output_path=str(pruned_tree_path))

        # Test loading cached pruned tree
        loaded = load_or_prune_tree(
            str(full_tree_path),
            simple_table_subset_asvs,
            pruned_tree_path=str(pruned_tree_path),
        )

        tips = [tip.name for tip in loaded.tips()]
        assert set(tips) == {"A", "B"}

    def test_prune_and_cache_tree(self, simple_tree, simple_table_subset_asvs, tmp_path):
        """Test pruning and caching tree when cache doesn't exist."""
        full_tree_path = tmp_path / "tree.nwk"
        pruned_tree_path = tmp_path / "tree.pruned.nwk"

        # Save full tree
        simple_tree.write(str(full_tree_path), format="newick")

        # Load and prune (cache doesn't exist yet)
        loaded = load_or_prune_tree(
            str(full_tree_path),
            simple_table_subset_asvs,
            pruned_tree_path=str(pruned_tree_path),
        )

        # Verify pruned tree was created
        assert Path(pruned_tree_path).exists()
        tips = [tip.name for tip in loaded.tips()]
        assert set(tips) == {"A", "B"}

    def test_force_prune_reexecutes(self, simple_tree, simple_table_subset_asvs, tmp_path):
        """Test that force_prune=True re-prunes even if cache exists."""
        full_tree_path = tmp_path / "tree.nwk"
        pruned_tree_path = tmp_path / "tree.pruned.nwk"

        # Save full tree
        simple_tree.write(str(full_tree_path), format="newick")

        # Create initial pruned tree
        prune_tree_to_table(simple_tree, simple_table_subset_asvs, output_path=str(pruned_tree_path))

        # Force re-pruning
        loaded = load_or_prune_tree(
            str(full_tree_path),
            simple_table_subset_asvs,
            pruned_tree_path=str(pruned_tree_path),
            force_prune=True,
        )

        tips = [tip.name for tip in loaded.tips()]
        assert set(tips) == {"A", "B"}


class TestTreePrunerIntegration:
    """Integration tests for tree pruning with UniFrac."""

    def test_pruned_tree_produces_same_unifrac(self, simple_tree, simple_table_all_asvs, tmp_path):
        """Test that pruned tree produces same UniFrac distances as full tree."""
        import unifrac

        # Create a table with all samples
        table = simple_table_all_asvs

        # Compute distances with full tree
        full_distances = unifrac.unweighted(table, simple_tree)

        # Prune tree (should be same since all ASVs present)
        pruned = prune_tree_to_table(simple_tree, table)

        # Compute distances with pruned tree
        pruned_distances = unifrac.unweighted(table, pruned)

        # Should be identical (or very close due to numerical precision)
        np.testing.assert_array_almost_equal(
            full_distances.data,
            pruned_distances.data,
            decimal=10,
        )

    def test_pruned_tree_subset_produces_valid_unifrac(self, simple_tree, simple_table_subset_asvs, tmp_path):
        """Test that pruned tree (subset) produces valid UniFrac distances."""
        import unifrac

        # Prune tree to subset
        pruned = prune_tree_to_table(simple_tree, simple_table_subset_asvs)

        # Compute distances with pruned tree
        distances = unifrac.unweighted(simple_table_subset_asvs, pruned)

        # Should be valid distance matrix
        assert distances.shape[0] == distances.shape[1]
        assert distances.shape[0] == len(simple_table_subset_asvs.ids(axis="sample"))
        # Diagonal should be 0
        np.testing.assert_array_almost_equal(np.diag(distances.data), np.zeros(distances.shape[0]))
