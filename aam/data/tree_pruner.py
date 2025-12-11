"""Tree pruning utilities for optimizing UniFrac computation."""

from typing import Set, Optional
from pathlib import Path
import logging

import biom
from biom import Table
import skbio
from skbio import TreeNode


logger = logging.getLogger(__name__)


def prune_tree_to_table(
    tree: TreeNode,
    table: Table,
    output_path: Optional[str] = None,
) -> TreeNode:
    """Prune phylogenetic tree to only include ASVs present in BIOM table.
    
    This dramatically reduces tree size by removing tips (ASVs) not present in
    the dataset, which speeds up both tree loading and UniFrac computation.
    
    Args:
        tree: Full phylogenetic tree (TreeNode)
        table: BIOM table containing ASVs to keep
        output_path: Optional path to save pruned tree (if None, tree is modified in-place)
    
    Returns:
        Pruned TreeNode containing only ASVs present in table
    
    Raises:
        ValueError: If no ASVs from table are found in tree
    """
    # Get ASV IDs from table
    table_asv_ids = set(table.ids(axis="observation"))
    
    # Get tip names from tree
    tree_tips = {tip.name for tip in tree.tips() if tip.name is not None}
    
    # Find overlap
    tips_to_keep = table_asv_ids.intersection(tree_tips)
    
    if not tips_to_keep:
        raise ValueError(
            f"No ASVs from table found in tree. "
            f"Table has {len(table_asv_ids)} ASVs, tree has {len(tree_tips)} tips, overlap: 0"
        )
    
    # Create a copy of the tree to avoid modifying the original
    # Note: skbio TreeNode doesn't have a deep copy method, so we'll work with the original
    # but shear() with inplace=False should create a new tree
    pruned_tree = tree.shear(tips_to_keep, strict=False, prune=True, inplace=False)
    
    # Save to file if output path provided
    if output_path is not None:
        pruned_tree.write(output_path, format="newick")
        logger.info(f"Pruned tree saved to {output_path}")
    
    return pruned_tree


def get_pruning_stats(tree: TreeNode, table: Table) -> dict:
    """Get statistics about tree pruning operation.
    
    Args:
        tree: Full phylogenetic tree
        table: BIOM table
    
    Returns:
        Dictionary with pruning statistics:
        - original_tree_tips: Number of tips in original tree
        - biom_observations: Number of ASVs in BIOM table
        - overlap_count: Number of ASVs in both tree and table
        - final_tree_tips: Number of tips after pruning (same as overlap_count)
        - pruned_tips: Number of tips removed
    """
    # Get ASV IDs from table
    table_asv_ids = set(table.ids(axis="observation"))
    
    # Get tip names from tree
    tree_tips = {tip.name for tip in tree.tips() if tip.name is not None}
    
    # Find overlap
    overlap = table_asv_ids.intersection(tree_tips)
    
    stats = {
        "original_tree_tips": len(tree_tips),
        "biom_observations": len(table_asv_ids),
        "overlap_count": len(overlap),
        "final_tree_tips": len(overlap),
        "pruned_tips": len(tree_tips) - len(overlap),
    }
    
    return stats


def load_or_prune_tree(
    tree_path: str,
    table: Table,
    pruned_tree_path: Optional[str] = None,
    force_prune: bool = False,
) -> TreeNode:
    """Load tree, using pruned version if available, otherwise prune and cache.
    
    Args:
        tree_path: Path to full tree file
        table: BIOM table
        pruned_tree_path: Optional path to pruned tree cache (default: {tree_path}.pruned.nwk)
        force_prune: If True, re-prune even if cached pruned tree exists
    
    Returns:
        TreeNode (pruned if cache exists or force_prune=True, otherwise full tree)
    """
    if pruned_tree_path is None:
        pruned_tree_path = str(Path(tree_path).with_suffix('.pruned.nwk'))
    
    pruned_path_obj = Path(pruned_tree_path)
    
    # Check if pruned tree cache exists and we're not forcing re-prune
    if pruned_path_obj.exists() and not force_prune:
        logger.info(f"Loading cached pruned tree from {pruned_tree_path}")
        return TreeNode.read(pruned_tree_path, format="newick")
    
    # Load full tree and prune
    logger.info(f"Loading full tree from {tree_path} for pruning...")
    full_tree = TreeNode.read(tree_path, format="newick")
    
    logger.info("Pruning tree to ASVs in table...")
    pruned_tree = prune_tree_to_table(full_tree, table, output_path=pruned_tree_path)
    
    logger.info(f"Pruned tree cached to {pruned_tree_path}")
    return pruned_tree
