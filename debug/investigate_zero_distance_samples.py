"""Investigate zero-distance UniFrac pairs in training data.

This script analyzes the distribution and origin of zero-distance UniFrac pairs
to determine whether they represent data quality issues or legitimate biological signal.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from typing import Dict, List, Tuple

# Add parent directory to path to import aam modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from aam.data.biom_loader import BIOMLoader
from aam.data.unifrac import UniFracComputer
from skbio import DistanceMatrix


def load_data(biom_path: str, tree_path: str, rarefaction_depth: int = 1000) -> Tuple:
    """Load BIOM table and compute UniFrac distances.
    
    Args:
        biom_path: Path to BIOM file
        tree_path: Path to phylogenetic tree file
        rarefaction_depth: Rarefaction depth for samples
        
    Returns:
        Tuple of (rarefied_table, distance_matrix, sample_ids)
    """
    print(f"Loading BIOM table from {biom_path}...")
    loader = BIOMLoader()
    table = loader.load_table(biom_path)
    
    print(f"Original table: {table.shape[1]} samples, {table.shape[0]} ASVs")
    
    print(f"Rarefying to depth {rarefaction_depth}...")
    rarefied_table = loader.rarefy(table, depth=rarefaction_depth, random_seed=42)
    print(f"Rarefied table: {rarefied_table.shape[1]} samples, {rarefied_table.shape[0]} ASVs")
    
    print(f"Computing UniFrac distances using tree {tree_path}...")
    computer = UniFracComputer()
    distance_matrix = computer.compute_unweighted(rarefied_table, tree_path)
    
    sample_ids = list(distance_matrix.ids)
    print(f"Distance matrix shape: {distance_matrix.shape}")
    
    return rarefied_table, distance_matrix, sample_ids


def analyze_zero_distances(distance_matrix: DistanceMatrix) -> Dict:
    """Analyze zero-distance pairs in the distance matrix.
    
    Args:
        distance_matrix: UniFrac distance matrix
        
    Returns:
        Dictionary with analysis results
    """
    print("\n" + "="*80)
    print("ZERO-DISTANCE ANALYSIS")
    print("="*80)
    
    # Convert to numpy array for analysis
    distances = distance_matrix.data
    n_samples = distances.shape[0]
    
    # Extract upper triangle (excluding diagonal) for pairwise distances
    triu_indices = np.triu_indices(n_samples, k=1)
    pairwise_distances = distances[triu_indices]
    
    # Count zero distances
    zero_mask = pairwise_distances == 0.0
    n_zero_pairs = np.sum(zero_mask)
    n_total_pairs = len(pairwise_distances)
    zero_percentage = (n_zero_pairs / n_total_pairs) * 100
    
    print(f"\nTotal pairwise comparisons: {n_total_pairs:,}")
    print(f"Zero-distance pairs: {n_zero_pairs:,} ({zero_percentage:.2f}%)")
    print(f"Non-zero distances: {n_total_pairs - n_zero_pairs:,} ({100 - zero_percentage:.2f}%)")
    
    # Find which samples are involved in zero-distance pairs
    zero_pair_indices = triu_indices[0][zero_mask], triu_indices[1][zero_mask]
    zero_sample_indices = set(zero_pair_indices[0]) | set(zero_pair_indices[1])
    n_samples_with_zeros = len(zero_sample_indices)
    
    print(f"\nSamples involved in zero-distance pairs: {n_samples_with_zeros} / {n_samples}")
    print(f"Percentage of samples with zero distances: {(n_samples_with_zeros / n_samples) * 100:.2f}%")
    
    # Count how many zero pairs each sample is involved in
    sample_zero_counts = np.zeros(n_samples, dtype=int)
    for i, j in zip(zero_pair_indices[0], zero_pair_indices[1]):
        sample_zero_counts[i] += 1
        sample_zero_counts[j] += 1
    
    print(f"\nZero-distance pair statistics per sample:")
    print(f"  Mean: {np.mean(sample_zero_counts):.2f}")
    print(f"  Median: {np.median(sample_zero_counts):.0f}")
    print(f"  Max: {np.max(sample_zero_counts)}")
    print(f"  Samples with 0 zero pairs: {np.sum(sample_zero_counts == 0)}")
    print(f"  Samples with 1+ zero pairs: {np.sum(sample_zero_counts > 0)}")
    print(f"  Samples with 10+ zero pairs: {np.sum(sample_zero_counts >= 10)}")
    print(f"  Samples with 50+ zero pairs: {np.sum(sample_zero_counts >= 50)}")
    
    # Analyze distribution of non-zero distances
    non_zero_distances = pairwise_distances[~zero_mask]
    
    print(f"\nNon-zero distance statistics:")
    print(f"  Count: {len(non_zero_distances):,}")
    print(f"  Mean: {np.mean(non_zero_distances):.6f}")
    print(f"  Median: {np.median(non_zero_distances):.6f}")
    print(f"  Std: {np.std(non_zero_distances):.6f}")
    print(f"  Min: {np.min(non_zero_distances):.6f}")
    print(f"  Max: {np.max(non_zero_distances):.6f}")
    print(f"  25th percentile: {np.percentile(non_zero_distances, 25):.6f}")
    print(f"  75th percentile: {np.percentile(non_zero_distances, 75):.6f}")
    
    # Check for near-zero distances (within epsilon)
    epsilon = 1e-6
    near_zero_mask = (pairwise_distances > 0.0) & (pairwise_distances < epsilon)
    n_near_zero = np.sum(near_zero_mask)
    print(f"\nNear-zero distances (< {epsilon}): {n_near_zero:,} ({n_near_zero / n_total_pairs * 100:.4f}%)")
    
    return {
        'n_total_pairs': n_total_pairs,
        'n_zero_pairs': n_zero_pairs,
        'zero_percentage': zero_percentage,
        'n_samples_with_zeros': n_samples_with_zeros,
        'sample_zero_counts': sample_zero_counts,
        'pairwise_distances': pairwise_distances,
        'non_zero_distances': non_zero_distances,
        'zero_pair_indices': zero_pair_indices,
        'zero_sample_indices': zero_sample_indices,
    }


def analyze_sample_metadata(table, sample_ids: List[str], analysis_results: Dict) -> Dict:
    """Analyze zero-distance pairs in context of sample metadata.
    
    Args:
        table: BIOM table with sample metadata
        sample_ids: List of sample IDs
        analysis_results: Results from analyze_zero_distances
        
    Returns:
        Dictionary with metadata analysis results
    """
    print("\n" + "="*80)
    print("METADATA ANALYSIS")
    print("="*80)
    
    # Check if table has sample metadata
    try:
        # Try to access sample metadata
        sample_metadata = table.metadata(axis='sample')
        if sample_metadata is None or len(sample_metadata) == 0:
            print("\nNo sample metadata available in BIOM table.")
            has_metadata = False
        else:
            has_metadata = True
    except (AttributeError, TypeError):
        print("\nNo sample metadata available in BIOM table.")
        has_metadata = False
    
    if not has_metadata:
        # Still analyze sample abundances even without metadata
        pass
    
    # Analyze sample abundances (total counts per sample)
    sample_counts = {}
    sample_id_list = list(table.ids(axis='sample'))
    
    # Get sample sums using BIOM table's built-in method
    for sample_id in sample_ids:
        if sample_id in sample_id_list:
            # Get the sample vector and sum it
            sample_vector = table.data(sample_id, axis='sample')
            total_count = float(sample_vector.sum())
            sample_counts[sample_id] = total_count
        else:
            # Sample not found in table (shouldn't happen, but handle gracefully)
            sample_counts[sample_id] = 0.0
    
    counts_array = np.array([sample_counts[sid] for sid in sample_ids])
    
    # Compare counts for samples with vs without zero distances
    zero_sample_indices = analysis_results['zero_sample_indices']
    non_zero_sample_indices = set(range(len(sample_ids))) - zero_sample_indices
    
    if zero_sample_indices and non_zero_sample_indices:
        zero_sample_counts = counts_array[list(zero_sample_indices)]
        non_zero_sample_counts = counts_array[list(non_zero_sample_indices)]
        
        print(f"\nSample abundance (total counts) analysis:")
        print(f"  Samples with zero distances:")
        print(f"    Mean: {np.mean(zero_sample_counts):.2f}")
        print(f"    Median: {np.median(zero_sample_counts):.0f}")
        print(f"    Std: {np.std(zero_sample_counts):.2f}")
        print(f"  Samples without zero distances:")
        print(f"    Mean: {np.mean(non_zero_sample_counts):.2f}")
        print(f"    Median: {np.median(non_zero_sample_counts):.0f}")
        print(f"    Std: {np.std(non_zero_sample_counts):.2f}")
    
    return {
        'sample_counts': sample_counts,
        'counts_array': counts_array,
    }


def create_visualizations(
    distance_matrix: DistanceMatrix,
    analysis_results: Dict,
    metadata_results: Dict,
    output_dir: Path
) -> None:
    """Create visualizations of zero-distance distribution.
    
    Args:
        distance_matrix: UniFrac distance matrix
        analysis_results: Results from analyze_zero_distances
        metadata_results: Results from analyze_sample_metadata
        output_dir: Directory to save plots
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pairwise_distances = analysis_results['pairwise_distances']
    non_zero_distances = analysis_results['non_zero_distances']
    
    # Set style
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Histogram of all distances (highlighting zero cluster)
    print("\nCreating histogram of UniFrac distances...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bins, ensuring zero is in its own bin
    bins = np.concatenate([[-0.01], np.linspace(0, 1, 101)])
    ax.hist(pairwise_distances, bins=bins, edgecolor='black', alpha=0.7)
    ax.axvline(x=0.0, color='red', linestyle='--', linewidth=2, label='Zero distance')
    ax.set_xlabel('UniFrac Distance', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of UniFrac Distances (Highlighting Zero Cluster)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'distance_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'distance_histogram.png'}")
    
    # 2. Histogram of non-zero distances only
    print("\nCreating histogram of non-zero distances...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(non_zero_distances, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('UniFrac Distance', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Non-Zero UniFrac Distances', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'non_zero_distance_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'non_zero_distance_histogram.png'}")
    
    # 3. Log-scale histogram of non-zero distances
    print("\nCreating log-scale histogram of non-zero distances...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use log scale for y-axis to better see distribution
    ax.hist(non_zero_distances, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_yscale('log')
    ax.set_xlabel('UniFrac Distance', fontsize=12)
    ax.set_ylabel('Frequency (log scale)', fontsize=12)
    ax.set_title('Distribution of Non-Zero UniFrac Distances (Log Scale)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'non_zero_distance_histogram_log.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'non_zero_distance_histogram_log.png'}")
    
    # 4. Sample zero-distance count distribution
    print("\nCreating histogram of zero-distance counts per sample...")
    sample_zero_counts = analysis_results['sample_zero_counts']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(sample_zero_counts, bins=min(50, len(np.unique(sample_zero_counts))), edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Zero-Distance Pairs per Sample', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Distribution of Zero-Distance Pair Counts per Sample', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_zero_count_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'sample_zero_count_histogram.png'}")
    
    # 5. Comparison: zero vs non-zero distance samples (if metadata available)
    if metadata_results and 'counts_array' in metadata_results:
        print("\nCreating comparison plot of sample abundances...")
        zero_sample_indices = analysis_results['zero_sample_indices']
        non_zero_sample_indices = set(range(len(analysis_results['sample_zero_counts']))) - zero_sample_indices
        
        if zero_sample_indices and non_zero_sample_indices:
            counts_array = metadata_results['counts_array']
            zero_counts = counts_array[list(zero_sample_indices)]
            non_zero_counts = counts_array[list(non_zero_sample_indices)]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Histogram comparison
            ax1.hist(zero_counts, bins=50, alpha=0.6, label='Samples with zero distances', color='red', edgecolor='black')
            ax1.hist(non_zero_counts, bins=50, alpha=0.6, label='Samples without zero distances', color='blue', edgecolor='black')
            ax1.set_xlabel('Total Sample Count', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.set_title('Sample Abundance Distribution', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot comparison
            data_to_plot = [zero_counts, non_zero_counts]
            ax2.boxplot(data_to_plot, labels=['With zeros', 'Without zeros'])
            ax2.set_ylabel('Total Sample Count', fontsize=12)
            ax2.set_title('Sample Abundance Comparison', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'sample_abundance_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {output_dir / 'sample_abundance_comparison.png'}")
    
    # 6. Distance matrix heatmap (subsample if too large)
    print("\nCreating distance matrix heatmap...")
    n_samples = distance_matrix.shape[0]
    max_samples_for_heatmap = 100
    
    if n_samples <= max_samples_for_heatmap:
        fig, ax = plt.subplots(figsize=(12, 10))
        if HAS_SEABORN:
            sns.heatmap(
                distance_matrix.data,
                cmap='viridis',
                square=True,
                cbar_kws={'label': 'UniFrac Distance'},
                ax=ax
            )
        else:
            im = ax.imshow(distance_matrix.data, cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax, label='UniFrac Distance')
        ax.set_title('UniFrac Distance Matrix Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Sample Index', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'distance_matrix_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir / 'distance_matrix_heatmap.png'}")
    else:
        print(f"  Skipping heatmap (too many samples: {n_samples} > {max_samples_for_heatmap})")
    
    print(f"\nAll visualizations saved to: {output_dir}")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description='Investigate zero-distance UniFrac pairs in training data'
    )
    parser.add_argument(
        '--biom-path',
        type=str,
        required=True,
        help='Path to BIOM table file'
    )
    parser.add_argument(
        '--tree-path',
        type=str,
        required=True,
        help='Path to phylogenetic tree file (.nwk)'
    )
    parser.add_argument(
        '--rarefaction-depth',
        type=int,
        default=1000,
        help='Rarefaction depth for samples (default: 1000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='debug/zero_distance_analysis',
        help='Output directory for plots and analysis (default: debug/zero_distance_analysis)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    biom_path = Path(args.biom_path)
    tree_path = Path(args.tree_path)
    output_dir = Path(args.output_dir)
    
    if not biom_path.exists():
        print(f"ERROR: BIOM file not found: {biom_path}")
        sys.exit(1)
    
    if not tree_path.exists():
        print(f"ERROR: Tree file not found: {tree_path}")
        sys.exit(1)
    
    # Load data and compute distances
    rarefied_table, distance_matrix, sample_ids = load_data(
        str(biom_path),
        str(tree_path),
        args.rarefaction_depth
    )
    
    # Analyze zero distances
    analysis_results = analyze_zero_distances(distance_matrix)
    
    # Analyze metadata
    metadata_results = analyze_sample_metadata(rarefied_table, sample_ids, analysis_results)
    
    # Create visualizations
    create_visualizations(distance_matrix, analysis_results, metadata_results, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Review visualizations in the output directory")
    print("2. Check ZERO_DISTANCE_ANALYSIS.md for detailed findings")
    print("3. Determine handling strategy based on analysis results")


if __name__ == '__main__':
    main()
