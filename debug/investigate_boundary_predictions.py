"""Investigate why predictions cluster at 0.0 and 1.0 during inference.

This script analyzes prediction distributions to identify root causes of boundary clustering.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
from functools import partial

# Add parent directory to path to import aam modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from aam.models.sequence_encoder import SequenceEncoder
from aam.data.biom_loader import BIOMLoader
from aam.data.unifrac import UniFracComputer
from aam.data.dataset import ASVDataset, collate_fn
from skbio import DistanceMatrix
from sklearn.model_selection import train_test_split


def load_model_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    encoder_type: str = "unifrac",
    vocab_size: int = 6,
    embedding_dim: int = 128,
    max_bp: int = 150,
    token_limit: int = 1024,
    base_output_dim: int = None,
    attention_layers: int = 2,
    attention_heads: int = 4,
) -> SequenceEncoder:
    """Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        encoder_type: Type of encoder
        vocab_size: Vocabulary size
        embedding_dim: Embedding dimension
        max_bp: Maximum sequence length
        token_limit: Maximum number of ASVs per sample
        base_output_dim: Base output dimension
        attention_layers: Number of attention layers
        attention_heads: Number of attention heads
        
    Returns:
        Loaded model
    """
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with same architecture as checkpoint
    model = SequenceEncoder(
        encoder_type=encoder_type,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_bp=max_bp,
        token_limit=token_limit,
        asv_num_layers=attention_layers,
        asv_num_heads=attention_heads,
        sample_num_layers=attention_layers,
        sample_num_heads=attention_heads,
        encoder_num_layers=attention_layers,
        encoder_num_heads=attention_heads,
        base_output_dim=base_output_dim,
        predict_nucleotides=False,  # Don't need nucleotide predictions for this analysis
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully. Epoch: {checkpoint.get('epoch', 'unknown')}")
    return model


def get_raw_predictions(model: SequenceEncoder, tokens: torch.Tensor) -> torch.Tensor:
    """Get raw predictions before clipping.
    
    Args:
        model: Model to run inference on
        tokens: Input tokens [batch_size, num_asvs, seq_len]
        
    Returns:
        Raw predictions before clipping
    """
    # Run forward pass but capture before clipping
    asv_mask = (tokens.sum(dim=-1) > 0).long()
    
    # Get embeddings
    sample_embeddings = model.sample_encoder(tokens, return_nucleotides=False)
    encoder_embeddings = model.encoder_transformer(sample_embeddings, mask=asv_mask)
    pooled_embeddings = model.attention_pooling(encoder_embeddings, mask=asv_mask)
    
    # Get raw output before clipping
    raw_pred = model.output_head(pooled_embeddings)
    
    return raw_pred


def run_inference_analysis(
    model: SequenceEncoder,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Run inference and collect predictions at different stages.
    
    Args:
        model: Model to run inference on
        dataloader: DataLoader for validation/test data
        device: Device to run inference on
        
    Returns:
        Dictionary with:
            - 'raw_predictions': Raw model outputs before clipping
            - 'clipped_predictions': Predictions after forward pass clipping
            - 'actual_distances': Actual UniFrac distances
            - 'batch_sizes': Batch sizes for each batch
    """
    raw_predictions = []
    clipped_predictions = []
    actual_distances = []
    batch_sizes = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            tokens = batch["tokens"].to(device)
            batch_size = tokens.shape[0]
            batch_sizes.append(batch_size)
            
            # Get raw predictions (before clipping)
            raw_pred = get_raw_predictions(model, tokens)
            
            # Get clipped predictions (after forward pass)
            output = model(tokens, return_nucleotides=False)
            clipped_pred = output["base_prediction"]
            
            # Get actual distances
            if "unifrac_target" in batch:
                actual = batch["unifrac_target"].to(device)
            else:
                actual = None
            
            # For UniFrac, predictions are pairwise distance matrices
            if model.encoder_type == "unifrac":
                # Extract upper triangle (excluding diagonal) for analysis
                # Raw predictions
                raw_pred_np = raw_pred.cpu().numpy()
                for i in range(batch_size):
                    triu_indices = np.triu_indices(batch_size, k=1)
                    raw_vals = raw_pred_np[i, triu_indices[0], triu_indices[1]]
                    raw_predictions.extend(raw_vals)
                
                # Clipped predictions
                clipped_pred_np = clipped_pred.cpu().numpy()
                for i in range(batch_size):
                    triu_indices = np.triu_indices(batch_size, k=1)
                    clipped_vals = clipped_pred_np[i, triu_indices[0], triu_indices[1]]
                    clipped_predictions.extend(clipped_vals)
                
                # Actual distances
                if actual is not None:
                    actual_np = actual.cpu().numpy()
                    for i in range(batch_size):
                        triu_indices = np.triu_indices(batch_size, k=1)
                        actual_vals = actual_np[i, triu_indices[0], triu_indices[1]]
                        actual_distances.extend(actual_vals)
            else:
                # For non-UniFrac (e.g., Faith PD), predictions are vectors
                raw_predictions.extend(raw_pred.cpu().numpy().flatten())
                clipped_predictions.extend(clipped_pred.cpu().numpy().flatten())
                if actual is not None:
                    actual_distances.extend(actual.cpu().numpy().flatten())
    
    return {
        "raw_predictions": np.array(raw_predictions),
        "clipped_predictions": np.array(clipped_predictions),
        "actual_distances": np.array(actual_distances) if actual_distances else None,
        "batch_sizes": np.array(batch_sizes),
    }


def analyze_distribution(predictions: np.ndarray, name: str) -> Dict[str, float]:
    """Analyze prediction distribution.
    
    Args:
        predictions: Array of predictions
        name: Name for logging
        
    Returns:
        Dictionary with distribution statistics
    """
    stats = {
        "count": len(predictions),
        "mean": float(np.mean(predictions)),
        "std": float(np.std(predictions)),
        "min": float(np.min(predictions)),
        "max": float(np.max(predictions)),
        "p25": float(np.percentile(predictions, 25)),
        "p50": float(np.percentile(predictions, 50)),
        "p75": float(np.percentile(predictions, 75)),
    }
    
    # Count boundary predictions
    epsilon = 0.01
    at_zero = np.sum(np.abs(predictions) < epsilon)
    at_one = np.sum(np.abs(predictions - 1.0) < epsilon)
    exactly_zero = np.sum(predictions == 0.0)
    exactly_one = np.sum(predictions == 1.0)
    
    stats["at_zero"] = int(at_zero)
    stats["at_one"] = int(at_one)
    stats["exactly_zero"] = int(exactly_zero)
    stats["exactly_one"] = int(exactly_one)
    stats["pct_at_zero"] = float(at_zero / len(predictions) * 100)
    stats["pct_at_one"] = float(at_one / len(predictions) * 100)
    stats["pct_exactly_zero"] = float(exactly_zero / len(predictions) * 100)
    stats["pct_exactly_one"] = float(exactly_one / len(predictions) * 100)
    
    print(f"\n{name} Distribution:")
    print(f"  Count: {stats['count']:,}")
    print(f"  Mean: {stats['mean']:.6f}")
    print(f"  Std: {stats['std']:.6f}")
    print(f"  Min: {stats['min']:.6f}")
    print(f"  Max: {stats['max']:.6f}")
    print(f"  Percentiles: 25%={stats['p25']:.6f}, 50%={stats['p50']:.6f}, 75%={stats['p75']:.6f}")
    print(f"  At 0.0 (within {epsilon}): {stats['at_zero']:,} ({stats['pct_at_zero']:.2f}%)")
    print(f"  At 1.0 (within {epsilon}): {stats['at_one']:,} ({stats['pct_at_one']:.2f}%)")
    print(f"  Exactly 0.0: {stats['exactly_zero']:,} ({stats['pct_exactly_zero']:.2f}%)")
    print(f"  Exactly 1.0: {stats['exactly_one']:,} ({stats['pct_exactly_one']:.2f}%)")
    
    return stats


def create_visualizations(
    raw_predictions: np.ndarray,
    clipped_predictions: np.ndarray,
    actual_distances: np.ndarray,
    output_dir: Path,
):
    """Create visualization plots.
    
    Args:
        raw_predictions: Raw predictions before clipping
        clipped_predictions: Predictions after clipping
        actual_distances: Actual UniFrac distances
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Histogram of raw predictions
    plt.figure(figsize=(12, 6))
    plt.hist(raw_predictions, bins=100, alpha=0.7, edgecolor='black')
    plt.axvline(0.0, color='red', linestyle='--', linewidth=2, label='Boundary: 0.0')
    plt.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Boundary: 1.0')
    plt.xlabel('Raw Prediction Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Raw Predictions (Before Clipping)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'raw_predictions_histogram.png', dpi=150)
    plt.close()
    
    # 2. Histogram of clipped predictions
    plt.figure(figsize=(12, 6))
    plt.hist(clipped_predictions, bins=100, alpha=0.7, edgecolor='black')
    plt.axvline(0.0, color='red', linestyle='--', linewidth=2, label='Boundary: 0.0')
    plt.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Boundary: 1.0')
    plt.xlabel('Clipped Prediction Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Clipped Predictions (After Forward Pass Clipping)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'clipped_predictions_histogram.png', dpi=150)
    plt.close()
    
    # 3. Comparison: Raw vs Clipped
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].hist(raw_predictions, bins=100, alpha=0.7, edgecolor='black', label='Raw')
    axes[0].set_xlabel('Prediction Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Raw Predictions (Before Clipping)')
    axes[0].axvline(0.0, color='red', linestyle='--', linewidth=2)
    axes[0].axvline(1.0, color='red', linestyle='--', linewidth=2)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].hist(clipped_predictions, bins=100, alpha=0.7, edgecolor='black', label='Clipped', color='orange')
    axes[1].set_xlabel('Prediction Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Clipped Predictions (After Forward Pass Clipping)')
    axes[1].axvline(0.0, color='red', linestyle='--', linewidth=2)
    axes[1].axvline(1.0, color='red', linestyle='--', linewidth=2)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'raw_vs_clipped_comparison.png', dpi=150)
    plt.close()
    
    # 4. Scatter plot: Raw vs Clipped (if we have enough data)
    if len(raw_predictions) > 0:
        # Sample for visualization if too many points
        max_points = 10000
        if len(raw_predictions) > max_points:
            indices = np.random.choice(len(raw_predictions), max_points, replace=False)
            raw_sample = raw_predictions[indices]
            clipped_sample = clipped_predictions[indices]
        else:
            raw_sample = raw_predictions
            clipped_sample = clipped_predictions
        
        plt.figure(figsize=(10, 10))
        plt.scatter(raw_sample, clipped_sample, alpha=0.5, s=1)
        plt.plot([-1, 2], [-1, 2], 'r--', linewidth=2, label='y=x (no clipping)')
        plt.plot([0, 0], [-1, 2], 'g--', linewidth=1, label='Boundary: 0.0')
        plt.plot([1, 1], [-1, 2], 'g--', linewidth=1, label='Boundary: 1.0')
        plt.xlabel('Raw Prediction')
        plt.ylabel('Clipped Prediction')
        plt.title('Raw vs Clipped Predictions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.savefig(output_dir / 'raw_vs_clipped_scatter.png', dpi=150)
        plt.close()
    
    # 5. Actual vs Predicted (if we have actual distances)
    if actual_distances is not None and len(actual_distances) > 0:
        # Sample for visualization if too many points
        max_points = 10000
        if len(clipped_predictions) > max_points:
            indices = np.random.choice(len(clipped_predictions), max_points, replace=False)
            pred_sample = clipped_predictions[indices]
            actual_sample = actual_distances[indices]
        else:
            pred_sample = clipped_predictions
            actual_sample = actual_distances
        
        plt.figure(figsize=(10, 10))
        plt.scatter(actual_sample, pred_sample, alpha=0.5, s=1)
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
        plt.xlabel('Actual UniFrac Distance')
        plt.ylabel('Predicted UniFrac Distance')
        plt.title('Actual vs Predicted UniFrac Distances')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(output_dir / 'actual_vs_predicted_scatter.png', dpi=150)
        plt.close()
        
        # Histogram comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].hist(actual_distances, bins=100, alpha=0.7, edgecolor='black', label='Actual')
        axes[0].set_xlabel('UniFrac Distance')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Actual UniFrac Distance Distribution')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].hist(clipped_predictions, bins=100, alpha=0.7, edgecolor='black', label='Predicted', color='orange')
        axes[1].set_xlabel('UniFrac Distance')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Predicted UniFrac Distance Distribution')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'actual_vs_predicted_histogram.png', dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Investigate boundary prediction clustering")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--table", type=str, required=True, help="Path to BIOM table")
    parser.add_argument("--tree", type=str, required=True, help="Path to phylogenetic tree")
    parser.add_argument("--output-dir", type=str, default="debug/boundary_analysis", help="Output directory for analysis")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to analyze")
    parser.add_argument("--encoder-type", type=str, default="unifrac", choices=["unifrac", "faith_pd"], help="Encoder type")
    parser.add_argument("--vocab-size", type=int, default=6, help="Vocabulary size")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--max-bp", type=int, default=150, help="Maximum sequence length")
    parser.add_argument("--token-limit", type=int, default=1024, help="Maximum number of ASVs per sample")
    parser.add_argument("--attention-layers", type=int, default=2, help="Number of attention layers")
    parser.add_argument("--attention-heads", type=int, default=4, help="Number of attention heads")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Boundary Prediction Clustering Investigation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Table: {args.table}")
    print(f"Tree: {args.tree}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print()
    
    # Load data
    print("Loading data...")
    biom_loader = BIOMLoader()
    table_obj = biom_loader.load_table(args.table)
    
    # Limit samples if requested
    if args.max_samples is not None:
        sample_ids = list(table_obj.ids(axis="sample"))[:args.max_samples]
        table_obj = table_obj.filter(sample_ids, axis="sample", inplace=False)
    
    # Compute UniFrac distances
    print("Computing UniFrac distances...")
    unifrac_computer = UniFracComputer()
    if args.encoder_type == "unifrac":
        unifrac_distances = unifrac_computer.compute_unweighted(table_obj, args.tree)
        base_output_dim = args.batch_size
    else:
        unifrac_distances = unifrac_computer.compute_faith_pd(table_obj, args.tree)
        base_output_dim = 1
    
    # Create dataset
    print("Creating dataset...")
    dataset = ASVDataset(
        table=table_obj,
        metadata=None,
        unifrac_distances=unifrac_distances,
        max_bp=args.max_bp,
        token_limit=args.token_limit,
        target_column=None,
        unifrac_metric=args.encoder_type,
    )
    
    collate = partial(
        collate_fn,
        token_limit=args.token_limit,
        unifrac_distances=unifrac_distances,
        unifrac_metric=args.encoder_type,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        drop_last=True,
    )
    
    # Load model
    model = load_model_checkpoint(
        args.checkpoint,
        device,
        encoder_type=args.encoder_type,
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        max_bp=args.max_bp,
        token_limit=args.token_limit,
        base_output_dim=base_output_dim,
        attention_layers=args.attention_layers,
        attention_heads=args.attention_heads,
    )
    
    # Run inference analysis
    results = run_inference_analysis(model, dataloader, device)
    
    # Analyze distributions
    print("\n" + "=" * 80)
    print("Distribution Analysis")
    print("=" * 80)
    
    raw_stats = analyze_distribution(results["raw_predictions"], "Raw Predictions")
    clipped_stats = analyze_distribution(results["clipped_predictions"], "Clipped Predictions")
    
    if results["actual_distances"] is not None:
        actual_stats = analyze_distribution(results["actual_distances"], "Actual Distances")
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("Creating visualizations...")
    print("=" * 80)
    
    create_visualizations(
        results["raw_predictions"],
        results["clipped_predictions"],
        results["actual_distances"],
        output_dir,
    )
    
    print(f"\nVisualizations saved to: {output_dir}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
