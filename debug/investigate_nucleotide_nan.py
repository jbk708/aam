#!/usr/bin/env python3
"""Investigate where NaN appears in nucleotide predictions."""

import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

from aam.data.dataset import ASVDataset, collate_fn
from aam.data.biom_loader import BIOMLoader
from aam.data.tokenizer import SequenceTokenizer
from aam.data.unifrac import UniFracComputer
from aam.models.sequence_encoder import SequenceEncoder


def check_tensor_stats(tensor, name):
    """Print tensor statistics."""
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    if tensor.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.long):
        print(f"  Min: {tensor.min().item()}, Max: {tensor.max().item()}")
    else:
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        nan_count = torch.isnan(tensor).sum().item()
        total = tensor.numel()
        print(f"  Min: {tensor.min().item():.6f}, Max: {tensor.max().item():.6f}")
        print(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")
        if has_nan:
            print(f"  NaN count: {nan_count} / {total} ({100 * nan_count / total:.2f}%)")
            # Find which ASVs have NaN
            if len(tensor.shape) >= 2:
                # For [batch, num_asvs, ...] shape
                nan_per_asv = torch.isnan(tensor).any(dim=tuple(range(2, len(tensor.shape))))
                nan_asv_indices = torch.where(nan_per_asv)
                print(f"  ASVs with NaN: {len(nan_asv_indices[0])} out of {tensor.shape[1]}")
                if len(nan_asv_indices[0]) > 0:
                    print(f"    Sample 0, ASVs with NaN: {torch.where(nan_per_asv[0])[0].tolist()[:20]}")


def main():
    parser = argparse.ArgumentParser(description="Investigate NaN in nucleotide predictions")
    parser.add_argument("--token-limit", type=int, default=256, help="Token limit")
    parser.add_argument("--batch-size", type=int, default=6, help="Batch size")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    biom_file = data_dir / "fall_train_only_all_outdoor.biom"
    tree_file = data_dir / "all-outdoors_sepp_tree.nwk"

    if not biom_file.exists():
        print(f"ERROR: BIOM file not found: {biom_file}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    loader = BIOMLoader()
    table = loader.load_table(str(biom_file))
    table = loader.rarefy(table, depth=5000, random_seed=42)

    # Compute UniFrac distances
    print("Computing UniFrac distances...")
    computer = UniFracComputer()
    unifrac_distances = computer.compute_unweighted(table, str(tree_file))

    # Create dataset
    print("Creating dataset...")
    tokenizer = SequenceTokenizer()
    dataset = ASVDataset(
        table=table,
        tokenizer=tokenizer,
        unifrac_distances=unifrac_distances,
        token_limit=args.token_limit,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(
            batch,
            token_limit=args.token_limit,
            unifrac_distances=unifrac_distances,
            unifrac_metric="unweighted",
        ),
    )

    # Create model
    print("Creating model...")
    model = SequenceEncoder(
        vocab_size=6,
        embedding_dim=128,
        max_bp=150,
        token_limit=args.token_limit,
        encoder_type="unifrac",
        predict_nucleotides=True,
    )
    model = model.to(device)
    model.eval()

    # Get a batch
    batch = next(iter(dataloader))
    tokens = batch["tokens"].to(device)  # [batch_size, num_asvs, seq_len]
    counts = batch["counts"].to(device)  # [batch_size, num_asvs, 1]

    print(f"\n{'=' * 80}")
    print("INVESTIGATING NUCLEOTIDE PREDICTIONS")
    print(f"{'=' * 80}")

    check_tensor_stats(tokens, "Input tokens")

    # Create mask for valid ASVs
    asv_mask = (tokens.sum(dim=-1) > 0).long()  # [batch_size, num_asvs]
    print("\nASV mask (valid ASVs):")
    print(f"  Shape: {asv_mask.shape}")
    print(f"  Valid ASVs per sample: {asv_mask.sum(dim=1).tolist()}")

    # Trace through model
    with torch.no_grad():
        # Step 1: ASV Encoder
        print(f"\n{'=' * 80}")
        print("STEP 1: ASV Encoder")
        print(f"{'=' * 80}")

        batch_size, num_asvs, seq_len = tokens.shape
        tokens_flat = tokens.reshape(batch_size * num_asvs, seq_len)
        mask_flat = (tokens_flat > 0).long()

        # Token embedding
        token_emb = model.sample_encoder.asv_encoder.token_embedding(tokens_flat)
        check_tensor_stats(token_emb, "Token embeddings")

        # Position embedding
        pos_emb = model.sample_encoder.asv_encoder.position_embedding(token_emb)
        check_tensor_stats(pos_emb, "Position embeddings")

        # Transformer
        transformer_out = model.sample_encoder.asv_encoder.transformer(pos_emb, mask=mask_flat)
        check_tensor_stats(transformer_out, "Transformer output (before NaN fix)")

        # Apply NaN fix (matching ASVEncoder.forward())
        mask_sum = mask_flat.sum(dim=-1)  # [batch_size * num_asvs]
        all_padding = mask_sum == 0  # [batch_size * num_asvs]
        if all_padding.any():
            all_padding_expanded = all_padding.unsqueeze(-1).unsqueeze(-1)  # [batch_size * num_asvs, 1, 1]
            transformer_out = torch.where(all_padding_expanded, torch.zeros_like(transformer_out), transformer_out)
        check_tensor_stats(transformer_out, "Transformer output (after NaN fix)")

        # Attention pooling
        pooled = model.sample_encoder.asv_encoder.attention_pooling(transformer_out, mask=mask_flat)
        check_tensor_stats(pooled, "Pooled ASV embeddings")

        # Nucleotide head
        print(f"\n{'=' * 80}")
        print("STEP 2: Nucleotide Head")
        print(f"{'=' * 80}")

        nuc_logits = model.sample_encoder.asv_encoder.nucleotide_head(transformer_out)
        check_tensor_stats(nuc_logits, "Nucleotide logits (before reshape)")

        # Reshape
        nuc_predictions = nuc_logits.reshape(batch_size, num_asvs, seq_len, -1)
        check_tensor_stats(nuc_predictions, "Nucleotide predictions (after reshape)")

        # Check which ASVs have NaN
        nan_per_asv = torch.isnan(nuc_predictions).any(dim=(2, 3))  # [batch_size, num_asvs]
        print("\nNaN per ASV:")
        for sample_idx in range(batch_size):
            nan_asvs = torch.where(nan_per_asv[sample_idx])[0].tolist()
            valid_asvs = torch.where(asv_mask[sample_idx])[0].tolist()
            padding_asvs = torch.where(~asv_mask[sample_idx].bool())[0].tolist()

            nan_in_valid = [a for a in nan_asvs if a in valid_asvs]
            nan_in_padding = [a for a in nan_asvs if a in padding_asvs]

            print(f"\n  Sample {sample_idx}:")
            print(f"    Valid ASVs: {len(valid_asvs)}")
            print(f"    Padding ASVs: {len(padding_asvs)}")
            print(f"    ASVs with NaN: {len(nan_asvs)}")
            print(f"    NaN in valid ASVs: {len(nan_in_valid)}")
            print(f"    NaN in padding ASVs: {len(nan_in_padding)}")
            if nan_in_valid:
                print(f"    ⚠️  WARNING: Valid ASVs with NaN: {nan_in_valid[:10]}")
            if nan_in_padding:
                print(f"    Padding ASVs with NaN: {nan_in_padding[:10]}")

        # Full forward pass
        print(f"\n{'=' * 80}")
        print("STEP 3: Full Forward Pass")
        print(f"{'=' * 80}")

        outputs = model(tokens, return_nucleotides=True)
        if "nuc_predictions" in outputs:
            check_tensor_stats(outputs["nuc_predictions"], "Final nucleotide predictions")


if __name__ == "__main__":
    main()
