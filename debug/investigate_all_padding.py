#!/usr/bin/env python3
"""Investigate why many sequences have all padding."""

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


def analyze_batch(batch, token_limit):
    """Analyze a batch to understand padding patterns."""
    print("\n" + "=" * 80)
    print("BATCH ANALYSIS")
    print("=" * 80)

    tokens = batch["tokens"]  # [batch_size, token_limit, seq_len]
    counts = batch["counts"]  # [batch_size, token_limit, 1]
    sample_ids = batch["sample_ids"]

    batch_size, num_asvs, seq_len = tokens.shape

    print(f"\nBatch shape: {tokens.shape}")
    print(f"Sample IDs: {sample_ids}")

    # Analyze each sample
    for sample_idx, sample_id in enumerate(sample_ids):
        print(f"\n{'=' * 80}")
        print(f"SAMPLE {sample_idx}: {sample_id}")
        print(f"{'=' * 80}")

        sample_tokens = tokens[sample_idx]  # [token_limit, seq_len]
        sample_counts = counts[sample_idx]  # [token_limit, 1]

        # Count valid ASVs (non-zero counts)
        valid_asv_mask = sample_counts.squeeze() > 0
        num_valid_asvs = valid_asv_mask.sum().item()
        num_padding_asvs = token_limit - num_valid_asvs

        print("\nASV Statistics:")
        print(f"  Total ASV positions: {token_limit}")
        print(f"  Valid ASVs (count > 0): {num_valid_asvs}")
        print(f"  Padding ASVs (count = 0): {num_padding_asvs}")

        # Analyze sequences
        sequences_with_start_token = 0
        sequences_all_zeros = 0
        sequences_all_padding = 0

        for asv_idx in range(token_limit):
            asv_tokens = sample_tokens[asv_idx]  # [seq_len]
            asv_count = sample_counts[asv_idx, 0].item()

            # Check for START_TOKEN at position 0
            has_start_token = (asv_tokens[0] == 5).item()

            # Check if all zeros
            is_all_zeros = (asv_tokens == 0).all().item()

            # Check if all padding (0 or START_TOKEN only)
            non_zero_count = (asv_tokens > 0).sum().item()
            is_all_padding = (non_zero_count == 0) or (non_zero_count == 1 and has_start_token)

            if has_start_token:
                sequences_with_start_token += 1
            if is_all_zeros:
                sequences_all_zeros += 1
            if is_all_padding:
                sequences_all_padding += 1

            # Show details for first few ASVs and any problematic ones
            if asv_idx < 5 or (asv_count > 0 and is_all_padding):
                print(f"\n  ASV {asv_idx}:")
                print(f"    Count: {asv_count}")
                print(f"    Has START_TOKEN at pos 0: {has_start_token}")
                print(f"    Is all zeros: {is_all_zeros}")
                print(f"    Is all padding: {is_all_padding}")
                print(f"    Non-zero tokens: {non_zero_count}")
                if asv_idx < 3:
                    print(f"    First 10 tokens: {asv_tokens[:10].tolist()}")

        print("\nSequence Statistics:")
        print(f"  Sequences with START_TOKEN: {sequences_with_start_token}")
        print(f"  Sequences all zeros: {sequences_all_zeros}")
        print(f"  Sequences all padding: {sequences_all_padding}")

        # Check mask that will be used in attention pooling
        tokens_flat = sample_tokens.reshape(-1, seq_len)  # [token_limit, seq_len]
        mask = (tokens_flat > 0).long()  # [token_limit, seq_len]
        mask_sum_per_seq = mask.sum(dim=1)  # [token_limit]

        all_padding_mask = mask_sum_per_seq == 0
        num_all_padding_from_mask = all_padding_mask.sum().item()

        print("\nMask Analysis (for attention pooling):")
        print(f"  Sequences with mask sum = 0 (all padding): {num_all_padding_from_mask}")
        print(f"  Mask sum per sequence (first 10): {mask_sum_per_seq[:10].tolist()}")

        # Check if valid ASVs have all-padding sequences
        valid_asv_all_padding = (all_padding_mask & valid_asv_mask).sum().item()
        if valid_asv_all_padding > 0:
            print(f"\n  ⚠️  WARNING: {valid_asv_all_padding} VALID ASVs have all-padding sequences!")
            print("     This is a problem - valid ASVs should have valid sequences.")
            problem_indices = torch.where(all_padding_mask & valid_asv_mask)[0].tolist()
            print(f"     Problem ASV indices: {problem_indices[:10]}...")


def trace_sequence_through_pipeline(dataset, sample_idx, asv_idx, token_limit):
    """Trace a specific sequence through the pipeline."""
    print(f"\n{'=' * 80}")
    print(f"TRACING SEQUENCE: Sample {sample_idx}, ASV {asv_idx}")
    print(f"{'=' * 80}")

    # Get sample from dataset
    sample = dataset[sample_idx]
    sample_id = sample["sample_id"]
    tokens = sample["tokens"]  # [num_asvs, seq_len]
    counts = sample["counts"]  # [num_asvs, 1]

    print(f"\nSample ID: {sample_id}")
    print(f"Original number of ASVs: {tokens.shape[0]}")

    if asv_idx >= tokens.shape[0]:
        print(f"  ASV index {asv_idx} is beyond original ASVs (padding)")
        return

    original_tokens = tokens[asv_idx]
    original_count = counts[asv_idx, 0].item()

    print(f"\nOriginal ASV {asv_idx}:")
    print(f"  Count: {original_count}")
    print(f"  Has START_TOKEN at pos 0: {(original_tokens[0] == 5).item()}")
    print(f"  First 10 tokens: {original_tokens[:10].tolist()}")
    print(f"  Non-zero tokens: {(original_tokens > 0).sum().item()}")

    # Simulate collate_fn truncation
    num_asvs = tokens.shape[0]
    if num_asvs > token_limit:
        tokens_truncated = tokens[:token_limit]
        counts_truncated = counts[:token_limit]
        print(f"\nAfter truncation (token_limit={token_limit}):")
        print(f"  ASVs kept: {tokens_truncated.shape[0]}")
        if asv_idx < token_limit:
            truncated_tokens = tokens_truncated[asv_idx]
            print(f"  ASV {asv_idx} tokens (first 10): {truncated_tokens[:10].tolist()}")
        else:
            print(f"  ASV {asv_idx} was removed by truncation")

    # Simulate padding
    padded_tokens = torch.zeros(token_limit, tokens.shape[1], dtype=torch.long)
    padded_tokens[:num_asvs] = tokens[:token_limit] if num_asvs > token_limit else tokens

    if asv_idx < token_limit:
        padded_sequence = padded_tokens[asv_idx]
        print("\nAfter padding:")
        print(f"  ASV {asv_idx} tokens (first 10): {padded_sequence[:10].tolist()}")
        print(f"  Has START_TOKEN at pos 0: {(padded_sequence[0] == 5).item()}")
        print(f"  Non-zero tokens: {(padded_sequence > 0).sum().item()}")


def main():
    parser = argparse.ArgumentParser(description="Investigate all-padding sequences")
    parser.add_argument("--token-limit", type=int, default=256, help="Token limit")
    parser.add_argument("--batch-size", type=int, default=6, help="Batch size")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--num-batches", type=int, default=1, help="Number of batches to analyze")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    biom_file = data_dir / "fall_train_only_all_outdoor.biom"
    tree_file = data_dir / "all-outdoors_sepp_tree.nwk"

    if not biom_file.exists():
        print(f"ERROR: BIOM file not found: {biom_file}")
        sys.exit(1)

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

    print(f"Dataset size: {len(dataset)}")

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

    # Analyze batches
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.num_batches:
            break

        print(f"\n{'#' * 80}")
        print(f"BATCH {batch_idx}")
        print(f"{'#' * 80}")

        analyze_batch(batch, args.token_limit)

        # Trace a few specific sequences
        if batch_idx == 0:
            sample_ids = batch["sample_ids"]
            for sample_idx, sample_id in enumerate(sample_ids):
                # Find this sample in dataset
                dataset_sample_idx = dataset.sample_ids.index(sample_id)

                # Trace first ASV and a middle ASV
                trace_sequence_through_pipeline(dataset, dataset_sample_idx, 0, args.token_limit)
                if dataset[dataset_sample_idx]["tokens"].shape[0] > 10:
                    trace_sequence_through_pipeline(dataset, dataset_sample_idx, 10, args.token_limit)


if __name__ == "__main__":
    main()
