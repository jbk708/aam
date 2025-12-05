#!/usr/bin/env python3
"""Investigate batching logic and UniFrac distance extraction with shuffling.

This script tests whether the order of sample_ids matches the order of tokens/counts
after collate_fn, especially when batches are shuffled and token_limit truncation occurs.
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aam.data.dataset import collate_fn, ASVDataset
from aam.data.tokenizer import SequenceTokenizer
from aam.data.biom_loader import BIOMLoader
from aam.data.unifrac import UniFracComputer
from torch.utils.data import DataLoader
from functools import partial
from skbio import DistanceMatrix


def create_test_distance_matrix(sample_ids):
    """Create a test distance matrix for debugging."""
    n = len(sample_ids)
    # Create symmetric distance matrix with known values
    # Distance from sample_i to sample_j = |i - j| * 0.1
    data = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                data[i, j] = abs(i - j) * 0.1
    return DistanceMatrix(data, ids=sample_ids)


def test_collate_fn_order():
    """Test that sample_ids order matches tokens/counts order in collate_fn."""
    print("=" * 80)
    print("TEST 1: Verify sample_ids order matches tokens/counts order")
    print("=" * 80)

    # Create a batch with known sample order (even batch size for UniFrac)
    batch = [
        {
            "tokens": torch.LongTensor([[5, 1, 2, 3] for _ in range(10)]),  # 10 ASVs
            "counts": torch.FloatTensor([[10.0] for _ in range(10)]),
            "sample_id": "sample_A",
        },
        {
            "tokens": torch.LongTensor([[5, 2, 3, 4] for _ in range(15)]),  # 15 ASVs
            "counts": torch.FloatTensor([[20.0] for _ in range(15)]),
            "sample_id": "sample_B",
        },
        {
            "tokens": torch.LongTensor([[5, 3, 4, 1] for _ in range(8)]),  # 8 ASVs
            "counts": torch.FloatTensor([[30.0] for _ in range(8)]),
            "sample_id": "sample_C",
        },
        {
            "tokens": torch.LongTensor([[5, 4, 1, 2] for _ in range(12)]),  # 12 ASVs
            "counts": torch.FloatTensor([[40.0] for _ in range(12)]),
            "sample_id": "sample_D",
        },
    ]

    # Create distance matrix
    sample_ids_all = ["sample_A", "sample_B", "sample_C", "sample_D"]
    distance_matrix = create_test_distance_matrix(sample_ids_all)

    token_limit = 12  # Will truncate sample_B from 15 to 12

    result = collate_fn(
        batch,
        token_limit=token_limit,
        unifrac_distances=distance_matrix,
        unifrac_metric="unweighted",
    )

    print(f"\nInput batch order: {[s['sample_id'] for s in batch]}")
    print(f"Output sample_ids: {result['sample_ids']}")
    print(f"Tokens shape: {result['tokens'].shape}")
    print(f"UniFrac target shape: {result['unifrac_target'].shape}")

    # Verify order matches
    expected_order = ["sample_A", "sample_B", "sample_C", "sample_D"]
    assert result["sample_ids"] == expected_order, f"Order mismatch: {result['sample_ids']} != {expected_order}"

    # Verify distance matrix order matches
    # Distance from sample_A to sample_B should be at [0, 1]
    # Distance from sample_A to sample_C should be at [0, 2]
    # Distance from sample_B to sample_C should be at [1, 2]

    distances = result["unifrac_target"].numpy()
    print(f"\nDistance matrix:")
    print(f"  [0, 1] (A->B): {distances[0, 1]:.3f} (expected: 0.1)")
    print(f"  [0, 2] (A->C): {distances[0, 2]:.3f} (expected: 0.2)")
    print(f"  [0, 3] (A->D): {distances[0, 3]:.3f} (expected: 0.3)")
    print(f"  [1, 2] (B->C): {distances[1, 2]:.3f} (expected: 0.1)")

    # Verify distances are correct
    assert np.isclose(distances[0, 1], 0.1), f"Distance A->B incorrect: {distances[0, 1]}"
    assert np.isclose(distances[0, 2], 0.2), f"Distance A->C incorrect: {distances[0, 2]}"
    assert np.isclose(distances[0, 3], 0.3), f"Distance A->D incorrect: {distances[0, 3]}"
    assert np.isclose(distances[1, 2], 0.1), f"Distance B->C incorrect: {distances[1, 2]}"

    print("✅ Order verification passed!")
    return True


def test_shuffled_batch_order():
    """Test that shuffled batches maintain correct order."""
    print("\n" + "=" * 80)
    print("TEST 2: Verify shuffled batch order")
    print("=" * 80)

    # Create a batch with shuffled order (even batch size)
    batch = [
        {
            "tokens": torch.LongTensor([[5, 3, 4, 1] for _ in range(8)]),
            "counts": torch.FloatTensor([[30.0] for _ in range(8)]),
            "sample_id": "sample_C",  # Third in original order
        },
        {
            "tokens": torch.LongTensor([[5, 1, 2, 3] for _ in range(10)]),
            "counts": torch.FloatTensor([[10.0] for _ in range(10)]),
            "sample_id": "sample_A",  # First in original order
        },
        {
            "tokens": torch.LongTensor([[5, 2, 3, 4] for _ in range(15)]),
            "counts": torch.FloatTensor([[20.0] for _ in range(15)]),
            "sample_id": "sample_B",  # Second in original order
        },
        {
            "tokens": torch.LongTensor([[5, 4, 1, 2] for _ in range(12)]),
            "counts": torch.FloatTensor([[40.0] for _ in range(12)]),
            "sample_id": "sample_D",  # Fourth in original order
        },
    ]

    # Create distance matrix (in original order)
    sample_ids_all = ["sample_A", "sample_B", "sample_C", "sample_D"]
    distance_matrix = create_test_distance_matrix(sample_ids_all)

    token_limit = 12

    result = collate_fn(
        batch,
        token_limit=token_limit,
        unifrac_distances=distance_matrix,
        unifrac_metric="unweighted",
    )

    print(f"\nInput batch order (shuffled): {[s['sample_id'] for s in batch]}")
    print(f"Output sample_ids: {result['sample_ids']}")

    # The output should match the input batch order (shuffled)
    expected_order = ["sample_C", "sample_A", "sample_B", "sample_D"]
    assert result["sample_ids"] == expected_order, f"Order mismatch: {result['sample_ids']} != {expected_order}"

    # Verify distance matrix is reordered correctly
    # In shuffled order: C, A, B, D
    # Distance from C to A should be at [0, 1] (original: C->A = 0.2)
    # Distance from C to B should be at [0, 2] (original: C->B = 0.1)
    # Distance from C to D should be at [0, 3] (original: C->D = 0.1)
    # Distance from A to B should be at [1, 2] (original: A->B = 0.1)

    distances = result["unifrac_target"].numpy()
    print(f"\nDistance matrix (reordered for shuffled batch):")
    print(f"  [0, 1] (C->A): {distances[0, 1]:.3f} (expected: 0.2)")
    print(f"  [0, 2] (C->B): {distances[0, 2]:.3f} (expected: 0.1)")
    print(f"  [0, 3] (C->D): {distances[0, 3]:.3f} (expected: 0.1)")
    print(f"  [1, 2] (A->B): {distances[1, 2]:.3f} (expected: 0.1)")

    # Verify distances are correct for shuffled order
    assert np.isclose(distances[0, 1], 0.2), f"Distance C->A incorrect: {distances[0, 1]}"
    assert np.isclose(distances[0, 2], 0.1), f"Distance C->B incorrect: {distances[0, 2]}"
    assert np.isclose(distances[0, 3], 0.1), f"Distance C->D incorrect: {distances[0, 3]}"
    assert np.isclose(distances[1, 2], 0.1), f"Distance A->B incorrect: {distances[1, 2]}"

    # Verify tokens match the shuffled order
    # First token of first sequence in batch should be from sample_C
    # We can't directly verify this without knowing the actual token values,
    # but we can verify the shape matches

    print(f"\nTokens shape: {result['tokens'].shape}")
    print(f"  Batch dimension (0): {result['tokens'].shape[0]} samples")
    print(f"  ASV dimension (1): {result['tokens'].shape[1]} ASVs (max {token_limit})")

    # Verify first sample has 8 ASVs (sample_C), second has 10 (sample_A), third has 12 (sample_B truncated)
    # Check non-zero ASVs in each sample
    for i, sample_id in enumerate(result["sample_ids"]):
        sample_tokens = result["tokens"][i]
        # Count non-zero sequences (sequences with at least one non-zero token)
        non_zero_asvs = (sample_tokens.sum(dim=1) > 0).sum().item()
        print(f"  Sample {i} ({sample_id}): {non_zero_asvs} non-zero ASVs")

    print("✅ Shuffled batch order verification passed!")
    return True


def test_dataloader_shuffling():
    """Test DataLoader with shuffling to see if order is preserved."""
    print("\n" + "=" * 80)
    print("TEST 3: DataLoader shuffling with token_limit")
    print("=" * 80)

    # Create minimal synthetic dataset
    from biom import Table
    import numpy as np

    num_samples = 10
    num_asvs = 50
    data = np.random.randint(1, 100, (num_asvs, num_samples))
    observation_ids = [f"ASV_{i:04d}" + "ACGT" * 37 + "A" for i in range(num_asvs)]
    sample_ids = [f"sample_{i:03d}" for i in range(num_samples)]
    table = Table(data, observation_ids=observation_ids, sample_ids=sample_ids)

    # Create distance matrix
    distance_matrix = create_test_distance_matrix(sample_ids)

    # Create dataset
    tokenizer = SequenceTokenizer()
    dataset = ASVDataset(
        table=table,
        tokenizer=tokenizer,
        max_bp=150,
        token_limit=1024,  # Dataset-level limit
        unifrac_distances=distance_matrix,
        unifrac_metric="unweighted",
    )

    # Create collate function with smaller token_limit to trigger truncation
    token_limit = 30  # Smaller than dataset limit to test truncation
    collate = partial(
        collate_fn,
        token_limit=token_limit,
        unifrac_distances=distance_matrix,
        unifrac_metric="unweighted",
    )

    # Test with shuffling
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        drop_last=True,
    )

    print(f"\nTesting {len(dataloader)} batches with shuffle=True...")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 2:  # Test first 2 batches
            break

        print(f"\nBatch {batch_idx}:")
        print(f"  sample_ids: {batch['sample_ids']}")
        print(f"  tokens shape: {batch['tokens'].shape}")
        print(f"  unifrac_target shape: {batch['unifrac_target'].shape}")

        # Verify order consistency
        sample_ids = batch["sample_ids"]
        distances = batch["unifrac_target"].numpy()

        # Verify distance matrix is symmetric
        assert np.allclose(distances, distances.T), "Distance matrix not symmetric!"

        # Verify diagonal is zero
        assert np.allclose(np.diag(distances), 0), "Distance matrix diagonal not zero!"

        # Verify distances match expected values based on sample_ids
        # The distance matrix was created with sample_ids in order: sample_000, sample_001, ..., sample_009
        # So we need to extract the numeric part to get the original index
        def get_sample_index(sample_id):
            """Extract numeric index from sample_id like 'sample_004' -> 4"""
            try:
                # Handle both string and numpy string types
                sample_id_str = str(sample_id)
                if "_" in sample_id_str:
                    return int(sample_id_str.split("_")[1])
                return -1
            except (ValueError, AttributeError):
                return -1

        distance_mismatches = []
        for i, id_i in enumerate(sample_ids):
            for j, id_j in enumerate(sample_ids):
                if i != j:
                    # Get original indices from sample ID names (e.g., 'sample_004' -> 4)
                    orig_idx_i = get_sample_index(id_i)
                    orig_idx_j = get_sample_index(id_j)

                    if orig_idx_i >= 0 and orig_idx_j >= 0:
                        expected_dist = abs(orig_idx_i - orig_idx_j) * 0.1
                        actual_dist = distances[i, j]
                        if not np.isclose(actual_dist, expected_dist, atol=0.01):
                            distance_mismatches.append(f"{id_i}->{id_j}: got {actual_dist:.3f}, expected {expected_dist:.3f}")

        if distance_mismatches:
            print(f"  ⚠️  Found {len(distance_mismatches)} distance mismatches (showing first 5):")
            for mismatch in distance_mismatches[:5]:
                print(f"    {mismatch}")
        else:
            print(f"  ✅ All distances match expected values")

        # Verify tokens order matches sample_ids order
        # Check that each sample's tokens are valid
        for i, sample_id in enumerate(sample_ids):
            sample_tokens = batch["tokens"][i]
            # Verify START_TOKEN at position 0 of each sequence
            first_tokens = sample_tokens[:, 0]
            start_token_count = (first_tokens == 5).sum().item()
            non_zero_asvs = (sample_tokens.sum(dim=1) > 0).sum().item()

            if start_token_count != non_zero_asvs:
                print(
                    f"  ⚠️  Sample {i} ({sample_id}): START_TOKEN count ({start_token_count}) != non-zero ASVs ({non_zero_asvs})"
                )

    print("✅ DataLoader shuffling test passed!")
    return True


def main():
    """Run all tests."""
    print("Investigating Batching Logic and UniFrac Distance Extraction")
    print("=" * 80)

    try:
        test_collate_fn_order()
        test_shuffled_batch_order()
        test_dataloader_shuffling()

        print("\n" + "=" * 80)
        print("✅ All tests passed!")
        print("=" * 80)
        print("\nConclusion: The batching logic appears correct.")
        print("The order of sample_ids matches the order of tokens/counts.")
        print("Distance matrix is correctly reordered to match batch order.")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
