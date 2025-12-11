#!/usr/bin/env python
"""Test unifrac dense_pair functions for stripe computation."""

import numpy as np
import biom
from biom import Table
import skbio
from skbio import TreeNode

# Create simple test data
np.random.seed(42)
n_observations = 5
n_samples = 10

bases = "ACGT"
observation_ids = [''.join(np.random.choice(list(bases), size=20)) for _ in range(n_observations)]
sample_ids = [f"sample_{i:02d}" for i in range(n_samples)]

data = np.random.randint(0, 100, size=(n_observations, n_samples))
table = Table(data, observation_ids=observation_ids, sample_ids=sample_ids)

# Create simple tree
tree_str = "(" + ",".join([f"{obs_id}:0.1" for obs_id in observation_ids]) + ");"
import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.nwk', delete=False) as f:
    f.write(tree_str)
    tree_path = f.name

tree = skbio.read(tree_path, format="newick", into=TreeNode)

# Test unweighted_dense_pair
import unifrac

print("Testing unweighted_dense_pair function:")
print(f"Sample IDs: {sample_ids}")

# Get observation IDs
obs_ids = list(table.ids(axis="observation"))

# Test computing distance between two specific samples
sample1_id = sample_ids[0]
sample2_id = sample_ids[1]

print(f"\nComputing distance between {sample1_id} and {sample2_id}:")

# Get sample vectors (as numpy arrays)
sample1_counts = np.array([table.get_value_by_ids(obs_id, sample1_id) for obs_id in obs_ids], dtype=np.float64)
sample2_counts = np.array([table.get_value_by_ids(obs_id, sample2_id) for obs_id in obs_ids], dtype=np.float64)

print(f"Sample 1 counts: {sample1_counts}")
print(f"Sample 2 counts: {sample2_counts}")

# Compute using dense_pair
distance = unifrac.unweighted_dense_pair(
    ids=obs_ids,
    sample1=sample1_counts,
    sample2=sample2_counts,
    phylogeny=tree
)

print(f"Distance: {distance}")

# Compare with full matrix computation
full_matrix = unifrac.unweighted(table, tree)
full_distance = full_matrix[sample1_id, sample2_id]

print(f"Full matrix distance: {full_distance}")
print(f"Match: {abs(distance - full_distance) < 1e-10}")

# Now test stripe computation: compute distances from test samples to reference samples
reference_sample_ids = sample_ids[:3]
test_sample_ids = sample_ids[3:6]

print(f"\n" + "="*60)
print("Testing stripe computation using dense_pair:")
print(f"Reference samples: {reference_sample_ids}")
print(f"Test samples: {test_sample_ids}")

stripe = np.zeros((len(test_sample_ids), len(reference_sample_ids)))

for i, test_sid in enumerate(test_sample_ids):
    test_counts = np.array([table.get_value_by_ids(obs_id, test_sid) for obs_id in obs_ids], dtype=np.float64)
    for j, ref_sid in enumerate(reference_sample_ids):
        ref_counts = np.array([table.get_value_by_ids(obs_id, ref_sid) for obs_id in obs_ids], dtype=np.float64)
        stripe[i, j] = unifrac.unweighted_dense_pair(
            ids=obs_ids,
            sample1=test_counts,
            sample2=ref_counts,
            phylogeny=tree
        )

print(f"\nStripe shape: {stripe.shape}")
print(f"Stripe:\n{stripe}")

# Compare with full matrix extraction
full_matrix = unifrac.unweighted(table, tree)
test_indices = [full_matrix.ids.index(sid) for sid in test_sample_ids]
ref_indices = [full_matrix.ids.index(sid) for sid in reference_sample_ids]
expected_stripe = full_matrix.data[np.ix_(test_indices, ref_indices)]

print(f"\nExpected stripe (from full matrix):\n{expected_stripe}")
print(f"\nDifference:\n{np.abs(stripe - expected_stripe)}")
print(f"Max difference: {np.max(np.abs(stripe - expected_stripe))}")
print(f"Match: {np.max(np.abs(stripe - expected_stripe)) < 1e-10}")

import os
os.unlink(tree_path)
