"""Shared pytest fixtures for AAM tests."""

import random

import numpy as np
import pytest
import torch
from biom import Table

from aam.data.tokenizer import SequenceTokenizer


def generate_150bp_sequence(seed=None):
    """Generate a random 150bp DNA sequence."""
    if seed is not None:
        random.seed(seed)
    return "".join(random.choice("ACGT") for _ in range(150))


def create_sample_tokens(batch_size: int = 2, num_asvs: int = 10, seq_len: int = 50) -> torch.Tensor:
    """Create sample tokens for testing [B, S, L].

    Args:
        batch_size: Number of samples in batch
        num_asvs: Number of ASVs per sample
        seq_len: Sequence length

    Returns:
        Tensor of shape [batch_size, num_asvs, seq_len]
    """
    tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len))
    tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
    tokens[:, :, 40:] = 0
    return tokens


@pytest.fixture
def sample_tokens():
    """Create sample tokens for testing [B, S, L]."""
    return create_sample_tokens(batch_size=2, num_asvs=10, seq_len=50)


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing [B, S, D]."""
    return torch.randn(2, 10, 64)


@pytest.fixture
def sample_mask():
    """Create sample mask for testing [B, S]."""
    return torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])


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
