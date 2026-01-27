"""Shared pytest fixtures for AAM tests."""

import pytest
import torch

from aam.data.tokenizer import SequenceTokenizer


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
