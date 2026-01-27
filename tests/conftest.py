"""Shared pytest fixtures for AAM tests."""

import random

import numpy as np
import pytest
import torch
from biom import Table

from aam.data.tokenizer import SequenceTokenizer
from aam.models.sequence_encoder import SequenceEncoder
from aam.training.losses import MultiTaskLoss


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


@pytest.fixture
def device():
    """Get device for testing with CUDA cleanup."""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return torch.device("cuda" if cuda_available else "cpu")


class MockBatchDataset:
    """Mock dataset that yields dict batches with sample_ids for testing."""

    def __init__(self, num_samples: int, device: torch.device):
        self.num_samples = num_samples
        self.device = device

    def __len__(self) -> int:
        return self.num_samples // 2

    def __iter__(self):
        batch_size = 2
        for i in range(0, self.num_samples, batch_size):
            tokens = torch.randint(1, 5, (batch_size, 10, 50))
            tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
            yield {
                "tokens": tokens.to(self.device),
                "counts": torch.rand(batch_size, 10, 1).to(self.device),
                "y_target": torch.rand(batch_size, 1).to(self.device),
                "sample_ids": [f"sample_{i + j}" for j in range(batch_size)],
            }


@pytest.fixture
def small_model():
    """Create a small SequenceEncoder for testing."""
    return SequenceEncoder(
        vocab_size=6,
        embedding_dim=32,
        max_bp=50,
        token_limit=64,
        asv_num_layers=1,
        asv_num_heads=2,
        sample_num_layers=1,
        sample_num_heads=2,
        encoder_num_layers=1,
        encoder_num_heads=2,
        base_output_dim=None,
        encoder_type="unifrac",
        predict_nucleotides=False,
    )


@pytest.fixture
def loss_fn():
    """Create a basic MultiTaskLoss instance for testing."""
    return MultiTaskLoss(penalty=1.0, nuc_penalty=0.0)
