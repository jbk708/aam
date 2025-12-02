"""Unit tests for PositionEmbedding class."""

import pytest
import torch
import torch.nn as nn

from aam.models.position_embedding import PositionEmbedding


@pytest.fixture
def position_embedding():
    """Create a PositionEmbedding instance."""
    return PositionEmbedding(max_length=100, hidden_dim=64)


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    batch_size = 2
    seq_len = 10
    hidden_dim = 64
    return torch.randn(batch_size, seq_len, hidden_dim)


class TestPositionEmbedding:
    """Test suite for PositionEmbedding class."""

    def test_init(self, position_embedding):
        """Test PositionEmbedding initialization."""
        assert position_embedding is not None
        assert isinstance(position_embedding, nn.Module)
        assert position_embedding.max_length == 100
        assert position_embedding.hidden_dim == 64
        assert isinstance(position_embedding.position_embedding, nn.Embedding)

    def test_forward_shape(self, position_embedding, sample_embeddings):
        """Test forward pass output shape."""
        result = position_embedding(sample_embeddings)
        assert result.shape == sample_embeddings.shape
        assert result.shape == (2, 10, 64)

    def test_forward_different_batch_sizes(self, position_embedding):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 4, 8]:
            embeddings = torch.randn(batch_size, 10, 64)
            result = position_embedding(embeddings)
            assert result.shape == (batch_size, 10, 64)

    def test_forward_different_seq_lengths(self, position_embedding):
        """Test forward pass with different sequence lengths."""
        for seq_len in [5, 10, 20, 50]:
            embeddings = torch.randn(2, seq_len, 64)
            result = position_embedding(embeddings)
            assert result.shape == (2, seq_len, 64)

    def test_forward_adds_position_info(self, position_embedding, sample_embeddings):
        """Test that position information is added (not concatenated)."""
        result = position_embedding(sample_embeddings)
        assert result.shape == sample_embeddings.shape

    def test_forward_same_device(self, position_embedding, sample_embeddings):
        """Test that output is on same device as input."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            position_embedding = position_embedding.to(device)
            sample_embeddings = sample_embeddings.to(device)
            result = position_embedding(sample_embeddings)
            assert result.device == device

    def test_forward_max_length_boundary(self, position_embedding):
        """Test forward pass at max_length boundary."""
        embeddings = torch.randn(2, 100, 64)
        result = position_embedding(embeddings)
        assert result.shape == (2, 100, 64)

    def test_forward_below_max_length(self, position_embedding):
        """Test forward pass with sequence length below max_length."""
        embeddings = torch.randn(2, 50, 64)
        result = position_embedding(embeddings)
        assert result.shape == (2, 50, 64)
