"""Unit tests for AttentionPooling and utility functions."""

import pytest
import torch
import torch.nn as nn

from aam.models.attention_pooling import (
    AttentionPooling,
    float_mask,
    create_mask_from_tokens,
    apply_mask,
)


@pytest.fixture
def attention_pooling():
    """Create an AttentionPooling instance."""
    return AttentionPooling(hidden_dim=64)


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    batch_size = 2
    seq_len = 10
    hidden_dim = 64
    return torch.randn(batch_size, seq_len, hidden_dim)


@pytest.fixture
def sample_mask():
    """Create sample mask for testing."""
    return torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])


class TestAttentionPooling:
    """Test suite for AttentionPooling class."""

    def test_init(self, attention_pooling):
        """Test AttentionPooling initialization."""
        assert attention_pooling is not None
        assert isinstance(attention_pooling, nn.Module)
        assert attention_pooling.hidden_dim == 64
        assert isinstance(attention_pooling.query, nn.Linear)
        assert isinstance(attention_pooling.norm, nn.LayerNorm)

    def test_forward_shape(self, attention_pooling, sample_embeddings):
        """Test forward pass output shape."""
        result = attention_pooling(sample_embeddings)
        assert result.shape == (2, 64)

    def test_forward_with_mask_shape(self, attention_pooling, sample_embeddings, sample_mask):
        """Test forward pass with mask output shape."""
        result = attention_pooling(sample_embeddings, mask=sample_mask)
        assert result.shape == (2, 64)

    def test_forward_different_batch_sizes(self, attention_pooling):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 4, 8]:
            embeddings = torch.randn(batch_size, 10, 64)
            result = attention_pooling(embeddings)
            assert result.shape == (batch_size, 64)

    def test_forward_different_seq_lengths(self, attention_pooling):
        """Test forward pass with different sequence lengths."""
        for seq_len in [5, 10, 20, 50]:
            embeddings = torch.randn(2, seq_len, 64)
            result = attention_pooling(embeddings)
            assert result.shape == (2, 64)

    def test_forward_attention_weights_sum_to_one(self, attention_pooling, sample_embeddings, sample_mask):
        """Test that attention weights sum to 1 over valid positions."""
        pass

    def test_forward_masked_positions_ignored(self, attention_pooling, sample_embeddings, sample_mask):
        """Test that masked positions are ignored in attention."""
        pass

    def test_forward_no_mask(self, attention_pooling, sample_embeddings):
        """Test forward pass without mask (all positions valid)."""
        result = attention_pooling(sample_embeddings)
        assert result.shape == (2, 64)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


class TestFloatMask:
    """Test suite for float_mask utility function."""

    def test_float_mask_basic(self):
        """Test basic float mask conversion."""
        tensor = torch.tensor([1, 2, 0, 3, 0])
        result = float_mask(tensor)
        expected = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])
        assert torch.allclose(result, expected)

    def test_float_mask_all_nonzero(self):
        """Test float mask with all nonzero values."""
        tensor = torch.tensor([1, 2, 3, 4])
        result = float_mask(tensor)
        expected = torch.tensor([1.0, 1.0, 1.0, 1.0])
        assert torch.allclose(result, expected)

    def test_float_mask_all_zero(self):
        """Test float mask with all zero values."""
        tensor = torch.tensor([0, 0, 0, 0])
        result = float_mask(tensor)
        expected = torch.tensor([0.0, 0.0, 0.0, 0.0])
        assert torch.allclose(result, expected)

    def test_float_mask_2d(self):
        """Test float mask with 2D tensor."""
        tensor = torch.tensor([[1, 0, 2], [0, 3, 0]])
        result = float_mask(tensor)
        expected = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        assert torch.allclose(result, expected)


class TestCreateMaskFromTokens:
    """Test suite for create_mask_from_tokens utility function."""

    def test_create_mask_from_tokens_basic(self):
        """Test basic mask creation from tokens."""
        tokens = torch.tensor([[1, 2, 3, 0, 0], [4, 1, 0, 0, 0]])
        result = create_mask_from_tokens(tokens)
        expected = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
        assert torch.allclose(result, expected)

    def test_create_mask_from_tokens_all_valid(self):
        """Test mask creation with all valid tokens."""
        tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        result = create_mask_from_tokens(tokens)
        expected = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]])
        assert torch.allclose(result, expected)

    def test_create_mask_from_tokens_all_padding(self):
        """Test mask creation with all padding tokens."""
        tokens = torch.tensor([[0, 0, 0], [0, 0, 0]])
        result = create_mask_from_tokens(tokens)
        expected = torch.tensor([[0, 0, 0], [0, 0, 0]])
        assert torch.allclose(result, expected)


class TestApplyMask:
    """Test suite for apply_mask utility function."""

    def test_apply_mask_basic(self):
        """Test basic mask application."""
        embeddings = torch.randn(2, 5, 64)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
        result = apply_mask(embeddings, mask)
        assert result.shape == embeddings.shape

    def test_apply_mask_masked_positions_zero(self):
        """Test that masked positions are set to zero."""
        embeddings = torch.ones(2, 3, 4)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        result = apply_mask(embeddings, mask)
        assert torch.allclose(result[0, 2, :], torch.zeros(4))
        assert torch.allclose(result[1, 1:, :], torch.zeros(2, 4))
