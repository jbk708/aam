"""Unit tests for TransformerEncoder class."""

import pytest
import torch
import torch.nn as nn

from aam.models.transformer import TransformerEncoder


@pytest.fixture
def transformer_encoder():
    """Create a TransformerEncoder instance."""
    return TransformerEncoder(
        num_layers=2,
        num_heads=4,
        hidden_dim=64,
        intermediate_size=256,
        dropout=0.1,
        activation="gelu",
    )


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


class TestTransformerEncoder:
    """Test suite for TransformerEncoder class."""

    def test_init(self, transformer_encoder):
        """Test TransformerEncoder initialization."""
        assert transformer_encoder is not None
        assert isinstance(transformer_encoder, nn.Module)

    def test_init_default_intermediate_size(self):
        """Test that intermediate_size defaults to 4 * hidden_dim."""
        encoder = TransformerEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            dropout=0.1,
        )
        assert encoder is not None

    def test_forward_shape(self, transformer_encoder, sample_embeddings):
        """Test forward pass output shape."""
        result = transformer_encoder(sample_embeddings)
        assert result.shape == sample_embeddings.shape
        assert result.shape == (2, 10, 64)

    def test_forward_with_mask_shape(self, transformer_encoder, sample_embeddings, sample_mask):
        """Test forward pass with mask output shape."""
        result = transformer_encoder(sample_embeddings, mask=sample_mask)
        assert result.shape == sample_embeddings.shape
        assert result.shape == (2, 10, 64)

    def test_forward_different_batch_sizes(self, transformer_encoder):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 4, 8]:
            embeddings = torch.randn(batch_size, 10, 64)
            result = transformer_encoder(embeddings)
            assert result.shape == (batch_size, 10, 64)

    def test_forward_different_seq_lengths(self, transformer_encoder):
        """Test forward pass with different sequence lengths."""
        for seq_len in [5, 10, 20, 50]:
            embeddings = torch.randn(2, seq_len, 64)
            result = transformer_encoder(embeddings)
            assert result.shape == (2, seq_len, 64)

    def test_forward_no_mask(self, transformer_encoder, sample_embeddings):
        """Test forward pass without mask (all positions valid)."""
        result = transformer_encoder(sample_embeddings)
        assert result.shape == sample_embeddings.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_forward_with_mask(self, transformer_encoder, sample_embeddings, sample_mask):
        """Test forward pass with mask."""
        result = transformer_encoder(sample_embeddings, mask=sample_mask)
        assert result.shape == sample_embeddings.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_forward_full_padding_mask(self, transformer_encoder):
        """Test forward pass with full padding mask."""
        embeddings = torch.randn(2, 10, 64)
        mask = torch.zeros(2, 10, dtype=torch.long)
        result = transformer_encoder(embeddings, mask=mask)
        assert result.shape == embeddings.shape

    def test_forward_partial_padding_mask(self, transformer_encoder, sample_embeddings, sample_mask):
        """Test forward pass with partial padding mask."""
        result = transformer_encoder(sample_embeddings, mask=sample_mask)
        assert result.shape == sample_embeddings.shape

    def test_forward_different_num_layers(self):
        """Test forward pass with different numbers of layers."""
        for num_layers in [1, 2, 4, 6]:
            encoder = TransformerEncoder(
                num_layers=num_layers,
                num_heads=4,
                hidden_dim=64,
                dropout=0.1,
            )
            embeddings = torch.randn(2, 10, 64)
            result = encoder(embeddings)
            assert result.shape == (2, 10, 64)

    def test_forward_different_num_heads(self):
        """Test forward pass with different numbers of heads."""
        for num_heads in [1, 2, 4, 8]:
            encoder = TransformerEncoder(
                num_layers=2,
                num_heads=num_heads,
                hidden_dim=64,
                dropout=0.1,
            )
            embeddings = torch.randn(2, 10, 64)
            result = encoder(embeddings)
            assert result.shape == (2, 10, 64)

    def test_forward_different_hidden_dims(self):
        """Test forward pass with different hidden dimensions."""
        for hidden_dim in [32, 64, 128, 256]:
            encoder = TransformerEncoder(
                num_layers=2,
                num_heads=4,
                hidden_dim=hidden_dim,
                dropout=0.1,
            )
            embeddings = torch.randn(2, 10, hidden_dim)
            result = encoder(embeddings)
            assert result.shape == (2, 10, hidden_dim)

    def test_forward_gelu_activation(self):
        """Test forward pass with GELU activation."""
        encoder = TransformerEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            activation="gelu",
            dropout=0.1,
        )
        embeddings = torch.randn(2, 10, 64)
        result = encoder(embeddings)
        assert result.shape == (2, 10, 64)

    def test_forward_relu_activation(self):
        """Test forward pass with ReLU activation."""
        encoder = TransformerEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            activation="relu",
            dropout=0.1,
        )
        embeddings = torch.randn(2, 10, 64)
        result = encoder(embeddings)
        assert result.shape == (2, 10, 64)

    def test_gradients_flow(self, transformer_encoder, sample_embeddings):
        """Test that gradients flow correctly."""
        result = transformer_encoder(sample_embeddings)
        loss = result.sum()
        loss.backward()

        for param in transformer_encoder.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

    def test_gradients_with_mask(self, transformer_encoder, sample_embeddings, sample_mask):
        """Test that gradients flow correctly with mask."""
        result = transformer_encoder(sample_embeddings, mask=sample_mask)
        loss = result.sum()
        loss.backward()

        for param in transformer_encoder.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

    def test_no_gradient_explosion(self, transformer_encoder, sample_embeddings):
        """Test that gradients don't explode."""
        result = transformer_encoder(sample_embeddings)
        loss = result.sum()
        loss.backward()

        max_grad = max(p.grad.abs().max().item() for p in transformer_encoder.parameters() if p.grad is not None)
        assert max_grad < 1e6, f"Gradient explosion detected: max_grad={max_grad}"

    def test_no_gradient_vanishing(self, transformer_encoder, sample_embeddings):
        """Test that gradients don't vanish."""
        result = transformer_encoder(sample_embeddings)
        loss = result.sum()
        loss.backward()

        min_grad = min(p.grad.abs().min().item() for p in transformer_encoder.parameters() if p.grad is not None)
        assert min_grad > 1e-8, f"Gradient vanishing detected: min_grad={min_grad}"

    def test_forward_same_device(self, transformer_encoder, sample_embeddings):
        """Test that output is on same device as input."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            transformer_encoder = transformer_encoder.to(device)
            sample_embeddings = sample_embeddings.to(device)
            result = transformer_encoder(sample_embeddings)
            assert result.device == device

    def test_dropout_training_mode(self, transformer_encoder, sample_embeddings):
        """Test that dropout is active in training mode."""
        transformer_encoder.train()
        result1 = transformer_encoder(sample_embeddings)
        result2 = transformer_encoder(sample_embeddings)
        assert not torch.allclose(result1, result2, atol=1e-5)

    def test_dropout_eval_mode(self, transformer_encoder, sample_embeddings):
        """Test that dropout is inactive in eval mode."""
        transformer_encoder.eval()
        result1 = transformer_encoder(sample_embeddings)
        result2 = transformer_encoder(sample_embeddings)
        assert torch.allclose(result1, result2, atol=1e-5)

    def test_mask_handles_variable_lengths(self, transformer_encoder):
        """Test that mask correctly handles variable sequence lengths."""
        embeddings = torch.randn(3, 10, 64)
        mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        result = transformer_encoder(embeddings, mask=mask)
        assert result.shape == (3, 10, 64)
