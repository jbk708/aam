"""Unit tests for BaseSequenceEncoder class."""

import pytest
import torch
import torch.nn as nn

from aam.models.base_sequence_encoder import BaseSequenceEncoder


@pytest.fixture
def base_encoder():
    """Create a BaseSequenceEncoder instance without nucleotide prediction."""
    return BaseSequenceEncoder(
        vocab_size=5,
        embedding_dim=64,
        max_bp=150,
        token_limit=1024,
        asv_num_layers=2,
        asv_num_heads=4,
        asv_dropout=0.1,
        sample_num_layers=2,
        sample_num_heads=4,
        sample_dropout=0.1,
        predict_nucleotides=False,
    )


@pytest.fixture
def base_encoder_with_nucleotides():
    """Create a BaseSequenceEncoder instance with nucleotide prediction."""
    return BaseSequenceEncoder(
        vocab_size=5,
        embedding_dim=64,
        max_bp=150,
        token_limit=1024,
        asv_num_layers=2,
        asv_num_heads=4,
        asv_dropout=0.1,
        sample_num_layers=2,
        sample_num_heads=4,
        sample_dropout=0.1,
        predict_nucleotides=True,
    )


@pytest.fixture
def sample_tokens():
    """Create sample tokens for testing [B, S, L]."""
    batch_size = 2
    num_asvs = 10
    seq_len = 50
    tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len))
    tokens[:, :, 40:] = 0
    return tokens


@pytest.fixture
def sample_tokens_with_partial_asvs():
    """Create sample tokens with some ASVs fully padded."""
    batch_size = 2
    num_asvs = 10
    seq_len = 50
    tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len))
    tokens[:, 5:, :] = 0
    return tokens


@pytest.fixture
def sample_tokens_full_length():
    """Create sample tokens with full length sequences."""
    batch_size = 2
    num_asvs = 5
    seq_len = 50
    return torch.randint(1, 5, (batch_size, num_asvs, seq_len))


class TestBaseSequenceEncoder:
    """Test suite for BaseSequenceEncoder class."""

    def test_init(self, base_encoder):
        """Test BaseSequenceEncoder initialization."""
        assert base_encoder is not None
        assert isinstance(base_encoder, nn.Module)

    def test_init_with_nucleotides(self, base_encoder_with_nucleotides):
        """Test BaseSequenceEncoder initialization with nucleotide prediction."""
        assert base_encoder_with_nucleotides is not None
        assert isinstance(base_encoder_with_nucleotides, nn.Module)

    def test_init_default_vocab_size(self):
        """Test that vocab_size defaults to 5."""
        encoder = BaseSequenceEncoder(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
        )
        assert encoder is not None

    def test_init_default_max_bp(self):
        """Test that max_bp defaults to 150."""
        encoder = BaseSequenceEncoder(
            embedding_dim=64,
            token_limit=1024,
        )
        assert encoder is not None

    def test_forward_shape_embeddings_only(self, base_encoder, sample_tokens):
        """Test forward pass output shape without nucleotide predictions."""
        result = base_encoder(sample_tokens, return_nucleotides=False)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 10, 64)

    def test_forward_shape_with_nucleotides(self, base_encoder_with_nucleotides, sample_tokens):
        """Test forward pass output shape with nucleotide predictions."""
        embeddings, nucleotides = base_encoder_with_nucleotides(sample_tokens, return_nucleotides=True)
        assert embeddings.shape == (2, 10, 64)
        assert nucleotides.shape == (2, 10, 50, 5)

    def test_forward_different_batch_sizes(self, base_encoder):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 4, 8]:
            tokens = torch.randint(1, 5, (batch_size, 10, 50))
            result = base_encoder(tokens)
            assert result.shape == (batch_size, 10, 64)

    def test_forward_different_num_asvs(self, base_encoder):
        """Test forward pass with different numbers of ASVs."""
        for num_asvs in [5, 10, 20, 50]:
            tokens = torch.randint(1, 5, (2, num_asvs, 50))
            result = base_encoder(tokens)
            assert result.shape == (2, num_asvs, 64)

    def test_forward_different_seq_lengths(self, base_encoder):
        """Test forward pass with different sequence lengths."""
        for seq_len in [10, 50, 100, 150]:
            tokens = torch.randint(1, 5, (2, 10, seq_len))
            result = base_encoder(tokens)
            assert result.shape == (2, 10, 64)

    def test_forward_with_padding(self, base_encoder, sample_tokens):
        """Test forward pass with padded sequences."""
        result = base_encoder(sample_tokens)
        assert result.shape == (2, 10, 64)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_forward_full_length_sequences(self, base_encoder, sample_tokens_full_length):
        """Test forward pass with full-length sequences."""
        result = base_encoder(sample_tokens_full_length)
        assert result.shape == (2, 5, 64)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_forward_with_partial_asv_masking(self, base_encoder, sample_tokens_with_partial_asvs):
        """Test forward pass with partial ASV masking."""
        result = base_encoder(sample_tokens_with_partial_asvs)
        assert result.shape == (2, 10, 64)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_forward_inference_mode(self, base_encoder, sample_tokens):
        """Test forward pass in inference mode (no nucleotide predictions)."""
        base_encoder.eval()
        result = base_encoder(sample_tokens, return_nucleotides=False)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 10, 64)

    def test_forward_training_mode_with_nucleotides(self, base_encoder_with_nucleotides, sample_tokens):
        """Test forward pass in training mode with nucleotide predictions."""
        base_encoder_with_nucleotides.train()
        embeddings, nucleotides = base_encoder_with_nucleotides(sample_tokens, return_nucleotides=True)
        assert embeddings.shape == (2, 10, 64)
        assert nucleotides.shape == (2, 10, 50, 5)

    def test_forward_training_mode_without_nucleotides(self, base_encoder_with_nucleotides, sample_tokens):
        """Test forward pass in training mode without requesting nucleotide predictions."""
        base_encoder_with_nucleotides.train()
        result = base_encoder_with_nucleotides(sample_tokens, return_nucleotides=False)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 10, 64)

    def test_forward_no_nucleotide_head(self, base_encoder, sample_tokens):
        """Test that encoder without nucleotide head doesn't return predictions."""
        result = base_encoder(sample_tokens, return_nucleotides=True)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 10, 64)

    def test_gradients_flow(self, base_encoder, sample_tokens):
        """Test that gradients flow correctly."""
        result = base_encoder(sample_tokens)
        loss = result.sum()
        loss.backward()

        for param in base_encoder.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

    def test_gradients_with_nucleotides(self, base_encoder_with_nucleotides, sample_tokens):
        """Test that gradients flow correctly with nucleotide predictions."""
        embeddings, nucleotides = base_encoder_with_nucleotides(sample_tokens, return_nucleotides=True)
        loss = embeddings.sum() + nucleotides.sum()
        loss.backward()

        for param in base_encoder_with_nucleotides.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

    def test_no_gradient_explosion(self, base_encoder, sample_tokens):
        """Test that gradients don't explode."""
        result = base_encoder(sample_tokens)
        loss = result.sum()
        loss.backward()

        max_grad = max(p.grad.abs().max().item() for p in base_encoder.parameters() if p.grad is not None)
        assert max_grad < 1e6, f"Gradient explosion detected: max_grad={max_grad}"

    def test_no_gradient_vanishing(self, base_encoder, sample_tokens):
        """Test that gradients don't vanish."""
        result = base_encoder(sample_tokens)
        loss = result.sum()
        loss.backward()

        max_grad = max(p.grad.abs().max().item() for p in base_encoder.parameters() if p.grad is not None)
        assert max_grad > 1e-8, f"Gradient vanishing detected: max_grad={max_grad}"

    def test_forward_same_device(self, base_encoder, sample_tokens):
        """Test that output is on same device as input."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            base_encoder = base_encoder.to(device)
            sample_tokens = sample_tokens.to(device)
            result = base_encoder(sample_tokens)
            assert result.device == device

    def test_dropout_training_mode(self, base_encoder, sample_tokens):
        """Test that dropout is active in training mode."""
        base_encoder.train()
        result1 = base_encoder(sample_tokens)
        result2 = base_encoder(sample_tokens)
        assert not torch.allclose(result1, result2, atol=1e-5)

    def test_dropout_eval_mode(self, base_encoder, sample_tokens):
        """Test that dropout is inactive in eval mode."""
        base_encoder.eval()
        result1 = base_encoder(sample_tokens)
        result2 = base_encoder(sample_tokens)
        assert torch.allclose(result1, result2, atol=1e-5)

    def test_forward_with_zero_padding(self, base_encoder):
        """Test forward pass with sequences that have zero padding."""
        tokens = torch.zeros(2, 10, 50)
        tokens[:, :, :30] = torch.randint(1, 5, (2, 10, 30))
        result = base_encoder(tokens)
        assert result.shape == (2, 10, 64)
        assert not torch.isnan(result).any()

    def test_forward_variable_length_sequences(self, base_encoder):
        """Test forward pass with variable-length sequences in same batch."""
        tokens = torch.zeros(2, 10, 50)
        tokens[0, :, :40] = torch.randint(1, 5, (1, 10, 40))
        tokens[1, :, :25] = torch.randint(1, 5, (1, 10, 25))
        result = base_encoder(tokens)
        assert result.shape == (2, 10, 64)

    def test_forward_different_activations(self):
        """Test forward pass with different activation functions."""
        for activation in ["gelu", "relu"]:
            encoder = BaseSequenceEncoder(
                embedding_dim=64,
                max_bp=150,
                token_limit=1024,
                asv_activation=activation,
                sample_activation=activation,
            )
            tokens = torch.randint(1, 5, (2, 10, 50))
            result = encoder(tokens)
            assert result.shape == (2, 10, 64)

    def test_forward_different_num_layers(self):
        """Test forward pass with different numbers of layers."""
        for num_layers in [1, 2, 4]:
            encoder = BaseSequenceEncoder(
                embedding_dim=64,
                max_bp=150,
                token_limit=1024,
                asv_num_layers=num_layers,
                sample_num_layers=num_layers,
            )
            tokens = torch.randint(1, 5, (2, 10, 50))
            result = encoder(tokens)
            assert result.shape == (2, 10, 64)

    def test_forward_different_num_heads(self):
        """Test forward pass with different numbers of heads."""
        for num_heads in [1, 2, 4, 8]:
            encoder = BaseSequenceEncoder(
                embedding_dim=64,
                max_bp=150,
                token_limit=1024,
                asv_num_heads=num_heads,
                sample_num_heads=num_heads,
            )
            tokens = torch.randint(1, 5, (2, 10, 50))
            result = encoder(tokens)
            assert result.shape == (2, 10, 64)

    def test_forward_different_embedding_dims(self):
        """Test forward pass with different embedding dimensions."""
        for embedding_dim in [32, 64, 128, 256]:
            encoder = BaseSequenceEncoder(
                embedding_dim=embedding_dim,
                max_bp=150,
                token_limit=1024,
            )
            tokens = torch.randint(1, 5, (2, 10, 50))
            result = encoder(tokens)
            assert result.shape == (2, 10, embedding_dim)

    def test_mask_creation_from_tokens(self, base_encoder):
        """Test that mask is correctly created from tokens."""
        tokens = torch.zeros(2, 10, 50)
        tokens[0, :5, :30] = torch.randint(1, 5, (1, 5, 30))
        tokens[1, :8, :40] = torch.randint(1, 5, (1, 8, 40))
        result = base_encoder(tokens)
        assert result.shape == (2, 10, 64)
        assert not torch.isnan(result).any()
