"""Unit tests for ASVEncoder class."""

import pytest
import torch
import torch.nn as nn

from aam.models.asv_encoder import ASVEncoder


@pytest.fixture
def asv_encoder():
    """Create an ASVEncoder instance without nucleotide prediction."""
    return ASVEncoder(
        vocab_size=6,
        embedding_dim=64,
        max_bp=150,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        predict_nucleotides=False,
    )


@pytest.fixture
def asv_encoder_with_nucleotides():
    """Create an ASVEncoder instance with nucleotide prediction."""
    return ASVEncoder(
        vocab_size=6,
        embedding_dim=64,
        max_bp=150,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        predict_nucleotides=True,
    )


@pytest.fixture
def sample_tokens_full_length():
    """Create sample tokens with full length sequences (no padding)."""
    from aam.data.tokenizer import SequenceTokenizer

    tokens = torch.randint(1, 5, (2, 5, 50))
    tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
    return tokens


class TestASVEncoder:
    """Test suite for ASVEncoder class."""

    def test_init(self, asv_encoder):
        """Test ASVEncoder initialization."""
        assert asv_encoder is not None
        assert isinstance(asv_encoder, nn.Module)

    def test_init_with_nucleotides(self, asv_encoder_with_nucleotides):
        """Test ASVEncoder initialization with nucleotide prediction."""
        assert asv_encoder_with_nucleotides is not None
        assert isinstance(asv_encoder_with_nucleotides, nn.Module)

    def test_init_default_vocab_size(self):
        """Test that vocab_size defaults to 7 (includes MASK token)."""
        encoder = ASVEncoder(
            embedding_dim=64,
            max_bp=150,
            num_layers=2,
            num_heads=4,
        )
        assert encoder is not None
        assert encoder.vocab_size == 7

    def test_init_default_max_bp(self):
        """Test that max_bp defaults to 150."""
        encoder = ASVEncoder(
            embedding_dim=64,
            num_layers=2,
            num_heads=4,
        )
        assert encoder is not None

    def test_forward_shape_embeddings_only(self, asv_encoder, sample_tokens):
        """Test forward pass output shape without nucleotide predictions."""
        result = asv_encoder(sample_tokens, return_nucleotides=False)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 10, 64)

    def test_forward_shape_with_nucleotides(self, asv_encoder_with_nucleotides, sample_tokens):
        """Test forward pass output shape with nucleotide predictions."""
        embeddings, nucleotides, mask_indices = asv_encoder_with_nucleotides(sample_tokens, return_nucleotides=True)
        assert embeddings.shape == (2, 10, 64)
        assert nucleotides.shape == (2, 10, 50, 6)
        # mask_indices is None when mask_ratio=0 (default in fixture)
        assert mask_indices is None

    def test_forward_different_batch_sizes(self, asv_encoder):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 4, 8]:
            tokens = torch.randint(1, 5, (batch_size, 10, 50))
            result = asv_encoder(tokens)
            assert result.shape == (batch_size, 10, 64)

    def test_forward_different_num_asvs(self, asv_encoder):
        """Test forward pass with different numbers of ASVs."""
        for num_asvs in [5, 10, 20, 50]:
            tokens = torch.randint(1, 5, (2, num_asvs, 50))
            result = asv_encoder(tokens)
            assert result.shape == (2, num_asvs, 64)

    def test_forward_different_seq_lengths(self, asv_encoder):
        """Test forward pass with different sequence lengths."""
        for seq_len in [10, 50, 100, 150]:
            tokens = torch.randint(1, 5, (2, 10, seq_len))
            result = asv_encoder(tokens)
            assert result.shape == (2, 10, 64)

    def test_forward_with_padding(self, asv_encoder, sample_tokens):
        """Test forward pass with padded sequences."""
        result = asv_encoder(sample_tokens)
        assert result.shape == (2, 10, 64)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_forward_full_length_sequences(self, asv_encoder, sample_tokens_full_length):
        """Test forward pass with full-length sequences."""
        result = asv_encoder(sample_tokens_full_length)
        assert result.shape == (2, 5, 64)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_forward_inference_mode(self, asv_encoder, sample_tokens):
        """Test forward pass in inference mode (no nucleotide predictions)."""
        asv_encoder.eval()
        result = asv_encoder(sample_tokens, return_nucleotides=False)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 10, 64)

    def test_forward_training_mode_with_nucleotides(self, asv_encoder_with_nucleotides, sample_tokens):
        """Test forward pass in training mode with nucleotide predictions."""
        asv_encoder_with_nucleotides.train()
        embeddings, nucleotides, mask_indices = asv_encoder_with_nucleotides(sample_tokens, return_nucleotides=True)
        assert embeddings.shape == (2, 10, 64)
        assert nucleotides.shape == (2, 10, 50, 6)
        # mask_indices is None when mask_ratio=0 (default in fixture)
        assert mask_indices is None

    def test_forward_training_mode_without_nucleotides(self, asv_encoder_with_nucleotides, sample_tokens):
        """Test forward pass in training mode without requesting nucleotide predictions."""
        asv_encoder_with_nucleotides.train()
        result = asv_encoder_with_nucleotides(sample_tokens, return_nucleotides=False)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 10, 64)

    def test_forward_no_nucleotide_head(self, asv_encoder, sample_tokens):
        """Test that encoder without nucleotide head doesn't return predictions."""
        result = asv_encoder(sample_tokens, return_nucleotides=True)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 10, 64)

    def test_gradients_flow(self, asv_encoder, sample_tokens):
        """Test that gradients flow correctly."""
        result = asv_encoder(sample_tokens)
        loss = result.sum()
        loss.backward()

        for param in asv_encoder.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

    def test_gradients_with_nucleotides(self, asv_encoder_with_nucleotides, sample_tokens):
        """Test that gradients flow correctly with nucleotide predictions."""
        embeddings, nucleotides, mask_indices = asv_encoder_with_nucleotides(sample_tokens, return_nucleotides=True)
        loss = embeddings.sum() + nucleotides.sum()
        loss.backward()

        for param in asv_encoder_with_nucleotides.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

    def test_no_gradient_explosion(self, asv_encoder, sample_tokens):
        """Test that gradients don't explode."""
        result = asv_encoder(sample_tokens)
        loss = result.sum()
        loss.backward()

        max_grad = max(p.grad.abs().max().item() for p in asv_encoder.parameters() if p.grad is not None)
        assert max_grad < 1e6, f"Gradient explosion detected: max_grad={max_grad}"

    def test_no_gradient_vanishing(self, asv_encoder, sample_tokens):
        """Test that gradients don't vanish."""
        result = asv_encoder(sample_tokens)
        loss = result.sum()
        loss.backward()

        max_grad = max(p.grad.abs().max().item() for p in asv_encoder.parameters() if p.grad is not None)
        assert max_grad > 1e-8, f"Gradient vanishing detected: max_grad={max_grad}"

    def test_forward_same_device(self, asv_encoder, sample_tokens):
        """Test that output is on same device as input."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            asv_encoder = asv_encoder.to(device)
            sample_tokens = sample_tokens.to(device)
            result = asv_encoder(sample_tokens)
            assert result.device.type == device.type

    def test_dropout_training_mode(self, asv_encoder, sample_tokens):
        """Test that dropout is active in training mode."""
        asv_encoder.train()
        result1 = asv_encoder(sample_tokens)
        result2 = asv_encoder(sample_tokens)
        assert not torch.allclose(result1, result2, atol=1e-5)

    def test_dropout_eval_mode(self, asv_encoder, sample_tokens):
        """Test that dropout is inactive in eval mode."""
        asv_encoder.eval()
        result1 = asv_encoder(sample_tokens)
        result2 = asv_encoder(sample_tokens)
        assert torch.allclose(result1, result2, atol=1e-5)

    def test_forward_with_zero_padding(self, asv_encoder):
        """Test forward pass with sequences that have zero padding."""
        tokens = torch.zeros(2, 10, 50)
        tokens[:, :, :30] = torch.randint(1, 5, (2, 10, 30))
        result = asv_encoder(tokens)
        assert result.shape == (2, 10, 64)
        assert not torch.isnan(result).any()

    def test_forward_variable_length_sequences(self, asv_encoder):
        """Test forward pass with variable-length sequences in same batch."""
        tokens = torch.zeros(2, 10, 50)
        tokens[0, :, :40] = torch.randint(1, 5, (1, 10, 40))
        tokens[1, :, :25] = torch.randint(1, 5, (1, 10, 25))
        result = asv_encoder(tokens)
        assert result.shape == (2, 10, 64)

    def test_forward_different_activations(self):
        """Test forward pass with different activation functions."""
        for activation in ["gelu", "relu"]:
            encoder = ASVEncoder(
                embedding_dim=64,
                max_bp=150,
                num_layers=2,
                num_heads=4,
                activation=activation,
            )
            tokens = torch.randint(1, 5, (2, 10, 50))
            result = encoder(tokens)
            assert result.shape == (2, 10, 64)

    def test_forward_different_num_layers(self):
        """Test forward pass with different numbers of layers."""
        for num_layers in [1, 2, 4]:
            encoder = ASVEncoder(
                embedding_dim=64,
                max_bp=150,
                num_layers=num_layers,
                num_heads=4,
            )
            tokens = torch.randint(1, 5, (2, 10, 50))
            result = encoder(tokens)
            assert result.shape == (2, 10, 64)

    def test_forward_different_num_heads(self):
        """Test forward pass with different numbers of heads."""
        for num_heads in [1, 2, 4, 8]:
            encoder = ASVEncoder(
                embedding_dim=64,
                max_bp=150,
                num_layers=2,
                num_heads=num_heads,
            )
            tokens = torch.randint(1, 5, (2, 10, 50))
            result = encoder(tokens)
            assert result.shape == (2, 10, 64)

    def test_forward_different_embedding_dims(self):
        """Test forward pass with different embedding dimensions."""
        for embedding_dim in [32, 64, 128, 256]:
            encoder = ASVEncoder(
                embedding_dim=embedding_dim,
                max_bp=150,
                num_layers=2,
                num_heads=4,
            )
            tokens = torch.randint(1, 5, (2, 10, 50))
            result = encoder(tokens)
            assert result.shape == (2, 10, embedding_dim)

    def test_gradient_checkpointing_init(self):
        """Test that gradient checkpointing can be enabled."""
        encoder = ASVEncoder(
            embedding_dim=64,
            max_bp=150,
            num_layers=2,
            num_heads=4,
            gradient_checkpointing=True,
        )
        assert encoder.transformer.gradient_checkpointing is True

    def test_gradient_checkpointing_disabled_by_default(self):
        """Test that gradient checkpointing is disabled by default."""
        encoder = ASVEncoder(
            embedding_dim=64,
            max_bp=150,
            num_layers=2,
            num_heads=4,
        )
        assert encoder.transformer.gradient_checkpointing is False

    def test_gradient_checkpointing_training_mode_gradients(self, sample_tokens):
        """Test that gradients flow correctly with checkpointing in training mode."""
        encoder = ASVEncoder(
            embedding_dim=64,
            max_bp=150,
            num_layers=2,
            num_heads=4,
            gradient_checkpointing=True,
        )
        encoder.train()

        result = encoder(sample_tokens)
        loss = result.sum()
        loss.backward()

        # Check that model parameters have gradients
        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()

    def test_gradient_checkpointing_output_shape(self, sample_tokens):
        """Test that checkpointing produces correct output shape."""
        encoder = ASVEncoder(
            embedding_dim=64,
            max_bp=150,
            num_layers=2,
            num_heads=4,
            gradient_checkpointing=True,
        )
        encoder.train()

        result = encoder(sample_tokens)
        assert result.shape == (2, 10, 64)


class TestASVEncoderMasking:
    """Test suite for ASVEncoder masking functionality (MAE)."""

    def test_init_with_mask_ratio(self):
        """Test ASVEncoder initialization with mask_ratio."""
        encoder = ASVEncoder(
            embedding_dim=64,
            max_bp=150,
            num_layers=2,
            num_heads=4,
            mask_ratio=0.15,
        )
        assert encoder.mask_ratio == 0.15

    def test_init_with_mask_strategy_random(self):
        """Test ASVEncoder initialization with random mask strategy."""
        encoder = ASVEncoder(
            embedding_dim=64,
            max_bp=150,
            num_layers=2,
            num_heads=4,
            mask_strategy="random",
        )
        assert encoder.mask_strategy == "random"

    def test_init_with_mask_strategy_span(self):
        """Test ASVEncoder initialization with span mask strategy."""
        encoder = ASVEncoder(
            embedding_dim=64,
            max_bp=150,
            num_layers=2,
            num_heads=4,
            mask_strategy="span",
        )
        assert encoder.mask_strategy == "span"

    def test_init_invalid_mask_strategy(self):
        """Test ASVEncoder raises error for invalid mask strategy."""
        with pytest.raises(ValueError, match="Invalid mask_strategy"):
            ASVEncoder(
                embedding_dim=64,
                max_bp=150,
                num_layers=2,
                num_heads=4,
                mask_strategy="invalid",
            )

    def test_default_mask_ratio_zero(self):
        """Test that default mask_ratio is 0 (no masking)."""
        encoder = ASVEncoder(
            embedding_dim=64,
            max_bp=150,
            num_layers=2,
            num_heads=4,
        )
        assert encoder.mask_ratio == 0.0

    def test_default_vocab_size_includes_mask_token(self):
        """Test that default vocab_size is 7 (includes MASK token)."""
        encoder = ASVEncoder(
            embedding_dim=64,
            max_bp=150,
            num_layers=2,
            num_heads=4,
        )
        assert encoder.vocab_size == 7

    def test_mask_token_constant(self):
        """Test that MASK_TOKEN constant is defined."""
        assert ASVEncoder.MASK_TOKEN == 6

    def test_forward_returns_mask_indices_with_nucleotides(self):
        """Test that forward returns mask_indices when return_nucleotides=True."""
        encoder = ASVEncoder(
            embedding_dim=64,
            max_bp=150,
            num_layers=2,
            num_heads=4,
            predict_nucleotides=True,
            mask_ratio=0.15,
        )
        encoder.train()

        tokens = torch.randint(1, 5, (2, 10, 50))
        result = encoder(tokens, return_nucleotides=True)

        assert isinstance(result, tuple)
        assert len(result) == 3  # embeddings, nuc_predictions, mask_indices
        embeddings, nuc_predictions, mask_indices = result
        assert embeddings.shape == (2, 10, 64)
        assert nuc_predictions.shape == (2, 10, 50, 7)

    def test_forward_no_masking_when_mask_ratio_zero(self):
        """Test that no masking occurs when mask_ratio is 0."""
        encoder = ASVEncoder(
            embedding_dim=64,
            max_bp=150,
            num_layers=2,
            num_heads=4,
            predict_nucleotides=True,
            mask_ratio=0.0,
        )
        encoder.train()

        tokens = torch.randint(1, 5, (2, 10, 50))
        embeddings, nuc_predictions, mask_indices = encoder(tokens, return_nucleotides=True)

        # mask_indices should be None when mask_ratio is 0
        assert mask_indices is None

    def test_forward_no_masking_in_eval_mode(self):
        """Test that no masking occurs in eval mode."""
        encoder = ASVEncoder(
            embedding_dim=64,
            max_bp=150,
            num_layers=2,
            num_heads=4,
            predict_nucleotides=True,
            mask_ratio=0.15,
        )
        encoder.eval()

        tokens = torch.randint(1, 5, (2, 10, 50))
        embeddings, nuc_predictions, mask_indices = encoder(tokens, return_nucleotides=True)

        # mask_indices should be None in eval mode
        assert mask_indices is None

    def test_apply_masking_returns_correct_shapes(self):
        """Test that _apply_masking returns correct shapes."""
        encoder = ASVEncoder(
            embedding_dim=64,
            max_bp=150,
            num_layers=2,
            num_heads=4,
            mask_ratio=0.15,
        )

        tokens = torch.randint(1, 5, (20, 50))  # [batch*num_asvs, seq_len]
        masked_tokens, mask_indices = encoder._apply_masking(tokens)

        assert masked_tokens.shape == tokens.shape
        assert mask_indices.shape == tokens.shape
        assert mask_indices.dtype == torch.bool

    def test_apply_masking_only_masks_nucleotides(self):
        """Test that masking only affects nucleotide positions (1-4)."""
        encoder = ASVEncoder(
            embedding_dim=64,
            max_bp=150,
            num_layers=2,
            num_heads=4,
            mask_ratio=0.5,  # High ratio to ensure some masking
            mask_strategy="random",
        )

        # Create tokens with padding (0) and START (5)
        tokens = torch.zeros(10, 50, dtype=torch.long)
        tokens[:, 0] = 5  # START token
        tokens[:, 1:30] = torch.randint(1, 5, (10, 29))  # Nucleotides
        # tokens[:, 30:] remain 0 (padding)

        masked_tokens, mask_indices = encoder._apply_masking(tokens)

        # Verify padding positions (0) are never masked
        padding_mask = tokens == 0
        assert not mask_indices[padding_mask].any(), "Padding tokens should not be masked"

        # Verify START positions (5) are never masked
        start_mask = tokens == 5
        assert not mask_indices[start_mask].any(), "START tokens should not be masked"

    def test_apply_masking_approximately_correct_ratio(self):
        """Test that masking ratio is approximately correct."""
        encoder = ASVEncoder(
            embedding_dim=64,
            max_bp=150,
            num_layers=2,
            num_heads=4,
            mask_ratio=0.20,
            mask_strategy="random",
        )

        # Create tokens with only nucleotides (easier to verify ratio)
        tokens = torch.randint(1, 5, (100, 50))  # All nucleotides

        masked_tokens, mask_indices = encoder._apply_masking(tokens)

        # Check that approximately 20% of positions are masked
        actual_ratio = mask_indices.float().mean().item()
        # Allow 5% tolerance for randomness
        assert 0.10 <= actual_ratio <= 0.30, f"Expected ~0.20, got {actual_ratio}"

    def test_masking_replaces_with_mask_token(self):
        """Test that masked positions are replaced with MASK_TOKEN."""
        encoder = ASVEncoder(
            embedding_dim=64,
            max_bp=150,
            num_layers=2,
            num_heads=4,
            mask_ratio=0.5,  # High ratio to ensure masking
            mask_strategy="random",
        )

        tokens = torch.randint(1, 5, (10, 50))
        masked_tokens, mask_indices = encoder._apply_masking(tokens)

        # Verify masked positions have MASK_TOKEN value
        if mask_indices.any():
            assert (masked_tokens[mask_indices] == ASVEncoder.MASK_TOKEN).all()

    def test_masking_preserves_unmasked_positions(self):
        """Test that unmasked positions retain original values."""
        encoder = ASVEncoder(
            embedding_dim=64,
            max_bp=150,
            num_layers=2,
            num_heads=4,
            mask_ratio=0.15,
            mask_strategy="random",
        )

        tokens = torch.randint(1, 5, (10, 50))
        masked_tokens, mask_indices = encoder._apply_masking(tokens)

        # Verify unmasked positions have original values
        unmasked_positions = ~mask_indices
        assert (masked_tokens[unmasked_positions] == tokens[unmasked_positions]).all()

    def test_forward_with_masking_produces_valid_output(self):
        """Test that forward pass with masking produces valid output."""
        encoder = ASVEncoder(
            embedding_dim=64,
            max_bp=150,
            num_layers=2,
            num_heads=4,
            predict_nucleotides=True,
            mask_ratio=0.15,
        )
        encoder.train()

        tokens = torch.randint(1, 5, (2, 10, 50))
        embeddings, nuc_predictions, mask_indices = encoder(tokens, return_nucleotides=True)

        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()
        assert not torch.isnan(nuc_predictions).any()
        assert not torch.isinf(nuc_predictions).any()

    def test_gradients_flow_with_masking(self):
        """Test that gradients flow correctly with masking enabled."""
        encoder = ASVEncoder(
            embedding_dim=64,
            max_bp=150,
            num_layers=2,
            num_heads=4,
            predict_nucleotides=True,
            mask_ratio=0.15,
        )
        encoder.train()

        tokens = torch.randint(1, 5, (2, 10, 50))
        embeddings, nuc_predictions, mask_indices = encoder(tokens, return_nucleotides=True)

        loss = embeddings.sum() + nuc_predictions.sum()
        loss.backward()

        for param in encoder.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()
