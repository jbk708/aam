"""Unit tests for SequenceRegressor class."""

import pytest
import torch
import torch.nn as nn

from aam.models.sequence_regressor import SequenceRegressor
from aam.models.sequence_encoder import SequenceEncoder


@pytest.fixture
def base_encoder():
    """Create a SequenceEncoder instance for use as base model."""
    return SequenceEncoder(
        vocab_size=5,
        embedding_dim=64,
        max_bp=150,
        token_limit=1024,
        asv_num_layers=2,
        asv_num_heads=4,
        sample_num_layers=2,
        sample_num_heads=4,
        encoder_num_layers=2,
        encoder_num_heads=4,
        base_output_dim=32,
        encoder_type="unifrac",
        predict_nucleotides=False,
    )


@pytest.fixture
def base_encoder_with_nucleotides():
    """Create a SequenceEncoder instance with nucleotide prediction."""
    return SequenceEncoder(
        vocab_size=5,
        embedding_dim=64,
        max_bp=150,
        token_limit=1024,
        asv_num_layers=2,
        asv_num_heads=4,
        sample_num_layers=2,
        sample_num_heads=4,
        encoder_num_layers=2,
        encoder_num_heads=4,
        base_output_dim=32,
        encoder_type="unifrac",
        predict_nucleotides=True,
    )


@pytest.fixture
def sequence_regressor(base_encoder):
    """Create a SequenceRegressor instance with provided base model."""
    return SequenceRegressor(
        base_model=base_encoder,
        out_dim=1,
        is_classifier=False,
        freeze_base=False,
    )


@pytest.fixture
def sequence_regressor_frozen(base_encoder):
    """Create a SequenceRegressor instance with frozen base model."""
    return SequenceRegressor(
        base_model=base_encoder,
        out_dim=1,
        is_classifier=False,
        freeze_base=True,
    )


@pytest.fixture
def sequence_regressor_classifier(base_encoder):
    """Create a SequenceRegressor instance for classification."""
    return SequenceRegressor(
        base_model=base_encoder,
        out_dim=7,
        is_classifier=True,
        freeze_base=False,
    )


@pytest.fixture
def sequence_regressor_with_nucleotides(base_encoder_with_nucleotides):
    """Create a SequenceRegressor instance with nucleotide prediction."""
    return SequenceRegressor(
        base_model=base_encoder_with_nucleotides,
        out_dim=1,
        is_classifier=False,
        freeze_base=False,
    )


@pytest.fixture
def sequence_regressor_no_base():
    """Create a SequenceRegressor instance without providing base model."""
    return SequenceRegressor(
        encoder_type="unifrac",
        embedding_dim=64,
        max_bp=150,
        token_limit=1024,
        base_output_dim=32,
        out_dim=1,
        is_classifier=False,
        freeze_base=False,
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


class TestSequenceRegressor:
    """Test suite for SequenceRegressor class."""

    def test_init_with_base_model(self, sequence_regressor):
        """Test SequenceRegressor initialization with provided base model."""
        assert sequence_regressor is not None
        assert isinstance(sequence_regressor, nn.Module)

    def test_init_without_base_model(self, sequence_regressor_no_base):
        """Test SequenceRegressor initialization without base model."""
        assert sequence_regressor_no_base is not None
        assert isinstance(sequence_regressor_no_base, nn.Module)
        assert sequence_regressor_no_base.base_model is not None

    def test_init_frozen_base(self, sequence_regressor_frozen):
        """Test SequenceRegressor initialization with frozen base."""
        assert sequence_regressor_frozen is not None
        for param in sequence_regressor_frozen.base_model.parameters():
            assert not param.requires_grad

    def test_init_unfrozen_base(self, sequence_regressor):
        """Test SequenceRegressor initialization with unfrozen base."""
        assert sequence_regressor is not None
        for param in sequence_regressor.base_model.parameters():
            assert param.requires_grad

    def test_init_classifier(self, sequence_regressor_classifier):
        """Test SequenceRegressor initialization for classification."""
        assert sequence_regressor_classifier is not None
        assert sequence_regressor_classifier.is_classifier

    def test_init_regressor(self, sequence_regressor):
        """Test SequenceRegressor initialization for regression."""
        assert sequence_regressor is not None
        assert not sequence_regressor.is_classifier

    def test_forward_shape_basic(self, sequence_regressor, sample_tokens):
        """Test forward pass output shape."""
        result = sequence_regressor(sample_tokens, return_nucleotides=False)
        assert isinstance(result, dict)
        assert "target_prediction" in result
        assert "count_prediction" in result
        assert "base_embeddings" in result
        assert result["target_prediction"].shape == (2, 1)
        assert result["count_prediction"].shape == (2, 10, 1)
        assert result["base_embeddings"].shape == (2, 10, 64)

    def test_forward_shape_with_nucleotides(self, sequence_regressor_with_nucleotides, sample_tokens):
        """Test forward pass output shape with nucleotide predictions."""
        result = sequence_regressor_with_nucleotides(sample_tokens, return_nucleotides=True)
        assert isinstance(result, dict)
        assert "target_prediction" in result
        assert "count_prediction" in result
        assert "base_embeddings" in result
        assert "base_prediction" in result
        assert "nuc_predictions" in result
        assert result["target_prediction"].shape == (2, 1)
        assert result["count_prediction"].shape == (2, 10, 1)
        assert result["base_embeddings"].shape == (2, 10, 64)
        assert result["base_prediction"].shape == (2, 32)
        assert result["nuc_predictions"].shape == (2, 10, 50, 5)

    def test_forward_shape_classifier(self, sequence_regressor_classifier, sample_tokens):
        """Test forward pass output shape for classification."""
        result = sequence_regressor_classifier(sample_tokens)
        assert result["target_prediction"].shape == (2, 7)

    def test_forward_different_batch_sizes(self, sequence_regressor):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 4, 8]:
            tokens = torch.randint(1, 5, (batch_size, 10, 50))
            result = sequence_regressor(tokens)
            assert result["target_prediction"].shape == (batch_size, 1)
            assert result["count_prediction"].shape == (batch_size, 10, 1)
            assert result["base_embeddings"].shape == (batch_size, 10, 64)

    def test_forward_different_num_asvs(self, sequence_regressor):
        """Test forward pass with different numbers of ASVs."""
        for num_asvs in [5, 10, 20, 50]:
            tokens = torch.randint(1, 5, (2, num_asvs, 50))
            result = sequence_regressor(tokens)
            assert result["target_prediction"].shape == (2, 1)
            assert result["count_prediction"].shape == (2, num_asvs, 1)
            assert result["base_embeddings"].shape == (2, num_asvs, 64)

    def test_forward_different_seq_lengths(self, sequence_regressor):
        """Test forward pass with different sequence lengths."""
        for seq_len in [10, 50, 100, 150]:
            tokens = torch.randint(1, 5, (2, 10, seq_len))
            result = sequence_regressor(tokens)
            assert result["target_prediction"].shape == (2, 1)
            assert result["count_prediction"].shape == (2, 10, 1)
            assert result["base_embeddings"].shape == (2, 10, 64)

    def test_forward_with_padding(self, sequence_regressor, sample_tokens):
        """Test forward pass with padded sequences."""
        result = sequence_regressor(sample_tokens)
        assert not torch.isnan(result["target_prediction"]).any()
        assert not torch.isnan(result["count_prediction"]).any()
        assert not torch.isnan(result["base_embeddings"]).any()

    def test_forward_inference_mode(self, sequence_regressor, sample_tokens):
        """Test forward pass in inference mode (no nucleotide predictions)."""
        sequence_regressor.eval()
        result = sequence_regressor(sample_tokens, return_nucleotides=False)
        assert isinstance(result, dict)
        assert "nuc_predictions" not in result
        assert "base_prediction" not in result

    def test_forward_training_mode_with_nucleotides(self, sequence_regressor_with_nucleotides, sample_tokens):
        """Test forward pass in training mode with nucleotide predictions."""
        sequence_regressor_with_nucleotides.train()
        result = sequence_regressor_with_nucleotides(sample_tokens, return_nucleotides=True)
        assert "nuc_predictions" in result
        assert "base_prediction" in result

    def test_forward_training_mode_without_nucleotides(self, sequence_regressor_with_nucleotides, sample_tokens):
        """Test forward pass in training mode without requesting nucleotide predictions."""
        sequence_regressor_with_nucleotides.train()
        result = sequence_regressor_with_nucleotides(sample_tokens, return_nucleotides=False)
        assert isinstance(result, dict)
        assert "nuc_predictions" not in result
        assert "base_prediction" not in result

    def test_gradients_flow_unfrozen(self, sequence_regressor, sample_tokens):
        """Test that gradients flow correctly with unfrozen base."""
        sequence_regressor.train()
        result = sequence_regressor(sample_tokens)
        loss = result["target_prediction"].sum() + result["count_prediction"].sum() + result["base_embeddings"].sum()
        loss.backward()

        has_base_gradients = False
        has_count_gradients = False
        has_target_gradients = False

        for name, param in sequence_regressor.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()
                if "base_model" in name:
                    has_base_gradients = True
                elif "count" in name:
                    has_count_gradients = True
                elif "target" in name:
                    has_target_gradients = True

        assert has_base_gradients or has_count_gradients or has_target_gradients

    def test_gradients_frozen_base(self, sequence_regressor_frozen, sample_tokens):
        """Test that gradients don't flow to frozen base model."""
        result = sequence_regressor_frozen(sample_tokens)
        loss = result["target_prediction"].sum() + result["count_prediction"].sum()
        loss.backward()

        for param in sequence_regressor_frozen.base_model.parameters():
            assert param.grad is None

        for name, param in sequence_regressor_frozen.named_parameters():
            if "base_model" not in name:
                assert param.grad is not None

    def test_gradients_flow_to_heads(self, sequence_regressor_frozen, sample_tokens):
        """Test that gradients flow to count and target encoders."""
        result = sequence_regressor_frozen(sample_tokens)
        loss = result["target_prediction"].sum() + result["count_prediction"].sum()
        loss.backward()

        has_gradients = False
        for name, param in sequence_regressor_frozen.named_parameters():
            if "base_model" not in name and param.requires_grad:
                if param.grad is not None:
                    has_gradients = True
                    break
        assert has_gradients

    def test_forward_same_device(self, sequence_regressor, sample_tokens):
        """Test that output is on same device as input."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            sequence_regressor = sequence_regressor.to(device)
            sample_tokens = sample_tokens.to(device)
            result = sequence_regressor(sample_tokens)
            assert result["target_prediction"].device == device
            assert result["count_prediction"].device == device
            assert result["base_embeddings"].device == device

    def test_base_embeddings_used(self, sequence_regressor, sample_tokens):
        """Test that base embeddings are used (not base predictions)."""
        result = sequence_regressor(sample_tokens)
        assert "base_embeddings" in result
        assert result["base_embeddings"].shape == (2, 10, 64)
        assert "base_prediction" not in result or result.get("base_prediction") is None

    def test_base_predictions_not_used_as_input(self, sequence_regressor_with_nucleotides, sample_tokens):
        """Test that base predictions are not used as input to heads."""
        result = sequence_regressor_with_nucleotides(sample_tokens, return_nucleotides=True)
        assert "base_prediction" in result
        assert "base_embeddings" in result
        base_emb_shape = result["base_embeddings"].shape
        base_pred_shape = result["base_prediction"].shape
        assert base_emb_shape != base_pred_shape
        assert base_emb_shape == (2, 10, 64)
        assert base_pred_shape == (2, 32)

    def test_different_encoder_types(self, sample_tokens):
        """Test forward pass with different encoder types."""
        for encoder_type in ["unifrac", "faith_pd", "taxonomy"]:
            regressor = SequenceRegressor(
                encoder_type=encoder_type,
                embedding_dim=64,
                max_bp=150,
                token_limit=1024,
                base_output_dim=32,
                out_dim=1,
            )
            result = regressor(sample_tokens)
            assert isinstance(result, dict)
            assert "target_prediction" in result
            assert "count_prediction" in result
            assert "base_embeddings" in result

    def test_combined_encoder_type(self, sample_tokens):
        """Test forward pass with combined encoder type."""
        regressor = SequenceRegressor(
            encoder_type="combined",
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
        )
        result = regressor(sample_tokens, return_nucleotides=True)
        assert isinstance(result, dict)
        assert "target_prediction" in result
        assert "count_prediction" in result
        assert "base_embeddings" in result
        assert "unifrac_pred" in result or "base_prediction" in result

    def test_different_out_dims(self, base_encoder, sample_tokens):
        """Test forward pass with different out_dim values."""
        for out_dim in [1, 5, 10]:
            regressor = SequenceRegressor(
                base_model=base_encoder,
                out_dim=out_dim,
            )
            result = regressor(sample_tokens)
            assert result["target_prediction"].shape == (2, out_dim)

    def test_composition_pattern(self, base_encoder):
        """Test that SequenceRegressor composes SequenceEncoder."""
        regressor = SequenceRegressor(base_model=base_encoder)
        assert regressor.base_model is base_encoder
        assert isinstance(regressor.base_model, SequenceEncoder)

    def test_base_model_swapping(self, base_encoder, sample_tokens):
        """Test that base model can be swapped."""
        regressor1 = SequenceRegressor(base_model=base_encoder, out_dim=1)
        result1 = regressor1(sample_tokens)

        new_base = SequenceEncoder(
            vocab_size=5,
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            base_output_dim=32,
            encoder_type="faith_pd",
        )
        regressor2 = SequenceRegressor(base_model=new_base, out_dim=1)
        result2 = regressor2(sample_tokens)

        assert result1["target_prediction"].shape == result2["target_prediction"].shape
        assert result1["count_prediction"].shape == result2["count_prediction"].shape
