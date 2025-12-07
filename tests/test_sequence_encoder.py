"""Unit tests for SequenceEncoder class."""

import pytest
import torch
import torch.nn as nn

from aam.models.sequence_encoder import SequenceEncoder


@pytest.fixture
def sequence_encoder():
    """Create a SequenceEncoder instance without nucleotide prediction (non-UniFrac for legacy tests)."""
    return SequenceEncoder(
        vocab_size=6,
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
        encoder_type="faith_pd",  # Use non-UniFrac for legacy tests
        predict_nucleotides=False,
    )


@pytest.fixture
def sequence_encoder_with_nucleotides():
    """Create a SequenceEncoder instance with nucleotide prediction (non-UniFrac for legacy tests)."""
    return SequenceEncoder(
        vocab_size=6,
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
        encoder_type="faith_pd",  # Use non-UniFrac for legacy tests
        predict_nucleotides=True,
    )


@pytest.fixture
def sequence_encoder_combined():
    """Create a SequenceEncoder instance with combined encoder type."""
    return SequenceEncoder(
        vocab_size=6,
        embedding_dim=64,
        max_bp=150,
        token_limit=1024,
        asv_num_layers=2,
        asv_num_heads=4,
        sample_num_layers=2,
        sample_num_heads=4,
        encoder_num_layers=2,
        encoder_num_heads=4,
        base_output_dim=None,
        encoder_type="combined",
        predict_nucleotides=False,
    )


@pytest.fixture
def sample_tokens():
    """Create sample tokens for testing [B, S, L]."""
    from aam.data.tokenizer import SequenceTokenizer

    batch_size = 2
    num_asvs = 10
    seq_len = 50
    tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len))
    tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
    tokens[:, :, 40:] = 0
    return tokens


class TestSequenceEncoder:
    """Test suite for SequenceEncoder class."""

    def test_init(self, sequence_encoder):
        """Test SequenceEncoder initialization."""
        assert sequence_encoder is not None
        assert isinstance(sequence_encoder, nn.Module)

    def test_init_with_nucleotides(self, sequence_encoder_with_nucleotides):
        """Test SequenceEncoder initialization with nucleotide prediction."""
        assert sequence_encoder_with_nucleotides is not None
        assert isinstance(sequence_encoder_with_nucleotides, nn.Module)

    def test_init_combined_type(self, sequence_encoder_combined):
        """Test SequenceEncoder initialization with combined encoder type."""
        assert sequence_encoder_combined is not None
        assert isinstance(sequence_encoder_combined, nn.Module)

    def test_init_default_vocab_size(self):
        """Test that vocab_size defaults to 6."""
        encoder = SequenceEncoder(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            base_output_dim=32,
        )
        assert encoder is not None

    def test_init_default_max_bp(self):
        """Test that max_bp defaults to 150."""
        encoder = SequenceEncoder(
            embedding_dim=64,
            token_limit=1024,
            base_output_dim=32,
        )
        assert encoder is not None

    def test_forward_shape_basic(self, sequence_encoder, sample_tokens):
        """Test forward pass output shape without nucleotide predictions."""
        result = sequence_encoder(sample_tokens, return_nucleotides=False)
        assert isinstance(result, dict)
        assert "base_prediction" in result
        assert "sample_embeddings" in result
        assert result["base_prediction"].shape == (2, 32)
        assert result["sample_embeddings"].shape == (2, 10, 64)

    def test_forward_shape_with_nucleotides(self, sequence_encoder_with_nucleotides, sample_tokens):
        """Test forward pass output shape with nucleotide predictions."""
        result = sequence_encoder_with_nucleotides(sample_tokens, return_nucleotides=True)
        assert isinstance(result, dict)
        assert "base_prediction" in result
        assert "sample_embeddings" in result
        assert "nuc_predictions" in result
        assert result["base_prediction"].shape == (2, 32)
        assert result["sample_embeddings"].shape == (2, 10, 64)
        assert result["nuc_predictions"].shape == (2, 10, 50, 6)

    def test_forward_unifrac_returns_embeddings(self, sample_tokens):
        """Test that UniFrac encoder returns embeddings instead of base_prediction."""
        encoder = SequenceEncoder(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            base_output_dim=None,
            encoder_type="unifrac",
        )
        result = encoder(sample_tokens)
        # For UniFrac, should return embeddings, not base_prediction
        assert "embeddings" in result
        assert "base_prediction" not in result
        assert result["embeddings"].shape == (2, 64)  # [batch_size, embedding_dim]
        assert result["sample_embeddings"].shape == (2, 10, 64)

    def test_forward_unifrac_no_output_head(self, sample_tokens):
        """Test that UniFrac encoder has no output_head."""
        encoder = SequenceEncoder(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            base_output_dim=None,
            encoder_type="unifrac",
        )
        assert encoder.output_head is None

    def test_forward_non_unifrac_returns_base_prediction(self, sample_tokens):
        """Test that non-UniFrac encoders still return base_prediction."""
        for encoder_type in ["faith_pd", "taxonomy"]:
            encoder = SequenceEncoder(
                embedding_dim=64,
                max_bp=150,
                token_limit=1024,
                base_output_dim=32,
                encoder_type=encoder_type,
            )
            result = encoder(sample_tokens)
            assert "base_prediction" in result
            assert "embeddings" not in result
            assert result["base_prediction"].shape == (2, 32)

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_forward_different_batch_sizes(self, sequence_encoder, batch_size):
        """Test forward pass with different batch sizes."""
        tokens = torch.randint(1, 5, (batch_size, 10, 50))
        result = sequence_encoder(tokens)
        assert result["base_prediction"].shape == (batch_size, 32)
        assert result["sample_embeddings"].shape == (batch_size, 10, 64)

    @pytest.mark.parametrize("num_asvs", [5, 10, 20, 50])
    def test_forward_different_num_asvs(self, sequence_encoder, num_asvs):
        """Test forward pass with different numbers of ASVs."""
        tokens = torch.randint(1, 5, (2, num_asvs, 50))
        result = sequence_encoder(tokens)
        assert result["base_prediction"].shape == (2, 32)
        assert result["sample_embeddings"].shape == (2, num_asvs, 64)

    @pytest.mark.parametrize("seq_len", [10, 50, 100, 150])
    def test_forward_different_seq_lengths(self, sequence_encoder, seq_len):
        """Test forward pass with different sequence lengths."""
        tokens = torch.randint(1, 5, (2, 10, seq_len))
        result = sequence_encoder(tokens)
        assert result["base_prediction"].shape == (2, 32)
        assert result["sample_embeddings"].shape == (2, 10, 64)

    def test_forward_with_padding(self, sequence_encoder, sample_tokens):
        """Test forward pass with padded sequences."""
        result = sequence_encoder(sample_tokens)
        assert result["base_prediction"].shape == (2, 32)
        assert result["sample_embeddings"].shape == (2, 10, 64)
        assert not torch.isnan(result["base_prediction"]).any()
        assert not torch.isnan(result["sample_embeddings"]).any()

    def test_forward_inference_mode(self, sequence_encoder, sample_tokens):
        """Test forward pass in inference mode (no nucleotide predictions)."""
        sequence_encoder.eval()
        result = sequence_encoder(sample_tokens, return_nucleotides=False)
        assert isinstance(result, dict)
        assert "nuc_predictions" not in result
        assert result["base_prediction"].shape == (2, 32)
        assert result["sample_embeddings"].shape == (2, 10, 64)

    def test_forward_training_mode_with_nucleotides(self, sequence_encoder_with_nucleotides, sample_tokens):
        """Test forward pass in training mode with nucleotide predictions."""
        sequence_encoder_with_nucleotides.train()
        result = sequence_encoder_with_nucleotides(sample_tokens, return_nucleotides=True)
        assert "nuc_predictions" in result
        assert result["nuc_predictions"].shape == (2, 10, 50, 6)

    def test_forward_training_mode_without_nucleotides(self, sequence_encoder_with_nucleotides, sample_tokens):
        """Test forward pass in training mode without requesting nucleotide predictions."""
        sequence_encoder_with_nucleotides.train()
        result = sequence_encoder_with_nucleotides(sample_tokens, return_nucleotides=False)
        assert isinstance(result, dict)
        assert "nuc_predictions" not in result

    def test_forward_no_nucleotide_head(self, sequence_encoder, sample_tokens):
        """Test that encoder without nucleotide head doesn't return predictions."""
        result = sequence_encoder(sample_tokens, return_nucleotides=True)
        assert isinstance(result, dict)
        assert "nuc_predictions" not in result

    def test_gradients_flow(self, sequence_encoder, sample_tokens):
        """Test that gradients flow correctly."""
        result = sequence_encoder(sample_tokens)
        loss = result["base_prediction"].sum() + result["sample_embeddings"].sum()
        loss.backward()

        for param in sequence_encoder.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

    def test_gradients_with_nucleotides(self, sequence_encoder_with_nucleotides, sample_tokens):
        """Test that gradients flow correctly with nucleotide predictions."""
        result = sequence_encoder_with_nucleotides(sample_tokens, return_nucleotides=True)
        loss = result["base_prediction"].sum() + result["sample_embeddings"].sum() + result["nuc_predictions"].sum()
        loss.backward()

        for param in sequence_encoder_with_nucleotides.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

    def test_forward_same_device(self, sequence_encoder, sample_tokens):
        """Test that output is on same device as input."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            sequence_encoder = sequence_encoder.to(device)
            sample_tokens = sample_tokens.to(device)
            result = sequence_encoder(sample_tokens)
            assert result["base_prediction"].device.type == device.type
            assert result["sample_embeddings"].device.type == device.type

    def test_forward_different_encoder_types(self, sample_tokens):
        """Test forward pass with different encoder types."""
        for encoder_type in ["faith_pd", "taxonomy"]:
            encoder = SequenceEncoder(
                embedding_dim=64,
                max_bp=150,
                token_limit=1024,
                base_output_dim=32,
                encoder_type=encoder_type,
            )
            result = encoder(sample_tokens)
            assert isinstance(result, dict)
            assert "base_prediction" in result
            assert result["base_prediction"].shape == (2, 32)
        
        # Test UniFrac separately (new architecture)
        encoder = SequenceEncoder(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            base_output_dim=None,
            encoder_type="unifrac",
        )
        result = encoder(sample_tokens)
        assert isinstance(result, dict)
        assert "embeddings" in result
        assert "base_prediction" not in result
        assert result["embeddings"].shape == (2, 64)

    def test_forward_combined_encoder_type(self, sequence_encoder_combined, sample_tokens):
        """Test forward pass with combined encoder type."""
        result = sequence_encoder_combined(sample_tokens)
        assert isinstance(result, dict)
        assert "unifrac_pred" in result
        assert "faith_pred" in result
        assert "tax_pred" in result
        assert "sample_embeddings" in result
        assert result["unifrac_pred"].shape == (2, 2)
        assert result["faith_pred"].shape == (2, 1)
        assert result["tax_pred"].shape == (2, 7)
        assert result["sample_embeddings"].shape == (2, 10, 64)

    def test_forward_different_base_output_dims(self, sample_tokens):
        """Test forward pass with different base_output_dim values (non-UniFrac)."""
        for base_output_dim in [16, 32, 64, 128]:
            encoder = SequenceEncoder(
                embedding_dim=64,
                max_bp=150,
                token_limit=1024,
                base_output_dim=base_output_dim,
                encoder_type="faith_pd",  # Use non-UniFrac
            )
            result = encoder(sample_tokens)
            assert result["base_prediction"].shape == (2, base_output_dim)

    def test_sample_embeddings_always_returned(self, sequence_encoder, sample_tokens):
        """Test that sample embeddings are always returned."""
        result = sequence_encoder(sample_tokens)
        assert "sample_embeddings" in result
        assert result["sample_embeddings"].shape == (2, 10, 64)

    def test_base_prediction_returned_for_non_unifrac(self, sample_tokens):
        """Test that base prediction is returned for non-UniFrac encoders."""
        for encoder_type in ["faith_pd", "taxonomy"]:
            encoder = SequenceEncoder(
                embedding_dim=64,
                max_bp=150,
                token_limit=1024,
                base_output_dim=32,
                encoder_type=encoder_type,
            )
            result = encoder(sample_tokens)
            assert "base_prediction" in result
        result = sequence_encoder(sample_tokens)
        assert "base_prediction" in result
        assert result["base_prediction"].shape == (2, 32)

    def test_nucleotide_predictions_side_output(self, sequence_encoder_with_nucleotides, sample_tokens):
        """Test that nucleotide predictions are side output, not used as input."""
        result = sequence_encoder_with_nucleotides(sample_tokens, return_nucleotides=True)
        assert "nuc_predictions" in result
        assert result["nuc_predictions"].shape == (2, 10, 50, 6)
        assert "base_prediction" in result
        assert "sample_embeddings" in result

    def test_forward_unifrac_embeddings_not_clipped(self, sample_tokens):
        """Test that UniFrac embeddings are not clipped (distances computed from embeddings)."""
        encoder = SequenceEncoder(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            base_output_dim=None,
            encoder_type="unifrac",
        )
        output = encoder(sample_tokens)
        embeddings = output["embeddings"]

        # Embeddings should not be constrained to [0, 1] (they can be any value)
        # They are used to compute distances, which are naturally >= 0
        assert embeddings.shape == (2, 64)
        
        # Test that distances computed from embeddings are non-negative
        from aam.training.losses import compute_pairwise_distances
        distances = compute_pairwise_distances(embeddings)
        assert torch.all(distances >= 0.0), "Distances should be >= 0.0"
        assert torch.allclose(torch.diag(distances), torch.zeros(2)), "Diagonal should be 0.0"

    def test_forward_combined_unifrac_sigmoid_activation(self, sequence_encoder_combined, sample_tokens):
        """Test that combined encoder type uses sigmoid for UniFrac predictions."""
        output = sequence_encoder_combined(sample_tokens)
        unifrac_pred = output["unifrac_pred"]

        # Verify UniFrac predictions are in [0, 1] range
        assert torch.all(unifrac_pred >= 0.0), "UniFrac predictions should be >= 0.0"
        assert torch.all(unifrac_pred <= 1.0), "UniFrac predictions should be <= 1.0"

        # Verify sigmoid behavior (continuous values, not hard-clipped)
        unique_values = torch.unique(unifrac_pred)
        assert len(unique_values) > 2, "Sigmoid should produce continuous values"
