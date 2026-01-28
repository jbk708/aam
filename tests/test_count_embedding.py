"""Unit tests for count embedding functionality."""

import pytest
import torch
import torch.nn as nn

from aam.models.sample_sequence_encoder import SampleSequenceEncoder
from aam.models.sequence_encoder import SequenceEncoder
from aam.models.sequence_predictor import SequencePredictor


@pytest.fixture
def sample_tokens():
    """Create sample tokens for testing [B, S, L]."""
    from aam.data.tokenizer import SequenceTokenizer

    tokens = torch.randint(1, 5, (2, 10, 50))
    tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
    tokens[:, :, 40:] = 0
    return tokens


@pytest.fixture
def sample_counts():
    """Create sample counts for testing [B, S]."""
    counts = torch.randint(1, 100, (2, 10)).float()
    # Set some counts to zero to match padding
    counts[:, 5:] = 0
    return counts


class TestSampleSequenceEncoderCountEmbedding:
    """Test count embedding in SampleSequenceEncoder."""

    @pytest.fixture
    def encoder_add(self):
        """Create SampleSequenceEncoder with add count embedding."""
        return SampleSequenceEncoder(
            vocab_size=6,
            embedding_dim=64,
            max_bp=150,
            token_limit=64,
            asv_num_layers=1,
            asv_num_heads=2,
            sample_num_layers=1,
            sample_num_heads=2,
            count_embedding=True,
            count_embedding_method="add",
        )

    @pytest.fixture
    def encoder_concat(self):
        """Create SampleSequenceEncoder with concat count embedding."""
        return SampleSequenceEncoder(
            vocab_size=6,
            embedding_dim=64,
            max_bp=150,
            token_limit=64,
            asv_num_layers=1,
            asv_num_heads=2,
            sample_num_layers=1,
            sample_num_heads=2,
            count_embedding=True,
            count_embedding_method="concat",
        )

    @pytest.fixture
    def encoder_film(self):
        """Create SampleSequenceEncoder with FiLM count embedding."""
        return SampleSequenceEncoder(
            vocab_size=6,
            embedding_dim=64,
            max_bp=150,
            token_limit=64,
            asv_num_layers=1,
            asv_num_heads=2,
            sample_num_layers=1,
            sample_num_heads=2,
            count_embedding=True,
            count_embedding_method="film",
        )

    @pytest.fixture
    def encoder_no_count(self):
        """Create SampleSequenceEncoder without count embedding."""
        return SampleSequenceEncoder(
            vocab_size=6,
            embedding_dim=64,
            max_bp=150,
            token_limit=64,
            asv_num_layers=1,
            asv_num_heads=2,
            sample_num_layers=1,
            sample_num_heads=2,
            count_embedding=False,
        )

    def test_init_count_embedding_add(self, encoder_add):
        """Test initialization with add count embedding method."""
        assert encoder_add.count_embedding is True
        assert encoder_add.count_embedding_method == "add"
        assert hasattr(encoder_add, "count_embed")
        assert isinstance(encoder_add.count_embed, nn.Linear)
        assert encoder_add.count_embed.in_features == 1
        assert encoder_add.count_embed.out_features == 64

    def test_init_count_embedding_concat(self, encoder_concat):
        """Test initialization with concat count embedding method."""
        assert encoder_concat.count_embedding is True
        assert encoder_concat.count_embedding_method == "concat"
        assert hasattr(encoder_concat, "count_embed")
        assert hasattr(encoder_concat, "count_proj")
        assert encoder_concat.count_proj.in_features == 128  # 2 * embedding_dim
        assert encoder_concat.count_proj.out_features == 64

    def test_init_count_embedding_film(self, encoder_film):
        """Test initialization with FiLM count embedding method."""
        assert encoder_film.count_embedding is True
        assert encoder_film.count_embedding_method == "film"
        assert hasattr(encoder_film, "count_film")
        assert encoder_film.count_film.in_features == 1
        assert encoder_film.count_film.out_features == 128  # 2 * embedding_dim

    def test_init_no_count_embedding(self, encoder_no_count):
        """Test initialization without count embedding."""
        assert encoder_no_count.count_embedding is False
        assert not hasattr(encoder_no_count, "count_embed")
        assert not hasattr(encoder_no_count, "count_proj")
        assert not hasattr(encoder_no_count, "count_film")

    def test_init_invalid_method(self):
        """Test that invalid count embedding method raises error."""
        with pytest.raises(ValueError, match="Invalid count_embedding_method"):
            SampleSequenceEncoder(
                embedding_dim=64,
                count_embedding=True,
                count_embedding_method="invalid",
            )

    def test_forward_add(self, encoder_add, sample_tokens, sample_counts):
        """Test forward pass with add count embedding."""
        result = encoder_add(sample_tokens, counts=sample_counts)
        assert result.shape == (2, 10, 64)
        assert not torch.isnan(result).any()

    def test_forward_concat(self, encoder_concat, sample_tokens, sample_counts):
        """Test forward pass with concat count embedding."""
        result = encoder_concat(sample_tokens, counts=sample_counts)
        assert result.shape == (2, 10, 64)
        assert not torch.isnan(result).any()

    def test_forward_film(self, encoder_film, sample_tokens, sample_counts):
        """Test forward pass with FiLM count embedding."""
        result = encoder_film(sample_tokens, counts=sample_counts)
        assert result.shape == (2, 10, 64)
        assert not torch.isnan(result).any()

    def test_forward_no_count_embedding(self, encoder_no_count, sample_tokens):
        """Test forward pass without count embedding (counts not required)."""
        result = encoder_no_count(sample_tokens)
        assert result.shape == (2, 10, 64)
        assert not torch.isnan(result).any()

    def test_forward_count_required_when_enabled(self, encoder_add, sample_tokens):
        """Test that counts are required when count_embedding=True."""
        with pytest.raises(ValueError, match="counts must be provided"):
            encoder_add(sample_tokens, counts=None)

    def test_forward_counts_optional_when_disabled(self, encoder_no_count, sample_tokens, sample_counts):
        """Test that counts are optional when count_embedding=False."""
        # Should work with counts=None
        result1 = encoder_no_count(sample_tokens, counts=None)
        assert result1.shape == (2, 10, 64)

        # Should also work with counts provided (just ignored)
        result2 = encoder_no_count(sample_tokens, counts=sample_counts)
        assert result2.shape == (2, 10, 64)

    def test_counts_2d_shape(self, encoder_add, sample_tokens, sample_counts):
        """Test that 2D counts [B, S] work correctly."""
        assert sample_counts.dim() == 2
        result = encoder_add(sample_tokens, counts=sample_counts)
        assert result.shape == (2, 10, 64)

    def test_counts_3d_shape(self, encoder_add, sample_tokens, sample_counts):
        """Test that 3D counts [B, S, 1] work correctly."""
        counts_3d = sample_counts.unsqueeze(-1)
        assert counts_3d.dim() == 3
        result = encoder_add(sample_tokens, counts=counts_3d)
        assert result.shape == (2, 10, 64)

    def test_log_transform_applied(self, encoder_add, sample_tokens):
        """Test that counts are log-transformed (log(count + 1))."""
        # Test with zero counts - should not produce -inf
        zero_counts = torch.zeros(2, 10)
        result = encoder_add(sample_tokens, counts=zero_counts)
        assert not torch.isinf(result).any()
        assert not torch.isnan(result).any()

    def test_count_embedding_affects_output(self, encoder_add, sample_tokens):
        """Test that different counts produce different outputs."""
        counts1 = torch.ones(2, 10) * 10
        counts2 = torch.ones(2, 10) * 100

        result1 = encoder_add(sample_tokens, counts=counts1)
        result2 = encoder_add(sample_tokens, counts=counts2)

        # Results should be different due to different counts
        assert not torch.allclose(result1, result2)


class TestSequenceEncoderCountEmbedding:
    """Test count embedding passthrough in SequenceEncoder."""

    def test_init_with_count_embedding(self):
        """Test SequenceEncoder initialization with count embedding."""
        encoder = SequenceEncoder(
            vocab_size=6,
            embedding_dim=64,
            max_bp=50,
            token_limit=64,
            asv_num_layers=1,
            asv_num_heads=2,
            sample_num_layers=1,
            sample_num_heads=2,
            encoder_num_layers=1,
            encoder_num_heads=2,
            count_embedding=True,
            count_embedding_method="add",
        )
        assert encoder.count_embedding is True
        assert encoder.count_embedding_method == "add"
        assert encoder.sample_encoder.count_embedding is True

    def test_forward_with_counts(self, sample_tokens, sample_counts):
        """Test SequenceEncoder forward with counts."""
        encoder = SequenceEncoder(
            vocab_size=6,
            embedding_dim=64,
            max_bp=50,
            token_limit=64,
            asv_num_layers=1,
            asv_num_heads=2,
            sample_num_layers=1,
            sample_num_heads=2,
            encoder_num_layers=1,
            encoder_num_heads=2,
            count_embedding=True,
            count_embedding_method="add",
        )
        result = encoder(sample_tokens, counts=sample_counts)
        assert "embeddings" in result
        assert result["embeddings"].shape == (2, 64)


class TestSequencePredictorCountEmbedding:
    """Test count embedding passthrough in SequencePredictor."""

    def test_init_with_count_embedding(self):
        """Test SequencePredictor initialization with count embedding."""
        predictor = SequencePredictor(
            vocab_size=6,
            embedding_dim=64,
            max_bp=50,
            token_limit=64,
            asv_num_layers=1,
            asv_num_heads=2,
            sample_num_layers=1,
            sample_num_heads=2,
            encoder_num_layers=1,
            encoder_num_heads=2,
            target_num_layers=1,
            target_num_heads=2,
            out_dim=1,
            count_embedding=True,
            count_embedding_method="concat",
        )
        assert predictor.count_embedding is True
        assert predictor.count_embedding_method == "concat"
        assert predictor.base_model.count_embedding is True

    def test_forward_with_counts(self, sample_tokens, sample_counts):
        """Test SequencePredictor forward with counts."""
        predictor = SequencePredictor(
            vocab_size=6,
            embedding_dim=64,
            max_bp=50,
            token_limit=64,
            asv_num_layers=1,
            asv_num_heads=2,
            sample_num_layers=1,
            sample_num_heads=2,
            encoder_num_layers=1,
            encoder_num_heads=2,
            target_num_layers=1,
            target_num_heads=2,
            out_dim=1,
            count_embedding=True,
            count_embedding_method="add",
        )
        result = predictor(sample_tokens, counts=sample_counts)
        assert "target_prediction" in result
        assert result["target_prediction"].shape == (2, 1)

    def test_backward_compatible_no_counts(self, sample_tokens):
        """Test SequencePredictor backward compatibility without count embedding."""
        predictor = SequencePredictor(
            vocab_size=6,
            embedding_dim=64,
            max_bp=50,
            token_limit=64,
            asv_num_layers=1,
            asv_num_heads=2,
            sample_num_layers=1,
            sample_num_heads=2,
            encoder_num_layers=1,
            encoder_num_heads=2,
            target_num_layers=1,
            target_num_heads=2,
            out_dim=1,
            count_embedding=False,
        )
        # Should work without counts
        result = predictor(sample_tokens)
        assert "target_prediction" in result
        assert result["target_prediction"].shape == (2, 1)
