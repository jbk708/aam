"""Unit tests for SequencePredictor class."""

import pytest
import torch
import torch.nn as nn

from aam.models.sequence_predictor import SequencePredictor
from aam.models.sequence_encoder import SequenceEncoder
from aam.data.tokenizer import SequenceTokenizer


def _create_sample_tokens(batch_size: int = 2, num_asvs: int = 10, seq_len: int = 50) -> torch.Tensor:
    """Create sample tokens for testing [B, S, L]."""
    tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len))
    tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
    tokens[:, :, 40:] = 0
    return tokens


@pytest.fixture
def base_encoder():
    """Create a SequenceEncoder instance for use as base model."""
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
        encoder_type="unifrac",
        predict_nucleotides=False,
    )


@pytest.fixture
def base_encoder_with_nucleotides():
    """Create a SequenceEncoder instance with nucleotide prediction."""
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
        encoder_type="unifrac",
        predict_nucleotides=True,
    )


@pytest.fixture
def sequence_predictor(base_encoder):
    """Create a SequencePredictor instance with provided base model."""
    return SequencePredictor(
        base_model=base_encoder,
        out_dim=1,
        is_classifier=False,
        freeze_base=False,
    )


@pytest.fixture
def sequence_predictor_frozen(base_encoder):
    """Create a SequencePredictor instance with frozen base model."""
    return SequencePredictor(
        base_model=base_encoder,
        out_dim=1,
        is_classifier=False,
        freeze_base=True,
    )


@pytest.fixture
def sequence_predictor_classifier(base_encoder):
    """Create a SequencePredictor instance for classification."""
    return SequencePredictor(
        base_model=base_encoder,
        out_dim=7,
        is_classifier=True,
        freeze_base=False,
    )


@pytest.fixture
def sequence_predictor_with_nucleotides(base_encoder_with_nucleotides):
    """Create a SequencePredictor instance with nucleotide prediction."""
    return SequencePredictor(
        base_model=base_encoder_with_nucleotides,
        out_dim=1,
        is_classifier=False,
        freeze_base=False,
    )


@pytest.fixture
def sequence_predictor_no_base():
    """Create a SequencePredictor instance without providing base model."""
    return SequencePredictor(
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
    return _create_sample_tokens()


class TestSequencePredictor:
    """Test suite for SequencePredictor class."""

    def test_init_with_base_model(self, sequence_predictor):
        """Test SequencePredictor initialization with provided base model."""
        assert sequence_predictor is not None
        assert isinstance(sequence_predictor, nn.Module)

    def test_init_without_base_model(self, sequence_predictor_no_base):
        """Test SequencePredictor initialization without base model."""
        assert sequence_predictor_no_base is not None
        assert isinstance(sequence_predictor_no_base, nn.Module)
        assert sequence_predictor_no_base.base_model is not None

    def test_init_frozen_base(self, sequence_predictor_frozen):
        """Test SequencePredictor initialization with frozen base."""
        assert sequence_predictor_frozen is not None
        for param in sequence_predictor_frozen.base_model.parameters():
            assert not param.requires_grad

    def test_init_unfrozen_base(self, sequence_predictor):
        """Test SequencePredictor initialization with unfrozen base."""
        assert sequence_predictor is not None
        for param in sequence_predictor.base_model.parameters():
            assert param.requires_grad

    def test_init_classifier(self, sequence_predictor_classifier):
        """Test SequencePredictor initialization for classification."""
        assert sequence_predictor_classifier is not None
        assert sequence_predictor_classifier.is_classifier

    def test_init_regressor(self, sequence_predictor):
        """Test SequencePredictor initialization for regression."""
        assert sequence_predictor is not None
        assert not sequence_predictor.is_classifier

    def test_forward_shape_basic(self, sequence_predictor, sample_tokens):
        """Test forward pass output shape."""
        result = sequence_predictor(sample_tokens, return_nucleotides=False)
        assert isinstance(result, dict)
        assert "target_prediction" in result
        assert "count_prediction" in result
        assert "base_embeddings" in result
        assert result["target_prediction"].shape == (2, 1)
        assert result["count_prediction"].shape == (2, 10, 1)
        assert result["base_embeddings"].shape == (2, 10, 64)

    def test_forward_shape_with_nucleotides(self, sequence_predictor_with_nucleotides, sample_tokens):
        """Test forward pass output shape with nucleotide predictions."""
        result = sequence_predictor_with_nucleotides(sample_tokens, return_nucleotides=True)
        assert isinstance(result, dict)
        assert "target_prediction" in result
        assert "count_prediction" in result
        assert "base_embeddings" in result
        # For UniFrac, embeddings are returned instead of base_prediction
        assert "embeddings" in result
        assert "nuc_predictions" in result
        assert result["target_prediction"].shape == (2, 1)
        assert result["nuc_predictions"].shape == (2, 10, 50, 6)
        assert result["count_prediction"].shape == (2, 10, 1)
        assert result["base_embeddings"].shape == (2, 10, 64)
        assert result["embeddings"].shape == (2, 64)  # Sample-level embeddings for UniFrac
        assert result["nuc_predictions"].shape == (2, 10, 50, 6)

    def test_forward_shape_classifier(self, sequence_predictor_classifier, sample_tokens):
        """Test forward pass output shape for classification."""
        result = sequence_predictor_classifier(sample_tokens)
        assert result["target_prediction"].shape == (2, 7)

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_forward_different_batch_sizes(self, sequence_predictor, batch_size):
        """Test forward pass with different batch sizes."""
        tokens = torch.randint(1, 5, (batch_size, 10, 50))
        result = sequence_predictor(tokens)
        assert result["target_prediction"].shape == (batch_size, 1)
        assert result["count_prediction"].shape == (batch_size, 10, 1)
        assert result["base_embeddings"].shape == (batch_size, 10, 64)

    @pytest.mark.parametrize("num_asvs", [5, 10, 20, 50])
    def test_forward_different_num_asvs(self, sequence_predictor, num_asvs):
        """Test forward pass with different numbers of ASVs."""
        tokens = torch.randint(1, 5, (2, num_asvs, 50))
        result = sequence_predictor(tokens)
        assert result["target_prediction"].shape == (2, 1)
        assert result["count_prediction"].shape == (2, num_asvs, 1)
        assert result["base_embeddings"].shape == (2, num_asvs, 64)

    @pytest.mark.parametrize("seq_len", [10, 50, 100, 150])
    def test_forward_different_seq_lengths(self, sequence_predictor, seq_len):
        """Test forward pass with different sequence lengths."""
        tokens = torch.randint(1, 5, (2, 10, seq_len))
        result = sequence_predictor(tokens)
        assert result["target_prediction"].shape == (2, 1)
        assert result["count_prediction"].shape == (2, 10, 1)
        assert result["base_embeddings"].shape == (2, 10, 64)

    def test_forward_with_padding(self, sequence_predictor, sample_tokens):
        """Test forward pass with padded sequences."""
        result = sequence_predictor(sample_tokens)
        assert not torch.isnan(result["target_prediction"]).any()
        assert not torch.isnan(result["count_prediction"]).any()
        assert not torch.isnan(result["base_embeddings"]).any()

    def test_forward_inference_mode(self, sequence_predictor, sample_tokens):
        """Test forward pass in inference mode (no nucleotide predictions)."""
        sequence_predictor.eval()
        result = sequence_predictor(sample_tokens, return_nucleotides=False)
        assert isinstance(result, dict)
        assert "nuc_predictions" not in result
        assert "base_prediction" not in result

    def test_forward_training_mode_with_nucleotides(self, sequence_predictor_with_nucleotides, sample_tokens):
        """Test forward pass in training mode with nucleotide predictions."""
        sequence_predictor_with_nucleotides.train()
        result = sequence_predictor_with_nucleotides(sample_tokens, return_nucleotides=True)
        assert "nuc_predictions" in result
        # For UniFrac, embeddings are returned instead of base_prediction
        assert "embeddings" in result

    def test_forward_training_mode_without_nucleotides(self, sequence_predictor_with_nucleotides, sample_tokens):
        """Test forward pass in training mode without requesting nucleotide predictions."""
        sequence_predictor_with_nucleotides.train()
        result = sequence_predictor_with_nucleotides(sample_tokens, return_nucleotides=False)
        assert isinstance(result, dict)
        assert "nuc_predictions" not in result
        assert "base_prediction" not in result

    def test_gradients_flow_unfrozen(self, sequence_predictor, sample_tokens):
        """Test that gradients flow correctly with unfrozen base."""
        sequence_predictor.train()
        result = sequence_predictor(sample_tokens)
        loss = result["target_prediction"].sum() + result["count_prediction"].sum() + result["base_embeddings"].sum()
        loss.backward()

        has_base_gradients = False
        has_count_gradients = False
        has_target_gradients = False

        for name, param in sequence_predictor.named_parameters():
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

    def test_gradients_frozen_base(self, sequence_predictor_frozen, sample_tokens):
        """Test that gradients don't flow to frozen base model."""
        result = sequence_predictor_frozen(sample_tokens)
        loss = result["target_prediction"].sum() + result["count_prediction"].sum()
        loss.backward()

        for param in sequence_predictor_frozen.base_model.parameters():
            assert param.grad is None

        for name, param in sequence_predictor_frozen.named_parameters():
            if "base_model" not in name:
                assert param.grad is not None

    def test_gradients_flow_to_heads(self, sequence_predictor_frozen, sample_tokens):
        """Test that gradients flow to count and target encoders."""
        result = sequence_predictor_frozen(sample_tokens)
        loss = result["target_prediction"].sum() + result["count_prediction"].sum()
        loss.backward()

        has_gradients = False
        for name, param in sequence_predictor_frozen.named_parameters():
            if "base_model" not in name and param.requires_grad:
                if param.grad is not None:
                    has_gradients = True
                    break
        assert has_gradients

    def test_forward_same_device(self, sequence_predictor, sample_tokens):
        """Test that output is on same device as input."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            sequence_predictor = sequence_predictor.to(device)
            sample_tokens = sample_tokens.to(device)
            result = sequence_predictor(sample_tokens)
            assert result["target_prediction"].device.type == device.type
            assert result["count_prediction"].device.type == device.type
            assert result["base_embeddings"].device.type == device.type

    def test_base_embeddings_used(self, sequence_predictor, sample_tokens):
        """Test that base embeddings are used (not base predictions)."""
        result = sequence_predictor(sample_tokens)
        assert "base_embeddings" in result
        assert result["base_embeddings"].shape == (2, 10, 64)
        assert "base_prediction" not in result or result.get("base_prediction") is None

    def test_base_predictions_not_used_as_input(self, sequence_predictor_with_nucleotides, sample_tokens):
        """Test that base predictions are not used as input to heads."""
        result = sequence_predictor_with_nucleotides(sample_tokens, return_nucleotides=True)
        # For UniFrac, embeddings are returned instead of base_prediction
        assert "embeddings" in result
        assert "base_embeddings" in result
        base_emb_shape = result["base_embeddings"].shape
        embeddings_shape = result["embeddings"].shape
        assert base_emb_shape != embeddings_shape
        assert base_emb_shape == (2, 10, 64)  # Per-ASV embeddings
        assert embeddings_shape == (2, 64)  # Sample-level pooled embeddings for UniFrac

    def test_different_encoder_types(self, sample_tokens):
        """Test forward pass with different encoder types."""
        for encoder_type in ["unifrac", "faith_pd", "taxonomy"]:
            regressor = SequencePredictor(
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
        regressor = SequencePredictor(
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

    @pytest.mark.parametrize("out_dim", [1, 5, 10])
    def test_different_out_dims(self, base_encoder, sample_tokens, out_dim):
        """Test forward pass with different out_dim values."""
        regressor = SequencePredictor(
            base_model=base_encoder,
            out_dim=out_dim,
        )
        result = regressor(sample_tokens)
        assert result["target_prediction"].shape == (2, out_dim)

    def test_composition_pattern(self, base_encoder):
        """Test that SequencePredictor composes SequenceEncoder."""
        regressor = SequencePredictor(base_model=base_encoder)
        assert regressor.base_model is base_encoder
        assert isinstance(regressor.base_model, SequenceEncoder)

    def test_base_model_swapping(self, base_encoder, sample_tokens):
        """Test that base model can be swapped."""
        regressor1 = SequencePredictor(base_model=base_encoder, out_dim=1)
        result1 = regressor1(sample_tokens)

        new_base = SequenceEncoder(
            vocab_size=6,
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            base_output_dim=32,
            encoder_type="faith_pd",
        )
        regressor2 = SequencePredictor(base_model=new_base, out_dim=1)
        result2 = regressor2(sample_tokens)

        assert result1["target_prediction"].shape == result2["target_prediction"].shape
        assert result1["count_prediction"].shape == result2["count_prediction"].shape


class TestRegressorHeadOptions:
    """Test suite for regressor head configuration options."""

    @pytest.fixture
    def sample_tokens(self):
        """Create sample tokens for testing."""
        return _create_sample_tokens()

    def test_default_has_layer_norm(self, sample_tokens):
        """Test that LayerNorm is enabled by default."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
        )
        assert model.target_layer_norm_enabled is True
        assert model.target_norm is not None
        assert isinstance(model.target_norm, nn.LayerNorm)

    def test_disable_layer_norm(self, sample_tokens):
        """Test disabling LayerNorm."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            target_layer_norm=False,
        )
        assert model.target_layer_norm_enabled is False
        assert model.target_norm is None

    def test_default_unbounded_regression(self, sample_tokens):
        """Test that regression is unbounded by default (no sigmoid)."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            is_classifier=False,
        )
        assert model.bounded_targets is False

        result = model(sample_tokens)
        predictions = result["target_prediction"]
        # Unbounded: values can be outside [0, 1]
        # With random weights, outputs won't be constrained
        assert predictions.shape == (2, 1)

    def test_bounded_targets_applies_sigmoid(self, sample_tokens):
        """Test that bounded_targets applies sigmoid."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            is_classifier=False,
            bounded_targets=True,
        )
        assert model.bounded_targets is True

        result = model(sample_tokens)
        predictions = result["target_prediction"]
        # Bounded: values must be in [0, 1]
        assert (predictions >= 0).all()
        assert (predictions <= 1).all()

    def test_learnable_output_scale_disabled_by_default(self):
        """Test that learnable output scale is disabled by default."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
        )
        assert model.learnable_output_scale is False
        assert model.output_scale is None
        assert model.output_bias is None

    def test_learnable_output_scale_enabled(self, sample_tokens):
        """Test learnable output scale when enabled."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=3,
            learnable_output_scale=True,
        )
        assert model.learnable_output_scale is True
        assert model.output_scale is not None
        assert model.output_bias is not None
        assert model.output_scale.shape == (3,)
        assert model.output_bias.shape == (3,)
        # Check initial values
        assert torch.allclose(model.output_scale, torch.ones(3))
        assert torch.allclose(model.output_bias, torch.zeros(3))

        result = model(sample_tokens)
        assert result["target_prediction"].shape == (2, 3)

    def test_learnable_scale_affects_output(self, sample_tokens):
        """Test that learnable scale/bias affect the output."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            learnable_output_scale=True,
        )

        result_before = model(sample_tokens)

        # Modify scale and bias
        with torch.no_grad():
            model.output_scale.fill_(2.0)
            model.output_bias.fill_(1.0)

        result_after = model(sample_tokens)

        # Output should be different
        assert not torch.allclose(
            result_before["target_prediction"],
            result_after["target_prediction"],
        )

    def test_weight_initialization(self):
        """Test that weights are initialized correctly."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
        )
        # Check that biases are zero-initialized
        assert torch.allclose(model.target_head.bias, torch.zeros_like(model.target_head.bias))
        assert torch.allclose(model.count_head.bias, torch.zeros_like(model.count_head.bias))
        # Check that weights are not all zeros (Xavier should give non-zero values)
        assert model.target_head.weight.abs().sum() > 0
        assert model.count_head.weight.abs().sum() > 0

    def test_classifier_ignores_bounded_targets(self, sample_tokens):
        """Test that classifier mode ignores bounded_targets flag."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=5,
            is_classifier=True,
            bounded_targets=True,  # Should be ignored for classifiers
        )

        result = model(sample_tokens)
        predictions = result["target_prediction"]
        # Classifier uses log_softmax, so values should be <= 0
        assert (predictions <= 0).all()
        # Sum of exp should be 1 (log_softmax property)
        assert torch.allclose(predictions.exp().sum(dim=-1), torch.ones(2), atol=1e-5)

    def test_all_options_combined(self, sample_tokens):
        """Test using all regressor options together."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=2,
            target_layer_norm=True,
            bounded_targets=True,
            learnable_output_scale=True,
        )

        assert model.target_norm is not None
        assert model.bounded_targets is True
        assert model.output_scale is not None

        result = model(sample_tokens)
        predictions = result["target_prediction"]
        assert predictions.shape == (2, 2)
        assert (predictions >= 0).all()
        assert (predictions <= 1).all()

    def test_gradients_flow_through_new_layers(self, sample_tokens):
        """Test that gradients flow through new layers."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            target_layer_norm=True,
            learnable_output_scale=True,
        )

        result = model(sample_tokens)
        loss = result["target_prediction"].sum()
        loss.backward()

        # Check LayerNorm gradients
        assert model.target_norm.weight.grad is not None
        assert model.target_norm.bias.grad is not None

        # Check output scale/bias gradients
        assert model.output_scale.grad is not None
        assert model.output_bias.grad is not None


class TestCategoricalIntegration:
    """Test suite for categorical conditioning in SequencePredictor."""

    @pytest.fixture
    def sample_tokens(self):
        """Create sample tokens for testing."""
        return _create_sample_tokens()

    @pytest.fixture
    def categorical_cardinalities(self):
        """Create categorical cardinalities for testing."""
        return {"location": 5, "season": 4}

    @pytest.fixture
    def categorical_ids(self):
        """Create categorical ids for testing."""
        return {
            "location": torch.tensor([1, 2]),
            "season": torch.tensor([3, 1]),
        }

    def test_init_with_categoricals_concat(self, categorical_cardinalities):
        """Test initialization with categorical conditioning (concat fusion)."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            categorical_cardinalities=categorical_cardinalities,
            categorical_embed_dim=16,
            categorical_fusion="concat",
        )
        assert model.categorical_embedder is not None
        assert model.categorical_projection is not None
        assert model.categorical_fusion == "concat"
        # Projection: D + cat_dim -> D
        total_cat_dim = 16 * 2  # 2 columns * 16 embed_dim
        assert model.categorical_projection.in_features == 64 + total_cat_dim
        assert model.categorical_projection.out_features == 64

    def test_init_with_categoricals_add(self, categorical_cardinalities):
        """Test initialization with categorical conditioning (add fusion)."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            categorical_cardinalities=categorical_cardinalities,
            categorical_embed_dim=16,
            categorical_fusion="add",
        )
        assert model.categorical_embedder is not None
        assert model.categorical_projection is not None
        assert model.categorical_fusion == "add"
        # Projection: cat_dim -> D
        total_cat_dim = 16 * 2
        assert model.categorical_projection.in_features == total_cat_dim
        assert model.categorical_projection.out_features == 64

    def test_init_without_categoricals(self):
        """Test initialization without categorical conditioning (backward compat)."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
        )
        assert model.categorical_embedder is None
        assert model.categorical_projection is None

    def test_invalid_fusion_strategy(self, categorical_cardinalities):
        """Test that invalid fusion strategy raises ValueError."""
        with pytest.raises(ValueError, match="categorical_fusion must be"):
            SequencePredictor(
                embedding_dim=64,
                max_bp=150,
                token_limit=1024,
                out_dim=1,
                categorical_cardinalities=categorical_cardinalities,
                categorical_fusion="invalid",
            )

    def test_forward_with_categoricals_concat(self, sample_tokens, categorical_cardinalities, categorical_ids):
        """Test forward pass with categorical conditioning (concat fusion)."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            categorical_cardinalities=categorical_cardinalities,
            categorical_fusion="concat",
        )
        result = model(sample_tokens, categorical_ids=categorical_ids)
        assert "target_prediction" in result
        assert "count_prediction" in result
        assert result["target_prediction"].shape == (2, 1)
        assert result["count_prediction"].shape == (2, 10, 1)

    def test_forward_with_categoricals_add(self, sample_tokens, categorical_cardinalities, categorical_ids):
        """Test forward pass with categorical conditioning (add fusion)."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            categorical_cardinalities=categorical_cardinalities,
            categorical_fusion="add",
        )
        result = model(sample_tokens, categorical_ids=categorical_ids)
        assert "target_prediction" in result
        assert "count_prediction" in result
        assert result["target_prediction"].shape == (2, 1)
        assert result["count_prediction"].shape == (2, 10, 1)

    def test_forward_without_categorical_ids(self, sample_tokens, categorical_cardinalities):
        """Test forward pass without providing categorical_ids."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            categorical_cardinalities=categorical_cardinalities,
        )
        # Should work without categorical_ids (graceful fallback)
        result = model(sample_tokens, categorical_ids=None)
        assert "target_prediction" in result
        assert result["target_prediction"].shape == (2, 1)

    def test_backward_compat_no_categoricals(self, sample_tokens):
        """Test backward compatibility: model without categoricals works identically."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
        )
        result = model(sample_tokens)
        assert "target_prediction" in result
        assert result["target_prediction"].shape == (2, 1)

    def test_categorical_does_not_affect_count_encoder(self, sample_tokens, categorical_cardinalities, categorical_ids):
        """Test that categorical embeddings don't affect count encoder pathway."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            categorical_cardinalities=categorical_cardinalities,
        )
        torch.manual_seed(42)
        result_with_cat = model(sample_tokens, categorical_ids=categorical_ids)
        torch.manual_seed(42)
        result_without_cat = model(sample_tokens, categorical_ids=None)

        # Count predictions should be identical
        assert torch.allclose(
            result_with_cat["count_prediction"],
            result_without_cat["count_prediction"],
        )
        # Target predictions should be different
        assert not torch.allclose(
            result_with_cat["target_prediction"],
            result_without_cat["target_prediction"],
        )

    def test_categorical_affects_target_prediction(self, sample_tokens, categorical_cardinalities):
        """Test that different categorical values produce different predictions."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            categorical_cardinalities=categorical_cardinalities,
        )
        cat_ids_1 = {"location": torch.tensor([1, 1]), "season": torch.tensor([1, 1])}
        cat_ids_2 = {"location": torch.tensor([2, 2]), "season": torch.tensor([2, 2])}

        result_1 = model(sample_tokens, categorical_ids=cat_ids_1)
        result_2 = model(sample_tokens, categorical_ids=cat_ids_2)

        # Different categorical values should produce different predictions
        assert not torch.allclose(
            result_1["target_prediction"],
            result_2["target_prediction"],
        )

    def test_gradients_flow_through_categorical_embedder(self, sample_tokens, categorical_cardinalities, categorical_ids):
        """Test that gradients flow through categorical embedder."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            categorical_cardinalities=categorical_cardinalities,
        )
        model.train()
        result = model(sample_tokens, categorical_ids=categorical_ids)
        loss = result["target_prediction"].sum()
        loss.backward()

        # Check categorical embedder gradients
        for name, param in model.categorical_embedder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

        # Check categorical projection gradients
        assert model.categorical_projection.weight.grad is not None
        assert model.categorical_projection.bias.grad is not None

    def test_gradients_flow_with_frozen_base(self, sample_tokens, categorical_cardinalities, categorical_ids):
        """Test gradients flow to categorical embedder with frozen base model."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            freeze_base=True,
            categorical_cardinalities=categorical_cardinalities,
        )
        model.train()
        result = model(sample_tokens, categorical_ids=categorical_ids)
        loss = result["target_prediction"].sum()
        loss.backward()

        # Base model should have no gradients
        for param in model.base_model.parameters():
            assert param.grad is None

        # Categorical embedder should have gradients
        has_categorical_grad = False
        for param in model.categorical_embedder.parameters():
            if param.requires_grad and param.grad is not None:
                has_categorical_grad = True
                break
        assert has_categorical_grad

        # Target encoder should have gradients
        has_target_grad = False
        for param in model.target_encoder.parameters():
            if param.requires_grad and param.grad is not None:
                has_target_grad = True
                break
        assert has_target_grad

    def test_categorical_weight_initialization(self, categorical_cardinalities):
        """Test that categorical projection weights are properly initialized."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            categorical_cardinalities=categorical_cardinalities,
        )
        # Bias should be zero-initialized
        assert torch.allclose(
            model.categorical_projection.bias,
            torch.zeros_like(model.categorical_projection.bias),
        )
        # Weight should have non-zero values (Xavier init)
        assert model.categorical_projection.weight.abs().sum() > 0

    @pytest.mark.parametrize("fusion", ["concat", "add"])
    def test_forward_no_nan_outputs(self, sample_tokens, categorical_cardinalities, categorical_ids, fusion):
        """Test that forward pass produces no NaN outputs."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            categorical_cardinalities=categorical_cardinalities,
            categorical_fusion=fusion,
        )
        result = model(sample_tokens, categorical_ids=categorical_ids)
        assert not torch.isnan(result["target_prediction"]).any()
        assert not torch.isnan(result["count_prediction"]).any()

    def test_categorical_with_classifier(self, sample_tokens, categorical_cardinalities, categorical_ids):
        """Test categorical conditioning with classification mode."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=5,
            is_classifier=True,
            categorical_cardinalities=categorical_cardinalities,
        )
        result = model(sample_tokens, categorical_ids=categorical_ids)
        predictions = result["target_prediction"]
        # Log-softmax outputs should be <= 0
        assert (predictions <= 0).all()
        # Sum of exp should be 1
        assert torch.allclose(predictions.exp().sum(dim=-1), torch.ones(2), atol=1e-5)

    def test_categorical_with_bounded_targets(self, sample_tokens, categorical_cardinalities, categorical_ids):
        """Test categorical conditioning with bounded regression."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            bounded_targets=True,
            categorical_cardinalities=categorical_cardinalities,
        )
        result = model(sample_tokens, categorical_ids=categorical_ids)
        predictions = result["target_prediction"]
        assert (predictions >= 0).all()
        assert (predictions <= 1).all()


class TestOutputActivation:
    """Test suite for output activation constraints in SequencePredictor."""

    @pytest.fixture
    def sample_tokens(self):
        """Create sample tokens for testing (batch_size=4 for this suite)."""
        return _create_sample_tokens(batch_size=4)

    def test_default_no_activation(self, sample_tokens):
        """Test that default is no output activation."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
        )
        assert model.output_activation == "none"

    def test_relu_produces_non_negative(self, sample_tokens):
        """Test that relu activation produces non-negative outputs."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            output_activation="relu",
        )
        result = model(sample_tokens)
        predictions = result["target_prediction"]
        assert (predictions >= 0).all()

    def test_softplus_produces_non_negative(self, sample_tokens):
        """Test that softplus activation produces non-negative outputs."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            output_activation="softplus",
        )
        result = model(sample_tokens)
        predictions = result["target_prediction"]
        assert (predictions >= 0).all()

    def test_exp_produces_positive(self, sample_tokens):
        """Test that exp activation produces strictly positive outputs."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            output_activation="exp",
        )
        result = model(sample_tokens)
        predictions = result["target_prediction"]
        assert (predictions > 0).all()

    def test_invalid_activation_raises(self):
        """Test that invalid activation raises ValueError."""
        with pytest.raises(ValueError, match="output_activation must be one of"):
            SequencePredictor(
                embedding_dim=64,
                max_bp=150,
                token_limit=1024,
                out_dim=1,
                output_activation="invalid",
            )

    def test_activation_with_bounded_targets_raises(self):
        """Test that using output_activation with bounded_targets raises ValueError."""
        with pytest.raises(ValueError, match="Cannot use output_activation with bounded_targets"):
            SequencePredictor(
                embedding_dim=64,
                max_bp=150,
                token_limit=1024,
                out_dim=1,
                output_activation="softplus",
                bounded_targets=True,
            )

    def test_activation_with_classifier_raises(self):
        """Test that using output_activation with is_classifier raises ValueError."""
        with pytest.raises(ValueError, match="Cannot use output_activation with is_classifier"):
            SequencePredictor(
                embedding_dim=64,
                max_bp=150,
                token_limit=1024,
                out_dim=5,
                output_activation="softplus",
                is_classifier=True,
            )

    def test_softplus_gradients_flow(self, sample_tokens):
        """Test that gradients flow through softplus activation."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            output_activation="softplus",
        )
        model.train()
        result = model(sample_tokens)
        loss = result["target_prediction"].sum()
        loss.backward()

        # Check target head gradients exist
        assert model.target_head.weight.grad is not None
        assert not torch.isnan(model.target_head.weight.grad).any()

    def test_relu_gradients_flow(self, sample_tokens):
        """Test that gradients flow through relu activation (when positive)."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            output_activation="relu",
        )
        model.train()
        result = model(sample_tokens)
        loss = result["target_prediction"].sum()
        loss.backward()

        assert model.target_head.weight.grad is not None

    @pytest.mark.parametrize("activation", ["relu", "softplus", "exp"])
    def test_activation_with_multi_output(self, sample_tokens, activation):
        """Test output activations work with multiple output dimensions."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=3,
            output_activation=activation,
        )
        result = model(sample_tokens)
        predictions = result["target_prediction"]
        assert predictions.shape == (4, 3)
        if activation == "exp":
            assert (predictions > 0).all()
        else:
            assert (predictions >= 0).all()

    def test_softplus_with_learnable_scale(self, sample_tokens):
        """Test softplus works with learnable output scale."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            output_activation="softplus",
            learnable_output_scale=True,
        )
        result = model(sample_tokens)
        predictions = result["target_prediction"]
        assert (predictions >= 0).all()

    def test_softplus_with_layer_norm(self, sample_tokens):
        """Test softplus works with target layer norm."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            output_activation="softplus",
            target_layer_norm=True,
        )
        result = model(sample_tokens)
        predictions = result["target_prediction"]
        assert (predictions >= 0).all()

    def test_none_activation_allows_negative(self, sample_tokens):
        """Test that 'none' activation allows negative values."""
        # Force negative output by setting bias to large negative value
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            output_activation="none",
        )
        with torch.no_grad():
            model.target_head.bias.fill_(-100.0)

        result = model(sample_tokens)
        predictions = result["target_prediction"]
        # Should allow negative values
        assert (predictions < 0).any()


class TestMLPRegressionHead:
    """Test suite for MLP regression head configuration."""

    @pytest.fixture
    def sample_tokens(self):
        """Create sample tokens for testing."""
        return _create_sample_tokens()

    def test_default_single_linear_layer(self):
        """Test that default target head is a single linear layer."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
        )
        assert model.regressor_hidden_dims is None
        assert model.regressor_dropout == 0.0
        assert isinstance(model.target_head, nn.Linear)
        assert model.target_head.in_features == 64
        assert model.target_head.out_features == 1

    def test_mlp_single_hidden_layer(self, sample_tokens):
        """Test MLP with single hidden layer."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            regressor_hidden_dims=[32],
        )
        assert model.regressor_hidden_dims == [32]
        assert isinstance(model.target_head, nn.Sequential)
        # Structure: Linear(64, 32), ReLU, Linear(32, 1)
        assert len(model.target_head) == 3
        assert isinstance(model.target_head[0], nn.Linear)
        assert model.target_head[0].in_features == 64
        assert model.target_head[0].out_features == 32
        assert isinstance(model.target_head[1], nn.ReLU)
        assert isinstance(model.target_head[2], nn.Linear)
        assert model.target_head[2].in_features == 32
        assert model.target_head[2].out_features == 1

        result = model(sample_tokens)
        assert result["target_prediction"].shape == (2, 1)

    def test_mlp_two_hidden_layers(self, sample_tokens):
        """Test MLP with two hidden layers (64,32)."""
        model = SequencePredictor(
            embedding_dim=128,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            regressor_hidden_dims=[64, 32],
        )
        assert model.regressor_hidden_dims == [64, 32]
        assert isinstance(model.target_head, nn.Sequential)
        # Structure: Linear(128, 64), ReLU, Linear(64, 32), ReLU, Linear(32, 1)
        assert len(model.target_head) == 5
        assert isinstance(model.target_head[0], nn.Linear)
        assert model.target_head[0].in_features == 128
        assert model.target_head[0].out_features == 64
        assert isinstance(model.target_head[1], nn.ReLU)
        assert isinstance(model.target_head[2], nn.Linear)
        assert model.target_head[2].in_features == 64
        assert model.target_head[2].out_features == 32
        assert isinstance(model.target_head[3], nn.ReLU)
        assert isinstance(model.target_head[4], nn.Linear)
        assert model.target_head[4].in_features == 32
        assert model.target_head[4].out_features == 1

        result = model(sample_tokens)
        assert result["target_prediction"].shape == (2, 1)

    def test_mlp_with_dropout(self, sample_tokens):
        """Test MLP with dropout between layers."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            regressor_hidden_dims=[32],
            regressor_dropout=0.1,
        )
        assert model.regressor_dropout == 0.1
        assert isinstance(model.target_head, nn.Sequential)
        # Structure: Linear(64, 32), ReLU, Dropout, Linear(32, 1)
        assert len(model.target_head) == 4
        assert isinstance(model.target_head[0], nn.Linear)
        assert isinstance(model.target_head[1], nn.ReLU)
        assert isinstance(model.target_head[2], nn.Dropout)
        assert model.target_head[2].p == 0.1
        assert isinstance(model.target_head[3], nn.Linear)

        result = model(sample_tokens)
        assert result["target_prediction"].shape == (2, 1)

    def test_mlp_two_layers_with_dropout(self, sample_tokens):
        """Test MLP with multiple hidden layers and dropout."""
        model = SequencePredictor(
            embedding_dim=128,
            max_bp=150,
            token_limit=1024,
            out_dim=3,
            regressor_hidden_dims=[64, 32],
            regressor_dropout=0.2,
        )
        # Structure: Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear
        assert len(model.target_head) == 7
        dropout_count = sum(1 for m in model.target_head if isinstance(m, nn.Dropout))
        assert dropout_count == 2

        result = model(sample_tokens)
        assert result["target_prediction"].shape == (2, 3)

    def test_mlp_zero_dropout_no_dropout_layers(self):
        """Test that zero dropout does not add Dropout layers."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            regressor_hidden_dims=[32],
            regressor_dropout=0.0,
        )
        # Should have no Dropout layers
        dropout_count = sum(1 for m in model.target_head if isinstance(m, nn.Dropout))
        assert dropout_count == 0

    def test_mlp_empty_list_single_linear(self):
        """Test that empty hidden dims list creates single linear layer."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            regressor_hidden_dims=[],
        )
        assert isinstance(model.target_head, nn.Linear)

    def test_mlp_with_bounded_targets(self, sample_tokens):
        """Test MLP works with bounded_targets sigmoid."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            regressor_hidden_dims=[32],
            bounded_targets=True,
        )
        result = model(sample_tokens)
        predictions = result["target_prediction"]
        assert (predictions >= 0).all()
        assert (predictions <= 1).all()

    def test_mlp_with_output_activation(self, sample_tokens):
        """Test MLP works with output activations."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            regressor_hidden_dims=[32],
            output_activation="softplus",
        )
        result = model(sample_tokens)
        predictions = result["target_prediction"]
        assert (predictions >= 0).all()

    def test_mlp_with_learnable_scale(self, sample_tokens):
        """Test MLP works with learnable output scale."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=2,
            regressor_hidden_dims=[32, 16],
            learnable_output_scale=True,
        )
        assert model.output_scale is not None
        result = model(sample_tokens)
        assert result["target_prediction"].shape == (2, 2)

    def test_mlp_with_layer_norm(self, sample_tokens):
        """Test MLP works with target layer norm."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            regressor_hidden_dims=[32],
            target_layer_norm=True,
        )
        assert model.target_norm is not None
        result = model(sample_tokens)
        assert result["target_prediction"].shape == (2, 1)

    def test_mlp_with_classifier(self, sample_tokens):
        """Test MLP works with classification mode."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=5,
            regressor_hidden_dims=[32],
            is_classifier=True,
        )
        result = model(sample_tokens)
        predictions = result["target_prediction"]
        # Log-softmax outputs should be <= 0
        assert (predictions <= 0).all()
        assert torch.allclose(predictions.exp().sum(dim=-1), torch.ones(2), atol=1e-5)

    def test_mlp_with_categorical_concat(self, sample_tokens):
        """Test MLP works with categorical conditioning (concat)."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            regressor_hidden_dims=[32],
            categorical_cardinalities={"location": 5},
            categorical_fusion="concat",
        )
        cat_ids = {"location": torch.tensor([1, 2])}
        result = model(sample_tokens, categorical_ids=cat_ids)
        assert result["target_prediction"].shape == (2, 1)

    def test_mlp_with_categorical_add(self, sample_tokens):
        """Test MLP works with categorical conditioning (add)."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            regressor_hidden_dims=[32],
            categorical_cardinalities={"season": 4},
            categorical_fusion="add",
        )
        cat_ids = {"season": torch.tensor([0, 3])}
        result = model(sample_tokens, categorical_ids=cat_ids)
        assert result["target_prediction"].shape == (2, 1)

    def test_mlp_gradients_flow(self, sample_tokens):
        """Test that gradients flow through MLP layers."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            regressor_hidden_dims=[32, 16],
        )
        model.train()
        result = model(sample_tokens)
        loss = result["target_prediction"].sum()
        loss.backward()

        # Check that all MLP linear layers have gradients
        for module in model.target_head.modules():
            if isinstance(module, nn.Linear):
                assert module.weight.grad is not None
                assert not torch.isnan(module.weight.grad).any()
                assert module.bias.grad is not None

    def test_mlp_weight_initialization(self):
        """Test that MLP weights are properly initialized."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            regressor_hidden_dims=[32],
        )
        for module in model.target_head.modules():
            if isinstance(module, nn.Linear):
                # Biases should be zero-initialized
                assert torch.allclose(module.bias, torch.zeros_like(module.bias))
                # Weights should be non-zero (Xavier init)
                assert module.weight.abs().sum() > 0

    def test_mlp_no_nan_outputs(self, sample_tokens):
        """Test that MLP produces no NaN outputs."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            regressor_hidden_dims=[64, 32, 16],
            regressor_dropout=0.3,
        )
        model.eval()
        result = model(sample_tokens)
        assert not torch.isnan(result["target_prediction"]).any()

    @pytest.mark.parametrize("out_dim", [1, 3, 5])
    def test_mlp_different_output_dims(self, sample_tokens, out_dim):
        """Test MLP with different output dimensions."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=out_dim,
            regressor_hidden_dims=[32],
        )
        result = model(sample_tokens)
        assert result["target_prediction"].shape == (2, out_dim)

    @pytest.mark.parametrize("hidden_dims", [[64], [64, 32], [128, 64, 32], [256, 128, 64, 32]])
    def test_mlp_various_architectures(self, sample_tokens, hidden_dims):
        """Test MLP with various hidden layer configurations."""
        model = SequencePredictor(
            embedding_dim=128,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            regressor_hidden_dims=hidden_dims,
        )
        result = model(sample_tokens)
        assert result["target_prediction"].shape == (2, 1)

    def test_mlp_frozen_base_gradients(self, sample_tokens):
        """Test that MLP receives gradients even with frozen base."""
        model = SequencePredictor(
            embedding_dim=64,
            max_bp=150,
            token_limit=1024,
            out_dim=1,
            regressor_hidden_dims=[32],
            freeze_base=True,
        )
        model.train()
        result = model(sample_tokens)
        loss = result["target_prediction"].sum()
        loss.backward()

        # Base model should have no gradients
        for param in model.base_model.parameters():
            assert param.grad is None

        # MLP should have gradients
        mlp_has_grads = False
        for module in model.target_head.modules():
            if isinstance(module, nn.Linear) and module.weight.grad is not None:
                mlp_has_grads = True
                break
        assert mlp_has_grads
