"""Unit tests for SequencePredictor class."""

import pytest
import torch
import torch.nn as nn

from aam.models.sequence_predictor import SequencePredictor
from aam.models.sequence_encoder import SequenceEncoder


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
    from aam.data.tokenizer import SequenceTokenizer

    batch_size = 2
    num_asvs = 10
    seq_len = 50
    tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len))
    tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
    tokens[:, :, 40:] = 0
    return tokens


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
        from aam.data.tokenizer import SequenceTokenizer

        batch_size = 2
        num_asvs = 10
        seq_len = 50
        tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len))
        tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
        tokens[:, :, 40:] = 0
        return tokens

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
        from aam.data.tokenizer import SequenceTokenizer

        batch_size = 2
        num_asvs = 10
        seq_len = 50
        tokens = torch.randint(1, 5, (batch_size, num_asvs, seq_len))
        tokens[:, :, 0] = SequenceTokenizer.START_TOKEN
        tokens[:, :, 40:] = 0
        return tokens

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
