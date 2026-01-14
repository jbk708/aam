"""Tests for FiLM (Feature-wise Linear Modulation) components."""

import pytest
import torch
import torch.nn as nn

from aam.models.film import FiLMGenerator, FiLMLayer, FiLMTargetHead


class TestFiLMGenerator:
    """Tests for FiLMGenerator."""

    def test_init(self):
        """Test FiLMGenerator initialization."""
        gen = FiLMGenerator(categorical_dim=64, hidden_dim=128)
        assert gen.gamma_proj.in_features == 64
        assert gen.gamma_proj.out_features == 128
        assert gen.beta_proj.in_features == 64
        assert gen.beta_proj.out_features == 128

    def test_forward_shape(self):
        """Test forward pass output shapes."""
        gen = FiLMGenerator(categorical_dim=64, hidden_dim=128)
        cat_emb = torch.randn(8, 64)
        gamma, beta = gen(cat_emb)
        assert gamma.shape == (8, 128)
        assert beta.shape == (8, 128)

    def test_identity_initialization(self):
        """Test that gamma initialized to 1, beta to 0."""
        gen = FiLMGenerator(categorical_dim=64, hidden_dim=128)
        cat_emb = torch.zeros(4, 64)  # Zero input
        gamma, beta = gen(cat_emb)
        # With zero input and zero weights, gamma should be bias (1) and beta should be bias (0)
        assert torch.allclose(gamma, torch.ones_like(gamma), atol=1e-6)
        assert torch.allclose(beta, torch.zeros_like(beta), atol=1e-6)

    def test_different_dims(self):
        """Test with various dimension combinations."""
        for cat_dim, hidden_dim in [(16, 32), (128, 64), (32, 256)]:
            gen = FiLMGenerator(categorical_dim=cat_dim, hidden_dim=hidden_dim)
            cat_emb = torch.randn(4, cat_dim)
            gamma, beta = gen(cat_emb)
            assert gamma.shape == (4, hidden_dim)
            assert beta.shape == (4, hidden_dim)


class TestFiLMLayer:
    """Tests for FiLMLayer."""

    def test_init(self):
        """Test FiLMLayer initialization."""
        layer = FiLMLayer(in_dim=128, out_dim=64, categorical_dim=32)
        assert layer.linear.in_features == 128
        assert layer.linear.out_features == 64
        assert layer.film.gamma_proj.in_features == 32
        assert layer.film.gamma_proj.out_features == 64

    def test_forward_shape(self):
        """Test forward pass output shapes."""
        layer = FiLMLayer(in_dim=128, out_dim=64, categorical_dim=32)
        x = torch.randn(8, 128)
        cat_emb = torch.randn(8, 32)
        out = layer(x, cat_emb)
        assert out.shape == (8, 64)

    def test_modulation_effect(self):
        """Test that FiLM modulation affects output after training."""
        layer = FiLMLayer(in_dim=128, out_dim=64, categorical_dim=32)

        # With identity initialization, weights are zero so we need to set non-zero weights
        # to verify that different categorical embeddings produce different outputs
        nn.init.xavier_uniform_(layer.film.gamma_proj.weight)
        nn.init.xavier_uniform_(layer.film.beta_proj.weight)

        x = torch.randn(8, 128)
        cat_emb1 = torch.randn(8, 32)
        cat_emb2 = torch.randn(8, 32)

        out1 = layer(x, cat_emb1)
        out2 = layer(x, cat_emb2)

        # Different categorical embeddings should produce different outputs
        assert not torch.allclose(out1, out2)

    def test_identity_modulation(self):
        """Test identity modulation (gamma=1, beta=0) preserves ReLU(linear) output."""
        layer = FiLMLayer(in_dim=128, out_dim=64, categorical_dim=32, dropout=0.0)
        layer.eval()  # Disable dropout

        x = torch.randn(8, 128)
        cat_emb = torch.zeros(8, 32)  # Zero input -> identity modulation

        # Manual computation: ReLU(linear(x))
        expected = torch.relu(layer.linear(x))
        actual = layer(x, cat_emb)

        assert torch.allclose(actual, expected, atol=1e-5)

    def test_dropout(self):
        """Test dropout is applied in training mode."""
        layer = FiLMLayer(in_dim=128, out_dim=64, categorical_dim=32, dropout=0.5)
        x = torch.randn(8, 128)
        cat_emb = torch.randn(8, 32)

        layer.train()
        out_train1 = layer(x, cat_emb)
        out_train2 = layer(x, cat_emb)

        layer.eval()
        out_eval1 = layer(x, cat_emb)
        out_eval2 = layer(x, cat_emb)

        # In eval mode, outputs should be deterministic
        assert torch.allclose(out_eval1, out_eval2)
        # In train mode with 50% dropout, outputs should differ (with high probability)
        # Note: This could theoretically fail but is extremely unlikely
        assert not torch.allclose(out_train1, out_train2)

    def test_no_dropout(self):
        """Test with dropout=0.0."""
        layer = FiLMLayer(in_dim=128, out_dim=64, categorical_dim=32, dropout=0.0)
        x = torch.randn(8, 128)
        cat_emb = torch.randn(8, 32)

        layer.train()
        out1 = layer(x, cat_emb)
        out2 = layer(x, cat_emb)

        assert torch.allclose(out1, out2)


class TestFiLMTargetHead:
    """Tests for FiLMTargetHead."""

    def test_init(self):
        """Test FiLMTargetHead initialization."""
        head = FiLMTargetHead(
            in_dim=256,
            out_dim=1,
            hidden_dims=[128, 64],
            categorical_dim=32,
        )
        assert len(head.film_layers) == 2  # Two hidden layers
        assert head.output_layer.in_features == 64
        assert head.output_layer.out_features == 1

    def test_forward_shape(self):
        """Test forward pass output shapes."""
        head = FiLMTargetHead(
            in_dim=256,
            out_dim=1,
            hidden_dims=[128, 64],
            categorical_dim=32,
        )
        x = torch.randn(8, 256)
        cat_emb = torch.randn(8, 32)
        out = head(x, cat_emb)
        assert out.shape == (8, 1)

    def test_forward_without_categorical(self):
        """Test forward pass without categorical embedding uses identity."""
        head = FiLMTargetHead(
            in_dim=256,
            out_dim=1,
            hidden_dims=[128, 64],
            categorical_dim=32,
            dropout=0.0,
        )
        head.eval()

        x = torch.randn(8, 256)
        # Without categorical embedding
        out_no_cat = head(x, categorical_emb=None)
        # With zero categorical embedding (identity transform)
        out_zero_cat = head(x, categorical_emb=torch.zeros(8, 32))

        assert torch.allclose(out_no_cat, out_zero_cat, atol=1e-5)

    def test_multi_layer(self):
        """Test MLP with multiple hidden layers."""
        head = FiLMTargetHead(
            in_dim=256,
            out_dim=2,
            hidden_dims=[128, 64, 32],
            categorical_dim=16,
        )
        assert len(head.film_layers) == 3

        x = torch.randn(4, 256)
        cat_emb = torch.randn(4, 16)
        out = head(x, cat_emb)
        assert out.shape == (4, 2)

    def test_single_layer(self):
        """Test MLP with single hidden layer."""
        head = FiLMTargetHead(
            in_dim=256,
            out_dim=1,
            hidden_dims=[64],
            categorical_dim=32,
        )
        assert len(head.film_layers) == 1

        x = torch.randn(8, 256)
        cat_emb = torch.randn(8, 32)
        out = head(x, cat_emb)
        assert out.shape == (8, 1)

    def test_gradient_flow(self):
        """Test gradients flow through FiLM layers."""
        head = FiLMTargetHead(
            in_dim=256,
            out_dim=1,
            hidden_dims=[128, 64],
            categorical_dim=32,
        )

        # Set non-zero FiLM weights so categorical embedding gradients are non-zero
        for layer in head.film_layers:
            nn.init.xavier_uniform_(layer.film.gamma_proj.weight)
            nn.init.xavier_uniform_(layer.film.beta_proj.weight)

        x = torch.randn(8, 256, requires_grad=True)
        cat_emb = torch.randn(8, 32, requires_grad=True)

        out = head(x, cat_emb)
        loss = out.sum()
        loss.backward()

        # Gradients should flow to both inputs
        assert x.grad is not None
        assert cat_emb.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
        assert not torch.allclose(cat_emb.grad, torch.zeros_like(cat_emb.grad))

        # Gradients should flow to FiLM parameters
        for layer in head.film_layers:
            assert layer.film.gamma_proj.weight.grad is not None
            assert layer.film.beta_proj.weight.grad is not None

    def test_multi_output(self):
        """Test with multiple output dimensions."""
        head = FiLMTargetHead(
            in_dim=256,
            out_dim=5,
            hidden_dims=[128],
            categorical_dim=32,
        )
        x = torch.randn(8, 256)
        cat_emb = torch.randn(8, 32)
        out = head(x, cat_emb)
        assert out.shape == (8, 5)


class TestFiLMIntegration:
    """Integration tests for FiLM with SequencePredictor."""

    def test_sequence_predictor_with_film(self):
        """Test SequencePredictor with FiLM conditioning."""
        from aam.models.sequence_predictor import SequencePredictor

        model = SequencePredictor(
            embedding_dim=64,
            out_dim=1,
            categorical_cardinalities={"season": 5, "site": 9},
            categorical_embed_dim=16,
            regressor_hidden_dims=[32, 16],
            film_conditioning_columns=["season"],
        )

        tokens = torch.randint(0, 5, (4, 10, 50))
        categorical_ids = {
            "season": torch.randint(0, 5, (4,)),
            "site": torch.randint(0, 9, (4,)),
        }

        outputs = model(tokens, categorical_ids=categorical_ids)
        assert outputs["target_prediction"].shape == (4, 1)

    def test_film_requires_mlp_head(self):
        """Test that FiLM requires regressor_hidden_dims."""
        from aam.models.sequence_predictor import SequencePredictor

        with pytest.raises(ValueError, match="film_conditioning_columns requires regressor_hidden_dims"):
            SequencePredictor(
                embedding_dim=64,
                out_dim=1,
                categorical_cardinalities={"season": 5},
                categorical_embed_dim=16,
                regressor_hidden_dims=None,  # No MLP
                film_conditioning_columns=["season"],
            )

    def test_film_requires_categorical(self):
        """Test that FiLM requires categorical columns."""
        from aam.models.sequence_predictor import SequencePredictor

        with pytest.raises(ValueError, match="film_conditioning_columns requires categorical_cardinalities"):
            SequencePredictor(
                embedding_dim=64,
                out_dim=1,
                categorical_cardinalities=None,  # No categorical
                regressor_hidden_dims=[32, 16],
                film_conditioning_columns=["season"],
            )

    def test_film_column_must_exist(self):
        """Test that FiLM columns must be in categorical_cardinalities."""
        from aam.models.sequence_predictor import SequencePredictor

        with pytest.raises(ValueError, match="not found in categorical_cardinalities"):
            SequencePredictor(
                embedding_dim=64,
                out_dim=1,
                categorical_cardinalities={"season": 5},
                categorical_embed_dim=16,
                regressor_hidden_dims=[32, 16],
                film_conditioning_columns=["unknown_column"],
            )

    def test_film_with_conditional_scaling(self):
        """Test FiLM works with conditional output scaling."""
        from aam.models.sequence_predictor import SequencePredictor

        model = SequencePredictor(
            embedding_dim=64,
            out_dim=1,
            categorical_cardinalities={"season": 5, "site": 9},
            categorical_embed_dim=16,
            regressor_hidden_dims=[32, 16],
            film_conditioning_columns=["season"],
            conditional_scaling_columns=["site"],
        )

        tokens = torch.randint(0, 5, (4, 10, 50))
        categorical_ids = {
            "season": torch.randint(0, 5, (4,)),
            "site": torch.randint(0, 9, (4,)),
        }

        outputs = model(tokens, categorical_ids=categorical_ids)
        assert outputs["target_prediction"].shape == (4, 1)

    def test_film_multiple_columns(self):
        """Test FiLM with multiple conditioning columns."""
        from aam.models.sequence_predictor import SequencePredictor

        model = SequencePredictor(
            embedding_dim=64,
            out_dim=1,
            categorical_cardinalities={"season": 5, "site": 9},
            categorical_embed_dim=16,
            regressor_hidden_dims=[32, 16],
            film_conditioning_columns=["season", "site"],
        )

        tokens = torch.randint(0, 5, (4, 10, 50))
        categorical_ids = {
            "season": torch.randint(0, 5, (4,)),
            "site": torch.randint(0, 9, (4,)),
        }

        outputs = model(tokens, categorical_ids=categorical_ids)
        assert outputs["target_prediction"].shape == (4, 1)

    def test_film_affects_predictions(self):
        """Test that different categorical values produce different predictions."""
        from aam.models.sequence_predictor import SequencePredictor

        torch.manual_seed(42)
        model = SequencePredictor(
            embedding_dim=64,
            out_dim=1,
            categorical_cardinalities={"season": 5},
            categorical_embed_dim=16,
            regressor_hidden_dims=[32, 16],
            film_conditioning_columns=["season"],
        )
        model.eval()

        tokens = torch.randint(0, 5, (4, 10, 50))

        # Same tokens, different seasons
        outputs1 = model(tokens, categorical_ids={"season": torch.tensor([1, 1, 1, 1])})
        outputs2 = model(tokens, categorical_ids={"season": torch.tensor([2, 2, 2, 2])})

        # Predictions should differ due to FiLM conditioning
        assert not torch.allclose(
            outputs1["target_prediction"],
            outputs2["target_prediction"],
        )

    def test_film_with_categorical_fusion(self):
        """Test FiLM works alongside categorical fusion (concat/add)."""
        from aam.models.sequence_predictor import SequencePredictor

        for fusion in ["concat", "add"]:
            model = SequencePredictor(
                embedding_dim=64,
                out_dim=1,
                categorical_cardinalities={"season": 5},
                categorical_embed_dim=16,
                categorical_fusion=fusion,
                regressor_hidden_dims=[32, 16],
                film_conditioning_columns=["season"],
            )

            tokens = torch.randint(0, 5, (4, 10, 50))
            categorical_ids = {"season": torch.randint(0, 5, (4,))}

            outputs = model(tokens, categorical_ids=categorical_ids)
            assert outputs["target_prediction"].shape == (4, 1)

    def test_film_warns_on_missing_columns(self):
        """Test that warning is raised when FiLM columns are missing from categorical_ids."""
        import warnings
        from aam.models.sequence_predictor import SequencePredictor

        model = SequencePredictor(
            embedding_dim=64,
            out_dim=1,
            categorical_cardinalities={"season": 5, "site": 9},
            categorical_embed_dim=16,
            regressor_hidden_dims=[32, 16],
            film_conditioning_columns=["season", "site"],
        )

        tokens = torch.randint(0, 5, (4, 10, 50))

        # Only provide 'season', missing 'site'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            outputs = model(tokens, categorical_ids={"season": torch.randint(0, 5, (4,))})

            # Should still produce output
            assert outputs["target_prediction"].shape == (4, 1)

            # Should have raised a warning about missing 'site'
            assert len(w) == 1
            assert "site" in str(w[0].message)
            assert "FiLM conditioning columns" in str(w[0].message)
