"""Unit tests for CategoricalEmbedder module."""

import pytest
import torch
import torch.nn as nn

from aam.models.categorical_embedder import CategoricalEmbedder


class TestCategoricalEmbedderInit:
    """Test suite for CategoricalEmbedder initialization."""

    def test_init_single_column(self):
        """Test initialization with single categorical column."""
        cardinalities = {"location": 4}  # 3 categories + unknown
        embedder = CategoricalEmbedder(cardinalities, embed_dim=16)

        assert embedder.num_columns == 1
        assert embedder.total_embed_dim == 16
        assert embedder.column_names == ["location"]

    def test_init_multiple_columns(self):
        """Test initialization with multiple categorical columns."""
        cardinalities = {"location": 4, "season": 5}
        embedder = CategoricalEmbedder(cardinalities, embed_dim=16)

        assert embedder.num_columns == 2
        assert embedder.total_embed_dim == 32  # 2 * 16
        assert set(embedder.column_names) == {"location", "season"}

    def test_init_per_column_embed_dim(self):
        """Test initialization with per-column embedding dimensions."""
        cardinalities = {"location": 4, "season": 5}
        embed_dims = {"location": 8, "season": 32}
        embedder = CategoricalEmbedder(cardinalities, embed_dim=embed_dims)

        assert embedder.total_embed_dim == 40  # 8 + 32

    def test_init_creates_embeddings(self):
        """Test that init creates nn.Embedding for each column."""
        cardinalities = {"location": 4, "season": 5}
        embedder = CategoricalEmbedder(cardinalities, embed_dim=16)

        # Should have embedding modules
        assert isinstance(embedder, nn.Module)
        # Check that embeddings exist in ModuleDict for each column
        assert hasattr(embedder, "embeddings")
        for col in cardinalities:
            assert col in embedder.embeddings
            assert isinstance(embedder.embeddings[col], nn.Embedding)

    def test_init_empty_cardinalities(self):
        """Test initialization with empty cardinalities dict."""
        embedder = CategoricalEmbedder({}, embed_dim=16)

        assert embedder.num_columns == 0
        assert embedder.total_embed_dim == 0
        assert embedder.column_names == []


class TestCategoricalEmbedderForward:
    """Test suite for CategoricalEmbedder forward pass."""

    @pytest.fixture
    def embedder(self):
        """Create embedder for testing."""
        cardinalities = {"location": 4, "season": 5}
        return CategoricalEmbedder(cardinalities, embed_dim=16)

    def test_forward_basic(self, embedder):
        """Test basic forward pass."""
        categorical_ids = {
            "location": torch.tensor([1, 2, 1]),  # batch_size=3
            "season": torch.tensor([1, 3, 2]),
        }

        output = embedder(categorical_ids)

        assert output.shape == (3, 32)  # [B, total_embed_dim]
        assert output.dtype == torch.float32

    def test_forward_single_sample(self, embedder):
        """Test forward with single sample batch."""
        categorical_ids = {
            "location": torch.tensor([1]),
            "season": torch.tensor([2]),
        }

        output = embedder(categorical_ids)

        assert output.shape == (1, 32)

    def test_forward_padding_index(self, embedder):
        """Test that padding index 0 produces valid embeddings."""
        categorical_ids = {
            "location": torch.tensor([0, 1, 0]),  # 0 = unknown/missing
            "season": torch.tensor([0, 0, 1]),
        }

        output = embedder(categorical_ids)

        assert output.shape == (3, 32)
        # Output should not contain NaN
        assert not torch.isnan(output).any()

    def test_forward_consistent_for_same_input(self):
        """Test that same input produces same output (in eval mode)."""
        cardinalities = {"location": 4}
        embedder = CategoricalEmbedder(cardinalities, embed_dim=16, dropout=0.0)
        embedder.eval()

        categorical_ids = {"location": torch.tensor([1, 2, 1])}

        output1 = embedder(categorical_ids)
        output2 = embedder(categorical_ids)

        torch.testing.assert_close(output1, output2)

    def test_forward_same_category_same_embedding(self):
        """Test that same category indices produce same embeddings."""
        cardinalities = {"location": 4}
        embedder = CategoricalEmbedder(cardinalities, embed_dim=16, dropout=0.0)
        embedder.eval()

        categorical_ids = {"location": torch.tensor([1, 2, 1])}

        output = embedder(categorical_ids)

        # Indices 0 and 2 both have category 1, should have same embedding
        torch.testing.assert_close(output[0], output[2])
        # Index 1 has category 2, should be different
        assert not torch.allclose(output[0], output[1])

    def test_forward_missing_column_uses_padding(self, embedder):
        """Test that missing column in input uses padding index."""
        # Only provide location, season missing
        categorical_ids = {
            "location": torch.tensor([1, 2]),
        }

        output = embedder(categorical_ids)

        assert output.shape == (2, 32)
        assert not torch.isnan(output).any()

    def test_forward_empty_dict(self):
        """Test forward with empty categorical_ids dict."""
        embedder = CategoricalEmbedder({}, embed_dim=16)

        output = embedder({})

        # Should return empty tensor or handle gracefully
        assert output.shape[1] == 0 or output.numel() == 0


class TestCategoricalEmbedderBroadcast:
    """Test suite for broadcast_to_sequence method."""

    @pytest.fixture
    def embedder(self):
        """Create embedder for testing."""
        cardinalities = {"location": 4}
        return CategoricalEmbedder(cardinalities, embed_dim=16)

    def test_broadcast_basic(self, embedder):
        """Test basic broadcast to sequence."""
        embeddings = torch.randn(3, 16)  # [B, embed_dim]

        broadcast = embedder.broadcast_to_sequence(embeddings, seq_len=10)

        assert broadcast.shape == (3, 10, 16)  # [B, S, embed_dim]

    def test_broadcast_preserves_values(self, embedder):
        """Test that broadcast copies same value to all sequence positions."""
        embeddings = torch.randn(2, 16)

        broadcast = embedder.broadcast_to_sequence(embeddings, seq_len=5)

        # All sequence positions should have same embedding
        for i in range(5):
            torch.testing.assert_close(broadcast[:, i, :], embeddings)

    def test_broadcast_seq_len_1(self, embedder):
        """Test broadcast with sequence length 1."""
        embeddings = torch.randn(3, 16)

        broadcast = embedder.broadcast_to_sequence(embeddings, seq_len=1)

        assert broadcast.shape == (3, 1, 16)
        torch.testing.assert_close(broadcast[:, 0, :], embeddings)


class TestCategoricalEmbedderDropout:
    """Test suite for dropout behavior."""

    def test_dropout_in_training_mode(self):
        """Test that dropout is applied in training mode."""
        cardinalities = {"location": 4}
        embedder = CategoricalEmbedder(cardinalities, embed_dim=16, dropout=0.5)
        embedder.train()

        categorical_ids = {"location": torch.tensor([1, 1, 1, 1])}

        # Run multiple times, outputs should differ due to dropout
        outputs = [embedder(categorical_ids) for _ in range(5)]

        # At least some outputs should be different
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, "Dropout should cause different outputs in training mode"

    def test_dropout_disabled_in_eval_mode(self):
        """Test that dropout is disabled in eval mode."""
        cardinalities = {"location": 4}
        embedder = CategoricalEmbedder(cardinalities, embed_dim=16, dropout=0.5)
        embedder.eval()

        categorical_ids = {"location": torch.tensor([1, 1, 1])}

        output1 = embedder(categorical_ids)
        output2 = embedder(categorical_ids)

        torch.testing.assert_close(output1, output2)

    def test_zero_dropout(self):
        """Test embedder with zero dropout."""
        cardinalities = {"location": 4}
        embedder = CategoricalEmbedder(cardinalities, embed_dim=16, dropout=0.0)
        embedder.train()

        categorical_ids = {"location": torch.tensor([1, 2])}

        output1 = embedder(categorical_ids)
        output2 = embedder(categorical_ids)

        torch.testing.assert_close(output1, output2)


class TestCategoricalEmbedderGradients:
    """Test suite for gradient flow."""

    def test_gradients_flow_through_embeddings(self):
        """Test that gradients flow back through embeddings."""
        cardinalities = {"location": 4}
        embedder = CategoricalEmbedder(cardinalities, embed_dim=16, dropout=0.0)

        categorical_ids = {"location": torch.tensor([1, 2])}
        output = embedder(categorical_ids)

        # Compute loss and backprop
        loss = output.sum()
        loss.backward()

        # Check that embedding weights have gradients
        has_grad = False
        for param in embedder.parameters():
            if param.grad is not None:
                has_grad = True
                break

        assert has_grad, "Gradients should flow to embedding parameters"


class TestCategoricalEmbedderDevice:
    """Test suite for device handling."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test forward pass on CUDA device."""
        cardinalities = {"location": 4}
        embedder = CategoricalEmbedder(cardinalities, embed_dim=16).cuda()

        categorical_ids = {"location": torch.tensor([1, 2]).cuda()}

        output = embedder(categorical_ids)

        assert output.device.type == "cuda"
        assert output.shape == (2, 16)

    def test_cpu_forward(self):
        """Test forward pass on CPU."""
        cardinalities = {"location": 4}
        embedder = CategoricalEmbedder(cardinalities, embed_dim=16)

        categorical_ids = {"location": torch.tensor([1, 2])}

        output = embedder(categorical_ids)

        assert output.device.type == "cpu"
