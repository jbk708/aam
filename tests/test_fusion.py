"""Tests for multimodal fusion modules."""

import pytest
import torch

from aam.models.fusion import CrossAttentionFusion, GMU


class TestGMU:
    """Tests for Gated Multimodal Unit."""

    # Initialization tests
    def test_init_basic(self):
        """Test GMU initializes with valid dimensions."""
        gmu = GMU(seq_dim=128, cat_dim=32)
        assert gmu.seq_dim == 128
        assert gmu.cat_dim == 32

    def test_init_asymmetric_dims(self):
        """Test GMU with different seq_dim and cat_dim."""
        gmu = GMU(seq_dim=256, cat_dim=64)
        assert gmu.seq_dim == 256
        assert gmu.cat_dim == 64

    def test_init_large_dims(self):
        """Test GMU with large dimensions."""
        gmu = GMU(seq_dim=1024, cat_dim=512)
        assert gmu.seq_dim == 1024
        assert gmu.cat_dim == 512

    def test_init_has_expected_layers(self):
        """Test GMU has cat_transform and gate layers."""
        gmu = GMU(seq_dim=128, cat_dim=32)
        assert hasattr(gmu, "cat_transform")
        assert hasattr(gmu, "gate")
        assert isinstance(gmu.cat_transform, torch.nn.Linear)
        assert isinstance(gmu.gate, torch.nn.Linear)

    def test_init_layer_dimensions(self):
        """Test layer dimensions are correct."""
        seq_dim, cat_dim = 128, 32
        gmu = GMU(seq_dim=seq_dim, cat_dim=cat_dim)
        # cat_transform: cat_dim -> seq_dim
        assert gmu.cat_transform.in_features == cat_dim
        assert gmu.cat_transform.out_features == seq_dim
        # gate: (seq_dim + cat_dim) -> seq_dim
        assert gmu.gate.in_features == seq_dim + cat_dim
        assert gmu.gate.out_features == seq_dim

    # Forward pass shape tests
    def test_forward_output_shape(self):
        """Test GMU output has correct shape [batch_size, seq_dim]."""
        gmu = GMU(seq_dim=128, cat_dim=32)
        h_seq = torch.randn(16, 128)
        h_cat = torch.randn(16, 32)
        output, _ = gmu(h_seq, h_cat)
        assert output.shape == (16, 128)

    def test_forward_batch_sizes(self):
        """Test GMU works with various batch sizes."""
        gmu = GMU(seq_dim=64, cat_dim=16)
        for batch_size in [1, 4, 16, 32, 128]:
            h_seq = torch.randn(batch_size, 64)
            h_cat = torch.randn(batch_size, 16)
            output, _ = gmu(h_seq, h_cat)
            assert output.shape == (batch_size, 64)

    def test_forward_single_sample(self):
        """Test GMU works with batch_size=1."""
        gmu = GMU(seq_dim=128, cat_dim=32)
        h_seq = torch.randn(1, 128)
        h_cat = torch.randn(1, 32)
        output, _ = gmu(h_seq, h_cat)
        assert output.shape == (1, 128)

    # Gate behavior tests
    def test_forward_return_gate(self):
        """Test return_gate=True returns gate values."""
        gmu = GMU(seq_dim=128, cat_dim=32)
        h_seq = torch.randn(8, 128)
        h_cat = torch.randn(8, 32)
        output, gate = gmu(h_seq, h_cat, return_gate=True)
        assert gate is not None
        assert gate.shape == (8, 128)

    def test_forward_no_gate(self):
        """Test return_gate=False returns None for gate."""
        gmu = GMU(seq_dim=128, cat_dim=32)
        h_seq = torch.randn(8, 128)
        h_cat = torch.randn(8, 32)
        output, gate = gmu(h_seq, h_cat, return_gate=False)
        assert gate is None

    def test_gate_shape(self):
        """Test gate values have correct shape [batch_size, seq_dim]."""
        gmu = GMU(seq_dim=256, cat_dim=64)
        h_seq = torch.randn(12, 256)
        h_cat = torch.randn(12, 64)
        _, gate = gmu(h_seq, h_cat, return_gate=True)
        assert gate.shape == (12, 256)

    def test_gate_values_bounded(self):
        """Test gate values are in [0, 1] range (sigmoid output)."""
        gmu = GMU(seq_dim=128, cat_dim=32)
        h_seq = torch.randn(100, 128) * 10  # Large values
        h_cat = torch.randn(100, 32) * 10
        _, gate = gmu(h_seq, h_cat, return_gate=True)
        assert gate.min() >= 0.0
        assert gate.max() <= 1.0

    # Gradient tests
    def test_backward_pass(self):
        """Test GMU supports backpropagation."""
        gmu = GMU(seq_dim=64, cat_dim=16)
        h_seq = torch.randn(8, 64, requires_grad=True)
        h_cat = torch.randn(8, 16, requires_grad=True)
        output, _ = gmu(h_seq, h_cat)
        loss = output.sum()
        loss.backward()
        # Check that gradients exist for GMU parameters
        assert gmu.cat_transform.weight.grad is not None
        assert gmu.gate.weight.grad is not None

    def test_gradients_flow_to_inputs(self):
        """Test gradients flow to both sequence and categorical inputs."""
        gmu = GMU(seq_dim=64, cat_dim=16)
        h_seq = torch.randn(8, 64, requires_grad=True)
        h_cat = torch.randn(8, 16, requires_grad=True)
        output, _ = gmu(h_seq, h_cat)
        loss = output.sum()
        loss.backward()
        assert h_seq.grad is not None
        assert h_cat.grad is not None
        assert h_seq.grad.shape == h_seq.shape
        assert h_cat.grad.shape == h_cat.shape

    # Device tests
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_if_available(self):
        """Test GMU works on CUDA if available."""
        gmu = GMU(seq_dim=128, cat_dim=32).cuda()
        h_seq = torch.randn(8, 128).cuda()
        h_cat = torch.randn(8, 32).cuda()
        output, gate = gmu(h_seq, h_cat, return_gate=True)
        assert output.device.type == "cuda"
        assert gate.device.type == "cuda"

    def test_device_consistency(self):
        """Test output is on same device as inputs."""
        gmu = GMU(seq_dim=64, cat_dim=16)
        h_seq = torch.randn(4, 64)
        h_cat = torch.randn(4, 16)
        output, _ = gmu(h_seq, h_cat)
        assert output.device == h_seq.device

    # Edge cases
    def test_zero_categorical_input(self):
        """Test GMU handles zero categorical embeddings."""
        gmu = GMU(seq_dim=128, cat_dim=32)
        h_seq = torch.randn(8, 128)
        h_cat = torch.zeros(8, 32)
        output, gate = gmu(h_seq, h_cat, return_gate=True)
        assert output.shape == (8, 128)
        assert not torch.isnan(output).any()

    def test_zero_sequence_input(self):
        """Test GMU handles zero sequence embeddings."""
        gmu = GMU(seq_dim=128, cat_dim=32)
        h_seq = torch.zeros(8, 128)
        h_cat = torch.randn(8, 32)
        output, gate = gmu(h_seq, h_cat, return_gate=True)
        assert output.shape == (8, 128)
        assert not torch.isnan(output).any()

    # Determinism tests
    def test_deterministic_output(self):
        """Test same inputs produce same outputs."""
        gmu = GMU(seq_dim=64, cat_dim=16)
        gmu.eval()
        h_seq = torch.randn(8, 64)
        h_cat = torch.randn(8, 16)
        output1, gate1 = gmu(h_seq, h_cat, return_gate=True)
        output2, gate2 = gmu(h_seq, h_cat, return_gate=True)
        assert torch.allclose(output1, output2)
        assert torch.allclose(gate1, gate2)

    # Weight initialization tests
    def test_weights_initialized(self):
        """Test that weights are properly initialized (not all zeros)."""
        gmu = GMU(seq_dim=128, cat_dim=32)
        assert gmu.cat_transform.weight.abs().sum() > 0
        assert gmu.gate.weight.abs().sum() > 0

    # Parameter count test
    def test_parameter_count(self):
        """Test GMU has expected number of parameters."""
        seq_dim, cat_dim = 128, 32
        gmu = GMU(seq_dim=seq_dim, cat_dim=cat_dim)
        # cat_transform: cat_dim * seq_dim + seq_dim (bias)
        # gate: (seq_dim + cat_dim) * seq_dim + seq_dim (bias)
        expected = (
            cat_dim * seq_dim
            + seq_dim  # cat_transform
            + (seq_dim + cat_dim) * seq_dim
            + seq_dim  # gate
        )
        actual = sum(p.numel() for p in gmu.parameters())
        assert actual == expected


class TestCrossAttentionFusion:
    """Tests for Cross-Attention Fusion module."""

    # Initialization tests
    def test_init_basic(self):
        """Test CrossAttentionFusion initializes with valid dimensions."""
        fusion = CrossAttentionFusion(seq_dim=128, cat_dim=32)
        assert fusion.seq_dim == 128
        assert fusion.cat_dim == 32
        assert fusion.num_heads == 8  # default

    def test_init_custom_heads(self):
        """Test CrossAttentionFusion with custom number of heads."""
        fusion = CrossAttentionFusion(seq_dim=128, cat_dim=32, num_heads=4)
        assert fusion.num_heads == 4

    def test_init_asymmetric_dims(self):
        """Test CrossAttentionFusion with different seq_dim and cat_dim."""
        fusion = CrossAttentionFusion(seq_dim=256, cat_dim=64)
        assert fusion.seq_dim == 256
        assert fusion.cat_dim == 64

    def test_init_large_dims(self):
        """Test CrossAttentionFusion with large dimensions."""
        fusion = CrossAttentionFusion(seq_dim=1024, cat_dim=512, num_heads=16)
        assert fusion.seq_dim == 1024
        assert fusion.cat_dim == 512

    def test_init_has_expected_layers(self):
        """Test CrossAttentionFusion has expected layers."""
        fusion = CrossAttentionFusion(seq_dim=128, cat_dim=32)
        assert hasattr(fusion, "cat_projection")
        assert hasattr(fusion, "cross_attn")
        assert hasattr(fusion, "norm")
        assert isinstance(fusion.cat_projection, torch.nn.Linear)
        assert isinstance(fusion.cross_attn, torch.nn.MultiheadAttention)
        assert isinstance(fusion.norm, torch.nn.LayerNorm)

    def test_init_layer_dimensions(self):
        """Test layer dimensions are correct."""
        seq_dim, cat_dim = 128, 32
        fusion = CrossAttentionFusion(seq_dim=seq_dim, cat_dim=cat_dim)
        # cat_projection: cat_dim -> seq_dim
        assert fusion.cat_projection.in_features == cat_dim
        assert fusion.cat_projection.out_features == seq_dim
        # norm: seq_dim
        assert fusion.norm.normalized_shape == (seq_dim,)

    # Forward pass shape tests
    def test_forward_output_shape(self):
        """Test CrossAttentionFusion output has correct shape [B, S, D]."""
        fusion = CrossAttentionFusion(seq_dim=128, cat_dim=32)
        seq_repr = torch.randn(8, 16, 128)  # [B, S, D]
        cat_emb = torch.randn(8, 32)  # [B, cat_dim]
        output = fusion(seq_repr, cat_emb)
        assert output.shape == (8, 16, 128)

    def test_forward_batch_sizes(self):
        """Test CrossAttentionFusion works with various batch sizes."""
        fusion = CrossAttentionFusion(seq_dim=64, cat_dim=16)
        for batch_size in [1, 4, 16, 32]:
            seq_repr = torch.randn(batch_size, 10, 64)
            cat_emb = torch.randn(batch_size, 16)
            output = fusion(seq_repr, cat_emb)
            assert output.shape == (batch_size, 10, 64)

    def test_forward_sequence_lengths(self):
        """Test CrossAttentionFusion works with various sequence lengths."""
        fusion = CrossAttentionFusion(seq_dim=64, cat_dim=16)
        for seq_len in [1, 5, 20, 100]:
            seq_repr = torch.randn(4, seq_len, 64)
            cat_emb = torch.randn(4, 16)
            output = fusion(seq_repr, cat_emb)
            assert output.shape == (4, seq_len, 64)

    def test_forward_single_sample(self):
        """Test CrossAttentionFusion works with batch_size=1."""
        fusion = CrossAttentionFusion(seq_dim=128, cat_dim=32)
        seq_repr = torch.randn(1, 8, 128)
        cat_emb = torch.randn(1, 32)
        output = fusion(seq_repr, cat_emb)
        assert output.shape == (1, 8, 128)

    # Attention weight behavior tests
    def test_forward_return_weights(self):
        """Test return_weights=True returns attention weights."""
        fusion = CrossAttentionFusion(seq_dim=128, cat_dim=32, num_heads=8)
        seq_repr = torch.randn(4, 16, 128)
        cat_emb = torch.randn(4, 32)
        output, weights = fusion(seq_repr, cat_emb, return_weights=True)
        assert weights is not None
        # Weights shape: [B, num_heads, seq_len, num_metadata_tokens]
        # With single cat_emb projected, num_metadata_tokens=1
        assert weights.shape == (4, 8, 16, 1)

    def test_forward_no_weights(self):
        """Test return_weights=False returns tensor (not tuple)."""
        fusion = CrossAttentionFusion(seq_dim=128, cat_dim=32)
        seq_repr = torch.randn(4, 8, 128)
        cat_emb = torch.randn(4, 32)
        output = fusion(seq_repr, cat_emb, return_weights=False)
        # Should be tensor, not tuple
        assert isinstance(output, torch.Tensor)
        assert output.shape == (4, 8, 128)

    def test_attention_weights_shape(self):
        """Test attention weights have correct shape."""
        fusion = CrossAttentionFusion(seq_dim=64, cat_dim=16, num_heads=4)
        seq_repr = torch.randn(8, 20, 64)
        cat_emb = torch.randn(8, 16)
        _, weights = fusion(seq_repr, cat_emb, return_weights=True)
        # [B, num_heads, seq_len, num_metadata_tokens=1]
        assert weights.shape == (8, 4, 20, 1)

    def test_attention_weights_sum_to_one(self):
        """Test attention weights sum to 1 across metadata dimension."""
        fusion = CrossAttentionFusion(seq_dim=64, cat_dim=16, num_heads=4, dropout=0.0)
        fusion.eval()
        seq_repr = torch.randn(4, 10, 64)
        cat_emb = torch.randn(4, 16)
        _, weights = fusion(seq_repr, cat_emb, return_weights=True)
        # With single metadata token, weights should be 1.0
        assert torch.allclose(weights.sum(dim=-1), torch.ones(4, 4, 10))

    # Gradient tests
    def test_backward_pass(self):
        """Test CrossAttentionFusion supports backpropagation."""
        fusion = CrossAttentionFusion(seq_dim=64, cat_dim=16)
        seq_repr = torch.randn(4, 8, 64, requires_grad=True)
        cat_emb = torch.randn(4, 16, requires_grad=True)
        output = fusion(seq_repr, cat_emb)
        loss = output.sum()
        loss.backward()
        # Check gradients exist for module parameters
        assert fusion.cat_projection.weight.grad is not None

    def test_gradients_flow_to_inputs(self):
        """Test gradients flow to both sequence and categorical inputs."""
        fusion = CrossAttentionFusion(seq_dim=64, cat_dim=16)
        seq_repr = torch.randn(4, 8, 64, requires_grad=True)
        cat_emb = torch.randn(4, 16, requires_grad=True)
        output = fusion(seq_repr, cat_emb)
        loss = output.sum()
        loss.backward()
        assert seq_repr.grad is not None
        assert cat_emb.grad is not None
        assert seq_repr.grad.shape == seq_repr.shape
        assert cat_emb.grad.shape == cat_emb.shape

    # Device tests
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_if_available(self):
        """Test CrossAttentionFusion works on CUDA if available."""
        fusion = CrossAttentionFusion(seq_dim=128, cat_dim=32).cuda()
        seq_repr = torch.randn(4, 8, 128).cuda()
        cat_emb = torch.randn(4, 32).cuda()
        output, weights = fusion(seq_repr, cat_emb, return_weights=True)
        assert output.device.type == "cuda"
        assert weights.device.type == "cuda"

    def test_device_consistency(self):
        """Test output is on same device as inputs."""
        fusion = CrossAttentionFusion(seq_dim=64, cat_dim=16)
        seq_repr = torch.randn(4, 8, 64)
        cat_emb = torch.randn(4, 16)
        output = fusion(seq_repr, cat_emb)
        assert output.device == seq_repr.device

    # Edge cases
    def test_zero_categorical_input(self):
        """Test CrossAttentionFusion handles zero categorical embeddings."""
        fusion = CrossAttentionFusion(seq_dim=128, cat_dim=32)
        seq_repr = torch.randn(4, 8, 128)
        cat_emb = torch.zeros(4, 32)
        output = fusion(seq_repr, cat_emb)
        assert output.shape == (4, 8, 128)
        assert not torch.isnan(output).any()

    def test_zero_sequence_input(self):
        """Test CrossAttentionFusion handles zero sequence embeddings."""
        fusion = CrossAttentionFusion(seq_dim=128, cat_dim=32)
        seq_repr = torch.zeros(4, 8, 128)
        cat_emb = torch.randn(4, 32)
        output = fusion(seq_repr, cat_emb)
        assert output.shape == (4, 8, 128)
        assert not torch.isnan(output).any()

    def test_single_position(self):
        """Test CrossAttentionFusion with single sequence position."""
        fusion = CrossAttentionFusion(seq_dim=64, cat_dim=16)
        seq_repr = torch.randn(4, 1, 64)
        cat_emb = torch.randn(4, 16)
        output, weights = fusion(seq_repr, cat_emb, return_weights=True)
        assert output.shape == (4, 1, 64)
        assert weights.shape == (4, 8, 1, 1)

    # Determinism tests
    def test_deterministic_output(self):
        """Test same inputs produce same outputs in eval mode."""
        fusion = CrossAttentionFusion(seq_dim=64, cat_dim=16, dropout=0.0)
        fusion.eval()
        seq_repr = torch.randn(4, 8, 64)
        cat_emb = torch.randn(4, 16)
        output1, weights1 = fusion(seq_repr, cat_emb, return_weights=True)
        output2, weights2 = fusion(seq_repr, cat_emb, return_weights=True)
        assert torch.allclose(output1, output2)
        assert torch.allclose(weights1, weights2)

    # Weight initialization tests
    def test_weights_initialized(self):
        """Test that weights are properly initialized (not all zeros)."""
        fusion = CrossAttentionFusion(seq_dim=128, cat_dim=32)
        assert fusion.cat_projection.weight.abs().sum() > 0

    # Parameter count test
    def test_parameter_count(self):
        """Test CrossAttentionFusion has expected number of parameters."""
        seq_dim, cat_dim, num_heads = 128, 32, 8
        fusion = CrossAttentionFusion(seq_dim=seq_dim, cat_dim=cat_dim, num_heads=num_heads)
        # cat_projection: cat_dim * seq_dim + seq_dim (bias)
        # cross_attn: in_proj_weight (3 * seq_dim * seq_dim) + in_proj_bias (3 * seq_dim)
        #            + out_proj.weight (seq_dim * seq_dim) + out_proj.bias (seq_dim)
        # norm: weight (seq_dim) + bias (seq_dim)
        cat_proj_params = cat_dim * seq_dim + seq_dim
        # MultiheadAttention with embed_dim=seq_dim: 4 * seq_dim^2 + 4 * seq_dim
        attn_params = 4 * seq_dim * seq_dim + 4 * seq_dim
        norm_params = 2 * seq_dim
        expected = cat_proj_params + attn_params + norm_params
        actual = sum(p.numel() for p in fusion.parameters())
        assert actual == expected

    # Residual connection test
    def test_residual_connection(self):
        """Test that output includes residual connection from input."""
        fusion = CrossAttentionFusion(seq_dim=64, cat_dim=16, dropout=0.0)
        fusion.eval()
        # Zero categorical embedding should result in output close to input
        # (since attention output will be zeros after zero values)
        seq_repr = torch.randn(4, 8, 64)
        cat_emb = torch.zeros(4, 16)
        output = fusion(seq_repr, cat_emb)
        # Output should be normalized version of input (LayerNorm applied to residual)
        # Check that output is still correlated with input
        correlation = torch.corrcoef(
            torch.stack([seq_repr.flatten(), output.flatten()])
        )[0, 1]
        assert correlation > 0.5  # Should be positively correlated

    # Dropout test
    def test_dropout_effect(self):
        """Test that dropout has effect in training mode."""
        fusion = CrossAttentionFusion(seq_dim=64, cat_dim=16, dropout=0.5)
        fusion.train()
        seq_repr = torch.randn(4, 8, 64)
        cat_emb = torch.randn(4, 16)
        torch.manual_seed(42)
        output1 = fusion(seq_repr, cat_emb)
        torch.manual_seed(43)
        output2 = fusion(seq_repr, cat_emb)
        # Outputs should differ due to dropout
        assert not torch.allclose(output1, output2)
