"""Tests for multimodal fusion modules."""

import pytest
import torch

from aam.models.fusion import GMU


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
        """Test GMU has seq_transform, cat_transform, and gate layers."""
        gmu = GMU(seq_dim=128, cat_dim=32)
        assert hasattr(gmu, "seq_transform")
        assert hasattr(gmu, "cat_transform")
        assert hasattr(gmu, "gate")
        assert isinstance(gmu.seq_transform, torch.nn.Linear)
        assert isinstance(gmu.cat_transform, torch.nn.Linear)
        assert isinstance(gmu.gate, torch.nn.Linear)

    def test_init_layer_dimensions(self):
        """Test layer dimensions are correct."""
        seq_dim, cat_dim = 128, 32
        gmu = GMU(seq_dim=seq_dim, cat_dim=cat_dim)
        # seq_transform: seq_dim -> seq_dim
        assert gmu.seq_transform.in_features == seq_dim
        assert gmu.seq_transform.out_features == seq_dim
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
        assert gmu.seq_transform.weight.grad is not None
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
        assert gmu.seq_transform.weight.abs().sum() > 0
        assert gmu.cat_transform.weight.abs().sum() > 0
        assert gmu.gate.weight.abs().sum() > 0

    # Parameter count test
    def test_parameter_count(self):
        """Test GMU has expected number of parameters."""
        seq_dim, cat_dim = 128, 32
        gmu = GMU(seq_dim=seq_dim, cat_dim=cat_dim)
        # seq_transform: seq_dim * seq_dim + seq_dim (bias)
        # cat_transform: cat_dim * seq_dim + seq_dim (bias)
        # gate: (seq_dim + cat_dim) * seq_dim + seq_dim (bias)
        expected = (
            seq_dim * seq_dim + seq_dim  # seq_transform
            + cat_dim * seq_dim + seq_dim  # cat_transform
            + (seq_dim + cat_dim) * seq_dim + seq_dim  # gate
        )
        actual = sum(p.numel() for p in gmu.parameters())
        assert actual == expected
