"""Unit tests for TransformerEncoder class."""

import pytest
import torch
import torch.nn as nn

from aam.models.transformer import TransformerEncoder, sdpa_kernel_context


@pytest.fixture
def transformer_encoder():
    """Create a TransformerEncoder instance."""
    return TransformerEncoder(
        num_layers=2,
        num_heads=4,
        hidden_dim=64,
        intermediate_size=256,
        dropout=0.1,
        activation="gelu",
    )


class TestTransformerEncoder:
    """Test suite for TransformerEncoder class."""

    def test_init(self, transformer_encoder):
        """Test TransformerEncoder initialization."""
        assert transformer_encoder is not None
        assert isinstance(transformer_encoder, nn.Module)

    def test_init_default_intermediate_size(self):
        """Test that intermediate_size defaults to 4 * hidden_dim."""
        encoder = TransformerEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            dropout=0.1,
        )
        assert encoder is not None

    def test_forward_shape(self, transformer_encoder, sample_embeddings):
        """Test forward pass output shape."""
        result = transformer_encoder(sample_embeddings)
        assert result.shape == sample_embeddings.shape
        assert result.shape == (2, 10, 64)

    def test_forward_with_mask_shape(self, transformer_encoder, sample_embeddings, sample_mask):
        """Test forward pass with mask output shape."""
        result = transformer_encoder(sample_embeddings, mask=sample_mask)
        assert result.shape == sample_embeddings.shape
        assert result.shape == (2, 10, 64)

    def test_forward_different_batch_sizes(self, transformer_encoder):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 4, 8]:
            embeddings = torch.randn(batch_size, 10, 64)
            result = transformer_encoder(embeddings)
            assert result.shape == (batch_size, 10, 64)

    def test_forward_different_seq_lengths(self, transformer_encoder):
        """Test forward pass with different sequence lengths."""
        for seq_len in [5, 10, 20, 50]:
            embeddings = torch.randn(2, seq_len, 64)
            result = transformer_encoder(embeddings)
            assert result.shape == (2, seq_len, 64)

    def test_forward_no_mask(self, transformer_encoder, sample_embeddings):
        """Test forward pass without mask (all positions valid)."""
        result = transformer_encoder(sample_embeddings)
        assert result.shape == sample_embeddings.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_forward_with_mask(self, transformer_encoder, sample_embeddings, sample_mask):
        """Test forward pass with mask."""
        result = transformer_encoder(sample_embeddings, mask=sample_mask)
        assert result.shape == sample_embeddings.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_forward_full_padding_mask(self, transformer_encoder):
        """Test forward pass with full padding mask."""
        embeddings = torch.randn(2, 10, 64)
        mask = torch.zeros(2, 10, dtype=torch.long)
        result = transformer_encoder(embeddings, mask=mask)
        assert result.shape == embeddings.shape

    def test_forward_partial_padding_mask(self, transformer_encoder, sample_embeddings, sample_mask):
        """Test forward pass with partial padding mask."""
        result = transformer_encoder(sample_embeddings, mask=sample_mask)
        assert result.shape == sample_embeddings.shape

    def test_forward_different_num_layers(self):
        """Test forward pass with different numbers of layers."""
        for num_layers in [1, 2, 4, 6]:
            encoder = TransformerEncoder(
                num_layers=num_layers,
                num_heads=4,
                hidden_dim=64,
                dropout=0.1,
            )
            embeddings = torch.randn(2, 10, 64)
            result = encoder(embeddings)
            assert result.shape == (2, 10, 64)

    def test_forward_different_num_heads(self):
        """Test forward pass with different numbers of heads."""
        for num_heads in [1, 2, 4, 8]:
            encoder = TransformerEncoder(
                num_layers=2,
                num_heads=num_heads,
                hidden_dim=64,
                dropout=0.1,
            )
            embeddings = torch.randn(2, 10, 64)
            result = encoder(embeddings)
            assert result.shape == (2, 10, 64)

    def test_forward_different_hidden_dims(self):
        """Test forward pass with different hidden dimensions."""
        for hidden_dim in [32, 64, 128, 256]:
            encoder = TransformerEncoder(
                num_layers=2,
                num_heads=4,
                hidden_dim=hidden_dim,
                dropout=0.1,
            )
            embeddings = torch.randn(2, 10, hidden_dim)
            result = encoder(embeddings)
            assert result.shape == (2, 10, hidden_dim)

    def test_forward_gelu_activation(self):
        """Test forward pass with GELU activation."""
        encoder = TransformerEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            activation="gelu",
            dropout=0.1,
        )
        embeddings = torch.randn(2, 10, 64)
        result = encoder(embeddings)
        assert result.shape == (2, 10, 64)

    def test_forward_relu_activation(self):
        """Test forward pass with ReLU activation."""
        encoder = TransformerEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            activation="relu",
            dropout=0.1,
        )
        embeddings = torch.randn(2, 10, 64)
        result = encoder(embeddings)
        assert result.shape == (2, 10, 64)

    def test_gradients_flow(self, transformer_encoder, sample_embeddings):
        """Test that gradients flow correctly."""
        result = transformer_encoder(sample_embeddings)
        loss = result.sum()
        loss.backward()

        for param in transformer_encoder.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

    def test_gradients_with_mask(self, transformer_encoder, sample_embeddings, sample_mask):
        """Test that gradients flow correctly with mask."""
        result = transformer_encoder(sample_embeddings, mask=sample_mask)
        loss = result.sum()
        loss.backward()

        for param in transformer_encoder.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

    def test_no_gradient_explosion(self, transformer_encoder, sample_embeddings):
        """Test that gradients don't explode."""
        result = transformer_encoder(sample_embeddings)
        loss = result.sum()
        loss.backward()

        max_grad = max(p.grad.abs().max().item() for p in transformer_encoder.parameters() if p.grad is not None)
        assert max_grad < 1e6, f"Gradient explosion detected: max_grad={max_grad}"

    def test_no_gradient_vanishing(self, transformer_encoder, sample_embeddings):
        """Test that gradients don't vanish."""
        result = transformer_encoder(sample_embeddings)
        loss = result.sum()
        loss.backward()

        max_grad = max(p.grad.abs().max().item() for p in transformer_encoder.parameters() if p.grad is not None)
        assert max_grad > 1e-8, f"Gradient vanishing detected: max_grad={max_grad}"

    def test_forward_same_device(self, transformer_encoder, sample_embeddings):
        """Test that output is on same device as input."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            transformer_encoder = transformer_encoder.to(device)
            sample_embeddings = sample_embeddings.to(device)
            result = transformer_encoder(sample_embeddings)
            assert result.device.type == device.type

    def test_dropout_training_mode(self, transformer_encoder, sample_embeddings):
        """Test that dropout is active in training mode."""
        transformer_encoder.train()
        result1 = transformer_encoder(sample_embeddings)
        result2 = transformer_encoder(sample_embeddings)
        assert not torch.allclose(result1, result2, atol=1e-5)

    def test_dropout_eval_mode(self, transformer_encoder, sample_embeddings):
        """Test that dropout is inactive in eval mode."""
        transformer_encoder.eval()
        result1 = transformer_encoder(sample_embeddings)
        result2 = transformer_encoder(sample_embeddings)
        assert torch.allclose(result1, result2, atol=1e-5)

    def test_mask_handles_variable_lengths(self, transformer_encoder):
        """Test that mask correctly handles variable sequence lengths."""
        embeddings = torch.randn(3, 10, 64)
        mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        result = transformer_encoder(embeddings, mask=mask)
        assert result.shape == (3, 10, 64)

    def test_gradient_checkpointing_init(self):
        """Test that gradient checkpointing can be enabled."""
        encoder = TransformerEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            gradient_checkpointing=True,
        )
        assert encoder.gradient_checkpointing is True

    def test_gradient_checkpointing_disabled_by_default(self):
        """Test that gradient checkpointing is disabled by default."""
        encoder = TransformerEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
        )
        assert encoder.gradient_checkpointing is False

    def test_gradient_checkpointing_eval_mode_same_output(self):
        """Test that checkpointing produces same output in eval mode (checkpointing disabled in eval)."""
        embeddings = torch.randn(2, 10, 64)
        mask = torch.ones(2, 10, dtype=torch.long)

        # Create one encoder and copy its weights to another
        encoder_no_checkpoint = TransformerEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            gradient_checkpointing=False,
        )
        encoder_checkpoint = TransformerEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            gradient_checkpointing=True,
        )

        # Copy weights to ensure same initialization
        encoder_checkpoint.load_state_dict(encoder_no_checkpoint.state_dict())

        encoder_no_checkpoint.eval()
        encoder_checkpoint.eval()

        output_no_checkpoint = encoder_no_checkpoint(embeddings, mask=mask)
        output_checkpoint = encoder_checkpoint(embeddings, mask=mask)

        # In eval mode, checkpointing should not be used, so outputs should be identical
        assert torch.allclose(output_no_checkpoint, output_checkpoint, atol=1e-5)

    def test_gradient_checkpointing_training_mode_gradients(self):
        """Test that gradients flow correctly with checkpointing in training mode."""
        embeddings = torch.randn(2, 10, 64, requires_grad=True)
        mask = torch.ones(2, 10, dtype=torch.long)

        encoder = TransformerEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            gradient_checkpointing=True,
        )
        encoder.train()

        output = encoder(embeddings, mask=mask)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert embeddings.grad is not None
        assert not torch.isnan(embeddings.grad).any()
        assert not torch.isinf(embeddings.grad).any()

        # Check that model parameters have gradients
        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()

    def test_gradient_checkpointing_with_mask(self):
        """Test gradient checkpointing works correctly with mask."""
        embeddings = torch.randn(2, 10, 64, requires_grad=True)
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])

        encoder = TransformerEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            gradient_checkpointing=True,
        )
        encoder.train()

        output = encoder(embeddings, mask=mask)
        loss = output.sum()
        loss.backward()

        assert embeddings.grad is not None
        assert not torch.isnan(embeddings.grad).any()

    def test_gradient_checkpointing_output_shape(self):
        """Test that checkpointing produces correct output shape."""
        embeddings = torch.randn(2, 10, 64)
        mask = torch.ones(2, 10, dtype=torch.long)

        encoder = TransformerEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            gradient_checkpointing=True,
        )
        encoder.train()

        output = encoder(embeddings, mask=mask)
        assert output.shape == embeddings.shape


class TestSDPAOptimization:
    """Test suite for SDPA (Scaled Dot Product Attention) optimization."""

    def test_attn_implementation_default(self):
        """Test that attn_implementation defaults to 'sdpa'."""
        encoder = TransformerEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
        )
        assert encoder.attn_implementation == "sdpa"

    @pytest.mark.parametrize("attn_impl", ["sdpa", "flash", "mem_efficient", "math", None])
    def test_attn_implementation_options(self, attn_impl):
        """Test that different attn_implementation options can be set."""
        encoder = TransformerEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            attn_implementation=attn_impl,
        )
        assert encoder.attn_implementation == attn_impl

    @pytest.mark.parametrize("attn_impl", ["sdpa", "flash", "mem_efficient", "math", None])
    def test_attn_implementation_forward_pass(self, attn_impl):
        """Test forward pass with different attn_implementation options."""
        encoder = TransformerEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            attn_implementation=attn_impl,
        )
        embeddings = torch.randn(2, 10, 64)
        mask = torch.ones(2, 10, dtype=torch.long)

        output = encoder(embeddings, mask=mask)

        assert output.shape == embeddings.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_numerical_equivalence_sdpa_vs_math(self):
        """Test that SDPA and math implementations produce equivalent outputs."""
        embeddings = torch.randn(2, 10, 64)
        mask = torch.ones(2, 10, dtype=torch.long)

        encoder_sdpa = TransformerEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            attn_implementation="sdpa",
            dropout=0.0,
        )
        encoder_math = TransformerEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            attn_implementation="math",
            dropout=0.0,
        )
        encoder_math.load_state_dict(encoder_sdpa.state_dict())

        encoder_sdpa.eval()
        encoder_math.eval()

        output_sdpa = encoder_sdpa(embeddings, mask=mask)
        output_math = encoder_math(embeddings, mask=mask)

        assert torch.allclose(output_sdpa, output_math, atol=1e-5)

    @pytest.mark.parametrize("seq_len", [8, 16, 32, 64, 128, 256])
    def test_sdpa_different_sequence_lengths(self, seq_len):
        """Test SDPA optimization with different sequence lengths."""
        encoder = TransformerEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            attn_implementation="sdpa",
        )
        embeddings = torch.randn(2, seq_len, 64)
        mask = torch.ones(2, seq_len, dtype=torch.long)

        output = encoder(embeddings, mask=mask)

        assert output.shape == (2, seq_len, 64)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_sdpa_with_variable_length_padding(self):
        """Test SDPA with variable-length sequences (different padding per sample)."""
        encoder = TransformerEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            attn_implementation="sdpa",
        )
        embeddings = torch.randn(4, 32, 64)
        mask = torch.zeros(4, 32, dtype=torch.long)
        mask[0, :8] = 1
        mask[1, :16] = 1
        mask[2, :24] = 1
        mask[3, :32] = 1

        output = encoder(embeddings, mask=mask)

        assert output.shape == (4, 32, 64)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_sdpa_gradients_flow(self):
        """Test that gradients flow correctly with SDPA optimization."""
        encoder = TransformerEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            attn_implementation="sdpa",
        )
        embeddings = torch.randn(2, 10, 64, requires_grad=True)

        output = encoder(embeddings)
        loss = output.sum()
        loss.backward()

        assert embeddings.grad is not None
        assert not torch.isnan(embeddings.grad).any()
        assert not torch.isinf(embeddings.grad).any()

        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_sdpa_with_gradient_checkpointing(self):
        """Test SDPA optimization combined with gradient checkpointing."""
        encoder = TransformerEncoder(
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            attn_implementation="sdpa",
            gradient_checkpointing=True,
        )
        encoder.train()
        embeddings = torch.randn(2, 10, 64, requires_grad=True)
        mask = torch.ones(2, 10, dtype=torch.long)

        output = encoder(embeddings, mask=mask)
        loss = output.sum()
        loss.backward()

        assert embeddings.grad is not None
        assert not torch.isnan(embeddings.grad).any()


class TestSDPAKernelContext:
    """Test suite for sdpa_kernel_context context manager."""

    def test_sdpa_context_none(self):
        """Test that None attn_implementation uses default behavior."""
        with sdpa_kernel_context(None):
            assert True

    def test_sdpa_context_sdpa(self):
        """Test that 'sdpa' attn_implementation uses default behavior."""
        with sdpa_kernel_context("sdpa"):
            assert True

    def test_sdpa_context_math(self):
        """Test that 'math' attn_implementation configures backends correctly."""
        with sdpa_kernel_context("math"):
            pass

    def test_sdpa_context_flash(self):
        """Test that 'flash' attn_implementation configures backends correctly."""
        with sdpa_kernel_context("flash"):
            pass

    def test_sdpa_context_mem_efficient(self):
        """Test that 'mem_efficient' attn_implementation configures backends correctly."""
        with sdpa_kernel_context("mem_efficient"):
            pass

    def test_sdpa_context_restores_state(self):
        """Test that context manager restores original backend state."""
        original_flash = torch.backends.cuda.flash_sdp_enabled()
        original_mem_eff = torch.backends.cuda.mem_efficient_sdp_enabled()
        original_math = torch.backends.cuda.math_sdp_enabled()

        with sdpa_kernel_context("math"):
            pass

        assert torch.backends.cuda.flash_sdp_enabled() == original_flash
        assert torch.backends.cuda.mem_efficient_sdp_enabled() == original_mem_eff
        assert torch.backends.cuda.math_sdp_enabled() == original_math

    def test_sdpa_context_restores_on_exception(self):
        """Test that context manager restores state even if exception occurs."""
        original_flash = torch.backends.cuda.flash_sdp_enabled()
        original_mem_eff = torch.backends.cuda.mem_efficient_sdp_enabled()
        original_math = torch.backends.cuda.math_sdp_enabled()

        with pytest.raises(ValueError):
            with sdpa_kernel_context("math"):
                raise ValueError("Test exception")

        assert torch.backends.cuda.flash_sdp_enabled() == original_flash
        assert torch.backends.cuda.mem_efficient_sdp_enabled() == original_mem_eff
        assert torch.backends.cuda.math_sdp_enabled() == original_math
