"""Tests for cross-platform numerical validation."""

import json
import pytest
import torch
from pathlib import Path

from tests.validation.numerical_validation import (
    ValidationConfig,
    ValidationResult,
    generate_golden_outputs,
    compare_golden_outputs,
    run_validation,
    _create_model,
    _create_inputs,
    _get_platform_info,
)


class TestValidationConfig:
    """Tests for ValidationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ValidationConfig()
        assert config.seed == 42
        assert config.batch_size == 2
        assert config.num_asvs == 8
        assert config.seq_len == 150
        assert config.embedding_dim == 64
        assert config.vocab_size == 6
        assert config.out_dim == 1
        assert config.atol == 1e-5
        assert config.rtol == 1e-4

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ValidationConfig(seed=123, batch_size=4, atol=1e-6)
        assert config.seed == 123
        assert config.batch_size == 4
        assert config.atol == 1e-6

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = ValidationConfig(seed=99)
        data = config.to_dict()
        assert data["seed"] == 99
        assert "golden_dir" not in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {"seed": 99, "batch_size": 4, "atol": 1e-6}
        config = ValidationConfig.from_dict(data)
        assert config.seed == 99
        assert config.batch_size == 4
        assert config.atol == 1e-6

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        config = ValidationConfig(seed=99, batch_size=4)
        restored = ValidationConfig.from_dict(config.to_dict())
        assert restored.seed == config.seed
        assert restored.batch_size == config.batch_size


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_model(self):
        """Test model creation is deterministic."""
        config = ValidationConfig()
        device = torch.device("cpu")

        model1 = _create_model(config, device)
        model2 = _create_model(config, device)

        # Same seed should produce identical weights
        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2), f"Parameter {n1} differs"

    def test_create_inputs(self):
        """Test input creation is deterministic."""
        config = ValidationConfig()
        device = torch.device("cpu")

        inputs1 = _create_inputs(config, device)
        inputs2 = _create_inputs(config, device)

        assert "tokens" in inputs1
        assert torch.equal(inputs1["tokens"], inputs2["tokens"])

    def test_create_inputs_shape(self):
        """Test input tensor shapes match config."""
        config = ValidationConfig(batch_size=3, num_asvs=10, seq_len=100)
        device = torch.device("cpu")

        inputs = _create_inputs(config, device)
        tokens = inputs["tokens"]

        assert tokens.shape == (3, 10, 100)

    def test_get_platform_info(self):
        """Test platform info collection."""
        info = _get_platform_info()

        assert "torch_version" in info
        assert "platform" in info
        assert "device" in info


class TestGoldenOutputGeneration:
    """Tests for golden output generation."""

    def test_generate_creates_files(self, tmp_path):
        """Test that generate_golden_outputs creates expected files."""
        config = ValidationConfig(golden_dir=tmp_path)
        generate_golden_outputs(config=config, device=torch.device("cpu"))

        # Check expected files exist
        assert (tmp_path / "metadata.json").exists()
        assert (tmp_path / "forward_outputs.pt").exists()
        assert (tmp_path / "model_weights.pt").exists()

    def test_generate_metadata_content(self, tmp_path):
        """Test that metadata contains expected fields."""
        config = ValidationConfig(seed=123, golden_dir=tmp_path)
        generate_golden_outputs(config=config, device=torch.device("cpu"))

        metadata = json.loads((tmp_path / "metadata.json").read_text())

        assert metadata["config"]["seed"] == 123
        assert "platform_info" in metadata
        assert "torch_version" in metadata["platform_info"]

    def test_generate_outputs_content(self, tmp_path):
        """Test that forward outputs contain expected keys."""
        config = ValidationConfig(golden_dir=tmp_path)
        generate_golden_outputs(config=config, device=torch.device("cpu"))

        outputs = torch.load(tmp_path / "forward_outputs.pt", weights_only=True)

        assert "target_prediction" in outputs
        assert "count_prediction" in outputs
        assert "base_embeddings" in outputs

    def test_generate_deterministic(self, tmp_path):
        """Test that generation is deterministic across runs."""
        config = ValidationConfig(golden_dir=tmp_path)

        # Generate twice
        dir1 = tmp_path / "run1"
        dir1.mkdir()
        config1 = ValidationConfig(golden_dir=dir1)
        generate_golden_outputs(config=config1, device=torch.device("cpu"))

        dir2 = tmp_path / "run2"
        dir2.mkdir()
        config2 = ValidationConfig(golden_dir=dir2)
        generate_golden_outputs(config=config2, device=torch.device("cpu"))

        # Compare outputs
        outputs1 = torch.load(dir1 / "forward_outputs.pt", weights_only=True)
        outputs2 = torch.load(dir2 / "forward_outputs.pt", weights_only=True)

        for key in outputs1:
            assert torch.equal(outputs1[key], outputs2[key]), f"Output {key} differs"


class TestGoldenOutputComparison:
    """Tests for golden output comparison."""

    def test_compare_same_platform_passes(self, tmp_path):
        """Test that comparing on same platform passes."""
        config = ValidationConfig(golden_dir=tmp_path)
        device = torch.device("cpu")

        # Generate golden
        generate_golden_outputs(config=config, device=device)

        # Compare on same platform
        result = compare_golden_outputs(config=config, device=device, golden_dir=tmp_path)

        assert result.passed
        assert all(d["passed"] for d in result.details.values())

    def test_compare_detects_difference(self, tmp_path):
        """Test that comparison detects numerical differences."""
        config = ValidationConfig(golden_dir=tmp_path)
        device = torch.device("cpu")

        # Generate golden
        generate_golden_outputs(config=config, device=device)

        # Corrupt the golden file
        outputs = torch.load(tmp_path / "forward_outputs.pt", weights_only=True)
        outputs["target_prediction"] = outputs["target_prediction"] + 1.0
        torch.save(outputs, tmp_path / "forward_outputs.pt")

        # Compare should fail
        result = compare_golden_outputs(config=config, device=device, golden_dir=tmp_path)

        assert not result.passed
        assert not result.details["target_prediction"]["passed"]

    def test_compare_within_tolerance(self, tmp_path):
        """Test that small differences within tolerance pass."""
        config = ValidationConfig(golden_dir=tmp_path, atol=1e-3)
        device = torch.device("cpu")

        # Generate golden
        generate_golden_outputs(config=config, device=device)

        # Add small noise within tolerance
        outputs = torch.load(tmp_path / "forward_outputs.pt", weights_only=True)
        outputs["target_prediction"] = outputs["target_prediction"] + 1e-5
        torch.save(outputs, tmp_path / "forward_outputs.pt")

        # Compare should pass (noise is within tolerance)
        result = compare_golden_outputs(config=config, device=device, golden_dir=tmp_path)

        assert result.passed

    def test_compare_missing_golden_raises(self, tmp_path):
        """Test that missing golden files raises appropriate error."""
        config = ValidationConfig(golden_dir=tmp_path)

        with pytest.raises(FileNotFoundError):
            compare_golden_outputs(config=config, device=torch.device("cpu"), golden_dir=tmp_path)


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_summary_passed(self):
        """Test summary for passed validation."""
        result = ValidationResult(
            passed=True,
            details={
                "target_prediction": {"passed": True, "max_diff": 1e-7},
                "count_prediction": {"passed": True, "max_diff": 1e-8},
            },
            platform_info={"device": "cpu"},
        )

        summary = result.summary()
        assert "PASSED" in summary or "passed" in summary.lower()

    def test_summary_failed(self):
        """Test summary for failed validation."""
        result = ValidationResult(
            passed=False,
            details={
                "target_prediction": {"passed": False, "max_diff": 0.5},
                "count_prediction": {"passed": True, "max_diff": 1e-8},
            },
            platform_info={"device": "cpu"},
        )

        summary = result.summary()
        assert "FAILED" in summary or "failed" in summary.lower()
        assert "target_prediction" in summary


class TestRunValidation:
    """Tests for run_validation CLI wrapper."""

    def test_run_generate_mode(self, tmp_path):
        """Test running in generate mode."""
        config = ValidationConfig(golden_dir=tmp_path)
        run_validation(mode="generate", config=config, device=torch.device("cpu"), golden_dir=tmp_path)

        assert (tmp_path / "forward_outputs.pt").exists()

    def test_run_compare_mode(self, tmp_path):
        """Test running in compare mode."""
        config = ValidationConfig(golden_dir=tmp_path)

        # First generate
        run_validation(mode="generate", config=config, device=torch.device("cpu"), golden_dir=tmp_path)

        # Then compare (should not raise)
        run_validation(mode="compare", config=config, device=torch.device("cpu"), golden_dir=tmp_path)

    def test_run_invalid_mode_raises(self, tmp_path):
        """Test that invalid mode raises error."""
        config = ValidationConfig(golden_dir=tmp_path)

        with pytest.raises(ValueError, match="mode"):
            run_validation(
                mode="invalid",
                config=config,
                device=torch.device("cpu"),
                golden_dir=tmp_path,
            )


class TestGradientValidation:
    """Tests for gradient validation (optional feature)."""

    def test_gradient_norms_saved(self, tmp_path):
        """Test that gradient norms are saved when requested."""
        config = ValidationConfig(golden_dir=tmp_path)

        # Generate with gradients
        generate_golden_outputs(config=config, device=torch.device("cpu"))

        # Check gradient info is saved
        outputs = torch.load(tmp_path / "forward_outputs.pt", weights_only=True)

        # Gradient norms should be included
        assert "gradient_norm" in outputs
