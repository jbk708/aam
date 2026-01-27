"""Cross-platform numerical validation for CUDA vs ROCm consistency.

This module provides utilities for validating that model outputs are
numerically equivalent across different GPU platforms (CUDA, ROCm, CPU).

Usage:
    # On CUDA system: generate golden outputs
    python -m tests.validation.numerical_validation generate

    # On ROCm system: compare against golden outputs
    python -m tests.validation.numerical_validation compare
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import platform
import sys

import torch

from aam.models.sequence_predictor import SequencePredictor


@dataclass
class ValidationConfig:
    """Configuration for numerical validation."""

    seed: int = 42
    batch_size: int = 2
    num_asvs: int = 8
    seq_len: int = 150
    embedding_dim: int = 64
    vocab_size: int = 6
    out_dim: int = 1
    atol: float = 1e-5
    rtol: float = 1e-4
    gradient_atol: float = 0.5  # Relaxed tolerance for cross-platform gradient differences
    golden_dir: Path = field(default_factory=lambda: Path(__file__).parent / "golden")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "seed": self.seed,
            "batch_size": self.batch_size,
            "num_asvs": self.num_asvs,
            "seq_len": self.seq_len,
            "embedding_dim": self.embedding_dim,
            "vocab_size": self.vocab_size,
            "out_dim": self.out_dim,
            "atol": self.atol,
            "rtol": self.rtol,
            "gradient_atol": self.gradient_atol,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k != "golden_dir"})


@dataclass
class ValidationResult:
    """Results from comparing outputs against golden files."""

    passed: bool
    details: Dict[str, Dict[str, Any]]
    platform_info: Dict[str, str]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines: List[str] = []
        status = "PASSED" if self.passed else "FAILED"
        lines.append(f"Validation {status}")
        lines.append(f"Platform: {self.platform_info.get('device', 'unknown')}")
        lines.append("")

        for name, detail in self.details.items():
            check_status = "OK" if detail["passed"] else "FAIL"
            max_diff = detail.get("max_diff", 0)
            lines.append(f"  {name}: {check_status} (max_diff={max_diff:.2e})")

        return "\n".join(lines)


def _create_model(config: ValidationConfig, device: torch.device) -> SequencePredictor:
    """Create a deterministic model for validation."""
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    model = SequencePredictor(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        max_bp=config.seq_len,
        token_limit=config.num_asvs,
        asv_num_layers=2,
        asv_num_heads=4,
        sample_num_layers=2,
        sample_num_heads=4,
        encoder_num_layers=2,
        encoder_num_heads=4,
        count_num_layers=1,
        count_num_heads=4,
        target_num_layers=1,
        target_num_heads=4,
        out_dim=config.out_dim,
        is_classifier=False,
        predict_nucleotides=False,
        attn_implementation="math",  # Most portable across platforms
    )

    return model.to(device)


def _create_inputs(config: ValidationConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    """Create deterministic input tensors for validation."""
    torch.manual_seed(config.seed + 1000)  # Different seed from model

    # Create token tensor with valid nucleotide values (1-4)
    tokens = torch.randint(
        1,
        5,
        (config.batch_size, config.num_asvs, config.seq_len),
        dtype=torch.long,
        device=device,
    )

    return {"tokens": tokens}


def _get_platform_info() -> Dict[str, str]:
    """Get information about the current platform."""
    info: Dict[str, str] = {
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "device": "cpu",
    }

    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["cuda_version"] = str(torch.version.cuda)
        info["gpu_name"] = torch.cuda.get_device_name(0)
    elif hasattr(torch.version, "hip") and torch.version.hip is not None:
        info["device"] = "rocm"
        info["rocm_version"] = str(torch.version.hip)
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)

    return info


def generate_golden_outputs(
    config: Optional[ValidationConfig] = None,
    device: Optional[torch.device] = None,
) -> Path:
    """Generate golden reference outputs on the current platform.

    Args:
        config: Validation configuration. If None, uses defaults.
        device: Device to run on. If None, auto-selects GPU or CPU.

    Returns:
        Path to the golden output directory.
    """
    if config is None:
        config = ValidationConfig()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    golden_dir = config.golden_dir
    golden_dir.mkdir(parents=True, exist_ok=True)

    model = _create_model(config, device)
    inputs = _create_inputs(config, device)

    model.eval()
    with torch.no_grad():
        outputs = model(inputs["tokens"], return_sample_embeddings=True)

    # Move outputs to CPU for storage
    cpu_outputs: Dict[str, torch.Tensor] = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            cpu_outputs[key] = value.cpu()

    # Compute gradient norm for a simple loss
    model.train()
    outputs_train = model(inputs["tokens"], return_sample_embeddings=True)
    loss = outputs_train["target_prediction"].mean()
    loss.backward()

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    gradient_norm = total_norm**0.5

    cpu_outputs["gradient_norm"] = torch.tensor(gradient_norm)

    # Save outputs
    torch.save(cpu_outputs, golden_dir / "forward_outputs.pt")

    # Save model weights
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(model_state, golden_dir / "model_weights.pt")

    # Save metadata
    metadata = {
        "config": config.to_dict(),
        "platform_info": _get_platform_info(),
    }
    (golden_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return golden_dir


def compare_golden_outputs(
    config: Optional[ValidationConfig] = None,
    device: Optional[torch.device] = None,
    golden_dir: Optional[Path] = None,
) -> ValidationResult:
    """Compare current platform outputs against golden reference.

    Args:
        config: Validation configuration. If None, loads from golden metadata.
        device: Device to run on. If None, auto-selects GPU or CPU.
        golden_dir: Directory containing golden outputs. If None, uses default.

    Returns:
        ValidationResult with pass/fail status and detailed comparison.
    """
    if golden_dir is None:
        golden_dir = ValidationConfig().golden_dir

    metadata_path = golden_dir / "metadata.json"
    outputs_path = golden_dir / "forward_outputs.pt"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Golden metadata not found: {metadata_path}")
    if not outputs_path.exists():
        raise FileNotFoundError(f"Golden outputs not found: {outputs_path}")

    # Load metadata and config
    metadata = json.loads(metadata_path.read_text())
    if config is None:
        config = ValidationConfig.from_dict(metadata["config"])
    else:
        # Ensure golden_dir is set from argument
        pass

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load golden outputs
    golden_outputs = torch.load(outputs_path, weights_only=True)

    # Generate current outputs
    model = _create_model(config, device)
    inputs = _create_inputs(config, device)

    model.eval()
    with torch.no_grad():
        current_outputs = model(inputs["tokens"], return_sample_embeddings=True)

    # Move to CPU for comparison
    current_cpu: Dict[str, torch.Tensor] = {}
    for key, value in current_outputs.items():
        if isinstance(value, torch.Tensor):
            current_cpu[key] = value.cpu()

    # Compute gradient norm
    model.train()
    outputs_train = model(inputs["tokens"], return_sample_embeddings=True)
    loss = outputs_train["target_prediction"].mean()
    loss.backward()

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    current_cpu["gradient_norm"] = torch.tensor(total_norm**0.5)

    # Compare outputs
    details: Dict[str, Dict[str, Any]] = {}
    all_passed = True

    keys_to_check = ["target_prediction", "count_prediction", "base_embeddings", "gradient_norm"]

    for key in keys_to_check:
        if key not in golden_outputs:
            continue
        if key not in current_cpu:
            details[key] = {"passed": False, "max_diff": float("inf"), "error": "missing"}
            all_passed = False
            continue

        golden = golden_outputs[key]
        current = current_cpu[key]

        if golden.shape != current.shape:
            details[key] = {
                "passed": False,
                "max_diff": float("inf"),
                "error": f"shape mismatch: {golden.shape} vs {current.shape}",
            }
            all_passed = False
            continue

        max_diff = (golden - current).abs().max().item()

        # Use relaxed tolerance for gradient norm (expected cross-platform difference)
        if key == "gradient_norm":
            is_close = torch.allclose(golden, current, atol=config.gradient_atol, rtol=config.rtol)
        else:
            is_close = torch.allclose(golden, current, atol=config.atol, rtol=config.rtol)

        details[key] = {"passed": is_close, "max_diff": max_diff}
        if not is_close:
            all_passed = False

    return ValidationResult(
        passed=all_passed,
        details=details,
        platform_info=_get_platform_info(),
    )


def run_validation(
    mode: str = "compare",
    config: Optional[ValidationConfig] = None,
    device: Optional[torch.device] = None,
    golden_dir: Optional[Path] = None,
) -> None:
    """Run validation in generate or compare mode.

    Args:
        mode: Either 'generate' or 'compare'.
        config: Validation configuration.
        device: Device to run on.
        golden_dir: Directory for golden files.
    """
    if mode not in ("generate", "compare"):
        raise ValueError(f"Invalid mode: {mode}. Must be 'generate' or 'compare'.")

    if config is None:
        config = ValidationConfig()

    if golden_dir is not None:
        config = ValidationConfig(
            seed=config.seed,
            batch_size=config.batch_size,
            num_asvs=config.num_asvs,
            seq_len=config.seq_len,
            embedding_dim=config.embedding_dim,
            vocab_size=config.vocab_size,
            out_dim=config.out_dim,
            atol=config.atol,
            rtol=config.rtol,
            gradient_atol=config.gradient_atol,
            golden_dir=golden_dir,
        )

    if mode == "generate":
        output_dir = generate_golden_outputs(config=config, device=device)
        print(f"Golden outputs generated at: {output_dir}")
    else:
        result = compare_golden_outputs(config=config, device=device, golden_dir=golden_dir)
        print(result.summary())
        if not result.passed:
            sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-platform numerical validation")
    parser.add_argument(
        "mode",
        choices=["generate", "compare"],
        help="Mode: generate golden outputs or compare against them",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, rocm, cpu). Auto-detects if not specified.",
    )
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=None,
        help="Directory for golden files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    config = ValidationConfig(seed=args.seed)
    device = torch.device(args.device) if args.device else None

    run_validation(
        mode=args.mode,
        config=config,
        device=device,
        golden_dir=args.golden_dir,
    )
