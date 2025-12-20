"""Numerical validation for cross-platform (CUDA vs ROCm) consistency."""

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

__all__ = [
    "ValidationConfig",
    "ValidationResult",
    "generate_golden_outputs",
    "compare_golden_outputs",
    "run_validation",
    "_create_model",
    "_create_inputs",
    "_get_platform_info",
]
