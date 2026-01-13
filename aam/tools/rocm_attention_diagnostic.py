#!/usr/bin/env python3
"""ROCm attention backend diagnostic tool.

This script diagnoses attention implementation issues on ROCm/HIP systems,
comparing numerical accuracy and performance between SDPA backends.

Usage:
    python -m aam.tools.rocm_attention_diagnostic
"""

import sys
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def get_system_info() -> Dict[str, str]:
    """Gather system and PyTorch information."""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda or "N/A",
        "hip_version": getattr(torch.version, "hip", None) or "N/A",
    }

    if torch.cuda.is_available():
        info["device_count"] = str(torch.cuda.device_count())
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["device_memory_gb"] = f"{props.total_memory / (1024**3):.1f}"
        info["device_arch"] = props.gcnArchName if hasattr(props, "gcnArchName") else "N/A"

    return info


def check_sdpa_backends() -> Dict[str, bool]:
    """Check which SDPA backends are available."""
    backends = {}

    # Check if SDPA is available at all
    backends["sdpa_available"] = hasattr(F, "scaled_dot_product_attention")

    if not backends["sdpa_available"]:
        return backends

    # Check individual backend availability
    backends["flash_enabled"] = torch.backends.cuda.flash_sdp_enabled()
    backends["mem_efficient_enabled"] = torch.backends.cuda.mem_efficient_sdp_enabled()
    backends["math_enabled"] = torch.backends.cuda.math_sdp_enabled()

    # Check if flash attention can actually run (requires specific GPU arch)
    backends["flash_available"] = False
    backends["mem_efficient_available"] = False

    if torch.cuda.is_available():
        try:
            # Test with small tensors to see what backends actually work
            q = torch.randn(1, 4, 16, 32, device="cuda", dtype=torch.float16)
            k = torch.randn(1, 4, 16, 32, device="cuda", dtype=torch.float16)
            v = torch.randn(1, 4, 16, 32, device="cuda", dtype=torch.float16)

            # Test flash
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(False)
                _ = F.scaled_dot_product_attention(q, k, v)
                backends["flash_available"] = True
            except RuntimeError:
                pass

            # Test mem_efficient
            try:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(False)
                _ = F.scaled_dot_product_attention(q, k, v)
                backends["mem_efficient_available"] = True
            except RuntimeError:
                pass

            # Restore defaults
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)

        except Exception as e:
            backends["test_error"] = str(e)

    return backends


@contextmanager
def sdpa_backend(backend: str):
    """Context manager to force a specific SDPA backend."""
    orig_flash = torch.backends.cuda.flash_sdp_enabled()
    orig_mem = torch.backends.cuda.mem_efficient_sdp_enabled()
    orig_math = torch.backends.cuda.math_sdp_enabled()

    try:
        if backend == "flash":
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)
        elif backend == "mem_efficient":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
        elif backend == "math":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        yield
    finally:
        torch.backends.cuda.enable_flash_sdp(orig_flash)
        torch.backends.cuda.enable_mem_efficient_sdp(orig_mem)
        torch.backends.cuda.enable_math_sdp(orig_math)


def numerical_comparison(
    seq_len: int = 128,
    num_heads: int = 4,
    head_dim: int = 32,
    batch_size: int = 8,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Dict[str, float]]:
    """Compare numerical outputs between SDPA backends.

    Returns dict mapping backend pairs to their numerical differences.
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.device("cuda")
    torch.manual_seed(42)

    # Create test tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    results = {}
    outputs = {}

    # Compute with each available backend
    backends_to_test = ["math", "mem_efficient", "flash"]

    for backend in backends_to_test:
        try:
            with sdpa_backend(backend):
                out = F.scaled_dot_product_attention(q, k, v)
                outputs[backend] = out.clone()
        except RuntimeError as e:
            results[f"{backend}_error"] = str(e)

    # Compare outputs
    if "math" in outputs:
        for backend in ["mem_efficient", "flash"]:
            if backend in outputs:
                diff = (outputs["math"] - outputs[backend]).abs()
                results[f"math_vs_{backend}"] = {
                    "max_diff": diff.max().item(),
                    "mean_diff": diff.mean().item(),
                    "std_diff": diff.std().item(),
                    "relative_error": (diff / (outputs["math"].abs() + 1e-8)).mean().item(),
                }

    return results


def numerical_comparison_with_mask(
    seq_len: int = 128,
    num_heads: int = 4,
    head_dim: int = 32,
    batch_size: int = 8,
    dtype: torch.dtype = torch.float32,
    mask_ratio: float = 0.3,
) -> Dict[str, Dict[str, float]]:
    """Compare numerical outputs with attention mask (padding mask).

    The AAM model uses src_key_padding_mask which creates masked positions.
    This test checks if masking exacerbates numerical differences.
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.device("cuda")
    torch.manual_seed(42)

    # Create test tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    # Create attention mask (True = masked/ignored position)
    attn_mask = torch.rand(batch_size, seq_len, device=device) < mask_ratio
    # Expand for SDPA: (batch, 1, 1, seq_len) for broadcasting
    attn_mask_expanded = attn_mask.unsqueeze(1).unsqueeze(2)

    results = {}
    outputs = {}

    backends_to_test = ["math", "mem_efficient", "flash"]

    for backend in backends_to_test:
        try:
            with sdpa_backend(backend):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask_expanded)
                outputs[backend] = out.clone()
        except RuntimeError as e:
            results[f"{backend}_error"] = str(e)

    if "math" in outputs:
        for backend in ["mem_efficient", "flash"]:
            if backend in outputs:
                diff = (outputs["math"] - outputs[backend]).abs()
                results[f"math_vs_{backend}"] = {
                    "max_diff": diff.max().item(),
                    "mean_diff": diff.mean().item(),
                    "std_diff": diff.std().item(),
                    "relative_error": (diff / (outputs["math"].abs() + 1e-8)).mean().item(),
                }

    return results


def benchmark_backends(
    seq_len: int = 256,
    num_heads: int = 4,
    head_dim: int = 32,
    batch_size: int = 16,
    num_iterations: int = 100,
    warmup: int = 10,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Dict[str, float]]:
    """Benchmark SDPA backends for performance."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.device("cuda")
    results = {}

    # Create test tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    backends_to_test = ["math", "mem_efficient", "flash"]

    for backend in backends_to_test:
        try:
            with sdpa_backend(backend):
                # Warmup
                for _ in range(warmup):
                    _ = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()

                # Benchmark
                start = time.perf_counter()
                for _ in range(num_iterations):
                    _ = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                results[backend] = {
                    "total_time_ms": elapsed * 1000,
                    "avg_time_ms": (elapsed / num_iterations) * 1000,
                    "throughput_iter_per_sec": num_iterations / elapsed,
                }
        except RuntimeError as e:
            results[f"{backend}_error"] = str(e)

    # Calculate relative performance
    if "math" in results and isinstance(results["math"], dict):
        math_time = results["math"]["avg_time_ms"]
        for backend in ["mem_efficient", "flash"]:
            if backend in results and isinstance(results[backend], dict):
                results[backend]["speedup_vs_math"] = math_time / results[backend]["avg_time_ms"]

    return results


def memory_comparison(
    seq_len: int = 512,
    num_heads: int = 8,
    head_dim: int = 64,
    batch_size: int = 32,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Dict[str, float]]:
    """Compare memory usage between SDPA backends."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.device("cuda")
    results = {}

    backends_to_test = ["math", "mem_efficient", "flash"]

    for backend in backends_to_test:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
            k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

            with sdpa_backend(backend):
                out = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()

            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            results[backend] = {"peak_memory_mb": peak_mb}

            del q, k, v, out
            torch.cuda.empty_cache()

        except RuntimeError as e:
            results[f"{backend}_error"] = str(e)

    # Calculate relative memory
    if "math" in results and isinstance(results["math"], dict):
        math_mem = results["math"]["peak_memory_mb"]
        for backend in ["mem_efficient", "flash"]:
            if backend in results and isinstance(results[backend], dict):
                results[backend]["memory_ratio_vs_math"] = results[backend]["peak_memory_mb"] / math_mem

    return results


def check_flash_attention_rocm() -> Dict[str, str]:
    """Check if Flash Attention for ROCm is available."""
    results = {}

    # Check for flash_attn package (ROCm fork)
    try:
        import flash_attn

        results["flash_attn_installed"] = "yes"
        results["flash_attn_version"] = getattr(flash_attn, "__version__", "unknown")
    except ImportError:
        results["flash_attn_installed"] = "no"
        results["flash_attn_install_hint"] = "For ROCm, install from: https://github.com/ROCm/flash-attention"

    # Check for xformers
    try:
        import xformers
        import xformers.ops

        results["xformers_installed"] = "yes"
        results["xformers_version"] = xformers.__version__
        # Check if memory_efficient_attention is available
        results["xformers_mem_eff_available"] = str(hasattr(xformers.ops, "memory_efficient_attention"))
    except ImportError:
        results["xformers_installed"] = "no"

    return results


def gradient_test(
    seq_len: int = 64,
    num_heads: int = 4,
    head_dim: int = 32,
    batch_size: int = 4,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Dict[str, float]]:
    """Test gradient computation accuracy between backends.

    This is critical because training divergence could be caused by
    gradient differences, not just forward pass differences.
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.device("cuda")
    torch.manual_seed(42)

    results = {}
    gradients = {}

    backends_to_test = ["math", "mem_efficient", "flash"]

    for backend in backends_to_test:
        try:
            # Fresh tensors with gradients for each backend
            torch.manual_seed(42)
            q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)

            with sdpa_backend(backend):
                out = F.scaled_dot_product_attention(q, k, v)
                loss = out.sum()
                loss.backward()

            gradients[backend] = {
                "q_grad": q.grad.clone(),
                "k_grad": k.grad.clone(),
                "v_grad": v.grad.clone(),
            }

        except RuntimeError as e:
            results[f"{backend}_error"] = str(e)

    # Compare gradients
    if "math" in gradients:
        for backend in ["mem_efficient", "flash"]:
            if backend in gradients:
                for tensor_name in ["q_grad", "k_grad", "v_grad"]:
                    diff = (gradients["math"][tensor_name] - gradients[backend][tensor_name]).abs()
                    key = f"math_vs_{backend}_{tensor_name}"
                    results[key] = {
                        "max_diff": diff.max().item(),
                        "mean_diff": diff.mean().item(),
                        "relative_error": (diff / (gradients["math"][tensor_name].abs() + 1e-8)).mean().item(),
                    }

    return results


def run_full_diagnostic() -> Dict:
    """Run complete diagnostic suite."""
    print("=" * 70)
    print("ROCm Attention Backend Diagnostic")
    print("=" * 70)

    results = {}

    # System info
    print("\n[1/8] Gathering system information...")
    results["system_info"] = get_system_info()
    for k, v in results["system_info"].items():
        print(f"  {k}: {v}")

    # SDPA backends
    print("\n[2/8] Checking SDPA backends...")
    results["sdpa_backends"] = check_sdpa_backends()
    for k, v in results["sdpa_backends"].items():
        print(f"  {k}: {v}")

    # Flash Attention / xFormers
    print("\n[3/8] Checking Flash Attention / xFormers...")
    results["flash_attention"] = check_flash_attention_rocm()
    for k, v in results["flash_attention"].items():
        print(f"  {k}: {v}")

    # Numerical comparison (fp32)
    print("\n[4/8] Numerical comparison (fp32, no mask)...")
    results["numerical_fp32"] = numerical_comparison(dtype=torch.float32)
    for k, v in results["numerical_fp32"].items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2:.2e}")
        else:
            print(f"  {k}: {v}")

    # Numerical comparison (fp16)
    print("\n[5/8] Numerical comparison (fp16, no mask)...")
    results["numerical_fp16"] = numerical_comparison(dtype=torch.float16)
    for k, v in results["numerical_fp16"].items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2:.2e}")
        else:
            print(f"  {k}: {v}")

    # Numerical comparison with mask
    print("\n[6/8] Numerical comparison with attention mask (fp32)...")
    results["numerical_masked"] = numerical_comparison_with_mask(dtype=torch.float32)
    for k, v in results["numerical_masked"].items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2:.2e}")
        else:
            print(f"  {k}: {v}")

    # Gradient test
    print("\n[7/8] Gradient comparison (backward pass)...")
    results["gradient_test"] = gradient_test(dtype=torch.float32)
    for k, v in results["gradient_test"].items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2:.2e}")
        else:
            print(f"  {k}: {v}")

    # Performance benchmark
    print("\n[8/8] Performance benchmark...")
    results["benchmark"] = benchmark_backends()
    for k, v in results["benchmark"].items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                if isinstance(v2, float):
                    print(f"    {k2}: {v2:.3f}")
                else:
                    print(f"    {k2}: {v2}")
        else:
            print(f"  {k}: {v}")

    # Memory comparison
    print("\n[Bonus] Memory comparison...")
    results["memory"] = memory_comparison()
    for k, v in results["memory"].items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                if isinstance(v2, float):
                    print(f"    {k2}: {v2:.2f}")
                else:
                    print(f"    {k2}: {v2}")
        else:
            print(f"  {k}: {v}")

    # Summary and recommendations
    print("\n" + "=" * 70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 70)

    # Check for numerical issues
    num_issues = []
    for test_name in ["numerical_fp32", "numerical_fp16", "numerical_masked"]:
        if test_name in results:
            for k, v in results[test_name].items():
                if isinstance(v, dict) and "max_diff" in v:
                    if v["max_diff"] > 1e-3:  # Significant difference
                        num_issues.append(f"{test_name}/{k}: max_diff={v['max_diff']:.2e}")

    if num_issues:
        print("\n⚠️  NUMERICAL DIVERGENCE DETECTED:")
        for issue in num_issues:
            print(f"  - {issue}")
        print("\n  RECOMMENDATION: Use --attn-implementation math for correct results on ROCm")
    else:
        print("\n✓ No significant numerical divergence detected")

    # Check gradient issues
    grad_issues = []
    if "gradient_test" in results:
        for k, v in results["gradient_test"].items():
            if isinstance(v, dict) and "max_diff" in v:
                if v["max_diff"] > 1e-3:
                    grad_issues.append(f"{k}: max_diff={v['max_diff']:.2e}")

    if grad_issues:
        print("\n⚠️  GRADIENT DIVERGENCE DETECTED:")
        for issue in grad_issues:
            print(f"  - {issue}")
        print("\n  This explains training accuracy differences between backends")

    # Performance summary
    if "benchmark" in results:
        if "math" in results["benchmark"] and isinstance(results["benchmark"]["math"], dict):
            math_time = results["benchmark"]["math"]["avg_time_ms"]
            print(f"\nPerformance (math backend): {math_time:.3f} ms/iteration")

            for backend in ["mem_efficient", "flash"]:
                if backend in results["benchmark"] and isinstance(results["benchmark"][backend], dict):
                    speedup = results["benchmark"][backend].get("speedup_vs_math", 0)
                    print(f"  {backend}: {speedup:.2f}x vs math")

    return results


if __name__ == "__main__":
    results = run_full_diagnostic()

    # Optionally save results to JSON
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        import json

        output_file = sys.argv[2] if len(sys.argv) > 2 else "rocm_diagnostic.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_file}")
