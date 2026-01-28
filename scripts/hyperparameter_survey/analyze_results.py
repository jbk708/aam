#!/usr/bin/env python3
"""Analyze hyperparameter survey results and generate summary report."""

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd


def extract_best_metrics(log_path: Path) -> dict:
    """Extract best MAE and epoch from training log."""
    metrics = {"best_mae": None, "best_epoch": None, "final_mae": None}

    if not log_path.exists():
        return metrics

    content = log_path.read_text()

    # Extract MAE from validation output: "mae=0.1234"
    # Log format: "Epoch X/Y: train_loss=... | val_loss=..., r2=..., mae=..."
    mae_matches = re.findall(r"\bmae=([0-9.]+)", content)
    if mae_matches:
        maes = [float(m) for m in mae_matches]
        metrics["final_mae"] = maes[-1] if maes else None
        metrics["best_mae"] = min(maes)
        # Find epoch with best MAE
        best_mae = metrics["best_mae"]
        for i, m in enumerate(maes):
            if abs(m - best_mae) < 1e-6:
                metrics["best_epoch"] = i + 1  # Epochs are 1-indexed in logs
                break

    return metrics


def load_checkpoint_metrics(checkpoint_dir: Path) -> Optional[dict]:
    """Load metrics from best_model.pt checkpoint."""
    best_model = checkpoint_dir / "best_model.pt"
    if not best_model.exists():
        return None

    try:
        import torch

        ckpt = torch.load(best_model, map_location="cpu", weights_only=False)
        metrics = ckpt.get("metrics", {})
        return {
            "epoch": ckpt.get("epoch"),
            "best_val_loss": ckpt.get("best_val_loss"),
            "best_metric_value": ckpt.get("best_metric_value"),
            "mae": metrics.get("mae"),
            "r2": metrics.get("r2"),
        }
    except Exception as e:
        print(f"Warning: Could not load {best_model}: {e}")
        return None


def analyze_survey(output_base: Path) -> pd.DataFrame:
    """Analyze all survey runs and return summary DataFrame."""
    results = []

    for phase_dir in sorted(output_base.glob("phase*")):
        phase_name = phase_dir.name

        for run_dir in sorted(phase_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            run_name = run_dir.name
            log_path = run_dir / "training.log"
            checkpoint_dir = run_dir / "checkpoints"

            # Extract metrics
            log_metrics = extract_best_metrics(log_path)
            ckpt_metrics = load_checkpoint_metrics(checkpoint_dir)

            # Determine status
            if checkpoint_dir.exists() and (checkpoint_dir / "best_model.pt").exists():
                status = "COMPLETE"
            elif log_path.exists():
                status = "IN_PROGRESS"
            else:
                status = "NOT_STARTED"

            # Get best MAE from checkpoint if available, fallback to log
            best_mae = log_metrics["best_mae"]
            best_epoch = log_metrics["best_epoch"]
            r2 = None
            if ckpt_metrics:
                if ckpt_metrics.get("mae") is not None:
                    best_mae = ckpt_metrics["mae"]
                if ckpt_metrics.get("epoch") is not None:
                    best_epoch = ckpt_metrics["epoch"] + 1  # Convert 0-indexed to 1-indexed
                if ckpt_metrics.get("r2") is not None:
                    r2 = ckpt_metrics["r2"]

            results.append(
                {
                    "phase": phase_name,
                    "run": run_name,
                    "status": status,
                    "best_mae": best_mae,
                    "best_epoch": best_epoch,
                    "r2": r2,
                    "final_mae": log_metrics["final_mae"],
                }
            )

    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame, target_mae: float = 50.0):
    """Print formatted summary of survey results."""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER SURVEY RESULTS")
    print("=" * 70)

    # Overall status
    completed = len(df[df["status"] == "COMPLETE"])
    total = len(df)
    print(f"\nProgress: {completed}/{total} runs complete")

    # Best results by phase
    print("\n" + "-" * 70)
    print("RESULTS BY PHASE")
    print("-" * 70)

    for phase in df["phase"].unique():
        phase_df = df[df["phase"] == phase].copy()
        print(f"\n{phase.upper()}:")

        for _, row in phase_df.iterrows():
            mae_str = f"{row['best_mae']:.2f}" if pd.notna(row["best_mae"]) else "N/A"
            r2_str = f"{row['r2']:.4f}" if pd.notna(row.get("r2")) else "N/A"
            epoch_str = str(int(row["best_epoch"])) if pd.notna(row["best_epoch"]) else "-"
            status_icon = {"COMPLETE": "OK", "IN_PROGRESS": "...", "NOT_STARTED": "  "}[row["status"]]
            beat_target = "*" if pd.notna(row["best_mae"]) and row["best_mae"] < target_mae else " "

            print(f"  [{status_icon}] {row['run']:25} MAE: {mae_str:>8}  R2: {r2_str:>7} (epoch {epoch_str:>3}){beat_target}")

    # Best overall
    completed_df = df[df["status"] == "COMPLETE"].copy()
    # Filter to only rows with valid MAE values
    valid_mae_df = completed_df[completed_df["best_mae"].notna()].copy()

    if not valid_mae_df.empty:
        print("\n" + "-" * 70)
        print("TOP 5 CONFIGURATIONS")
        print("-" * 70)

        top5 = valid_mae_df.nsmallest(5, "best_mae")
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            beat = "< TARGET" if row["best_mae"] < target_mae else ""
            r2_str = f", R2 = {row['r2']:.4f}" if pd.notna(row.get("r2")) else ""
            print(f"  {i}. {row['phase']}/{row['run']}: MAE = {row['best_mae']:.2f}{r2_str} {beat}")
    else:
        print("\n" + "-" * 70)
        print("TOP 5 CONFIGURATIONS")
        print("-" * 70)
        print("  No valid MAE values found. Check log/checkpoint format.")

    # Recommendations
    print("\n" + "-" * 70)
    print("RECOMMENDATIONS FOR COMBINATION RUNS")
    print("-" * 70)

    # Find best in each phase
    for phase in ["phase1_loss", "phase2_fusion", "phase3_regressor", "phase4_lr"]:
        phase_df = valid_mae_df[valid_mae_df["phase"] == phase]
        if not phase_df.empty:
            best = phase_df.loc[phase_df["best_mae"].idxmin()]
            print(f"  Best {phase.split('_')[1]}: {best['run']} (MAE: {best['best_mae']:.2f})")
        else:
            print(f"  Best {phase.split('_')[1]}: N/A (no valid results)")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter survey results")
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("/cosmos/nfs/home/jkirkland/repos/aam/data/survey_add_0c"),
        help="Base directory containing survey results",
    )
    parser.add_argument("--target-mae", type=float, default=50.0, help="Target MAE to beat")
    parser.add_argument("--csv", type=Path, help="Save results to CSV file")
    args = parser.parse_args()

    if not args.output_base.exists():
        print(f"Error: Output directory not found: {args.output_base}")
        return 1

    df = analyze_survey(args.output_base)
    print_summary(df, args.target_mae)

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nResults saved to: {args.csv}")

    return 0


if __name__ == "__main__":
    exit(main())
