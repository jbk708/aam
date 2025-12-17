"""Learning rate finder utility to identify optimal initial learning rate.

Implements LR range test similar to fastai's approach: exponentially increase
learning rate while training and track loss to find optimal starting LR.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


class LearningRateFinder:
    """Find optimal learning rate using LR range test.

    The LR range test exponentially increases the learning rate while training
    and tracks the loss. The optimal learning rate is typically where the loss
    decreases fastest (steepest negative slope).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
    ):
        """Initialize learning rate finder.

        Args:
            model: Model to train
            optimizer: Optimizer (will be modified during LR finder)
            loss_fn: Loss function
            device: Device to run on
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

        # Store original learning rate
        self.original_lr = [group["lr"] for group in optimizer.param_groups]

        # Results storage
        self.lrs: List[float] = []
        self.losses: List[float] = []

    def find_lr(
        self,
        train_loader: DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 10.0,
        num_iter: int = 100,
        smooth_factor: float = 0.05,
        diverge_threshold: float = 5.0,
    ) -> Tuple[List[float], List[float], Optional[float]]:
        """Find optimal learning rate using LR range test.

        Args:
            train_loader: DataLoader for training data
            start_lr: Starting learning rate (default: 1e-7)
            end_lr: Ending learning rate (default: 10.0)
            num_iter: Number of iterations to run (default: 100)
            smooth_factor: Smoothing factor for loss (default: 0.05)
            diverge_threshold: Threshold for detecting divergence (default: 5.0)

        Returns:
            Tuple of (learning_rates, losses, suggested_lr)
            - learning_rates: List of learning rates tested
            - losses: List of corresponding losses
            - suggested_lr: Suggested optimal learning rate (None if not found)
        """
        self.lrs = []
        self.losses = []

        # Set initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = start_lr

        # Exponential learning rate multiplier
        lr_mult = (end_lr / start_lr) ** (1.0 / num_iter)

        # Create iterator from data loader
        data_iter = iter(train_loader)

        # Track best loss and smoothed loss
        best_loss = float("inf")
        smoothed_loss = None

        self.model.train()

        try:
            for iteration in range(num_iter):
                # Get next batch (cycle if needed)
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)

                # Update learning rate exponentially
                current_lr = start_lr * (lr_mult**iteration)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = current_lr

                # Forward pass
                self.optimizer.zero_grad()

                # Prepare batch (similar to trainer._prepare_batch)
                if isinstance(batch, dict):
                    tokens = batch["tokens"].to(self.device)
                    targets = {}
                    if "counts" in batch:
                        targets["counts"] = batch["counts"].to(self.device)
                    if "y_target" in batch:
                        targets["target"] = batch["y_target"].to(self.device)
                    if "unifrac_target" in batch:
                        targets["base_target"] = batch["unifrac_target"].to(self.device)
                    targets["tokens"] = tokens
                elif isinstance(batch, (list, tuple)):
                    tokens = batch[0].to(self.device)
                    targets = {"tokens": tokens}
                    if len(batch) > 1:
                        targets["base_target"] = batch[1].to(self.device)
                else:
                    tokens = batch.to(self.device)
                    targets = {"tokens": tokens}

                # Forward pass
                outputs = self.model(tokens)

                # Compute loss (need to determine encoder_type)
                encoder_type = "unifrac"  # Default, can be improved
                if hasattr(self.model, "encoder_type"):
                    encoder_type = self.model.encoder_type
                elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "encoder_type"):
                    encoder_type = self.model.base_model.encoder_type

                loss_dict = self.loss_fn(outputs, targets, encoder_type=encoder_type)
                loss = loss_dict["total_loss"]

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Track loss
                loss_value = loss.item()

                # Smooth loss
                if smoothed_loss is None:
                    smoothed_loss = loss_value
                else:
                    smoothed_loss = smooth_factor * loss_value + (1 - smooth_factor) * smoothed_loss

                # Store results
                self.lrs.append(current_lr)
                self.losses.append(smoothed_loss)

                # Check for divergence
                if smoothed_loss > diverge_threshold * best_loss:
                    print(f"LR finder: Loss diverged at LR={current_lr:.2e}, stopping early")
                    break

                # Update best loss
                if smoothed_loss < best_loss:
                    best_loss = smoothed_loss

        except KeyboardInterrupt:
            print("LR finder: Interrupted by user")

        # Restore original learning rate
        for param_group, orig_lr in zip(self.optimizer.param_groups, self.original_lr):
            param_group["lr"] = orig_lr

        # Find suggested learning rate (steepest negative slope)
        suggested_lr = self._suggest_lr()

        return self.lrs, self.losses, suggested_lr

    def _suggest_lr(self) -> Optional[float]:
        """Suggest optimal learning rate based on loss curve.

        Finds the learning rate where the loss decreases fastest
        (steepest negative slope).

        Returns:
            Suggested learning rate, or None if not enough data
        """
        if len(self.lrs) < 10:
            return None

        # Convert to numpy for easier computation
        lrs = np.array(self.lrs)
        losses = np.array(self.losses)

        # Find minimum loss point
        min_idx = np.argmin(losses)

        # Look for steepest negative slope before minimum
        # Use a window to compute gradients
        window_size = max(5, len(losses) // 20)
        best_lr = None
        best_gradient = float("inf")

        # Check gradients in the range before minimum loss
        for i in range(window_size, min_idx):
            # Compute gradient over window
            lr_window = lrs[i - window_size : i + window_size]
            loss_window = losses[i - window_size : i + window_size]

            # Fit linear regression to log space
            log_lrs = np.log10(lr_window)
            if len(loss_window) > 1 and np.std(loss_window) > 1e-8:
                gradient = np.polyfit(log_lrs, loss_window, 1)[0]

                # Find most negative gradient (steepest descent)
                if gradient < best_gradient:
                    best_gradient = gradient
                    best_lr = lrs[i]

        # If no good gradient found, use LR at 1/10th of minimum loss point
        if best_lr is None and min_idx > 0:
            best_lr = lrs[min(min_idx // 10, len(lrs) - 1)]

        return best_lr

    def plot(
        self,
        output_path: Optional[Path] = None,
        skip_start: int = 10,
        skip_end: int = 5,
    ) -> plt.Figure:
        """Plot learning rate vs loss curve.

        Args:
            output_path: Path to save plot (optional)
            skip_start: Number of initial points to skip (default: 10)
            skip_end: Number of final points to skip (default: 5)

        Returns:
            matplotlib Figure object
        """
        if len(self.lrs) == 0:
            raise ValueError("No LR finder data. Run find_lr() first.")

        # Skip noisy start and end
        start_idx = skip_start
        end_idx = len(self.lrs) - skip_end if skip_end > 0 else len(self.lrs)

        lrs_plot = self.lrs[start_idx:end_idx]
        losses_plot = self.losses[start_idx:end_idx]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot on log scale
        ax.semilogx(lrs_plot, losses_plot, "b-", linewidth=2, label="Loss")

        # Mark suggested LR if available
        suggested_lr = self._suggest_lr()
        if suggested_lr is not None:
            # Find corresponding loss
            idx = min(np.argmin(np.abs(np.array(lrs_plot) - suggested_lr)), len(losses_plot) - 1)
            ax.axvline(x=suggested_lr, color="r", linestyle="--", linewidth=2, label=f"Suggested LR: {suggested_lr:.2e}")
            ax.plot(suggested_lr, losses_plot[idx], "ro", markersize=10)

        ax.set_xlabel("Learning Rate", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Learning Rate Finder", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"LR finder plot saved to {output_path}")

        return fig
