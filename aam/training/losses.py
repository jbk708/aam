"""Multi-task loss functions for AAM model."""

import torch
import torch.nn as nn
from typing import Dict, Optional


def _format_tensor_stats(tensor: torch.Tensor) -> str:
    """Safely format tensor statistics for error messages.

    Handles both integer and floating point tensors.

    Args:
        tensor: Tensor to format

    Returns:
        String with tensor statistics
    """
    min_val = tensor.min().item()
    max_val = tensor.max().item()

    # Handle integer tensors (can't compute mean)
    if tensor.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.long):
        return f"min={min_val}, max={max_val} (integer tensor)"
    else:
        # Floating point tensors
        mean_val = tensor.mean().item()
        return f"min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}"


def compute_pairwise_distances(
    embeddings: torch.Tensor,
    normalize: bool = False,
    scale: float = 5.0,
) -> torch.Tensor:
    """Compute pairwise Euclidean distances from embeddings.

    Args:
        embeddings: Sample embeddings [batch_size, embedding_dim]
        normalize: If True, normalize distances to [0, 1] using sigmoid (default: False)
        scale: Scaling factor for sigmoid normalization (default: 5.0)

    Returns:
        Pairwise distance matrix [batch_size, batch_size]
        If normalize=True, distances are bounded to [0, 1]
    """
    # Check for NaN or Inf in embeddings
    if torch.any(torch.isnan(embeddings)):
        import sys

        error_msg = f"NaN values found in embeddings before distance computation, shape={embeddings.shape}"
        print(f"ERROR: {error_msg}", file=sys.stderr, flush=True)
        raise ValueError(error_msg)
    if torch.any(torch.isinf(embeddings)):
        import sys

        error_msg = f"Inf values found in embeddings before distance computation, shape={embeddings.shape}"
        print(f"ERROR: {error_msg}", file=sys.stderr, flush=True)
        raise ValueError(error_msg)

    # Compute squared differences: (a - b)^2 for all pairs
    # Using broadcasting: [batch_size, 1, embedding_dim] - [1, batch_size, embedding_dim]
    # Result: [batch_size, batch_size, embedding_dim]
    diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)
    squared_diff = diff**2

    # Sum over embedding dimension: [batch_size, batch_size]
    squared_distances = squared_diff.sum(dim=-1)

    # Check for NaN in squared distances (shouldn't happen, but safety check)
    if torch.any(torch.isnan(squared_distances)):
        import sys

        error_msg = f"NaN values found in squared_distances, shape={squared_distances.shape}"
        print(f"ERROR: {error_msg}", file=sys.stderr, flush=True)
        raise ValueError(error_msg)

    # Take square root to get Euclidean distances
    # Clamp to prevent numerical issues (sqrt of very small negative values)
    # Use eps to prevent sqrt(0) numerical issues, but allow zero distances (diagonal)
    eps = 1e-8
    # For diagonal elements, we want exactly 0.0, so handle separately
    distances = torch.sqrt(torch.clamp(squared_distances, min=eps))
    # Set diagonal to exactly 0.0 (distance from sample to itself)
    # Use non-inplace operation to preserve gradient computation
    eye_mask = torch.eye(distances.shape[0], device=distances.device, dtype=distances.dtype)
    distances = distances * (1.0 - eye_mask)

    # Final check for NaN in distances
    if torch.any(torch.isnan(distances)):
        import sys

        error_msg = f"NaN values found in computed distances, shape={distances.shape}"
        print(f"ERROR: {error_msg}", file=sys.stderr, flush=True)
        print(
            f"embeddings stats: min={embeddings.min().item():.6f}, max={embeddings.max().item():.6f}, mean={embeddings.mean().item():.6f}",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"squared_distances stats: min={squared_distances.min().item():.6f}, max={squared_distances.max().item():.6f}, mean={squared_distances.mean().item():.6f}",
            file=sys.stderr,
            flush=True,
        )
        raise ValueError(error_msg)

    # Normalize distances to [0, 1] if requested (for UniFrac distances)
    if normalize:
        # Apply sigmoid with scaling to bound distances to [0, 1]
        # scale parameter controls sensitivity: larger scale = more sensitive to distance changes
        # sigmoid(scale * distance) maps distances to [0, 1]
        # Note: diagonal is already 0.0, sigmoid(0) = 0.5, so we need to handle diagonal separately
        normalized_distances = torch.sigmoid(scale * distances)
        # Preserve diagonal as 0.0 (distance from sample to itself)
        eye_mask = torch.eye(distances.shape[0], device=distances.device, dtype=distances.dtype)
        normalized_distances = normalized_distances * (1.0 - eye_mask)
        return normalized_distances

    return distances


class MultiTaskLoss(nn.Module):
    """Multi-task loss computation for AAM model."""

    def __init__(
        self,
        penalty: float = 1.0,
        nuc_penalty: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """Initialize MultiTaskLoss.

        Args:
            penalty: Weight for base loss (UniFrac)
            nuc_penalty: Weight for nucleotide loss
            class_weights: Optional class weights for classification (registered as buffer)
        """
        super().__init__()
        self.penalty = penalty
        self.nuc_penalty = nuc_penalty

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.register_buffer("class_weights", None)

    def compute_target_loss(
        self,
        target_pred: torch.Tensor,
        target_true: torch.Tensor,
        is_classifier: bool = False,
    ) -> torch.Tensor:
        """Compute target loss (MSE for regression, NLL for classification).

        Args:
            target_pred: Predicted targets [batch_size, out_dim]
            target_true: True targets [batch_size, out_dim] or [batch_size] for classification
            is_classifier: Whether using classification mode

        Returns:
            Target loss scalar tensor
        """
        if is_classifier:
            return nn.functional.nll_loss(target_pred, target_true, weight=self.class_weights)
        else:
            return nn.functional.mse_loss(target_pred, target_true)

    def compute_count_loss(
        self,
        count_pred: torch.Tensor,
        count_true: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked MSE loss for ASV counts.

        Args:
            count_pred: Predicted counts [batch_size, num_asvs, 1]
            count_true: True counts [batch_size, num_asvs, 1]
            mask: Mask for valid ASVs [batch_size, num_asvs] (1=valid, 0=padding)

        Returns:
            Count loss scalar tensor
        """
        valid_mask = mask.unsqueeze(-1)
        squared_diff = (count_pred - count_true) ** 2
        masked_diff = squared_diff * valid_mask
        num_valid = valid_mask.sum()
        if num_valid == 0:
            return torch.zeros_like(count_pred.sum(), requires_grad=True)
        return masked_diff.sum() / num_valid

    def compute_base_loss(
        self,
        base_pred: torch.Tensor,
        base_true: torch.Tensor,
        encoder_type: str = "unifrac",
        embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute MSE loss for base prediction (UniFrac/Faith PD).

        Args:
            base_pred: Predicted base values [batch_size, base_output_dim] or [batch_size, batch_size] for unifrac
                      For UniFrac with embeddings, this is ignored (use embeddings instead)
            base_true: True base values [batch_size, base_output_dim] or [batch_size, batch_size] for unifrac
            encoder_type: Type of encoder ('unifrac', 'faith_pd', 'taxonomy', 'combined')
            embeddings: Optional embeddings [batch_size, embedding_dim] for UniFrac distance computation

        Returns:
            Base loss scalar tensor
        """
        # For UniFrac, compute distances from embeddings if provided
        if encoder_type == "unifrac" and embeddings is not None:
            # Check for NaN in embeddings before computing distances
            if torch.any(torch.isnan(embeddings)):
                import sys

                error_msg = f"NaN values found in embeddings with shape {embeddings.shape}"
                print(f"ERROR: {error_msg}", file=sys.stderr, flush=True)
                print(
                    f"embeddings {_format_tensor_stats(embeddings)}",
                    file=sys.stderr,
                    flush=True,
                )
                raise ValueError(error_msg)
            if torch.any(torch.isinf(embeddings)):
                import sys

                error_msg = f"Inf values found in embeddings with shape {embeddings.shape}"
                print(f"ERROR: {error_msg}", file=sys.stderr, flush=True)
                raise ValueError(error_msg)
            # Compute pairwise distances from embeddings
            # Normalize to [0, 1] for UniFrac distances (UniFrac distances are bounded)
            try:
                base_pred = compute_pairwise_distances(embeddings, normalize=True)
            except ValueError as e:
                # Re-raise with more context
                import sys

                print(f"ERROR: Failed to compute pairwise distances from embeddings", file=sys.stderr, flush=True)
                print(f"embeddings shape={embeddings.shape}, {_format_tensor_stats(embeddings)}", file=sys.stderr, flush=True)
                raise
        elif encoder_type == "unifrac" and embeddings is None:
            # Legacy mode: use base_pred directly (for backward compatibility)
            pass

        # Validate shapes match
        if base_pred.shape != base_true.shape:
            import sys

            error_msg = (
                f"Shape mismatch in base loss: base_pred.shape={base_pred.shape}, "
                f"base_true.shape={base_true.shape}, encoder_type={encoder_type}"
            )
            print(f"ERROR: {error_msg}", file=sys.stderr, flush=True)
            raise ValueError(error_msg)

        # Check for NaN or Inf values
        if torch.any(torch.isnan(base_pred)):
            import sys

            error_msg = f"NaN values found in base_pred with shape {base_pred.shape}"
            print(f"ERROR: {error_msg}", file=sys.stderr, flush=True)
            print(
                f"base_pred {_format_tensor_stats(base_pred)}",
                file=sys.stderr,
                flush=True,
            )
            # If embeddings were used, provide additional context
            if encoder_type == "unifrac" and embeddings is not None:
                print(f"ERROR: Computed from embeddings with shape {embeddings.shape}", file=sys.stderr, flush=True)
                print(f"embeddings {_format_tensor_stats(embeddings)}", file=sys.stderr, flush=True)
                if torch.any(torch.isnan(embeddings)):
                    print(f"ERROR: Embeddings themselves contain NaN!", file=sys.stderr, flush=True)
            raise ValueError(error_msg)
        if torch.any(torch.isnan(base_true)):
            import sys

            error_msg = f"NaN values found in base_true with shape {base_true.shape}"
            print(f"ERROR: {error_msg}", file=sys.stderr, flush=True)
            print(
                f"base_true {_format_tensor_stats(base_true)}",
                file=sys.stderr,
                flush=True,
            )
            raise ValueError(error_msg)
        if torch.any(torch.isinf(base_pred)):
            import sys

            error_msg = f"Inf values found in base_pred with shape {base_pred.shape}"
            print(f"ERROR: {error_msg}", file=sys.stderr, flush=True)
            raise ValueError(error_msg)
        if torch.any(torch.isinf(base_true)):
            import sys

            error_msg = f"Inf values found in base_true with shape {base_true.shape}"
            print(f"ERROR: {error_msg}", file=sys.stderr, flush=True)
            raise ValueError(error_msg)

        # For UniFrac pairwise distance matrices, mask diagonal elements
        # Diagonal elements are always 0.0 (distance from sample to itself) and provide no training signal
        if encoder_type == "unifrac" and base_pred.dim() == 2 and base_pred.shape[0] == base_pred.shape[1]:
            # Extract upper triangle (excluding diagonal) using offset=1
            batch_size = base_pred.shape[0]
            triu_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=base_pred.device)

            # Handle edge case: batch_size=1 has no off-diagonal elements
            if triu_indices.shape[1] == 0:
                # Return zero loss when there are no off-diagonal elements
                return torch.zeros(1, device=base_pred.device, requires_grad=True)

            base_pred_masked = base_pred[triu_indices[0], triu_indices[1]]
            base_true_masked = base_true[triu_indices[0], triu_indices[1]]
            return nn.functional.mse_loss(base_pred_masked, base_true_masked)
        else:
            # For non-UniFrac encoders or non-square matrices, use existing logic
            return nn.functional.mse_loss(base_pred, base_true)

    def compute_nucleotide_loss(
        self,
        nuc_pred: torch.Tensor,
        nuc_true: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked CrossEntropy loss for nucleotide prediction.

        Args:
            nuc_pred: Predicted nucleotides [batch_size, num_asvs, seq_len, vocab_size]
            nuc_true: True nucleotides [batch_size, num_asvs, seq_len]
            mask: Mask for valid positions [batch_size, num_asvs, seq_len] (1=valid, 0=padding)

        Returns:
            Nucleotide loss scalar tensor
        """
        import sys

        # Check for NaN/Inf in inputs
        if torch.any(torch.isnan(nuc_pred)):
            print(f"ERROR: NaN in nuc_pred before nucleotide loss computation", file=sys.stderr, flush=True)
            print(
                f"nuc_pred shape={nuc_pred.shape}, min={nuc_pred.min().item()}, max={nuc_pred.max().item()}",
                file=sys.stderr,
                flush=True,
            )
            raise ValueError(f"NaN values found in nuc_pred with shape {nuc_pred.shape}")
        if torch.any(torch.isinf(nuc_pred)):
            print(f"ERROR: Inf in nuc_pred before nucleotide loss computation", file=sys.stderr, flush=True)
            raise ValueError(f"Inf values found in nuc_pred with shape {nuc_pred.shape}")

        # Check target values are valid (within vocab_size range)
        vocab_size = nuc_pred.size(-1)
        if torch.any(nuc_true >= vocab_size) or torch.any(nuc_true < 0):
            invalid_mask = (nuc_true >= vocab_size) | (nuc_true < 0)
            invalid_count = invalid_mask.sum().item()
            print(f"ERROR: Invalid target values in nuc_true", file=sys.stderr, flush=True)
            print(f"nuc_true shape={nuc_true.shape}, vocab_size={vocab_size}", file=sys.stderr, flush=True)
            print(
                f"nuc_true min={nuc_true.min().item()}, max={nuc_true.max().item()}, invalid_count={invalid_count}",
                file=sys.stderr,
                flush=True,
            )
            raise ValueError(
                f"Invalid target values in nuc_true: min={nuc_true.min().item()}, max={nuc_true.max().item()}, vocab_size={vocab_size}"
            )

        nuc_pred_flat = nuc_pred.view(-1, nuc_pred.size(-1))
        nuc_true_flat = nuc_true.view(-1)
        mask_flat = mask.view(-1)

        valid_indices = mask_flat.bool()
        num_valid = valid_indices.sum().item()

        if num_valid == 0:
            print(f"WARNING: No valid positions in mask for nucleotide loss", file=sys.stderr, flush=True)
            return torch.zeros_like(nuc_pred.sum(), requires_grad=True)

        valid_pred = nuc_pred_flat[valid_indices]
        valid_true = nuc_true_flat[valid_indices]

        # Final check before cross_entropy
        if torch.any(torch.isnan(valid_pred)):
            print(f"ERROR: NaN in valid_pred after masking", file=sys.stderr, flush=True)
            print(f"valid_pred shape={valid_pred.shape}, num_valid={num_valid}", file=sys.stderr, flush=True)
            raise ValueError(f"NaN values found in valid_pred after masking")

        loss = nn.functional.cross_entropy(valid_pred, valid_true)

        if torch.any(torch.isnan(loss)):
            print(f"ERROR: NaN in nucleotide loss after cross_entropy", file=sys.stderr, flush=True)
            print(f"valid_pred shape={valid_pred.shape}, valid_true shape={valid_true.shape}", file=sys.stderr, flush=True)
            print(
                f"valid_pred {_format_tensor_stats(valid_pred)}",
                file=sys.stderr,
                flush=True,
            )
            # valid_true is integer (token indices), so format it separately
            print(f"valid_true min={valid_true.min().item()}, max={valid_true.max().item()}", file=sys.stderr, flush=True)
            raise ValueError(f"NaN in nucleotide loss after cross_entropy computation")

        return loss

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        is_classifier: bool = False,
        encoder_type: str = "unifrac",
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses.

        Args:
            outputs: Dictionary with model outputs (target_prediction, count_prediction, base_prediction, nuc_predictions)
            targets: Dictionary with targets (target, counts, base_target, nucleotides, mask, nuc_mask)
            is_classifier: Whether using classification mode
            encoder_type: Type of encoder for base loss

        Returns:
            Dictionary with individual losses and total_loss
        """
        device = None
        reference_tensor = None
        if outputs:
            reference_tensor = next(iter(outputs.values()))
            device = reference_tensor.device

        losses = {}

        if "target_prediction" in outputs and "target" in targets:
            losses["target_loss"] = self.compute_target_loss(
                outputs["target_prediction"],
                targets["target"],
                is_classifier=is_classifier,
            )
        else:
            if reference_tensor is not None:
                losses["target_loss"] = torch.zeros_like(reference_tensor.sum(), requires_grad=True)
            else:
                losses["target_loss"] = torch.tensor(0.0, device=device if device else torch.device("cpu"), requires_grad=True)

        if "count_prediction" in outputs and "counts" in targets:
            if "mask" in targets:
                mask = targets["mask"]
            elif "tokens" in targets:
                mask = (targets["tokens"].sum(dim=-1) > 0).long()
            else:
                mask = (targets["counts"].squeeze(-1) > 0).long()
            losses["count_loss"] = self.compute_count_loss(
                outputs["count_prediction"],
                targets["counts"],
                mask,
            )
        else:
            if reference_tensor is not None:
                losses["count_loss"] = torch.zeros_like(reference_tensor.sum(), requires_grad=True)
            else:
                losses["count_loss"] = torch.tensor(0.0, device=device if device else torch.device("cpu"), requires_grad=True)

        if "base_target" in targets:
            # For UniFrac, use embeddings if available (new approach)
            # Otherwise fall back to base_prediction (for backward compatibility)
            if encoder_type == "unifrac" and "embeddings" in outputs:
                # New approach: compute distances from embeddings
                embeddings = outputs["embeddings"]
                # Pass dummy base_pred (will be replaced by computed distances in compute_base_loss)
                base_pred = torch.zeros(1, device=embeddings.device)
                losses["unifrac_loss"] = self.compute_base_loss(
                    base_pred,
                    targets["base_target"],
                    encoder_type=encoder_type,
                    embeddings=embeddings,
                )
            elif "base_prediction" in outputs:
                # Legacy approach: use base_prediction directly
                losses["unifrac_loss"] = self.compute_base_loss(
                    outputs["base_prediction"],
                    targets["base_target"],
                    encoder_type=encoder_type,
                )
            else:
                # No base prediction available
                if reference_tensor is not None:
                    losses["unifrac_loss"] = torch.zeros_like(reference_tensor.sum(), requires_grad=True)
                else:
                    losses["unifrac_loss"] = torch.tensor(
                        0.0, device=device if device else torch.device("cpu"), requires_grad=True
                    )
        else:
            if reference_tensor is not None:
                losses["unifrac_loss"] = torch.zeros_like(reference_tensor.sum(), requires_grad=True)
            else:
                losses["unifrac_loss"] = torch.tensor(0.0, device=device if device else torch.device("cpu"), requires_grad=True)

        if "nuc_predictions" in outputs:
            if "nucleotides" in targets:
                nuc_true = targets["nucleotides"]
            elif "tokens" in targets:
                nuc_true = targets["tokens"]
            else:
                nuc_true = None

            if nuc_true is not None:
                if "nuc_mask" in targets:
                    nuc_mask = targets["nuc_mask"]
                elif "tokens" in targets:
                    nuc_mask = (targets["tokens"] > 0).long()
                else:
                    nuc_mask = torch.ones_like(nuc_true, dtype=torch.long)
                losses["nuc_loss"] = self.compute_nucleotide_loss(
                    outputs["nuc_predictions"],
                    nuc_true,
                    nuc_mask,
                )
            else:
                if reference_tensor is not None:
                    losses["nuc_loss"] = torch.zeros_like(reference_tensor.sum(), requires_grad=True)
                else:
                    losses["nuc_loss"] = torch.tensor(0.0, device=device if device else torch.device("cpu"), requires_grad=True)
        else:
            if reference_tensor is not None:
                losses["nuc_loss"] = torch.zeros_like(reference_tensor.sum(), requires_grad=True)
            else:
                losses["nuc_loss"] = torch.tensor(0.0, device=device if device else torch.device("cpu"), requires_grad=True)

        total_loss = (
            losses["target_loss"]
            + losses["count_loss"]
            + losses["unifrac_loss"] * self.penalty
            + losses["nuc_loss"] * self.nuc_penalty
        )
        losses["total_loss"] = total_loss

        return losses
