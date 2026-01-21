"""Multi-task loss functions for AAM model."""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, Optional, List


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


def _gather_target_matrices(
    local_target: torch.Tensor,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather target matrices from all ranks and construct global matrix with mask.

    In distributed training, each rank has a local target matrix representing
    pairwise distances within its local batch. This function gathers all local
    matrices and constructs a block-diagonal global matrix.

    Args:
        local_target: Local target matrix [local_batch, local_batch]
        world_size: Number of ranks in distributed training

    Returns:
        Tuple of:
        - global_target: Block-diagonal matrix [global_batch, global_batch]
        - mask: Boolean mask indicating valid (local) entries [global_batch, global_batch]
    """
    local_batch_size = local_target.shape[0]
    global_batch_size = local_batch_size * world_size

    # Gather all local target matrices
    gathered: List[torch.Tensor] = [torch.zeros_like(local_target) for _ in range(world_size)]
    try:
        dist.all_gather(gathered, local_target)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to gather target matrices across {world_size} ranks "
            f"(shape={local_target.shape}, device={local_target.device}): {e}"
        ) from e

    # Construct global target matrix (block-diagonal)
    global_target = torch.zeros(global_batch_size, global_batch_size, dtype=local_target.dtype, device=local_target.device)
    mask = torch.zeros(global_batch_size, global_batch_size, dtype=torch.bool, device=local_target.device)

    # Place each gathered matrix on the diagonal
    for rank_idx, target_block in enumerate(gathered):
        start = rank_idx * local_batch_size
        end = start + local_batch_size
        global_target[start:end, start:end] = target_block
        mask[start:end, start:end] = True

    return global_target, mask


def compute_pairwise_distances(
    embeddings: torch.Tensor,
    normalize: bool = True,
    scale: float = 10.0,
) -> torch.Tensor:
    """Compute pairwise Euclidean distances from embeddings.

    Args:
        embeddings: Sample embeddings [batch_size, embedding_dim]
        normalize: If True, normalize distances to [0, 1] using tanh with fixed scale (default: True)
        scale: Scaling factor for normalization (default: 10.0).

    Returns:
        Pairwise distance matrix [batch_size, batch_size]
        If normalize=True, distances are bounded to [0, 1] using tanh normalization
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
        # Use tanh normalization with fixed scale to bound distances to [0, 1]
        # This avoids sigmoid saturation while maintaining consistent scaling across batches
        # Formula: (tanh(distances / scale) + 1) / 2 maps to [0, 1] with better gradient flow
        if distances.max() > 0:
            # Normalize using tanh: maps to [-1, 1], then shift to [0, 1]
            normalized = distances / scale
            normalized_distances = (torch.tanh(normalized) + 1.0) / 2.0
        else:
            # All distances are 0, return zeros
            normalized_distances = torch.zeros_like(distances)
        # Preserve diagonal as 0.0 (distance from sample to itself)
        eye_mask = torch.eye(distances.shape[0], device=distances.device, dtype=distances.dtype)
        normalized_distances = normalized_distances * (1.0 - eye_mask)
        return normalized_distances

    return distances


def compute_asymmetric_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    over_penalty: float = 1.0,
    under_penalty: float = 1.0,
) -> torch.Tensor:
    """Compute asymmetric loss with different penalties for over/under prediction.

    The loss penalizes over-predictions and under-predictions differently:
        L = over_penalty * max(pred - target, 0) + under_penalty * max(target - pred, 0)

    When over_penalty == under_penalty == 1.0, this reduces to MAE (L1 loss).

    Args:
        pred: Predicted values [batch_size, out_dim]
        target: True values [batch_size, out_dim]
        over_penalty: Penalty weight for over-predictions (pred > target)
        under_penalty: Penalty weight for under-predictions (pred < target)

    Returns:
        Scalar loss averaged over all samples and outputs
    """
    error = pred - target
    over_error = torch.clamp(error, min=0)
    under_error = torch.clamp(-error, min=0)
    loss = over_penalty * over_error + under_penalty * under_error
    return loss.mean()


def compute_pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
) -> torch.Tensor:
    """Compute pinball (quantile) loss for quantile regression.

    The pinball loss penalizes under-predictions and over-predictions asymmetrically
    based on the quantile level τ:
        L(y, ŷ, τ) = τ * max(y - ŷ, 0) + (1 - τ) * max(ŷ - y, 0)

    This can be rewritten as:
        L(y, ŷ, τ) = max(τ * (y - ŷ), (τ - 1) * (y - ŷ))

    Args:
        pred: Predicted quantiles [batch_size, out_dim, num_quantiles]
        target: True values [batch_size, out_dim]
        quantiles: Quantile levels [num_quantiles], values in (0, 1)

    Returns:
        Scalar loss averaged over all samples, outputs, and quantiles
    """
    # Ensure quantiles are on the same device as predictions
    quantiles = quantiles.to(pred.device)

    # Expand target to match pred shape: [batch, out_dim] -> [batch, out_dim, num_quantiles]
    target_expanded = target.unsqueeze(-1).expand_as(pred)

    # Compute error: y - ŷ (positive means underprediction)
    error = target_expanded - pred

    # Expand quantiles for broadcasting: [num_quantiles] -> [1, 1, num_quantiles]
    tau = quantiles.view(1, 1, -1)

    # Pinball loss: max(τ * error, (τ - 1) * error)
    # When error > 0 (underprediction): loss = τ * error
    # When error < 0 (overprediction): loss = (τ - 1) * error = (1 - τ) * |error|
    loss = torch.max(tau * error, (tau - 1) * error)

    # Average over all dimensions
    return loss.mean()


class MultiTaskLoss(nn.Module):
    """Multi-task loss computation for AAM model."""

    VALID_LOSS_TYPES = ("mse", "mae", "huber", "quantile", "asymmetric")

    # Type annotations for attributes
    penalty: float
    nuc_penalty: float
    target_penalty: float
    count_penalty: float
    target_loss_type: str
    class_weights: Optional[torch.Tensor]
    quantiles: Optional[torch.Tensor]
    over_penalty: float
    under_penalty: float

    def __init__(
        self,
        penalty: float = 1.0,
        nuc_penalty: float = 1.0,
        target_penalty: float = 1.0,
        count_penalty: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        target_loss_type: str = "huber",
        quantiles: Optional[List[float]] = None,
        over_penalty: float = 1.0,
        under_penalty: float = 1.0,
    ):
        """Initialize MultiTaskLoss.

        Args:
            penalty: Weight for base loss (UniFrac)
            nuc_penalty: Weight for nucleotide loss
            target_penalty: Weight for target loss (default: 1.0)
            count_penalty: Weight for count loss (default: 1.0)
            class_weights: Optional class weights for classification (registered as buffer)
            target_loss_type: Loss type for regression targets ('mse', 'mae', 'huber', 'quantile', 'asymmetric'). Default: 'huber'
            quantiles: List of quantile levels for quantile regression, e.g., [0.1, 0.5, 0.9].
                Required when target_loss_type='quantile'. Values must be in (0, 1).
            over_penalty: Penalty weight for over-predictions when using asymmetric loss. Default: 1.0.
            under_penalty: Penalty weight for under-predictions when using asymmetric loss. Default: 1.0.
        """
        super().__init__()
        self.penalty = penalty
        self.nuc_penalty = nuc_penalty
        self.target_penalty = target_penalty
        self.count_penalty = count_penalty

        if target_loss_type not in self.VALID_LOSS_TYPES:
            raise ValueError(f"Invalid target_loss_type: {target_loss_type}. Must be one of: {self.VALID_LOSS_TYPES}")
        self.target_loss_type = target_loss_type

        if target_loss_type == "quantile":
            if quantiles is None:
                raise ValueError("quantiles must be provided when target_loss_type='quantile'")
            if len(quantiles) == 0:
                raise ValueError("quantiles must contain at least one value")
            for q in quantiles:
                if not (0 < q < 1):
                    raise ValueError(f"Quantile values must be in (0, 1), got {q}")
            self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))
        else:
            self.register_buffer("quantiles", None)

        if target_loss_type == "asymmetric":
            if over_penalty <= 0:
                raise ValueError(f"over_penalty must be positive, got {over_penalty}")
            if under_penalty <= 0:
                raise ValueError(f"under_penalty must be positive, got {under_penalty}")
        self.over_penalty = over_penalty
        self.under_penalty = under_penalty

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
        """Compute target loss based on configured loss type.

        For classification: NLL loss
        For regression: MSE, MAE, Huber, quantile, or asymmetric loss based on target_loss_type

        Args:
            target_pred: Predicted targets [batch_size, out_dim] or
                [batch_size, out_dim, num_quantiles] for quantile regression
            target_true: True targets [batch_size, out_dim] or [batch_size] for classification
            is_classifier: Whether using classification mode

        Returns:
            Target loss scalar tensor
        """
        if is_classifier:
            return nn.functional.nll_loss(target_pred, target_true, weight=self.class_weights)
        else:
            if self.target_loss_type == "mse":
                return nn.functional.mse_loss(target_pred, target_true)
            elif self.target_loss_type == "mae":
                return nn.functional.l1_loss(target_pred, target_true)
            elif self.target_loss_type == "huber":
                # Huber loss (smooth L1): MSE for small errors, MAE for large errors
                # beta=1.0 is the threshold where it transitions from MSE to MAE
                return nn.functional.smooth_l1_loss(target_pred, target_true, beta=1.0)
            elif self.target_loss_type == "quantile":
                assert self.quantiles is not None  # Validated in __init__
                return compute_pinball_loss(target_pred, target_true, self.quantiles)
            elif self.target_loss_type == "asymmetric":
                return compute_asymmetric_loss(
                    target_pred, target_true, self.over_penalty, self.under_penalty
                )
            else:
                # Fallback to MSE (shouldn't happen due to validation in __init__)
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
        gather_for_distributed: bool = False,
    ) -> torch.Tensor:
        """Compute MSE loss for base prediction (UniFrac/Faith PD).

        Args:
            base_pred: Predicted base values [batch_size, base_output_dim] or [batch_size, batch_size] for unifrac
                      For UniFrac with embeddings, this is ignored (use embeddings instead)
            base_true: True base values [batch_size, base_output_dim] or [batch_size, batch_size] for unifrac
            encoder_type: Type of encoder ('unifrac', 'faith_pd', 'taxonomy', 'combined')
            embeddings: Optional embeddings [batch_size, embedding_dim] for UniFrac distance computation
            gather_for_distributed: If True and in distributed mode, gather embeddings and targets
                across all ranks for full pairwise distance computation. Used for FSDP pretraining.

        Returns:
            Base loss scalar tensor
        """
        # Track distributed gathering mask (set if gather_for_distributed and in multi-GPU mode)
        target_mask = None

        # For UniFrac, compute pairwise distances from embeddings if provided
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

            # Handle distributed gathering for FSDP pretraining
            if gather_for_distributed:
                if not dist.is_initialized():
                    import warnings

                    warnings.warn(
                        "gather_for_distributed=True but distributed not initialized. "
                        "Skipping gathering. Did you forget to initialize the process group?",
                        stacklevel=2,
                    )
                else:
                    from aam.training.distributed import gather_embeddings_for_unifrac, get_world_size

                    world_size = get_world_size()
                    if world_size == 1:
                        import warnings

                        warnings.warn(
                            "gather_for_distributed=True but world_size=1. Gathering has no effect with single GPU.",
                            stacklevel=2,
                        )
                    elif world_size > 1:
                        # Gather embeddings across all ranks for full pairwise computation
                        embeddings = gather_embeddings_for_unifrac(embeddings)
                        # Gather target matrices and construct block-diagonal global matrix
                        base_true, target_mask = _gather_target_matrices(base_true, world_size)

            # Compute pairwise distances from embeddings
            # Normalize to [0, 1] for UniFrac distances (UniFrac distances are bounded)
            try:
                base_pred = compute_pairwise_distances(embeddings)
            except ValueError:
                # Re-raise with more context
                import sys

                print("ERROR: Failed to compute pairwise distances from embeddings", file=sys.stderr, flush=True)
                print(f"embeddings shape={embeddings.shape}, {_format_tensor_stats(embeddings)}", file=sys.stderr, flush=True)
                raise

            # Note: target_mask is used later during MSE computation to only include valid pairs

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
                    print("ERROR: Embeddings themselves contain NaN!", file=sys.stderr, flush=True)
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

            base_pred_triu = base_pred[triu_indices[0], triu_indices[1]]
            base_true_triu = base_true[triu_indices[0], triu_indices[1]]

            # If we have a target_mask from distributed gathering, filter to valid pairs only
            if target_mask is not None:
                mask_triu = target_mask[triu_indices[0], triu_indices[1]]
                valid_count = mask_triu.sum().item()
                if valid_count == 0:
                    return torch.zeros(1, device=base_pred.device, requires_grad=True)
                # Compute MSE only on valid pairs (where target_mask is True)
                squared_errors = (base_pred_triu - base_true_triu) ** 2
                return (squared_errors * mask_triu.float()).sum() / valid_count
            else:
                return nn.functional.mse_loss(base_pred_triu, base_true_triu)
        else:
            # For non-UniFrac encoders, use all elements (no diagonal masking)
            return nn.functional.mse_loss(base_pred, base_true)

    def compute_nucleotide_loss(
        self,
        nuc_pred: torch.Tensor,
        nuc_true: torch.Tensor,
        mask: torch.Tensor,
        masked_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute masked CrossEntropy loss for nucleotide prediction.

        When masked_indices is provided (MAE mode), loss is computed only on
        positions that were masked during training. Otherwise, loss is computed
        on all valid (non-padding) positions.

        Args:
            nuc_pred: Predicted nucleotides [batch_size, num_asvs, seq_len, vocab_size]
            nuc_true: True nucleotides [batch_size, num_asvs, seq_len]
            mask: Mask for valid positions [batch_size, num_asvs, seq_len] (1=valid, 0=padding)
            masked_indices: Boolean mask indicating which positions were masked for MAE
                          [batch_size, num_asvs, seq_len] (True=masked, compute loss here)

        Returns:
            Nucleotide loss scalar tensor
        """
        import sys

        # Check for NaN/Inf in inputs
        if torch.any(torch.isnan(nuc_pred)):
            print("ERROR: NaN in nuc_pred before nucleotide loss computation", file=sys.stderr, flush=True)
            print(
                f"nuc_pred shape={nuc_pred.shape}, min={nuc_pred.min().item()}, max={nuc_pred.max().item()}",
                file=sys.stderr,
                flush=True,
            )
            raise ValueError(f"NaN values found in nuc_pred with shape {nuc_pred.shape}")
        if torch.any(torch.isinf(nuc_pred)):
            print("ERROR: Inf in nuc_pred before nucleotide loss computation", file=sys.stderr, flush=True)
            raise ValueError(f"Inf values found in nuc_pred with shape {nuc_pred.shape}")

        # Check target values are valid (within vocab_size range)
        vocab_size = nuc_pred.size(-1)
        if torch.any(nuc_true >= vocab_size) or torch.any(nuc_true < 0):
            invalid_mask = (nuc_true >= vocab_size) | (nuc_true < 0)
            invalid_count = invalid_mask.sum().item()
            print("ERROR: Invalid target values in nuc_true", file=sys.stderr, flush=True)
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

        # Determine which positions to compute loss on
        if masked_indices is not None:
            # MAE mode: compute loss only on positions that were masked during training
            # AND are valid (not padding)
            masked_indices_flat = masked_indices.view(-1)
            compute_loss_indices = masked_indices_flat & mask_flat.bool()
        else:
            # Standard mode: compute loss on all valid positions
            compute_loss_indices = mask_flat.bool()

        num_to_compute = compute_loss_indices.sum().item()

        if num_to_compute == 0:
            if masked_indices is not None:
                # MAE mode with no masked positions - return zero loss
                return torch.zeros(1, device=nuc_pred.device, dtype=nuc_pred.dtype, requires_grad=True).squeeze()
            else:
                print("WARNING: No valid positions in mask for nucleotide loss", file=sys.stderr, flush=True)
                return torch.zeros_like(nuc_pred.sum(), requires_grad=True)

        valid_pred = nuc_pred_flat[compute_loss_indices]
        valid_true = nuc_true_flat[compute_loss_indices]

        # Final check before cross_entropy
        if torch.any(torch.isnan(valid_pred)):
            print("ERROR: NaN in valid_pred after masking", file=sys.stderr, flush=True)
            print(f"valid_pred shape={valid_pred.shape}, num_to_compute={num_to_compute}", file=sys.stderr, flush=True)
            raise ValueError("NaN values found in valid_pred after masking")

        loss = nn.functional.cross_entropy(valid_pred, valid_true)

        if torch.any(torch.isnan(loss)):
            print("ERROR: NaN in nucleotide loss after cross_entropy", file=sys.stderr, flush=True)
            print(f"valid_pred shape={valid_pred.shape}, valid_true shape={valid_true.shape}", file=sys.stderr, flush=True)
            print(
                f"valid_pred {_format_tensor_stats(valid_pred)}",
                file=sys.stderr,
                flush=True,
            )
            # valid_true is integer (token indices), so format it separately
            print(f"valid_true min={valid_true.min().item()}, max={valid_true.max().item()}", file=sys.stderr, flush=True)
            raise ValueError("NaN in nucleotide loss after cross_entropy computation")

        return loss

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        is_classifier: bool = False,
        encoder_type: str = "unifrac",
        gather_for_distributed: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses.

        Args:
            outputs: Dictionary with model outputs (target_prediction, count_prediction, base_prediction, nuc_predictions)
            targets: Dictionary with targets (target, counts, base_target, nucleotides, mask, nuc_mask)
            is_classifier: Whether using classification mode
            encoder_type: Type of encoder for base loss
            gather_for_distributed: If True and in distributed mode, gather embeddings and targets
                across all ranks for full pairwise distance computation. Used for FSDP pretraining.

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
                # Compute pairwise distances from embeddings
                embeddings = outputs["embeddings"]
                # Pass dummy base_pred (will be replaced by computed distances in compute_base_loss)
                base_pred = torch.zeros(1, device=embeddings.device)
                losses["unifrac_loss"] = self.compute_base_loss(
                    base_pred,
                    targets["base_target"],
                    encoder_type=encoder_type,
                    embeddings=embeddings,
                    gather_for_distributed=gather_for_distributed,
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
                # Get mask_indices for MAE mode (if provided)
                masked_indices = outputs.get("mask_indices")
                losses["nuc_loss"] = self.compute_nucleotide_loss(
                    outputs["nuc_predictions"],
                    nuc_true,
                    nuc_mask,
                    masked_indices=masked_indices,
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
            losses["target_loss"] * self.target_penalty
            + losses["count_loss"] * self.count_penalty
            + losses["unifrac_loss"] * self.penalty
            + losses["nuc_loss"] * self.nuc_penalty
        )
        losses["total_loss"] = total_loss

        return losses
