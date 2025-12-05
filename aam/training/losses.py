"""Multi-task loss functions for AAM model."""

import torch
import torch.nn as nn
from typing import Dict, Optional


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
    ) -> torch.Tensor:
        """Compute MSE loss for base prediction (UniFrac/Faith PD).

        Args:
            base_pred: Predicted base values [batch_size, base_output_dim] or [batch_size, batch_size] for unifrac
            base_true: True base values [batch_size, base_output_dim] or [batch_size, batch_size] for unifrac
            encoder_type: Type of encoder ('unifrac', 'faith_pd', 'taxonomy', 'combined')

        Returns:
            Base loss scalar tensor
        """
        # Validate shapes match
        if base_pred.shape != base_true.shape:
            raise ValueError(
                f"Shape mismatch in base loss: base_pred.shape={base_pred.shape}, "
                f"base_true.shape={base_true.shape}, encoder_type={encoder_type}"
            )
        
        # Check for NaN or Inf values
        if torch.any(torch.isnan(base_pred)):
            raise ValueError(f"NaN values found in base_pred with shape {base_pred.shape}")
        if torch.any(torch.isnan(base_true)):
            raise ValueError(f"NaN values found in base_true with shape {base_true.shape}")
        if torch.any(torch.isinf(base_pred)):
            raise ValueError(f"Inf values found in base_pred with shape {base_pred.shape}")
        if torch.any(torch.isinf(base_true)):
            raise ValueError(f"Inf values found in base_true with shape {base_true.shape}")
        
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
        nuc_pred_flat = nuc_pred.view(-1, nuc_pred.size(-1))
        nuc_true_flat = nuc_true.view(-1)
        mask_flat = mask.view(-1)

        valid_indices = mask_flat.bool()
        if valid_indices.sum() == 0:
            return torch.zeros_like(nuc_pred.sum(), requires_grad=True)

        valid_pred = nuc_pred_flat[valid_indices]
        valid_true = nuc_true_flat[valid_indices]

        return nn.functional.cross_entropy(valid_pred, valid_true)

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

        if "base_prediction" in outputs and "base_target" in targets:
            losses["base_loss"] = self.compute_base_loss(
                outputs["base_prediction"],
                targets["base_target"],
                encoder_type=encoder_type,
            )
        else:
            if reference_tensor is not None:
                losses["base_loss"] = torch.zeros_like(reference_tensor.sum(), requires_grad=True)
            else:
                losses["base_loss"] = torch.tensor(0.0, device=device if device else torch.device("cpu"), requires_grad=True)

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
            + losses["base_loss"] * self.penalty
            + losses["nuc_loss"] * self.nuc_penalty
        )
        losses["total_loss"] = total_loss

        return losses
