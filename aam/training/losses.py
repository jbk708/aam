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
        pass

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
        pass

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
        pass

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
        pass

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
            targets: Dictionary with targets (target, counts, base_target, nucleotides)
            is_classifier: Whether using classification mode
            encoder_type: Type of encoder for base loss

        Returns:
            Dictionary with individual losses and total_loss
        """
        pass
