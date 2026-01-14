"""Main prediction model that composes SequenceEncoder as base model."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from aam.models.sequence_encoder import SequenceEncoder
from aam.models.attention_pooling import AttentionPooling
from aam.models.transformer import AttnImplementation, TransformerEncoder
from aam.models.categorical_embedder import CategoricalEmbedder


class SequencePredictor(nn.Module):
    """Main model for predicting sample-level targets and ASV counts.

    Supports both regression and classification tasks. Composes SequenceEncoder
    as base model to enable transfer learning and multi-task learning.
    """

    def __init__(
        self,
        base_model: Optional[SequenceEncoder] = None,
        encoder_type: str = "unifrac",
        vocab_size: int = 7,
        embedding_dim: int = 128,
        max_bp: int = 150,
        token_limit: int = 1024,
        asv_num_layers: int = 2,
        asv_num_heads: int = 4,
        asv_intermediate_size: Optional[int] = None,
        asv_dropout: float = 0.1,
        asv_activation: str = "gelu",
        sample_num_layers: int = 2,
        sample_num_heads: int = 4,
        sample_intermediate_size: Optional[int] = None,
        sample_dropout: float = 0.1,
        sample_activation: str = "gelu",
        encoder_num_layers: int = 2,
        encoder_num_heads: int = 4,
        encoder_intermediate_size: Optional[int] = None,
        encoder_dropout: float = 0.1,
        encoder_activation: str = "gelu",
        base_output_dim: Optional[int] = None,
        count_num_layers: int = 2,
        count_num_heads: int = 4,
        count_intermediate_size: Optional[int] = None,
        count_dropout: float = 0.1,
        count_activation: str = "gelu",
        target_num_layers: int = 2,
        target_num_heads: int = 4,
        target_intermediate_size: Optional[int] = None,
        target_dropout: float = 0.1,
        target_activation: str = "gelu",
        out_dim: int = 1,
        is_classifier: bool = False,
        freeze_base: bool = False,
        predict_nucleotides: bool = False,
        gradient_checkpointing: bool = False,
        attn_implementation: Optional[AttnImplementation] = "sdpa",
        asv_chunk_size: Optional[int] = None,
        mask_ratio: float = 0.0,
        mask_strategy: str = "random",
        target_layer_norm: bool = True,
        bounded_targets: bool = False,
        learnable_output_scale: bool = False,
        output_activation: str = "none",
        categorical_cardinalities: Optional[Dict[str, int]] = None,
        categorical_embed_dim: int = 16,
        categorical_fusion: str = "concat",
        categorical_dropout: float = 0.1,
        regressor_hidden_dims: Optional[List[int]] = None,
        regressor_dropout: float = 0.0,
        conditional_scaling_columns: Optional[List[str]] = None,
    ):
        """Initialize SequencePredictor.

        Args:
            base_model: Optional SequenceEncoder instance (if None, creates one)
            encoder_type: Type of encoder for base model ('unifrac', 'taxonomy', 'faith_pd', 'combined')
            vocab_size: Vocabulary size (default: 7 for pad, A, C, G, T, START, MASK)
            embedding_dim: Embedding dimension
            max_bp: Maximum sequence length (base pairs)
            token_limit: Maximum number of ASVs per sample
            asv_num_layers: Number of transformer layers for ASV encoder
            asv_num_heads: Number of attention heads for ASV encoder
            asv_intermediate_size: FFN intermediate size for ASV encoder
            asv_dropout: Dropout rate for ASV encoder
            asv_activation: Activation function for ASV encoder ('gelu' or 'relu')
            sample_num_layers: Number of transformer layers for sample-level transformer
            sample_num_heads: Number of attention heads for sample-level transformer
            sample_intermediate_size: FFN intermediate size for sample-level transformer
            sample_dropout: Dropout rate for sample-level transformer
            sample_activation: Activation function for sample-level transformer ('gelu' or 'relu')
            encoder_num_layers: Number of transformer layers for encoder transformer
            encoder_num_heads: Number of attention heads for encoder transformer
            encoder_intermediate_size: FFN intermediate size for encoder transformer
            encoder_dropout: Dropout rate for encoder transformer
            encoder_activation: Activation function for encoder transformer ('gelu' or 'relu')
            base_output_dim: Output dimension for base prediction (None = use embedding_dim)
            count_num_layers: Number of transformer layers for count encoder
            count_num_heads: Number of attention heads for count encoder
            count_intermediate_size: FFN intermediate size for count encoder
            count_dropout: Dropout rate for count encoder
            count_activation: Activation function for count encoder ('gelu' or 'relu')
            target_num_layers: Number of transformer layers for target encoder
            target_num_heads: Number of attention heads for target encoder
            target_intermediate_size: FFN intermediate size for target encoder
            target_dropout: Dropout rate for target encoder
            target_activation: Activation function for target encoder ('gelu' or 'relu')
            out_dim: Output dimension for target prediction
            is_classifier: Whether to use classification (log-softmax) or regression
            freeze_base: Whether to freeze base model parameters
            predict_nucleotides: Whether base model should predict nucleotides
            gradient_checkpointing: Whether to use gradient checkpointing to save memory
            attn_implementation: Which SDPA backend to use ('sdpa', 'flash', 'mem_efficient', 'math')
            asv_chunk_size: Process ASVs in chunks to reduce memory (None = process all)
            mask_ratio: Fraction of nucleotide positions to mask for MAE training (0.0 = no masking)
            mask_strategy: Masking strategy ('random' or 'span')
            target_layer_norm: Apply LayerNorm before target projection (default: True)
            bounded_targets: Apply sigmoid to bound regression output to [0, 1] (default: False)
            learnable_output_scale: Add learnable scale and bias after target projection (default: False)
            output_activation: Activation for non-negative regression outputs. Options: 'none' (default),
                'relu', 'softplus', 'exp'. Cannot be used with bounded_targets or is_classifier.
            categorical_cardinalities: Dict mapping column name to cardinality for categorical conditioning.
                If None, no categorical conditioning is applied.
            categorical_embed_dim: Embedding dimension per categorical column (default: 16)
            categorical_fusion: Fusion strategy for combining categorical embeddings with base embeddings.
                Options: 'concat' (concatenate + project) or 'add' (project + add). Default: 'concat'
            categorical_dropout: Dropout applied to categorical embeddings (default: 0.1)
            regressor_hidden_dims: Hidden layer dimensions for MLP regression head. If None, uses single
                linear layer. E.g., [64, 32] creates MLP: embedding_dim -> 64 -> 32 -> out_dim
            regressor_dropout: Dropout rate between MLP layers (default: 0.0, no dropout)
            conditional_scaling_columns: List of categorical column names to use for conditional output
                scaling. For each column, learns per-category scale and bias parameters applied after
                base prediction: output = prediction * scale[cat] + bias[cat]. Requires categorical_cardinalities.
        """
        super().__init__()

        if base_model is None:
            self.base_model = SequenceEncoder(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                max_bp=max_bp,
                token_limit=token_limit,
                asv_num_layers=asv_num_layers,
                asv_num_heads=asv_num_heads,
                asv_intermediate_size=asv_intermediate_size,
                asv_dropout=asv_dropout,
                asv_activation=asv_activation,
                sample_num_layers=sample_num_layers,
                sample_num_heads=sample_num_heads,
                sample_intermediate_size=sample_intermediate_size,
                sample_dropout=sample_dropout,
                sample_activation=sample_activation,
                encoder_num_layers=encoder_num_layers,
                encoder_num_heads=encoder_num_heads,
                encoder_intermediate_size=encoder_intermediate_size,
                encoder_dropout=encoder_dropout,
                encoder_activation=encoder_activation,
                base_output_dim=base_output_dim,
                encoder_type=encoder_type,
                predict_nucleotides=predict_nucleotides,
                gradient_checkpointing=gradient_checkpointing,
                attn_implementation=attn_implementation,
                asv_chunk_size=asv_chunk_size,
                mask_ratio=mask_ratio,
                mask_strategy=mask_strategy,
            )
            self.embedding_dim = embedding_dim
        else:
            self.base_model = base_model
            self.embedding_dim = base_model.embedding_dim

        self.out_dim = out_dim
        self.is_classifier = is_classifier

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        if count_intermediate_size is None:
            count_intermediate_size = 4 * self.embedding_dim

        self.count_encoder = TransformerEncoder(
            num_layers=count_num_layers,
            num_heads=count_num_heads,
            hidden_dim=self.embedding_dim,
            intermediate_size=count_intermediate_size,
            dropout=count_dropout,
            activation=count_activation,
            gradient_checkpointing=gradient_checkpointing,
            attn_implementation=attn_implementation,
        )

        self.count_head = nn.Linear(self.embedding_dim, 1)

        if target_intermediate_size is None:
            target_intermediate_size = 4 * self.embedding_dim

        self.target_encoder = TransformerEncoder(
            num_layers=target_num_layers,
            num_heads=target_num_heads,
            hidden_dim=self.embedding_dim,
            intermediate_size=target_intermediate_size,
            dropout=target_dropout,
            activation=target_activation,
            gradient_checkpointing=gradient_checkpointing,
            attn_implementation=attn_implementation,
        )

        self.target_pooling = AttentionPooling(hidden_dim=self.embedding_dim)

        self.target_layer_norm_enabled = target_layer_norm
        self.bounded_targets = bounded_targets
        self.learnable_output_scale = learnable_output_scale
        self.output_activation = output_activation

        valid_activations = ("none", "relu", "softplus", "exp")
        if output_activation not in valid_activations:
            raise ValueError(f"output_activation must be one of {valid_activations}, got '{output_activation}'")
        if output_activation != "none" and bounded_targets:
            raise ValueError("Cannot use output_activation with bounded_targets (both constrain output)")
        if output_activation != "none" and is_classifier:
            raise ValueError("Cannot use output_activation with is_classifier (use for regression only)")

        if target_layer_norm:
            self.target_norm = nn.LayerNorm(self.embedding_dim)
        else:
            self.target_norm = None

        self.regressor_hidden_dims = regressor_hidden_dims
        self.regressor_dropout = regressor_dropout
        self.target_head = self._build_target_head(self.embedding_dim, out_dim)

        if learnable_output_scale:
            self.output_scale = nn.Parameter(torch.ones(out_dim))
            self.output_bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.output_scale = None
            self.output_bias = None

        self.categorical_fusion = categorical_fusion
        if categorical_cardinalities:
            if categorical_fusion not in ("concat", "add"):
                raise ValueError(f"categorical_fusion must be 'concat' or 'add', got '{categorical_fusion}'")
            self.categorical_embedder: Optional[CategoricalEmbedder] = CategoricalEmbedder(
                column_cardinalities=categorical_cardinalities,
                embed_dim=categorical_embed_dim,
                dropout=categorical_dropout,
            )
            total_cat_dim = self.categorical_embedder.total_embed_dim
            if categorical_fusion == "concat":
                self.categorical_projection = nn.Linear(
                    self.embedding_dim + total_cat_dim,
                    self.embedding_dim,
                )
            else:  # add
                self.categorical_projection = nn.Linear(
                    total_cat_dim,
                    self.embedding_dim,
                )
        else:
            self.categorical_embedder = None
            self.categorical_projection = None

        self.conditional_scaling_columns = conditional_scaling_columns
        self._init_conditional_scaling(categorical_cardinalities)

        self._init_weights()

    def _build_target_head(self, in_dim: int, out_dim: int) -> nn.Module:
        """Build target prediction head (single layer or MLP).

        Args:
            in_dim: Input dimension (embedding_dim)
            out_dim: Output dimension (number of targets)

        Returns:
            nn.Module: Linear layer or Sequential MLP

        Raises:
            ValueError: If any hidden dimension is not a positive integer
        """
        if self.regressor_hidden_dims is None or len(self.regressor_hidden_dims) == 0:
            return nn.Linear(in_dim, out_dim)

        for i, dim in enumerate(self.regressor_hidden_dims):
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(
                    f"regressor_hidden_dims[{i}] must be a positive integer, got {dim}. "
                    f"Full hidden dims: {self.regressor_hidden_dims}"
                )

        layers: List[nn.Module] = []
        current_dim = in_dim
        for hidden_dim in self.regressor_hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if self.regressor_dropout > 0:
                layers.append(nn.Dropout(self.regressor_dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, out_dim))
        return nn.Sequential(*layers)

    def _init_conditional_scaling(
        self, categorical_cardinalities: Optional[Dict[str, int]]
    ) -> None:
        """Initialize conditional output scaling embeddings.

        Creates per-category scale and bias parameters for each specified column.
        Scale is initialized to 1.0 and bias to 0.0 for identity transform at start.

        Args:
            categorical_cardinalities: Dict mapping column name to number of categories.
                Required if conditional_scaling_columns is set.

        Raises:
            ValueError: If conditional_scaling_columns specified without categorical_cardinalities,
                or if a scaling column is not in categorical_cardinalities.
        """
        if not self.conditional_scaling_columns:
            self.output_scales: Optional[nn.ModuleDict] = None
            self.output_biases: Optional[nn.ModuleDict] = None
            return

        if categorical_cardinalities is None:
            raise ValueError(
                "conditional_scaling_columns requires categorical_cardinalities to be set"
            )

        self.output_scales = nn.ModuleDict()
        self.output_biases = nn.ModuleDict()

        for col in self.conditional_scaling_columns:
            if col not in categorical_cardinalities:
                raise ValueError(
                    f"Conditional scaling column '{col}' not found in categorical_cardinalities. "
                    f"Available columns: {list(categorical_cardinalities.keys())}"
                )
            num_categories = categorical_cardinalities[col]
            scale_emb = nn.Embedding(num_categories, self.out_dim)
            bias_emb = nn.Embedding(num_categories, self.out_dim)
            nn.init.ones_(scale_emb.weight)
            nn.init.zeros_(bias_emb.weight)
            self.output_scales[col] = scale_emb
            self.output_biases[col] = bias_emb

    def _apply_conditional_scaling(
        self,
        prediction: torch.Tensor,
        categorical_ids: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Apply per-category scale and bias to predictions.

        For each conditional scaling column, applies: output = prediction * scale[cat] + bias[cat]
        Multiple columns are applied sequentially (multiplicatively for scales).

        Args:
            prediction: Base predictions [batch_size, out_dim]
            categorical_ids: Dict mapping column name to category indices [batch_size]

        Returns:
            Scaled predictions [batch_size, out_dim]
        """
        if self.output_scales is None or self.conditional_scaling_columns is None:
            return prediction
        if categorical_ids is None:
            return prediction

        for col in self.conditional_scaling_columns:
            if col not in categorical_ids:
                continue
            ids = categorical_ids[col]
            scale = self.output_scales[col](ids)
            bias = self.output_biases[col](ids)
            prediction = prediction * scale + bias

        return prediction

    def _init_weights(self) -> None:
        """Initialize weights for target head, count head, and categorical projection."""
        if isinstance(self.target_head, nn.Linear):
            nn.init.xavier_uniform_(self.target_head.weight)
            nn.init.zeros_(self.target_head.bias)
        else:
            for module in self.target_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.count_head.weight)
        nn.init.zeros_(self.count_head.bias)
        if self.categorical_projection is not None:
            nn.init.xavier_uniform_(self.categorical_projection.weight)
            nn.init.zeros_(self.categorical_projection.bias)

    def _fuse_categorical(
        self,
        base_embeddings: torch.Tensor,
        categorical_ids: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Fuse categorical embeddings with base embeddings for target prediction.

        Categorical embeddings are only applied to the target prediction pathway,
        not the count encoder or base model.

        Args:
            base_embeddings: Base model embeddings [batch_size, num_asvs, embedding_dim]
            categorical_ids: Dict mapping column name to category indices [batch_size],
                or None if no categorical conditioning.

        Returns:
            Fused embeddings [batch_size, num_asvs, embedding_dim]
        """
        if self.categorical_embedder is None or categorical_ids is None:
            return base_embeddings

        assert self.categorical_projection is not None  # Type narrowing

        cat_emb = self.categorical_embedder(categorical_ids)  # [B, cat_dim]
        seq_len = base_embeddings.size(1)
        cat_emb_seq = self.categorical_embedder.broadcast_to_sequence(cat_emb, seq_len)  # [B, S, cat_dim]

        if self.categorical_fusion == "concat":
            fused = torch.cat([base_embeddings, cat_emb_seq], dim=-1)  # [B, S, D + cat_dim]
            return self.categorical_projection(fused)  # [B, S, D]
        else:  # add
            cat_proj = self.categorical_projection(cat_emb_seq)  # [B, S, D]
            return base_embeddings + cat_proj

    def _apply_output_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply output activation for non-negative regression.

        Args:
            x: Raw predictions [batch_size, out_dim]

        Returns:
            Activated predictions with non-negative values
        """
        if self.output_activation == "relu":
            return nn.functional.relu(x)
        elif self.output_activation == "softplus":
            return nn.functional.softplus(x)
        elif self.output_activation == "exp":
            return torch.exp(x)
        return x

    def forward(
        self,
        tokens: torch.Tensor,
        categorical_ids: Optional[Dict[str, torch.Tensor]] = None,
        return_nucleotides: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            tokens: Input tokens [batch_size, num_asvs, seq_len]
            categorical_ids: Optional dict mapping column name to category indices [batch_size].
                Used for categorical conditioning of target predictions.
            return_nucleotides: Whether to return nucleotide predictions

        Returns:
            Dictionary with keys:
                - 'target_prediction': [batch_size, out_dim]
                - 'count_prediction': [batch_size, num_asvs, 1]
                - 'base_embeddings': [batch_size, num_asvs, embedding_dim]
                - 'base_prediction': [batch_size, base_output_dim] (if return_nucleotides=True)
                - 'nuc_predictions': [batch_size, num_asvs, seq_len, vocab_size] (if return_nucleotides=True)
                - 'mask_indices': [batch_size, num_asvs, seq_len] boolean (if masking, else None)
        """
        asv_mask = (tokens.sum(dim=-1) > 0).long()

        base_outputs = self.base_model(tokens, return_nucleotides=return_nucleotides)

        base_embeddings = base_outputs["sample_embeddings"]
        # For UniFrac, embeddings are returned directly (no base_prediction)
        # For other encoder types, base_prediction may exist
        base_prediction = base_outputs.get("base_prediction")
        embeddings = base_outputs.get("embeddings")  # For UniFrac
        nuc_predictions = base_outputs.get("nuc_predictions")
        mask_indices = base_outputs.get("mask_indices")

        count_embeddings = self.count_encoder(base_embeddings, mask=asv_mask)
        count_prediction = torch.sigmoid(self.count_head(count_embeddings))

        target_input = self._fuse_categorical(base_embeddings, categorical_ids)

        target_embeddings = self.target_encoder(target_input, mask=asv_mask)
        pooled_target = self.target_pooling(target_embeddings, mask=asv_mask)

        if self.target_norm is not None:
            pooled_target = self.target_norm(pooled_target)

        target_prediction = self.target_head(pooled_target)

        target_prediction = self._apply_conditional_scaling(target_prediction, categorical_ids)

        if self.output_scale is not None:
            target_prediction = target_prediction * self.output_scale + self.output_bias

        if self.is_classifier:
            target_prediction = nn.functional.log_softmax(target_prediction, dim=-1)
        elif self.bounded_targets:
            target_prediction = torch.sigmoid(target_prediction)
        elif self.output_activation != "none":
            target_prediction = self._apply_output_activation(target_prediction)

        result = {
            "target_prediction": target_prediction,
            "count_prediction": count_prediction,
            "base_embeddings": base_embeddings,
        }

        # For UniFrac, pass embeddings through for distance computation
        if embeddings is not None:
            result["embeddings"] = embeddings

        if return_nucleotides and base_prediction is not None:
            result["base_prediction"] = base_prediction

        if return_nucleotides and nuc_predictions is not None:
            result["nuc_predictions"] = nuc_predictions
            result["mask_indices"] = mask_indices

        if return_nucleotides and "unifrac_pred" in base_outputs:
            result["unifrac_pred"] = base_outputs["unifrac_pred"]
            result["faith_pred"] = base_outputs["faith_pred"]
            result["tax_pred"] = base_outputs["tax_pred"]

        return result
