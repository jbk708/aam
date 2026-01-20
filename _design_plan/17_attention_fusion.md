# Attention-Based Categorical Fusion

**Status:** Proposed
**Tickets:** FUS-1 through FUS-3, CLN-1 through CLN-6

---

## Motivation

Current categorical conditioning mechanisms (concat/add fusion, FiLM) apply **uniform transforms across all ASV positions**. For microbiome data where different taxa respond differently to environmental conditions, this is architecturally limiting.

### Current Limitations

**Position-agnostic fusion:**
```
Categorical embedding: [B, cat_dim]
    ↓ broadcast
[B, S, cat_dim] (same embedding for all S positions)
    ↓ concat/add
Base embeddings: [B, S, D]
```

Every ASV receives identical categorical modulation. A taxa that thrives in summer vs one that's dormant receives the same "summer" embedding.

**FiLM operates post-pooling:**
```
Base embeddings [B, S, D] → AttentionPooling → [B, D] → FiLM-MLP → prediction
```

FiLM modulates the MLP hidden layers AFTER pooling, so it cannot selectively weight which ASVs matter for a given category.

### Architectural Goal

Enable **position-specific (ASV-specific) metadata modulation** where different taxa can attend to categorical information differently:

```
Cross-Attention Fusion:
    Sequence [B, S, D] --query--> Attention <--key/value-- Metadata [B, K, E]
                                      ↓
                            Position-specific update [B, S, D]
```

ASV at position i can learn to attend strongly to "summer" while ASV at position j attends to "location=outdoor".

---

## Architecture Options

### Option 1: Gated Multimodal Unit (GMU) - Baseline

Simple gated fusion operating on pooled representations. Not position-specific, but provides learned modality weighting.

```
┌─────────────────┐     ┌──────────────────┐
│ Pooled Sequence │     │ Cat Embeddings   │
│    [B, D]       │     │    [B, C]        │
└────────┬────────┘     └────────┬─────────┘
         │                       │
    ┌────▼────┐             ┌────▼────┐
    │  tanh   │             │  tanh   │
    │ Linear  │             │ Linear  │
    └────┬────┘             └────┬────┘
         │                       │
         └───────────┬───────────┘
                     │
              ┌──────▼──────┐
              │   σ(gate)   │
              │ z * seq +   │
              │ (1-z) * cat │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │   Output    │
              │   [B, D]    │
              └─────────────┘
```

**Pros:** Minimal code, fast iteration, interpretable gate values
**Cons:** Not position-specific, operates after pooling

### Option 2: Cross-Attention Fusion - Primary

Sequence tokens attend to metadata tokens, enabling position-specific conditioning.

```
┌─────────────────┐     ┌──────────────────┐
│ Sequence Repr   │     │ Metadata Tokens  │
│  [B, S, D]      │     │   [B, K, E]      │
│  (ASV embeds)   │     │ (cat embeds)     │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         │    ┌──────────────────┤
         │    │                  │
    ┌────▼────▼────┐        ┌────▼────┐
    │   Q = seq    │        │ K, V =  │
    │              │        │ metadata│
    └──────┬───────┘        └────┬────┘
           │                     │
           └─────────┬───────────┘
                     │
              ┌──────▼──────┐
              │ MultiHead   │
              │ CrossAttn   │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │ LayerNorm   │
              │ + Residual  │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │   Output    │
              │  [B, S, D]  │
              └─────────────┘
```

**Pros:** Position-specific modulation, attention weights interpretable
**Cons:** More parameters, O(S×K) attention complexity

### Option 3: Perceiver-Style Latent Fusion - Advanced

Learned latent bottleneck for handling variable/missing metadata.

```
┌─────────────────┐     ┌──────────────────┐
│ Sequence Repr   │     │ Metadata Tokens  │
│  [B, S, D]      │     │   [B, K, E]      │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     │
              ┌──────▼──────┐
              │   Concat    │
              │ [B, S+K, D] │
              └──────┬──────┘
                     │
         ┌───────────▼───────────┐
         │    Learned Latents    │
         │     [num_latents, D]  │
         └───────────┬───────────┘
                     │
              ┌──────▼──────┐
              │ Cross-Attn  │
              │ latents→inp │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │ Self-Attn   │
              │  × N layers │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │ Pool→Output │
              │   [B, D]    │
              └─────────────┘
```

**Pros:** Linear complexity O(L×(S+K)), handles missing data gracefully
**Cons:** Most complex, additional hyperparameters (num_latents, num_layers)

---

## Integration Points

### Current Forward Pass (sequence_predictor.py)

```python
def forward(self, tokens, categorical_ids=None, ...):
    # Base model produces embeddings
    base_outputs = self.base_model(tokens, ...)
    base_embeddings = base_outputs["sample_embeddings"]  # [B, S, D]

    # Current: Position-agnostic fusion
    target_input = self._fuse_categorical(base_embeddings, categorical_ids)  # Line 576

    # Target encoder processes fused embeddings
    target_embeddings = self.target_encoder(target_input, mask=asv_mask)
    pooled_target = self.target_pooling(target_embeddings, mask=asv_mask)

    # MLP head (with optional FiLM)
    target_prediction = self.target_head(pooled_target, ...)
```

### Proposed Integration

**Cross-Attention (Option 2):** Replace `_fuse_categorical` at line 576

```python
def forward(self, tokens, categorical_ids=None, ...):
    base_embeddings = base_outputs["sample_embeddings"]  # [B, S, D]

    # NEW: Position-specific cross-attention fusion
    if self.categorical_fusion == "cross-attention":
        target_input = self.cross_attn_fusion(base_embeddings, categorical_ids)
    else:
        target_input = self._fuse_categorical(base_embeddings, categorical_ids)

    target_embeddings = self.target_encoder(target_input, mask=asv_mask)
    # ... rest unchanged
```

**GMU (Option 1):** Apply after pooling, before target_head

```python
def forward(self, tokens, categorical_ids=None, ...):
    # ... existing code ...
    pooled_target = self.target_pooling(target_embeddings, mask=asv_mask)

    # NEW: GMU fusion on pooled representation
    if self.categorical_fusion == "gmu":
        cat_emb = self.categorical_embedder(categorical_ids)
        pooled_target = self.gmu(pooled_target, cat_emb)

    target_prediction = self.target_head(pooled_target, ...)
```

---

## Implementation Tickets

### FUS-1: Gated Multimodal Unit (GMU)
**Priority:** HIGH | **Effort:** 3-4 hours | **Status:** Not Started

Fast baseline for validating attention-based fusion approach.

**Scope:**
- Create `aam/models/fusion.py` with `GMU` class
- Add `--categorical-fusion gmu` option (extend existing choices)
- GMU operates on pooled representations after `target_pooling`
- Log gate values to TensorBoard for interpretability

**Implementation:**
```python
class GMU(nn.Module):
    """Gated Multimodal Unit for adaptive modality weighting."""

    def __init__(self, seq_dim: int, cat_dim: int):
        super().__init__()
        self.seq_transform = nn.Linear(seq_dim, seq_dim)
        self.cat_transform = nn.Linear(cat_dim, seq_dim)
        self.gate = nn.Linear(seq_dim + cat_dim, seq_dim)

    def forward(self, h_seq: torch.Tensor, h_cat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_seq: Pooled sequence representation [B, D]
            h_cat: Categorical embeddings [B, cat_dim]

        Returns:
            Fused representation [B, D]
        """
        h_seq_t = torch.tanh(self.seq_transform(h_seq))
        h_cat_t = torch.tanh(self.cat_transform(h_cat))
        z = torch.sigmoid(self.gate(torch.cat([h_seq, h_cat], dim=-1)))
        return z * h_seq_t + (1 - z) * h_cat_t
```

**Acceptance Criteria:**
- [ ] `--categorical-fusion gmu` enables GMU fusion
- [ ] Works with existing `--categorical-columns`
- [ ] Gate values (mean z per batch) logged to TensorBoard
- [ ] Validation metrics comparable or better than concat/add
- [ ] 15+ unit tests in `tests/test_fusion.py`

**Files:**
- `aam/models/fusion.py` - New: GMU class
- `aam/models/sequence_predictor.py` - Integrate GMU option
- `aam/cli/train.py` - Extend `--categorical-fusion` choices
- `tests/test_fusion.py` - New test file

**Dependencies:** None

---

### FUS-2: Cross-Attention Fusion
**Priority:** HIGH | **Effort:** 5-6 hours | **Status:** Not Started

Position-specific metadata modulation via cross-attention.

**Scope:**
- Add `CrossAttentionFusion` class to `aam/models/fusion.py`
- Sequence tokens attend to metadata tokens (categorical embeddings)
- Add `--categorical-fusion cross-attention` option
- Log attention weights to TensorBoard (which ASVs attend to which categories)

**Implementation:**
```python
class CrossAttentionFusion(nn.Module):
    """Cross-attention for position-specific categorical conditioning."""

    def __init__(
        self,
        seq_dim: int,
        cat_cardinalities: Dict[str, int],
        embed_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        embed_dim = embed_dim or seq_dim

        # Per-category embeddings (separate from CategoricalEmbedder for flexibility)
        self.cat_embeddings = nn.ModuleDict({
            col: nn.Embedding(card, embed_dim, padding_idx=0)
            for col, card in cat_cardinalities.items()
        })

        # Cross-attention: sequence queries, metadata keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=seq_dim,
            num_heads=num_heads,
            kdim=embed_dim,
            vdim=embed_dim,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(seq_dim)

    def forward(
        self,
        seq_repr: torch.Tensor,
        cat_ids: Dict[str, torch.Tensor],
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            seq_repr: Sequence representations [B, S, D]
            cat_ids: Dict mapping column name to indices [B]
            return_weights: Whether to return attention weights

        Returns:
            Fused representations [B, S, D], optionally with attention weights [B, S, K]
        """
        # Build metadata tokens: [B, K, embed_dim]
        metadata_tokens = []
        for col in sorted(self.cat_embeddings.keys()):
            if col in cat_ids:
                emb = self.cat_embeddings[col](cat_ids[col])  # [B, E]
                metadata_tokens.append(emb.unsqueeze(1))  # [B, 1, E]

        if not metadata_tokens:
            return (seq_repr, None) if return_weights else seq_repr

        metadata = torch.cat(metadata_tokens, dim=1)  # [B, K, E]

        # Cross-attention with residual
        attn_out, attn_weights = self.cross_attn(
            query=seq_repr, key=metadata, value=metadata
        )
        output = self.norm(seq_repr + attn_out)

        return (output, attn_weights) if return_weights else output
```

**Acceptance Criteria:**
- [ ] `--categorical-fusion cross-attention` enables cross-attention
- [ ] Different ASV positions can attend differently to metadata
- [ ] Attention weights logged to TensorBoard (heatmap visualization)
- [ ] Configurable `--cross-attn-heads` (default 8)
- [ ] 20+ unit tests

**Files:**
- `aam/models/fusion.py` - Add CrossAttentionFusion
- `aam/models/sequence_predictor.py` - Integrate cross-attention
- `aam/cli/train.py` - Add `--cross-attn-heads` flag
- `aam/training/trainer.py` - Log attention weights to TensorBoard
- `tests/test_fusion.py` - Test cross-attention

**Dependencies:** None

---

### FUS-3: Perceiver-Style Latent Fusion
**Priority:** LOW | **Effort:** 6-8 hours | **Status:** Not Started

For handling variable/missing metadata with linear complexity bottleneck.

**Scope:**
- Add `PerceiverFusion` class to `aam/models/fusion.py`
- Learned latents attend to concatenated sequence + metadata
- Self-attention refinement layers on latents
- Add `--categorical-fusion perceiver` option

**Implementation:**
```python
class PerceiverFusion(nn.Module):
    """Perceiver-style latent bottleneck for multimodal fusion."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 256,
        num_latents: int = 64,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # Input projection
        self.input_proj = nn.Linear(input_dim, latent_dim)

        # Cross-attention: latents attend to inputs
        self.cross_attn = nn.MultiheadAttention(
            latent_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.cross_norm = nn.LayerNorm(latent_dim)

        # Self-attention layers on latents
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                latent_dim, nhead=num_heads, batch_first=True, dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        seq_repr: torch.Tensor,
        cat_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            seq_repr: Sequence representations [B, S, D]
            cat_embeds: Categorical embeddings [B, K, D] or [B, C] (will broadcast)

        Returns:
            Latent representations [B, num_latents, latent_dim] or pooled [B, latent_dim]
        """
        B = seq_repr.size(0)

        # Ensure cat_embeds is 3D
        if cat_embeds.dim() == 2:
            cat_embeds = cat_embeds.unsqueeze(1)  # [B, 1, C]

        # Concatenate all inputs
        inputs = torch.cat([seq_repr, cat_embeds], dim=1)  # [B, S+K, D]
        inputs = self.input_proj(inputs)  # [B, S+K, latent_dim]

        # Expand latents for batch
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B, L, latent_dim]

        # Cross-attend latents to inputs
        attn_out, _ = self.cross_attn(query=latents, key=inputs, value=inputs)
        latents = self.cross_norm(latents + attn_out)

        # Refine via self-attention
        for layer in self.self_attn_layers:
            latents = layer(latents)

        # Pool for final representation
        return latents.mean(dim=1)  # [B, latent_dim]
```

**Acceptance Criteria:**
- [ ] `--categorical-fusion perceiver` enables perceiver fusion
- [ ] Configurable `--perceiver-num-latents`, `--perceiver-num-layers`
- [ ] Handles missing categorical columns gracefully
- [ ] 15+ unit tests

**Files:**
- `aam/models/fusion.py` - Add PerceiverFusion
- `aam/models/sequence_predictor.py` - Integrate perceiver
- `aam/cli/train.py` - Add perceiver configuration flags
- `tests/test_fusion.py` - Test perceiver

**Dependencies:** FUS-1 (for shared infrastructure)

---

## Code Cleanup Tickets

Based on codebase analysis, the following consolidation and cleanup work is recommended.

### CLN-1: Consolidate Output Constraint Flags
**Priority:** MEDIUM | **Effort:** 3-4 hours | **Status:** Not Started

Replace three overlapping mechanisms with unified interface.

**Current State:**
```python
--bounded-targets          # sigmoid → [0, 1]
--output-activation        # relu, softplus, exp
--learnable-output-scale   # learnable scale + bias
```

**Problems:**
- Mutual exclusivity validation scattered across code
- Users must understand subtle differences
- `--learnable-output-scale` overlaps with `--output-activation`

**Proposed:**
```python
--output-constraint none|bounded|nonnegative|nonnegative-learnable
```

| Old Flags | New Value |
|-----------|-----------|
| (none) | `none` |
| `--bounded-targets` | `bounded` |
| `--output-activation softplus` | `nonnegative` |
| `--output-activation softplus --learnable-output-scale` | `nonnegative-learnable` |

**Acceptance Criteria:**
- [ ] Single `--output-constraint` flag replaces three old flags
- [ ] Old flags deprecated with warning (removed in next major version)
- [ ] Validation logic simplified
- [ ] Documentation updated
- [ ] 10+ migration tests

**Files:**
- `aam/cli/train.py` - New flag, deprecation warnings
- `aam/models/sequence_predictor.py` - Simplify output logic
- `tests/test_cli.py` - Migration tests

---

### CLN-2: Unify Target Normalization
**Priority:** MEDIUM | **Effort:** 3-4 hours | **Status:** Not Started

Replace fragmented normalization options with single interface.

**Current State:**
```python
--normalize-targets/--no-normalize-targets    # Global min-max
--normalize-targets-by <columns>              # Per-category z-score
--log-transform-targets                       # log(y+1)
```

**Problems:**
- Mutual exclusivity validation
- Implicit flag dependencies (log+normalize auto-enables bounded)
- Three separate code paths

**Proposed:**
```python
--target-transform none|minmax|zscore|zscore-category|log-minmax|log-zscore
--normalize-by <columns>  # Only used with zscore-category
```

**Acceptance Criteria:**
- [ ] Single `--target-transform` flag replaces old flags
- [ ] `--normalize-by` only valid with `zscore-category`
- [ ] Old flags deprecated with warning
- [ ] Implicit behaviors made explicit
- [ ] 15+ tests

**Files:**
- `aam/cli/train.py` - New flag, deprecation warnings
- `aam/data/dataset.py` - Simplify normalization logic
- `aam/data/normalization.py` - Consolidate implementations
- `tests/test_normalization.py` - Migration tests

---

### CLN-3: Remove Unused Parameters
**Priority:** LOW | **Effort:** 1-2 hours | **Status:** Not Started

Remove dead code and unused function parameters.

**Items to Remove:**
1. `intermediate_size` parameters from model constructors (always auto-computed as 4×embedding_dim)
2. `is_rocm()` utility in `aam/cli/utils.py` (never called)
3. `unifrac_loader=None` parameter in `inference_collate` (predict.py)
4. `CategoricalSchema` class (unused, `CategoricalEncoder.from_dict` used instead)

**Acceptance Criteria:**
- [ ] Identified parameters removed
- [ ] No test failures
- [ ] API surface simplified
- [ ] Type hints updated

**Files:**
- `aam/models/sequence_predictor.py` - Remove intermediate_size params
- `aam/models/sequence_encoder.py` - Remove intermediate_size params
- `aam/cli/utils.py` - Remove is_rocm()
- `aam/cli/predict.py` - Remove unifrac_loader param
- `aam/data/categorical.py` - Remove CategoricalSchema (or document if planned for future)

---

### CLN-4: Extract Shared Training Utilities
**Priority:** LOW | **Effort:** 2-3 hours | **Status:** Not Started

Reduce code duplication between `pretrain.py` and `train.py`.

**Duplicated Code:**
1. Scheduler creation logic (~50 lines)
2. Distributed validation checks (~15 lines)
3. Checkpoint saving/loading patterns (~30 lines)
4. DataLoader creation (~40 lines)

**Proposed:**
Create `aam/cli/training_utils.py` with:
```python
def create_scheduler(optimizer, config) -> lr_scheduler
def validate_distributed_config(args) -> None
def create_data_loaders(dataset, config) -> Tuple[DataLoader, DataLoader]
```

**Acceptance Criteria:**
- [ ] Shared utilities extracted
- [ ] Both CLI scripts use shared code
- [ ] No behavior changes
- [ ] Tests verify equivalence

**Files:**
- `aam/cli/training_utils.py` - New shared utilities
- `aam/cli/pretrain.py` - Use shared utilities
- `aam/cli/train.py` - Use shared utilities

---

### CLN-5: Add DataParallel to train.py
**Priority:** MEDIUM | **Effort:** 2-3 hours | **Status:** Not Started

Feature parity: DataParallel exists in `pretrain.py` but not `train.py`.

**Context:**
DataParallel preserves full pairwise comparisons for UniFrac auxiliary loss during training. Currently only available for pretraining.

**Scope:**
- Add `--data-parallel` flag to `train.py`
- Copy DataParallel setup from `pretrain.py`
- Mutually exclusive with `--distributed` and `--fsdp`

**Acceptance Criteria:**
- [ ] `--data-parallel` available in train.py
- [ ] Works with UniFrac auxiliary loss
- [ ] Validation prevents combining with DDP/FSDP
- [ ] Documentation updated
- [ ] 5+ tests

**Files:**
- `aam/cli/train.py` - Add DataParallel support
- `tests/test_cli.py` - Test flag combinations

---

### CLN-6: Simplify Categorical Conditioning Architecture
**Priority:** MEDIUM | **Effort:** 4-5 hours | **Status:** Not Started

Consolidate three parallel categorical conditioning systems.

**Current Systems:**
1. **Base fusion** (`--categorical-fusion concat|add`)
2. **Conditional scaling** (`--conditional-output-scaling`)
3. **FiLM** (`--film-conditioning`)

**Problems:**
- Three independent code paths for same concept
- No guidance on when to use which
- Overlapping but not composable

**Proposed:**
Document clear use cases and add validation:

| Use Case | Recommended |
|----------|-------------|
| Simple metadata | `--categorical-fusion concat` |
| Per-category shift | `--categorical-fusion add` + `--conditional-output-scaling` |
| Feature modulation | `--film-conditioning` (replaces base fusion) |
| Position-specific | `--categorical-fusion cross-attention` (FUS-2) |

**Acceptance Criteria:**
- [ ] Add `--categorical-help` flag showing decision tree
- [ ] Warn if redundant flags used together
- [ ] Documentation with clear recommendations
- [ ] No breaking changes

**Files:**
- `aam/cli/train.py` - Add validation, help
- `README.md` - Document categorical options
- `tests/test_cli.py` - Test validation warnings

---

## Summary

### Fusion Tickets (MVP)

| Ticket | Description | Effort | Priority |
|--------|-------------|--------|----------|
| **FUS-1** | GMU baseline | 3-4h | HIGH |
| **FUS-2** | Cross-attention fusion | 5-6h | HIGH |
| **FUS-3** | Perceiver fusion | 6-8h | LOW |
| **Total** | | **14-18h** | |

**Recommended MVP:** FUS-1 + FUS-2 (~9 hours)

### Cleanup Tickets

| Ticket | Description | Effort | Priority |
|--------|-------------|--------|----------|
| **CLN-1** | Consolidate output constraints | 3-4h | MEDIUM |
| **CLN-2** | Unify target normalization | 3-4h | MEDIUM |
| **CLN-3** | Remove unused parameters | 1-2h | LOW |
| **CLN-4** | Extract shared utilities | 2-3h | LOW |
| **CLN-5** | Add DataParallel to train.py | 2-3h | MEDIUM |
| **CLN-6** | Simplify categorical architecture | 4-5h | MEDIUM |
| **Total** | | **15-21h** | |

**Recommended cleanup order:**
1. CLN-3 (quick win, removes dead code)
2. CLN-5 (feature parity fix)
3. CLN-1 + CLN-2 (user-facing simplification)
4. CLN-4 (internal quality)
5. CLN-6 (depends on FUS-2 being complete)

---

## References

- Arevalo et al., "Gated Multimodal Units for Information Fusion" (arXiv:1702.01992)
- Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data" (FT-Transformer, arXiv:2106.11959)
- Jaegle et al., "Perceiver IO: A General Architecture for Structured Inputs & Outputs" (arXiv:2107.14795)
- Nagrani et al., "Attention Bottlenecks for Multimodal Fusion" (MBT, NeurIPS 2021)
