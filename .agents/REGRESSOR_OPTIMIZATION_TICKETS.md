# Regressor Optimization Tickets

**Last Updated:** 2026-01-13
**Status:** 9 tickets (~28-42 hours)

---

## Overview

Optimization iterations for the regression prediction head, focusing on:
1. **Categorical compensation** for multi-environment/season data (Tier 1)
2. **Regression head architecture** improvements (Tier 2)
3. **Loss function flexibility** (Tier 3)
4. **Advanced techniques** (Tier 4)

**Key Context:**
- Primary categorical variables: `season`, `location`
- Some locations missing certain seasons (incomplete factorial)
- Large sample spread across categories
- Current pain point: distributional shift between environments/seasons

---

## Tier 1: Categorical Compensation (HIGH Priority)

### REG-1: MLP Regression Head
**Priority:** HIGH | **Effort:** 3-4 hours | **Status:** Complete

Replace single linear output layer with configurable MLP for better feature mapping.

**Background:**
Current architecture uses a single `nn.Linear(embedding_dim, out_dim)` for the final prediction. Adding hidden layers enables non-linear feature transformation before output projection, which is especially important when categorical modulation (REG-3) is added.

**Scope:**
- Add `--regressor-hidden-dims` flag accepting comma-separated dimensions (e.g., `64,32`)
- Add `--regressor-dropout` flag for dropout between MLP layers
- Default behavior (no flag) preserves current single-layer architecture
- MLP uses ReLU activation between layers

**Implementation Notes:**
```python
# In SequencePredictor.__init__
if regressor_hidden_dims:
    layers = []
    in_dim = embedding_dim
    for hidden_dim in regressor_hidden_dims:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        if regressor_dropout > 0:
            layers.append(nn.Dropout(regressor_dropout))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, out_dim))
    self.target_head = nn.Sequential(*layers)
else:
    self.target_head = nn.Linear(embedding_dim, out_dim)
```

**Acceptance Criteria:**
- [x] `--regressor-hidden-dims 64,32` creates 3-layer MLP (128→64→32→out)
- [x] `--regressor-dropout 0.1` adds dropout between layers
- [x] Default behavior unchanged (single linear layer)
- [x] Works with all existing output transforms (sigmoid, softplus, etc.)
- [x] Unit tests for MLP construction and forward pass

**Files:**
- `aam/models/sequence_predictor.py` - Add MLP head construction
- `aam/cli/train.py` - Add `--regressor-hidden-dims`, `--regressor-dropout` flags
- `tests/test_sequence_predictor.py` - Test MLP head configurations

**Dependencies:** None

---

### REG-2: Per-Category Target Normalization
**Priority:** HIGH | **Effort:** 3-4 hours | **Status:** Complete

Normalize targets within each category during training to remove distributional shift.

**Background:**
Different environments/seasons have different baseline target values and variances. By normalizing targets per-category, the model learns residuals from category-specific means rather than absolute values. At inference, predictions are denormalized using the category statistics.

**Problem Example:**
```
Location A: targets mean=0.3, std=0.1
Location B: targets mean=0.7, std=0.2
→ Model struggles to learn both distributions simultaneously
```

**Scope:**
- Add `--normalize-targets-by` flag accepting categorical column name(s)
- Compute per-category mean/std from training data
- Store statistics in checkpoint for inference
- Denormalize predictions at inference time
- Handle unseen categories gracefully (fall back to global stats)

**Implementation Notes:**
```python
# In dataset or training setup
class CategoryNormalizer:
    def __init__(self, targets: np.ndarray, categories: pd.Series):
        self.stats = {}
        for cat in categories.unique():
            mask = categories == cat
            self.stats[cat] = {
                'mean': targets[mask].mean(),
                'std': targets[mask].std() + 1e-8
            }
        self.global_mean = targets.mean()
        self.global_std = targets.std() + 1e-8

    def normalize(self, target, category):
        stats = self.stats.get(category, {'mean': self.global_mean, 'std': self.global_std})
        return (target - stats['mean']) / stats['std']

    def denormalize(self, prediction, category):
        stats = self.stats.get(category, {'mean': self.global_mean, 'std': self.global_std})
        return prediction * stats['std'] + stats['mean']
```

**Acceptance Criteria:**
- [x] `--normalize-targets-by location` normalizes per-location
- [x] `--normalize-targets-by location,season` normalizes per location-season combination
- [x] Statistics saved in checkpoint
- [x] Inference denormalizes correctly (Note: validation metrics on z-scores due to batch-level processing)
- [x] Unseen categories use global statistics with warning
- [x] Mutually exclusive with `--normalize-targets` (global normalization)
- [x] Unit tests for normalization roundtrip (26 tests)

**Files:**
- `aam/data/normalization.py` - New file for CategoryNormalizer
- `aam/cli/train.py` - Add `--normalize-targets-by` flag
- `aam/data/dataset.py` - Integrate normalizer into ASVDataset
- `aam/training/evaluation.py` - Handle category normalization in metrics
- `tests/test_normalization.py` - Test category normalization

**Dependencies:** None

---

### REG-3: Conditional Output Scaling
**Priority:** HIGH | **Effort:** 3-4 hours | **Status:** COMPLETE

Learn per-category scale and bias parameters for output adjustment.

**Completed:**
- Added `--conditional-output-scaling` flag accepting categorical column name(s)
- Per-category scale (init=1.0) and bias (init=0.0) as nn.Embedding
- Applied after target_head: `output = prediction * scale[cat] + bias[cat]`
- Multiple columns apply sequentially (multiplicatively for scales)
- Works with MLP head, bounded targets, output activations, learnable scale
- Added to model_config checkpoint and predict.py model reconstruction
- 22 unit tests in TestConditionalOutputScaling class

**Files:**
- `aam/models/sequence_predictor.py` - Add conditional scaling logic
- `aam/cli/train.py` - Add `--conditional-output-scaling` flag
- `aam/cli/predict.py` - Load conditional_scaling_columns from checkpoint
- `tests/test_sequence_predictor.py` - Test conditional scaling

**Dependencies:** Requires categorical columns already configured (CAT-1 through CAT-5)

---

### REG-4: FiLM Layers (Feature-wise Linear Modulation)
**Priority:** HIGH | **Effort:** 4-5 hours | **Status:** Not Started

Categorical embeddings modulate regression MLP activations via learned γ and β.

**Background:**
FiLM (Feature-wise Linear Modulation) is a powerful conditioning technique where context (categorical embeddings) modulates intermediate representations. Unlike concat/add fusion which mixes features early, FiLM allows categories to amplify, suppress, or shift specific learned features at each layer.

**Paper:** Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (2018)

**Scope:**
- Add `--film-conditioning` flag accepting categorical column name(s)
- Generate γ (scale) and β (shift) from categorical embeddings
- Apply FiLM after each MLP layer: `h_out = γ * h + β`
- Requires MLP head (REG-1) to have layers to modulate

**Implementation Notes:**
```python
class FiLMGenerator(nn.Module):
    """Generates FiLM parameters from categorical embeddings."""
    def __init__(self, categorical_dim: int, hidden_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(categorical_dim, hidden_dim)
        self.beta_proj = nn.Linear(categorical_dim, hidden_dim)
        # Initialize gamma to 1, beta to 0
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)

    def forward(self, categorical_emb):
        gamma = self.gamma_proj(categorical_emb)  # [B, hidden_dim]
        beta = self.beta_proj(categorical_emb)    # [B, hidden_dim]
        return gamma, beta

class FiLMLayer(nn.Module):
    """MLP layer with FiLM conditioning."""
    def __init__(self, in_dim: int, out_dim: int, categorical_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.film = FiLMGenerator(categorical_dim, out_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, categorical_emb):
        h = self.linear(x)
        gamma, beta = self.film(categorical_emb)
        h = gamma * h + beta  # FiLM modulation
        h = self.activation(h)
        h = self.dropout(h)
        return h
```

**Acceptance Criteria:**
- [ ] `--film-conditioning location,season` enables FiLM on those columns
- [ ] FiLM parameters generated from categorical embeddings
- [ ] Applied at each MLP hidden layer
- [ ] Initialized to identity transform (gamma=1, beta=0)
- [ ] Requires `--regressor-hidden-dims` to be set (error otherwise)
- [ ] Unit tests for FiLM layer and integration

**Files:**
- `aam/models/film.py` - New file for FiLMGenerator, FiLMLayer
- `aam/models/sequence_predictor.py` - Integrate FiLM into MLP head
- `aam/cli/train.py` - Add `--film-conditioning` flag
- `tests/test_film.py` - Test FiLM components

**Dependencies:** REG-1 (MLP head required for FiLM layers)

---

## Tier 2: Loss Functions (MEDIUM Priority)

### REG-5: Quantile Regression
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Not Started

Predict multiple quantiles for uncertainty estimation.

**Background:**
Point predictions don't capture uncertainty. Quantile regression predicts percentiles of the target distribution, providing prediction intervals. This is especially valuable when category effects vary—some categories may have inherently higher uncertainty.

**Scope:**
- Add `--loss-type quantile` option
- Add `--quantiles` flag (default: `0.1,0.5,0.9`)
- Output dimension multiplied by number of quantiles
- Pinball loss for each quantile

**Implementation Notes:**
```python
def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    """Pinball (quantile) loss."""
    error = target - pred
    return torch.max(quantile * error, (quantile - 1) * error).mean()

def quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: List[float]) -> torch.Tensor:
    """Combined loss for multiple quantiles."""
    # pred: [B, num_quantiles] or [B, out_dim * num_quantiles]
    # target: [B, 1] or [B, out_dim]
    total_loss = 0
    for i, q in enumerate(quantiles):
        total_loss += pinball_loss(pred[:, i], target.squeeze(), q)
    return total_loss / len(quantiles)
```

**Acceptance Criteria:**
- [ ] `--loss-type quantile --quantiles 0.1,0.5,0.9` trains quantile model
- [ ] Output dimension = original_out_dim * num_quantiles
- [ ] Median (0.5) quantile comparable to point prediction
- [ ] Prediction intervals cover expected proportion of targets
- [ ] Works with all Tier 1 features
- [ ] Unit tests for pinball loss and quantile output

**Files:**
- `aam/training/losses.py` - Add pinball_loss, quantile_loss
- `aam/models/sequence_predictor.py` - Adjust output dim for quantiles
- `aam/cli/train.py` - Add `--quantiles` flag, update `--loss-type` choices
- `tests/test_losses.py` - Test quantile loss

**Dependencies:** None (but benefits from REG-1 through REG-4)

---

### REG-6: Asymmetric Loss
**Priority:** MEDIUM | **Effort:** 2-3 hours | **Status:** Not Started

Penalize over-predictions and under-predictions differently.

**Background:**
In some applications, errors in one direction are more costly than the other. For example, underestimating contamination is worse than overestimating it.

**Scope:**
- Add `--loss-type asymmetric` option
- Add `--over-penalty` and `--under-penalty` flags
- Weighted MSE/MAE based on sign of error

**Implementation Notes:**
```python
def asymmetric_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    over_penalty: float = 1.0,
    under_penalty: float = 1.0,
    base_loss: str = "mse"
) -> torch.Tensor:
    """Asymmetric loss with different penalties for over/under prediction."""
    error = pred - target
    weights = torch.where(error > 0, over_penalty, under_penalty)

    if base_loss == "mse":
        return (weights * error ** 2).mean()
    else:  # mae
        return (weights * error.abs()).mean()
```

**Acceptance Criteria:**
- [ ] `--loss-type asymmetric --over-penalty 2.0` penalizes over-prediction 2x
- [ ] `--under-penalty` controls under-prediction penalty
- [ ] Default both penalties = 1.0 (equivalent to symmetric loss)
- [ ] Unit tests for asymmetric loss

**Files:**
- `aam/training/losses.py` - Add asymmetric_loss
- `aam/cli/train.py` - Add `--over-penalty`, `--under-penalty` flags
- `tests/test_losses.py` - Test asymmetric loss

**Dependencies:** None

---

## Tier 3: Architecture Variants (LOW Priority)

### REG-7: Residual Regression Head
**Priority:** LOW | **Effort:** 2-3 hours | **Status:** Not Started

Add skip connection around MLP head combining linear and non-linear paths.

**Background:**
Residual connections help gradient flow and allow the model to learn both simple linear relationships and complex non-linear patterns simultaneously.

**Scope:**
- Add `--residual-regression-head` flag
- Output = Linear(x) + MLP(x)
- Requires `--regressor-hidden-dims` to be set

**Implementation Notes:**
```python
if residual_regression_head:
    self.linear_skip = nn.Linear(embedding_dim, out_dim)
    # self.target_head is the MLP from REG-1

def forward_target_head(self, x):
    if self.residual_regression_head:
        return self.linear_skip(x) + self.target_head(x)
    return self.target_head(x)
```

**Acceptance Criteria:**
- [ ] `--residual-regression-head` adds skip connection
- [ ] Requires `--regressor-hidden-dims` (error if not set)
- [ ] Gradient flow improved (check gradient norms in tests)
- [ ] Unit tests for residual head

**Files:**
- `aam/models/sequence_predictor.py` - Add residual connection
- `aam/cli/train.py` - Add `--residual-regression-head` flag
- `tests/test_sequence_predictor.py` - Test residual head

**Dependencies:** REG-1

---

### REG-8: Per-Output Loss Configuration
**Priority:** LOW | **Effort:** 3-4 hours | **Status:** Not Started

Different loss functions per target column for multi-output regression.

**Background:**
When predicting multiple targets, different targets may have different error distributions. One target might be well-suited to MSE while another needs Huber loss for outlier robustness.

**Scope:**
- Add `--loss-config` flag accepting JSON file or inline JSON
- Each target column can specify its loss type
- Default: all targets use `--loss-type`

**Implementation Notes:**
```python
# --loss-config '{"pH": "mse", "temperature": "huber", "concentration": "mae"}'

class PerOutputLoss:
    def __init__(self, loss_config: Dict[str, str], default_loss: str = "huber"):
        self.loss_config = loss_config
        self.default_loss = default_loss

    def compute(self, pred: torch.Tensor, target: torch.Tensor, column_names: List[str]):
        total_loss = 0
        for i, col in enumerate(column_names):
            loss_type = self.loss_config.get(col, self.default_loss)
            total_loss += compute_loss(pred[:, i], target[:, i], loss_type)
        return total_loss / len(column_names)
```

**Acceptance Criteria:**
- [ ] `--loss-config` accepts JSON specification
- [ ] Per-column loss types applied correctly
- [ ] Falls back to `--loss-type` for unspecified columns
- [ ] Unit tests for per-output loss

**Files:**
- `aam/training/losses.py` - Add PerOutputLoss
- `aam/cli/train.py` - Add `--loss-config` flag
- `tests/test_losses.py` - Test per-output loss

**Dependencies:** None (for future multi-output work)

---

## Tier 4: Advanced (LOW Priority)

### REG-9: Mixture of Experts
**Priority:** LOW | **Effort:** 6-8 hours | **Status:** Not Started

Separate expert heads per category with learned routing.

**Background:**
The most expressive approach: each environment/season gets its own expert prediction head, with a router that learns to weight expert contributions. This can capture fundamentally different input-output relationships per category.

**Scope:**
- Add `--moe-experts` flag for number of experts
- Add `--moe-routing` flag for routing based on categorical column
- Experts share base embeddings but have independent prediction heads
- Soft routing: output = Σ router_weight[i] * expert[i](x)

**Implementation Notes:**
```python
class MixtureOfExperts(nn.Module):
    def __init__(self, embedding_dim: int, out_dim: int, num_experts: int,
                 hidden_dims: List[int], categorical_dim: int):
        super().__init__()
        self.experts = nn.ModuleList([
            self._build_mlp(embedding_dim, out_dim, hidden_dims)
            for _ in range(num_experts)
        ])
        self.router = nn.Linear(categorical_dim, num_experts)

    def forward(self, x, categorical_emb):
        # Compute expert outputs
        expert_outputs = torch.stack([exp(x) for exp in self.experts], dim=1)  # [B, E, out_dim]

        # Compute routing weights from categorical embedding
        routing_logits = self.router(categorical_emb)  # [B, E]
        routing_weights = F.softmax(routing_logits, dim=-1)  # [B, E]

        # Weighted combination
        output = (routing_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)  # [B, out_dim]
        return output
```

**Acceptance Criteria:**
- [ ] `--moe-experts 4` creates 4 expert heads
- [ ] `--moe-routing location` routes based on location embedding
- [ ] Soft routing with learned weights
- [ ] Load balancing regularization (optional, prevents expert collapse)
- [ ] Expert utilization logged during training
- [ ] Unit tests for MoE forward pass and routing

**Files:**
- `aam/models/moe.py` - New file for MixtureOfExperts
- `aam/models/sequence_predictor.py` - Integrate MoE as target head option
- `aam/cli/train.py` - Add `--moe-experts`, `--moe-routing` flags
- `tests/test_moe.py` - Test MoE components

**Dependencies:** REG-1 (experts use MLP architecture), CAT features

---

## Summary

| Ticket | Description | Effort | Priority | Dependencies |
|--------|-------------|--------|----------|--------------|
| **REG-1** | MLP regression head | 3-4h | HIGH | None |
| **REG-2** | Per-category target normalization | 3-4h | HIGH | None |
| **REG-3** | Conditional output scaling | 3-4h | HIGH | CAT features |
| **REG-4** | FiLM layers | 4-5h | HIGH | REG-1 |
| **REG-5** | Quantile regression | 4-6h | MEDIUM | None |
| **REG-6** | Asymmetric loss | 2-3h | MEDIUM | None |
| **REG-7** | Residual regression head | 2-3h | LOW | REG-1 |
| **REG-8** | Per-output loss config | 3-4h | LOW | None |
| **REG-9** | Mixture of Experts | 6-8h | LOW | REG-1, CAT |
| **Total** | | **32-44h** | | |

---

## Recommended Implementation Order

1. **REG-1** - MLP head (foundation)
2. **REG-2** - Per-category normalization (quick win)
3. **REG-3** - Conditional output scaling (complements REG-2)
4. **REG-4** - FiLM layers (most expressive modulation)
5. **REG-5** - Quantile regression (uncertainty)
6. **REG-6** - Asymmetric loss (if needed)
7. **REG-7, 8, 9** - As needed based on results

---

## Notes

**Categorical Context:**
- Primary columns: `season`, `location`
- Some location-season combinations missing (incomplete factorial)
- Large sample spread provides good coverage

**Compatibility:**
- All REG tickets should work with existing features:
  - `--output-activation` (relu, softplus, exp)
  - `--bounded-targets` (sigmoid output)
  - `--learnable-output-scale`
  - Categorical fusion (concat, add)
- REG-3 conditional scaling is more expressive than `--learnable-output-scale` and should be preferred when categorical columns are available
