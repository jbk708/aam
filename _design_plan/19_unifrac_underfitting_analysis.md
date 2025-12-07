# UniFrac Model Underfitting Analysis and Improvement Plan

**Status:** 📋 Planning  
**Priority:** HIGH  
**Created:** 2024  
**Related Tickets:** PYT-8.12 (Mask Diagonal in UniFrac Loss Computation)

## Executive Summary

The PyTorch AAM model exhibits severe underfitting with R² = 0.0455, indicating failure to learn meaningful UniFrac distance patterns. The model collapses to predicting approximately the mean UniFrac distance (~0.57-0.60) regardless of input, failing to capture the full range of distances (0.0-1.0). This document analyzes the root causes and proposes comprehensive improvements.

## Problem Statement

### Current Performance
- **R² = 0.0455** - Severe underfitting, model explains <5% of variance
- **Model Collapse**: Predictions cluster around 0.57-0.60 (approximately mean UniFrac distance)
- **Range Limitation**: Model fails to predict:
  - Zero distances (identical communities)
  - High distances (>0.75, highly dissimilar communities)
- **Distribution Anomaly**: Large cluster of samples at actual UniFrac distance = 0.0

### Key Observations
1. **Bimodal Distribution**: Substantial cluster at distance = 0.0 creates problematic distribution for regression
2. **Mean Convergence**: Model outputs approximately mean distance regardless of input
3. **Near-Horizontal Fit**: Linear regression fit is nearly horizontal, confirming mean prediction behavior
4. **Bounded Output**: UniFrac distances are constrained to [0, 1], but model doesn't respect this constraint

## Root Cause Analysis

### 1. Loss Function Issues

#### Current Implementation (PyTorch)
- **MSE Loss**: Direct MSE on full pairwise distance matrix including diagonal
- **Diagonal Included**: Diagonal elements (always 0.0) artificially lower loss
- **No Bounded Constraint**: Loss doesn't account for [0, 1] constraint
- **No Distribution Handling**: Doesn't handle bimodal distribution (zero-distance cluster)

#### TensorFlow Implementation Comparison
- **Pairwise Distance Computation**: Computes pairwise distances FROM embeddings using Euclidean distance
- **Diagonal Masking**: Uses `tf.linalg.band_part(reg_loss, 0, -1)` to extract upper triangle (excludes diagonal)
- **Architectural Difference**: TensorFlow computes distances from embeddings; PyTorch directly predicts distance matrix

**Key Finding**: TensorFlow masks diagonal elements in loss computation, but PyTorch does not (PYT-8.12 addresses this).

### 2. Architectural Differences

#### TensorFlow Approach
```python
# TensorFlow: Computes pairwise distances from embeddings
y_pred_dist = _pairwise_distances(y_pred, squared=False)  # From embeddings
differences = tf.math.square(y_pred_dist - y_true)
reg_loss = tf.linalg.band_part(reg_loss, 0, -1)  # Upper triangle only (no diagonal)
```

#### PyTorch Approach
```python
# PyTorch: Directly predicts pairwise distance matrix
base_prediction = self.output_head(pooled_embeddings)  # [batch_size, batch_size]
loss = nn.functional.mse_loss(base_pred, base_true)  # Full matrix including diagonal
```

**Key Difference**: 
- TensorFlow: Embeddings → Pairwise distances → Loss (with diagonal masking)
- PyTorch: Embeddings → Direct distance matrix prediction → Loss (no diagonal masking)

### 3. Data Distribution Issues

#### Zero-Distance Cluster
- Large number of samples with actual UniFrac distance = 0.0
- May represent:
  - **Data Quality Issues**: Identical samples due to preprocessing/rarefaction artifacts
  - **Legitimate Signal**: Truly identical or near-identical microbial communities
- **Impact**: Creates bimodal distribution that MSE struggles with

#### Distribution Characteristics
- **Bounded Range**: [0, 1] constraint not enforced
- **Non-Normal**: Bimodal distribution with zero cluster
- **Skewed**: Many low distances, fewer high distances

### 4. Training Configuration Issues

#### Potential Problems
- **Learning Rate**: May be too high/low for this task
- **Loss Weighting**: `penalty=1.0` may not be optimal
- **Model Capacity**: May be insufficient for pairwise distance prediction
- **Normalization**: No normalization of UniFrac distances before training
- **Feature Informativeness**: Model may not be extracting meaningful features from sequences

## Proposed Solutions

### Phase 1: Critical Fixes (Immediate Impact)

#### 1.1 Mask Diagonal in Loss Computation (PYT-8.12)
**Priority:** HIGH | **Effort:** Low (1-2 hours)

**Action**: Implement diagonal masking in `compute_base_loss()` for UniFrac pairwise matrices.

**Implementation**:
```python
if encoder_type == "unifrac" and base_pred.dim() == 2 and base_pred.shape[0] == base_pred.shape[1]:
    # Extract upper triangle (excluding diagonal)
    batch_size = base_pred.shape[0]
    triu_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=base_pred.device)
    base_pred_masked = base_pred[triu_indices[0], triu_indices[1]]
    base_true_masked = base_true[triu_indices[0], triu_indices[1]]
    return nn.functional.mse_loss(base_pred_masked, base_true_masked)
```

**Expected Impact**: 
- Loss values will increase (no longer artificially lowered by 0.0 diagonal)
- Stronger training signal (only meaningful pairwise comparisons)
- Better distance relationship learning

#### 1.2 Investigate Zero-Distance Samples
**Priority:** HIGH | **Effort:** Medium (2-3 hours)

**Actions**:
1. Analyze distribution of zero-distance samples
2. Determine if they represent data quality issues or legitimate signal
3. Consider separate handling (removal, weighting, or separate loss term)

**Investigation Steps**:
- Count samples with distance = 0.0
- Check if zero distances cluster by sample metadata (time, location, etc.)
- Verify if zero distances are due to rarefaction artifacts
- Determine if identical samples are duplicates or truly identical communities

**Potential Solutions**:
- **Option A**: Remove zero-distance pairs if they're data artifacts
- **Option B**: Down-weight zero-distance pairs in loss (e.g., `weight = 0.1` for zero distances)
- **Option C**: Separate loss term for zero vs non-zero distances
- **Option D**: Keep zero distances but use different loss function (see Phase 2)

### Phase 2: Loss Function Improvements (Medium-Term)

#### 2.1 Bounded Regression Loss
**Priority:** MEDIUM | **Effort:** Medium (3-4 hours)

**Problem**: MSE doesn't account for [0, 1] constraint on UniFrac distances.

**Solutions**:

**Option A: Clipped MSE**
```python
# Clip predictions to [0, 1] before computing loss
base_pred_clipped = torch.clamp(base_pred, 0.0, 1.0)
loss = nn.functional.mse_loss(base_pred_clipped, base_true)
```

**Option B: Beta Regression Loss**
- Use beta distribution likelihood (appropriate for bounded [0, 1] data)
- Requires reparameterization: `y = (x - a) / (b - a)` where a=0, b=1
- More complex but theoretically sound for bounded data

**Option C: Huber Loss**
- Less sensitive to outliers than MSE
- May help with zero-distance cluster

**Recommendation**: Start with Option A (clipped MSE) for simplicity, consider Option B if needed.

#### 2.2 Weighted Loss for Zero-Distance Pairs
**Priority:** MEDIUM | **Effort:** Low (1-2 hours)

**Action**: Down-weight zero-distance pairs in loss computation.

**Implementation**:
```python
# Create weights: lower weight for zero distances
weights = torch.where(base_true == 0.0, 0.1, 1.0)
loss = nn.functional.mse_loss(base_pred_masked, base_true_masked, reduction='none')
weighted_loss = (loss * weights).mean()
```

**Rationale**: Zero distances may be less informative or represent data artifacts.

#### 2.3 Focal Loss for Distance Regression
**Priority:** LOW | **Effort:** Medium (2-3 hours)

**Action**: Adapt focal loss concept for regression to focus on hard examples.

**Implementation**: Weight loss by prediction error magnitude to focus on difficult pairs.

### Phase 3: Architectural Improvements (Long-Term)

#### 3.1 Align with TensorFlow Architecture
**Priority:** MEDIUM | **Effort:** High (6-8 hours)

**Action**: Consider computing pairwise distances from embeddings (like TensorFlow) instead of directly predicting distance matrix.

**Current (PyTorch)**:
```python
# Direct prediction
base_prediction = self.output_head(pooled_embeddings)  # [batch_size, batch_size]
```

**Proposed (TensorFlow-like)**:
```python
# Compute distances from embeddings
sample_embeddings = pooled_embeddings  # [batch_size, embedding_dim]
# Compute pairwise Euclidean distances
pairwise_distances = compute_pairwise_distances(sample_embeddings)  # [batch_size, batch_size]
```

**Benefits**:
- More interpretable (embeddings → distances)
- Enforces distance metric properties (symmetry, triangle inequality)
- May improve generalization

**Trade-offs**:
- Requires architectural changes
- May need to normalize distances to [0, 1] range
- More complex implementation

#### 3.2 Model Capacity Increase
**Priority:** MEDIUM | **Effort:** Medium (2-3 hours)

**Actions**:
- Increase embedding dimension
- Add more transformer layers
- Increase attention heads
- Add residual connections if missing

**Investigation**: Profile model capacity vs dataset complexity.

#### 3.3 Feature Engineering
**Priority:** LOW | **Effort:** Medium (3-4 hours)

**Actions**:
- Add phylogenetic features (tree distances)
- Incorporate ASV abundance statistics
- Add sample-level metadata features
- Consider attention visualization to understand what model focuses on

### Phase 4: Training Configuration Improvements

#### 4.1 Learning Rate Tuning
**Priority:** MEDIUM | **Effort:** Low (1-2 hours)

**Actions**:
- Try lower learning rates (1e-5, 5e-5)
- Implement learning rate finder
- Use learning rate scheduling (ReduceLROnPlateau)

#### 4.2 Loss Weighting
**Priority:** MEDIUM | **Effort:** Low (1 hour)

**Actions**:
- Tune `penalty` parameter (currently 1.0)
- Try higher weights for UniFrac loss relative to nucleotide loss
- Consider adaptive weighting based on loss magnitudes

#### 4.3 Normalization
**Priority:** LOW | **Effort:** Low (1 hour)

**Actions**:
- Normalize UniFrac distances before training (z-score or min-max)
- Denormalize predictions for evaluation
- May help with training stability

#### 4.4 Data Augmentation
**Priority:** LOW | **Effort:** Medium (2-3 hours)

**Actions**:
- Add noise to sequences (small mutations)
- Vary ASV ordering (if order doesn't matter)
- Subsampling ASVs (if token_limit allows)

## Implementation Priority

### Immediate (This Sprint)
1. ✅ **PYT-8.12**: Mask diagonal in UniFrac loss computation
2. 🔄 **Investigate zero-distance samples**: Analyze distribution and determine handling strategy

### Short-Term (Next Sprint)
3. **Bounded regression loss**: Implement clipped MSE or beta regression
4. **Weighted loss for zero distances**: Down-weight zero-distance pairs
5. **Learning rate tuning**: Find optimal learning rate

### Medium-Term (Future Sprints)
6. **Architectural alignment**: Consider TensorFlow-like pairwise distance computation
7. **Model capacity**: Increase if needed based on investigation
8. **Advanced loss functions**: Focal loss, beta regression if needed

### Long-Term (Research)
9. **Feature engineering**: Add phylogenetic and metadata features
10. **Data augmentation**: Implement sequence-level augmentation

## Success Metrics

### Primary Metrics
- **R² > 0.3**: Model explains >30% of variance (7x improvement)
- **Range Coverage**: Model predicts full [0, 1] range
- **No Mean Collapse**: Predictions vary meaningfully with input

### Secondary Metrics
- **MAE < 0.15**: Mean absolute error below 0.15
- **RMSE < 0.20**: Root mean squared error below 0.20
- **Zero-Distance Handling**: Appropriate handling of zero-distance cluster

## Testing Strategy

### Unit Tests
- Test diagonal masking in loss computation
- Test bounded loss (clipping)
- Test weighted loss for zero distances
- Test pairwise distance computation (if architectural change)

### Integration Tests
- End-to-end training with new loss functions
- Verify loss values increase after diagonal masking
- Verify model predictions respect [0, 1] bounds

### Validation Tests
- Compare R² before/after fixes
- Verify prediction range coverage
- Check for mean collapse

## Risk Assessment

### Low Risk
- Diagonal masking (PYT-8.12): Well-defined change, clear expected impact
- Clipped MSE: Simple change, minimal risk

### Medium Risk
- Zero-distance handling: Requires data analysis, may affect model behavior
- Learning rate tuning: May require multiple training runs

### High Risk
- Architectural changes (TensorFlow alignment): Major refactoring, may introduce bugs
- Beta regression: Complex implementation, may not improve performance

## Dependencies

- **PYT-8.12**: Must be completed first (foundation for other improvements)
- **Zero-distance investigation**: Should inform loss function choices
- **Learning rate tuning**: Can be done in parallel with loss improvements

## References

### TensorFlow Implementation
- `aam/losses.py`: `PairwiseLoss` class with diagonal masking
- `aam/unifrac_model.py`: `_compute_loss` method with `band_part` masking

### Current PyTorch Implementation
- `aam/training/losses.py`: `compute_base_loss()` method (no diagonal masking)
- `aam/models/sequence_encoder.py`: Direct distance matrix prediction
- `aam/training/trainer.py`: Training loop with UniFrac loss

### Related Literature
- Beta regression for bounded data: Ferrari & Cribari-Neto (2004)
- Focal loss for regression: Lin et al. (2017) - adapt for regression
- Pairwise distance learning: Hoffer & Ailon (2015)

## Next Steps

1. **Review this document** with team
2. **Create tickets** for prioritized improvements
3. **Start with PYT-8.12** (diagonal masking) - already planned
4. **Investigate zero-distance samples** - create analysis script
5. **Implement bounded loss** - clipped MSE as first step
6. **Tune learning rate** - use learning rate finder
7. **Evaluate improvements** - compare R² and prediction range

## Questions for Discussion

1. **Zero-Distance Handling**: Should we remove, down-weight, or keep zero-distance pairs?
2. **Architectural Alignment**: Should we align with TensorFlow (pairwise from embeddings) or keep direct prediction?
3. **Loss Function**: Clipped MSE, beta regression, or weighted MSE?
4. **Model Capacity**: Is current capacity sufficient, or should we increase?
5. **Training Strategy**: Should we use staged training (pretrain → finetune) or end-to-end?

---

**Document Status**: Ready for review  
**Next Action**: Review with team, create implementation tickets
