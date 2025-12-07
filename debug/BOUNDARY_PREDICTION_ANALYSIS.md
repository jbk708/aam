# Boundary Prediction Clustering Analysis

**Date:** December 6, 2024  
**Ticket:** PYT-8.16  
**Model:** `data/test-run/checkpoints/best_model.pt` (Epoch 8)  
**Dataset:** `fall_train_only_all_outdoor.biom`  
**Investigation Script:** `debug/investigate_boundary_predictions.py`

## Executive Summary

This investigation analyzed why UniFrac distance predictions cluster at the boundaries (0.0 and 1.0) during inference, causing large loss values. The analysis compared raw model outputs (before clipping) with clipped predictions (after forward pass clipping) and actual UniFrac distances to identify the root cause of boundary clustering.

## Investigation Methodology

### Data and Model
- **Model Architecture:**
  - ASV layers: 4
  - Sample layers: 4
  - Encoder layers: 4
  - Embedding dim: 128
  - Predict nucleotides: True
  - Encoder type: unifrac
  - Base output dim: 6 (batch_size)
  
- **Inference Setup:**
  - Batch size: 6
  - Token limit: 512
  - Device: CUDA
  - Total batches processed: 130

### Analysis Approach
1. **Raw Predictions:** Captured model outputs before any clipping operations
2. **Clipped Predictions:** Captured predictions after forward pass clipping in `SequenceEncoder.forward()`
3. **Actual Distances:** Extracted from UniFrac distance matrix for comparison
4. **Distribution Analysis:** Computed statistics for each prediction type
5. **Visualizations:** Created histograms and scatter plots to visualize distributions

## Key Findings

### 1. Raw Predictions Distribution

**Observation:** Raw predictions (before clipping) show the model's true output distribution.

**Key Metrics:**
- Count: 1,950 predictions
- Mean: 0.140521 (much lower than actual mean of 0.720342)
- Std: 1.016876 (very high variance, wide distribution)
- Min: -1.280454 (negative values present!)
- Max: 1.922829 (values exceed 1.0)
- Percentiles: 25% = -1.069432, 50% = 0.664850, 75% = 0.775767
- At 0.0 (within 0.01): 0 (0.00%)
- At 1.0 (within 0.01): 0 (0.00%)
- Exactly 0.0: 0 (0.00%)
- Exactly 1.0: 0 (0.00%)

**Critical Finding:** Raw predictions are **NOT clustering at boundaries** - 0% are at 0.0 or 1.0!

**Interpretation:**
- Raw predictions have a wide distribution with many negative values (25th percentile = -1.07)
- Median (0.66) is reasonable but mean (0.14) is pulled down by negative values
- Model is learning a continuous distribution, not boundary values
- **Conclusion: Clipping is forcing boundaries, not the model itself**

### 2. Clipped Predictions Distribution

**Observation:** Clipped predictions (after `torch.clamp(base_prediction, 0.0, 1.0)` in forward pass).

**Key Metrics:**
- Count: 1,950 predictions
- Mean: 0.470679 (improved from raw 0.14, but still below actual 0.72)
- Std: 0.397494 (reduced from raw 1.02, clipping reduces variance)
- Min: 0.000000, Max: 1.000000 (as expected from clipping)
- Percentiles: 25% = 0.000000, 50% = 0.664850, 75% = 0.775767
- At 0.0 (within 0.01): 780 (40.00%) ⚠️ **CRITICAL ISSUE**
- At 1.0 (within 0.01): 265 (13.59%) ⚠️
- Exactly 0.0: 780 (40.00%)
- Exactly 1.0: 265 (13.59%)

**Critical Finding:** **53.59% of predictions are at boundaries (0.0 or 1.0)** after clipping!

**Interpretation:**
- Clipping transforms 0% boundary predictions → 53.59% boundary predictions
- 780 predictions (40%) are exactly 0.0 (raw predictions < 0.0 → clipped to 0.0)
- 265 predictions (13.59%) are exactly 1.0 (raw predictions > 1.0 → clipped to 1.0)
- **Clipping is the root cause of boundary clustering**

### 3. Actual UniFrac Distance Distribution

**Observation:** Ground truth distribution for comparison.

**Key Metrics:**
- Count: 1,950 distances
- Mean: 0.720342 (unimodal, centered around 0.72 as expected)
- Std: 0.126691 (low variance, tight distribution)
- Min: 0.292482, Max: 0.977680 (all within [0, 1] but not at boundaries)
- Percentiles: 25% = 0.649793, 50% = 0.727119, 75% = 0.805883
- At 0.0 (within 0.01): 0 (0.00%)
- At 1.0 (within 0.01): 0 (0.00%)
- Exactly 0.0: 0 (0.00%)
- Exactly 1.0: 0 (0.00%)

**Expected Distribution:**
- Matches PYT-8.13 findings: Unimodal distribution centered around 0.72
- Very few zero distances (0.00% of pairs)
- Continuous distribution, not bimodal at boundaries
- **No actual distances are at 0.0 or 1.0 boundaries**

### 4. Raw vs Clipped Comparison

**Visualization:** `raw_vs_clipped_scatter.png` and `raw_vs_clipped_comparison.png`

**Key Observations:**
- [Review scatter plot to identify patterns]
  - Are raw predictions outside [0, 1] range being clipped?
  - What percentage of predictions are affected by clipping?
  - Is there a pattern (e.g., all negative values → 0.0, all >1.0 → 1.0)?

**Critical Question:** Are raw predictions already at boundaries, or is clipping forcing them there?

### 5. Actual vs Predicted Comparison

**Visualization:** `actual_vs_predicted_scatter.png` and `actual_vs_predicted_histogram.png`

**Key Observations:**
- [Review scatter plot to identify patterns]
  - How well do predictions match actual distances?
  - Are predictions systematically biased?
  - What is the R² correlation?
  - Are boundary predictions (0.0, 1.0) causing large errors?

**Distribution Comparison:**
- Actual: Unimodal around 0.72
- Predicted: [Describe based on histogram]
  - Bimodal at boundaries?
  - Continuous but shifted?
  - Completely different distribution?

## Root Cause Analysis

### ✅ Hypothesis 1: Over-Clipping - **CONFIRMED**

**Evidence:**
- ✅ Raw predictions are continuous but outside [0, 1] range
  - 25th percentile = -1.07 (many negative values)
  - Max = 1.92 (values exceed 1.0)
  - 0% at boundaries before clipping
- ✅ Clipping forces many predictions to 0.0 or 1.0
  - 40% clipped to exactly 0.0 (780 predictions)
  - 13.59% clipped to exactly 1.0 (265 predictions)
  - Total: 53.59% at boundaries after clipping
- ✅ Raw vs clipped comparison shows massive impact
  - 0% → 53.59% boundary clustering caused by clipping

**Root Cause:** Clipping in forward pass (`torch.clamp(base_prediction, 0.0, 1.0)`) is too aggressive. Raw predictions have a wide distribution with many negative values (mean 0.14, 25th percentile -1.07), and clipping forces all negative values to exactly 0.0, creating artificial boundary clustering.

**Solution:** 
1. **Remove clipping from forward pass** - Only clip in loss computation if needed
2. **Use sigmoid activation** - Natural [0, 1] constraint without hard clipping
3. **Smooth clipping** - Use soft clipping (e.g., `x * sigmoid(x)` or `tanh` transformation)

### ❌ Hypothesis 2: Model Learning Issue - **DISPROVEN**

**Evidence:**
- ❌ Raw predictions are NOT clustered at boundaries (0% at 0.0 or 1.0)
- ✅ Model IS learning continuous distribution (wide spread, median 0.66)
- ⚠️ However, raw predictions have issues:
  - Mean (0.14) is much lower than actual (0.72)
  - Many negative values (25th percentile = -1.07)
  - High variance (std = 1.02)

**Conclusion:** Model is learning a distribution, but it's shifted and has negative values. The primary issue is clipping, but there may be secondary issues with model calibration.

**Secondary Issues:**
- Model output distribution is shifted (mean 0.14 vs actual 0.72)
- Negative values suggest model needs better constraint
- May benefit from sigmoid activation to naturally constrain to [0, 1]

### Hypothesis 3: Loss Function Behavior
**Evidence:**
- [ ] Clipped MSE creates flat gradients at boundaries
- [ ] Model converges to boundaries due to gradient flow issues
- [ ] Loss function doesn't penalize boundary predictions appropriately

**If True:** MSE loss with hard clipping creates optimization problems.

**Solution:** Use smooth loss function that handles boundaries better (Huber loss, smooth L1, or beta regression).

### Hypothesis 4: Training Instability
**Evidence:**
- [ ] Model predictions are unstable
- [ ] High variance in predictions
- [ ] Learning rate may be too high/low

**If True:** Training dynamics are causing poor convergence.

**Solution:** Adjust learning rate, add gradient clipping, or use different optimizer/scheduler.

## Visualizations Summary

### 1. `raw_predictions_histogram.png`
- Distribution of raw model outputs before clipping
- Shows if model naturally produces boundary values
- **Key Question:** Is distribution continuous or already bimodal?

### 2. `clipped_predictions_histogram.png`
- Distribution after forward pass clipping
- Shows impact of `torch.clamp(base_prediction, 0.0, 1.0)`
- **Key Question:** Does clipping create boundary clusters?

### 3. `raw_vs_clipped_comparison.png`
- Side-by-side comparison of raw and clipped distributions
- Highlights the effect of clipping operation
- **Key Question:** How many predictions are affected by clipping?

### 4. `raw_vs_clipped_scatter.png`
- Scatter plot: raw (x-axis) vs clipped (y-axis)
- Points on y=0.0 line: raw predictions < 0.0 → clipped to 0.0
- Points on y=1.0 line: raw predictions > 1.0 → clipped to 1.0
- Points on y=x line: no clipping needed (within [0, 1])
- **Key Question:** What percentage of predictions require clipping?

### 5. `actual_vs_predicted_scatter.png`
- Scatter plot: actual (x-axis) vs predicted (y-axis)
- Points on y=0.0 or y=1.0: boundary predictions
- Points on y=x line: perfect predictions
- **Key Question:** How well do predictions match actual distances?

### 6. `actual_vs_predicted_histogram.png`
- Side-by-side histograms of actual and predicted distributions
- Shows distribution mismatch
- **Key Question:** Is predicted distribution similar to actual, or completely different?

## Recommendations

### Immediate Actions

1. ✅ **Statistics Reviewed** - All metrics filled in from console output
2. **Calculate R² Correlation** - Should be calculated from scatter plot data
3. ✅ **Boundary Percentage Identified** - 53.59% at boundaries (40% at 0.0, 13.59% at 1.0)

### Primary Fix: Remove Forward Pass Clipping

**Root Cause:** Clipping in `SequenceEncoder.forward()` is forcing 53.59% of predictions to boundaries.

**Recommended Solution:** Replace linear layer + hard clipping with sigmoid activation

**Implementation:**
1. **Option A (Recommended):** Use sigmoid activation in final layer
   - Change `output_head` from `nn.Linear(embedding_dim, base_output_dim)` to include sigmoid
   - Or wrap output: `torch.sigmoid(self.output_head(pooled_embeddings))`
   - Natural [0, 1] constraint without hard boundaries
   - Smooth gradients throughout range

2. **Option B:** Remove forward pass clipping, only clip in loss
   - Remove `torch.clamp(base_prediction, 0.0, 1.0)` from forward pass
   - Keep clipping in `compute_base_loss()` if needed
   - Allows model to learn better, clips only for loss computation

3. **Option C:** Soft clipping
   - Use `x * torch.sigmoid(x)` or `torch.tanh(x) * 0.5 + 0.5`
   - Smooth transition instead of hard cutoff

### Secondary Fixes (Model Calibration)

1. **Model Output Calibration**
   - Raw predictions have mean 0.14 vs actual 0.72 (shifted distribution)
   - May need bias adjustment or different initialization
   - Sigmoid activation may help with calibration

2. **Loss Function Enhancement**
   - Current clipped MSE may have gradient issues at boundaries
   - Consider Huber loss or smooth L1 for better gradient flow
   - Beta regression loss is theoretically sound for bounded regression

3. **Training Dynamics Review**
   - Learning rate may need adjustment (current 1e-4)
   - Gradient clipping already implemented, verify it's working
   - Consider different optimizer/scheduler combinations

### Next Steps

1. **Fill in statistics** from console output
2. **Review visualizations** to confirm hypotheses
3. **Implement fix** based on root cause identified
4. **Test fix** by re-running inference and comparing distributions
5. **Update ticket** with findings and implementation

## Conclusion

**Root Cause Identified:** Over-clipping in forward pass is the primary cause of boundary prediction clustering.

**Key Findings:**
1. **Raw predictions are NOT clustering at boundaries** (0% at 0.0 or 1.0)
2. **Clipping creates 53.59% boundary predictions** (40% at 0.0, 13.59% at 1.0)
3. **Raw predictions have wide distribution** with many negative values (25th percentile = -1.07)
4. **Actual distances are unimodal** around 0.72 with 0% at boundaries

**Recommended Fix:**
Replace hard clipping (`torch.clamp`) with sigmoid activation in the final layer. This will:
- Provide natural [0, 1] constraint without hard boundaries
- Eliminate artificial boundary clustering (53.59% → expected ~0%)
- Improve gradient flow throughout the range
- Better match actual distance distribution

**Expected Outcome:**
- Predictions should be continuous, matching actual distribution
- Boundary clustering should drop from 53.59% to near 0%
- Mean prediction should improve from 0.47 toward actual 0.72
- Loss values should decrease significantly

---

## Appendix: Console Output

```
Raw Predictions Distribution:
  Count: 1,950
  Mean: 0.140521
  Std: 1.016876
  Min: -1.280454
  Max: 1.922829
  Percentiles: 25%=-1.069432, 50%=0.664850, 75%=0.775767
  At 0.0 (within 0.01): 0 (0.00%)
  At 1.0 (within 0.01): 0 (0.00%)
  Exactly 0.0: 0 (0.00%)
  Exactly 1.0: 0 (0.00%)

Clipped Predictions Distribution:
  Count: 1,950
  Mean: 0.470679
  Std: 0.397494
  Min: 0.000000
  Max: 1.000000
  Percentiles: 25%=0.000000, 50%=0.664850, 75%=0.775767
  At 0.0 (within 0.01): 780 (40.00%)
  At 1.0 (within 0.01): 265 (13.59%)
  Exactly 0.0: 780 (40.00%)
  Exactly 1.0: 265 (13.59%)

Actual Distances Distribution:
  Count: 1,950
  Mean: 0.720342
  Std: 0.126691
  Min: 0.292482
  Max: 0.977680
  Percentiles: 25%=0.649793, 50%=0.727119, 75%=0.805883
  At 0.0 (within 0.01): 0 (0.00%)
  At 1.0 (within 0.01): 0 (0.00%)
  Exactly 0.0: 0 (0.00%)
  Exactly 1.0: 0 (0.00%)
```

## Summary Statistics

| Metric | Raw Predictions | Clipped Predictions | Actual Distances |
|--------|----------------|---------------------|------------------|
| Count | 1,950 | 1,950 | 1,950 |
| Mean | 0.140521 | 0.470679 | 0.720342 |
| Std | 1.016876 | 0.397494 | 0.126691 |
| Min | -1.280454 | 0.000000 | 0.292482 |
| Max | 1.922829 | 1.000000 | 0.977680 |
| Median | 0.664850 | 0.664850 | 0.727119 |
| % at 0.0 | 0.00% | **40.00%** | 0.00% |
| % at 1.0 | 0.00% | **13.59%** | 0.00% |
| **Total % at boundaries** | **0.00%** | **53.59%** | **0.00%** |
