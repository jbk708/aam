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
- [To be filled from console output]
  - Mean: ___
  - Std: ___
  - Min: ___
  - Max: ___
  - Percentiles: 25% = ___, 50% = ___, 75% = ___
  - At 0.0 (within 0.01): ___ (___%)
  - At 1.0 (within 0.01): ___ (___%)
  - Exactly 0.0: ___ (___%)
  - Exactly 1.0: ___ (___%)

**Interpretation:**
- If raw predictions are already clustered at boundaries → **Model architecture/learning issue**
- If raw predictions are continuous → **Clipping is forcing boundaries**

### 2. Clipped Predictions Distribution

**Observation:** Clipped predictions (after `torch.clamp(base_prediction, 0.0, 1.0)` in forward pass).

**Key Metrics:**
- [To be filled from console output]
  - Mean: ___
  - Std: ___
  - Min: ___
  - Max: ___
  - Percentiles: 25% = ___, 50% = ___, 75% = ___
  - At 0.0 (within 0.01): ___ (___%)
  - At 1.0 (within 0.01): ___ (___%)
  - Exactly 0.0: ___ (___%)
  - Exactly 1.0: ___ (___%)

**Interpretation:**
- Comparison with raw predictions shows the impact of clipping
- If clipping significantly increases boundary clustering → **Clipping is the problem**

### 3. Actual UniFrac Distance Distribution

**Observation:** Ground truth distribution for comparison.

**Key Metrics:**
- [To be filled from console output]
  - Mean: ___
  - Std: ___
  - Min: ___
  - Max: ___
  - Percentiles: 25% = ___, 50% = ___, 75% = ___

**Expected Distribution:**
- Based on PYT-8.13 findings, actual distances should be:
  - Unimodal distribution centered around 0.72
  - Very few zero distances (0.00% of pairs)
  - Continuous distribution, not bimodal at boundaries

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

### Hypothesis 1: Over-Clipping
**Evidence:**
- [ ] Raw predictions are continuous but outside [0, 1] range
- [ ] Clipping forces many predictions to 0.0 or 1.0
- [ ] Raw vs clipped scatter shows many points on boundaries

**If True:** Clipping in forward pass is too aggressive. Raw predictions may be slightly outside [0, 1] but clipping forces them to exact boundaries.

**Solution:** Remove clipping from forward pass, only clip in loss computation if needed, or use a smoother constraint (e.g., sigmoid activation).

### Hypothesis 2: Model Learning Issue
**Evidence:**
- [ ] Raw predictions are already clustered at boundaries
- [ ] Model is not learning continuous distribution
- [ ] Training instability or inappropriate loss function

**If True:** Model architecture or training dynamics are causing boundary convergence.

**Solution:** 
- Use sigmoid activation instead of linear + clip
- Adjust loss function (Huber loss, smooth L1)
- Review learning rate and optimizer settings
- Consider beta regression loss for bounded regression

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

1. **Review Console Output Statistics**
   - Fill in the metrics sections above with actual numbers from console output
   - Calculate R² correlation between actual and predicted
   - Identify exact percentage of boundary predictions

2. **Analyze Visualizations**
   - Review each plot to identify patterns
   - Count boundary clusters in histograms
   - Measure distribution overlap between actual and predicted

### Potential Fixes (Based on Findings)

#### If Clipping is the Problem:
1. **Remove forward pass clipping** - Only clip in loss computation
2. **Use sigmoid activation** - Natural [0, 1] constraint without hard clipping
3. **Smooth clipping** - Use soft clipping (e.g., `x * sigmoid(x)` or `tanh` transformation)

#### If Model Learning is the Problem:
1. **Change final layer activation** - Use sigmoid instead of linear + clip
2. **Adjust loss function** - Use Huber loss or smooth L1 for better gradient flow
3. **Beta regression loss** - Theoretically sound for bounded regression

#### If Training Dynamics are the Problem:
1. **Learning rate tuning** - May need lower learning rate
2. **Gradient clipping** - Already implemented, may need adjustment
3. **Optimizer/scheduler** - Try different combinations (ReduceLROnPlateau, OneCycleLR)

### Next Steps

1. **Fill in statistics** from console output
2. **Review visualizations** to confirm hypotheses
3. **Implement fix** based on root cause identified
4. **Test fix** by re-running inference and comparing distributions
5. **Update ticket** with findings and implementation

## Conclusion

[To be completed after reviewing console output and visualizations]

The investigation has identified [root cause] as the primary reason for boundary prediction clustering. The recommended fix is [solution], which should [expected outcome].

---

## Appendix: Console Output

[Paste console output with distribution statistics here]

```
Raw Predictions Distribution:
  Count: ___
  Mean: ___
  Std: ___
  ...

Clipped Predictions Distribution:
  Count: ___
  Mean: ___
  Std: ___
  ...

Actual Distances Distribution:
  Count: ___
  Mean: ___
  Std: ___
  ...
```
