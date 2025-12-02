# Training Losses and Metrics

## Objective
Implement multi-task loss functions and metrics.

## Loss Functions

### 1. Target Loss
- **Regression**: MSE between predicted and actual targets
- **Classification**: NLL with optional class weights

### 2. Count Loss
- MSE between predicted and actual ASV counts
- Masked to ignore padding ASVs

### 3. Base Loss (UniFrac)
- MSE between predicted and actual UniFrac distances
- Weighted by `penalty` parameter
- Handles different metrics:
  - Unweighted UniFrac: Pairwise matrix `[B, B]`
  - Faith PD: Per-sample `[B, 1]`

### 4. Nucleotide Loss
- Masked CrossEntropy for nucleotide prediction
- Weighted by `nuc_penalty` parameter
- Mask padding positions

### 5. Total Loss
- Weighted sum of all losses
- Enables task balancing

## Metrics

### Regression
- MAE, MSE, R-squared (optional)

### Classification
- Accuracy, Confusion Matrix, Precision/Recall/F1 (optional)

## Implementation Requirements

### Loss Computation Function
- Input: Model outputs dictionary, targets dictionary
- Compute individual losses
- Return dictionary: `{target_loss, count_loss, base_loss, nuc_loss, total_loss}`
- Handle missing outputs (inference mode)

### Class Weights
- Register as buffer in loss module
- Apply in NLL loss computation

## Implementation Checklist

- [x] Implement target loss (MSE/NLL)
- [x] Implement count loss (masked MSE)
- [x] Implement base loss (MSE for UniFrac)
- [x] Implement nucleotide loss (masked CrossEntropy)
- [x] Implement total loss computation
- [x] Handle class weights for classification
- [x] Implement regression metrics
- [x] Implement classification metrics
- [x] Test loss computation
- [x] Test metrics computation

## Key Considerations

- Loss weighting: Balance different loss scales
- Masking: Handle padding for nucleotide and count losses
- Device handling: Ensure all tensors on same device
- Metrics: Move to CPU for sklearn computation

## Testing Requirements

- Test each loss function individually
- Test total loss computation
- Test with missing outputs (inference mode)
- Test metrics computation
- Test with class weights
