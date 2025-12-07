# Sigmoid Saturation Fix for Mode Collapse

## Problem

After implementing sigmoid activation to fix boundary clustering, the model is now converging to constant predictions (mode collapse) in the first 5 epochs. This is a serious underfitting issue.

## Root Cause

Sigmoid activation can saturate when the linear layer outputs are too large:
- Large positive outputs → sigmoid → ~1.0 (saturated)
- Large negative outputs → sigmoid → ~0.0 (saturated)
- When sigmoid saturates, gradients vanish → model can't learn
- Model collapses to predicting the same value for all inputs

## Solution Implemented

### 1. Scaled Sigmoid Input
- Scale the linear output by 0.1 before applying sigmoid: `torch.sigmoid(x * 0.1)`
- This keeps sigmoid in the linear region (not saturated)
- For input range [-10, 10], scaled sigmoid outputs [0.27, 0.73] (not saturated)
- Regular sigmoid would output [0.000, 1.000] (saturated at extremes)

### 2. Smaller Weight Initialization
- Initialize output head weights with std=0.01 (instead of default ~0.1)
- Initialize bias to 0.0
- This prevents initial outputs from being too large

## Expected Behavior

- Model should learn diverse predictions, not collapse to constant
- Predictions should be in [0, 1] range (sigmoid constraint)
- Gradients should flow properly (no saturation)
- Training should converge normally

## Testing

If the model still collapses to constant predictions:
1. Try even smaller scaling factor (0.05 or 0.02)
2. Try even smaller weight initialization (std=0.001)
3. Consider using tanh scaled to [0, 1]: `(torch.tanh(x) + 1) / 2`
4. Consider adding a learnable scale parameter

## Alternative Approaches

If scaled sigmoid doesn't work:
1. **Tanh scaled to [0, 1]**: `(torch.tanh(x) + 1) / 2` - smoother gradients
2. **Soft clipping**: `x * torch.sigmoid(x)` - smoother than hard clipping
3. **Learnable scale**: Add a learnable parameter to scale sigmoid input
4. **Different activation**: Consider ReLU + normalization for bounded output
