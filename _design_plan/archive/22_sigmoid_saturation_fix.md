# Sigmoid Saturation Fix

**Status:** Completed
**Ticket:** PYT-11.5

## Problem

Validation plots showed all predictions clustering at ~0.55 and not changing during training due to sigmoid saturation.

## Root Cause

After normalizing by max distance, values were in [0, 1]. Applying sigmoid to this range:
- `sigmoid(0) ≈ 0.5`
- `sigmoid(1) ≈ 0.73`

This compressed all distance relationships into a narrow range with poor gradient flow.

## Solution

Replaced sigmoid with tanh normalization:
```python
distances = (tanh(distances / scale) + 1) / 2
```

Using fixed scale (10.0) for consistent scaling across batches.

## Results

- Predictions distributed across [0, 1] range
- Healthy gradient magnitudes (> 1e-5)
- Consistent scaling across all batches

## Files Modified
- `aam/training/losses.py` - `compute_pairwise_distances()`, `compute_stripe_distances()`
