# TensorFlow vs PyTorch Implementation Comparison

## Key Finding: Different Approaches to UniFrac Distance Prediction

### TensorFlow Implementation (Main Branch)

**Approach:**
1. Model outputs **embeddings** (linear activation, no constraint)
2. Pairwise distances computed from embeddings using **Euclidean distance**
3. No sigmoid, no clipping on embeddings
4. Distances naturally bounded (Euclidean distance ≥ 0)

**Code:**
```python
# unifrac_model.py line 161
sample_embeddings = self.linear_activation(sample_embeddings)  # Linear activation (no-op)
# Returns embeddings directly

# losses.py _pairwise_distances()
distances = sqrt(||a - b||^2)  # Euclidean distance from embeddings
distances = tf.maximum(distances, 0.0)  # Only clip negative values (numerical stability)
```

**Key Points:**
- Embeddings are unconstrained (can be any value)
- Distances computed as `sqrt(sum((a - b)^2))`
- No sigmoid or clipping on embeddings
- Only clips negative distances (numerical stability, not constraint)

### PyTorch Implementation (Current)

**Approach:**
1. Model outputs **direct distance predictions** via linear head
2. Predictions constrained to [0, 1] using **sigmoid activation**
3. `base_output_dim = batch_size` means predicting full pairwise matrix directly

**Code:**
```python
# sequence_encoder.py
base_prediction = self.output_head(pooled_embeddings)  # [batch_size, batch_size]
base_prediction = torch.sigmoid(base_prediction)  # Constrain to [0, 1]
```

**Key Points:**
- Direct distance predictions (not computed from embeddings)
- Sigmoid constrains predictions to [0, 1]
- Different architecture than TensorFlow

## The Problem

The PyTorch implementation uses a fundamentally different approach:
- **TensorFlow**: Embeddings → Euclidean distance (natural, unbounded)
- **PyTorch**: Direct predictions → Sigmoid constraint (artificial, bounded)

This architectural difference may be causing the issues:
1. Boundary clustering (fixed with sigmoid, but causes mode collapse)
2. Mode collapse to 0.5 (sigmoid saturation issue)
3. Underfitting (model can't learn proper distribution)

## Key Insight

**TensorFlow computes distances from embeddings, PyTorch predicts them directly.**

This fundamental architectural difference explains why:
1. TensorFlow doesn't need sigmoid/clipping (distances computed, not predicted)
2. PyTorch needs constraints (direct predictions must be bounded)
3. Current issues with sigmoid (saturation, mode collapse)

## Recommendation

**Option 1: Match TensorFlow Approach (Recommended)**
1. Remove direct distance prediction head
2. Use embeddings directly (linear activation)
3. Compute pairwise distances from embeddings (Euclidean distance)
4. No sigmoid/clipping needed (distances naturally ≥ 0)

**Benefits:**
- Matches TensorFlow implementation exactly
- Avoids sigmoid saturation issues
- Allows model to learn proper distance relationships
- Naturally produces distances without artificial constraints

**Option 2: Fix Current Approach**
If keeping direct predictions:
1. Use tanh scaled to [0, 1]: `(torch.tanh(x) + 1) / 2`
2. Or use learnable scale parameter
3. Better initialization (already done with Xavier + bias)

**Current Status:**
- Model predicting 0.5 suggests sigmoid is saturating at 0 (input = 0)
- Xavier init + bias 0.85 should help, but may need further tuning
- Consider Option 1 for long-term solution
