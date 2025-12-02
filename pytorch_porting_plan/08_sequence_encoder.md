# SequenceEncoder

## Objective
Implement encoder that adds prediction head for UniFrac distance prediction. This serves as the base model for SequenceRegressor.

## Architecture Philosophy

**Purpose**: SequenceEncoder is designed to be used as a base model for SequenceRegressor. It:
- Processes sequences through BaseSequenceEncoder
- Adds encoder-specific prediction head
- Produces base embeddings for downstream use
- Can be frozen when used in SequenceRegressor

**Key Insight**: Nucleotide prediction and UniFrac prediction are **parallel tasks** that share the same base embeddings. They are NOT sequential - nucleotide predictions are not used as input to UniFrac prediction. Both tasks help learn good representations through multi-task learning.

**Data Flow**:
```
Input: [B, S, L] tokens
  ↓
BaseSequenceEncoder: 
  - Processes tokens through ASVEncoder
  - Produces base embeddings: [B, S, D]
  - Also produces nucleotide predictions: [B, S, L, 5] (self-supervised)
  ↓
Encoder Transformer: [B, S, D] (uses base embeddings, NOT nuc predictions)
  ↓
Attention Pooling: [B, D]
  ↓
Linear: [B, base_output_dim]
  ↓
Output: [B, base_output_dim] base prediction
```

**Important**: The nucleotide predictions are a **side output** used only for self-supervised learning loss. They do NOT feed into the UniFrac prediction head. Both tasks share the same base embeddings learned from the sequence processing.

## Components

**1. BaseSequenceEncoder**
- Core sequence processing
- Outputs base embeddings: `[B, S, D]`
- Also outputs nucleotide predictions `[B, S, L, 5]` if training (self-supervised task)
- Nucleotide predictions are NOT used as input to encoder head

**2. Encoder Transformer**
- Additional transformer layer
- Encoder-specific processing
- **Input**: Base embeddings from BaseSequenceEncoder (NOT nucleotide predictions)
- Learns features for UniFrac/Taxonomy prediction

**3. Attention Pooling**
- Pools ASV-level to sample-level
- Learns which ASVs are important

**4. Dense Output Layer**
- Linear layer to prediction dimension
- Output size depends on encoder type:
  - Unweighted UniFrac: `[batch_size, batch_size]` (pairwise distances)
  - Faith PD: `[batch_size, 1]` (per-sample diversity)
  - Taxonomy: `[batch_size, num_levels]` (taxonomic predictions)
  - Combined: Tuple of (unifrac, faith_pd, taxonomy) predictions

## Multi-Task Learning

**Parallel Tasks** (not sequential):
1. **Nucleotide Prediction**: Self-supervised task to learn sequence patterns
2. **UniFrac/Taxonomy Prediction**: Encoder-specific task to learn phylogenetic/taxonomic relationships

**Shared Representations**:
- Both tasks use the same base embeddings from BaseSequenceEncoder
- Learning to predict nucleotides helps learn good sequence representations
- Learning to predict UniFrac helps learn phylogenetic relationships
- Both tasks improve the shared base embeddings through multi-task learning

**Loss Computation**:
- Nucleotide loss: CrossEntropy on nucleotide predictions
- Encoder loss: MSE/CrossEntropy on UniFrac/Taxonomy predictions
- Total loss: `nuc_loss + encoder_loss` (both computed in parallel)

## Implementation Requirements

**Class Structure**:
- Inherit from `nn.Module`
- Composes BaseSequenceEncoder
- Forward pass: base → transform → pool → predict

**Output Handling**:
- Always return base embeddings: `[B, S, D]`
- Return base predictions: `[B, base_output_dim]` (if `base_output_dim` specified)
- Return nucleotide predictions: `[B, S, L, 5]` (if training) - **side output, not used as input**

**Output Dictionary**:
```python
{
    'base_prediction': [B, base_output_dim] or [B, D],
    'base_embeddings': [B, S, D],
    'nuc_predictions': [B, S, L, 5],  # if training - side output for loss only
}
```

**Training vs Inference**:
- Training: Return predictions, embeddings, nucleotide predictions (for loss)
- Inference: Return predictions and embeddings

**Combined Encoder Type**:
- When `encoder_type='combined'`:
  - Predicts UniFrac distances, Faith PD, and Taxonomy simultaneously
  - Uses separate heads: `uni_ff`, `faith_ff`, `tax_ff`
  - Returns tuple of predictions: `(unifrac_pred, faith_pred, tax_pred)`
  - All predictions use the same base embeddings (parallel, not sequential)

## Implementation Checklist

- [ ] Create `SequenceEncoder` class inheriting from `nn.Module`
- [ ] Initialize BaseSequenceEncoder
- [ ] Initialize encoder transformer
- [ ] Initialize attention pooling
- [ ] Initialize dense output layer(s) based on encoder_type
- [ ] Handle combined encoder type (multiple heads)
- [ ] Implement forward pass
- [ ] Return base embeddings (critical for SequenceRegressor)
- [ ] Return base predictions
- [ ] Return nucleotide predictions as side output (for loss only)
- [ ] Handle mask creation and conversion
- [ ] Handle optional base_output_dim
- [ ] Handle training vs inference modes
- [ ] Test with dummy data
- [ ] Verify output shapes
- [ ] Test with different base_output_dim values
- [ ] Test combined encoder type

## Key Considerations

### Base Embeddings
- **Critical**: Must return base embeddings for SequenceRegressor
- Base embeddings are shared between encoder and regressor
- Shape: `[B, S, D]` - ASV-level embeddings

### Nucleotide Predictions
- **Side output**: Not used as input to encoder head
- Used only for self-supervised learning loss
- Helps learn good sequence representations
- Parallel task, not sequential dependency

### Output Dimension
- `base_output_dim` determines prediction size
- If None, return pooled embeddings without projection
- Common values:
  - Unweighted UniFrac: `batch_size` (pairwise distances)
  - Faith PD: `1` (per-sample diversity)
  - Taxonomy: `7` (taxonomic levels)
  - Combined: Tuple of `(batch_size, 1, num_levels)` dimensions

### Integration with SequenceRegressor
- SequenceRegressor uses this as `base_model`
- Base embeddings extracted from output dictionary
- Base predictions used for loss computation
- Can be frozen when used in SequenceRegressor

### Encoder Types
- Can support different encoder types (UniFrac, Taxonomy, Faith PD, Combined)
- Each type has different output dimension
- Can be specified via `encoder_type` parameter
- Combined type predicts all three simultaneously (parallel heads)

## Testing Requirements

### Basic Forward Pass
- Input: `[batch_size, num_asvs, max_bp]` tokens
- Output: Dictionary with `base_prediction` and `base_embeddings`
- Shape: `[B, base_output_dim]` or `[B, D]` for predictions
- Shape: `[B, S, D]` for base embeddings

### Nucleotide Predictions
- Verify nucleotide predictions are returned as side output
- Verify they are NOT used as input to encoder head
- Verify they match expected shape `[B, S, L, 5]`

### Combined Encoder Type
- Verify all three predictions are returned
- Verify they use the same base embeddings
- Verify output shapes match expected dimensions

### Integration with SequenceRegressor
- Verify base embeddings can be extracted
- Verify base predictions are correct shape
- Verify nucleotide predictions passed through
- Test with frozen base model

## Notes

- **Base model role**: Designed to be used as base for SequenceRegressor
- **Base embeddings**: Critical output for downstream use
- **Nucleotide predictions**: Side output for self-supervised learning, NOT input to encoder
- **Multi-task learning**: Parallel tasks share base embeddings
- **Output dimension**: Determines prediction size
- **Freezing**: Can be frozen when used in SequenceRegressor
- **Multi-purpose**: Used standalone or as base model
- **Combined type**: Predicts multiple targets simultaneously using parallel heads
