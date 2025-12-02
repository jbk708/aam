# SequenceRegressor

## Objective
Implement main model for predicting sample-level targets and ASV counts. This model composes SequenceEncoder as its base model.

## Architecture Philosophy

**Key Design**: SequenceRegressor uses SequenceEncoder as its `base_model` through composition (not inheritance). This enables:
- Transfer learning (freeze base model)
- Shared base embeddings
- Flexible base model swapping
- Multi-task learning

**Important Clarification**: The model does NOT use base predictions (UniFrac) as input to target prediction. Instead:
- Base predictions are used only for **loss computation** (self-supervised learning)
- Target prediction uses **base embeddings** (shared representations)
- This is multi-task learning with shared representations, not sequential prediction

**Data Flow**:
```
Input: [B, S, L] tokens, [B, S, 1] counts
  ↓
SequenceEncoder (base_model): 
  - Processes sequences through SampleSequenceEncoder
  - Produces base embeddings: [B, S, D] (shared representations)
  - Produces base predictions: [B, base_output_dim] (for loss only)
  - Produces nucleotide predictions: [B, S, L, 5] (for loss only)
  ↓
┌─────────────────────┬─────────────────────┐
│                     │                     │
Count Encoder         Target Encoder
(uses base_embeddings) (uses base_embeddings)
Transformer           Transformer
  ↓                     ↓
Linear(1)           Attention Pooling
  ↓                     ↓
[B, S, 1]          [B, D]
count_prediction       ↓
                  Linear(out_dim)
                       ↓
                  [B, out_dim]
                  target_prediction
```

**Key Point**: Base predictions (UniFrac) and nucleotide predictions are **side outputs** used only for loss computation. They do NOT feed into the count or target encoders. All tasks share the same base embeddings.

## Components

**1. Base Model (SequenceEncoder)**
- **Composition**: SequenceRegressor contains SequenceEncoder instance as `base_model`
- **Purpose**: Processes sequences and predicts UniFrac (or other encoder targets)
- **Outputs**: 
  - Base embeddings: `[B, S, D]` (used by regressor heads)
  - Base predictions: `[B, base_output_dim]` (for loss computation only, NOT used as input)
  - Nucleotide predictions: `[B, S, L, 5]` (for loss computation only, NOT used as input)
- **Freezing**: Can freeze base model parameters (`freeze_base=True`)

**2. Count Encoder**
- **Input**: Base embeddings from `base_model` (`[B, S, D]`)
- **NOT Input**: Base predictions or nucleotide predictions
- **Processing**: Transformer encoder
- **Output**: Count predictions per ASV `[B, S, 1]`
- **Purpose**: Learn ASV abundance patterns

**3. Target Encoder**
- **Input**: Base embeddings from `base_model` (`[B, S, D]`)
- **NOT Input**: Base predictions or nucleotide predictions
- **Processing**: Transformer encoder → Attention pooling
- **Output**: Sample-level target predictions `[B, out_dim]`
- **Purpose**: Main prediction task

**4. Classification vs Regression**
- **Regression**: Raw output
- **Classification**: Log-softmax output

## Multi-Task Learning Architecture

**Parallel Tasks** (all share base embeddings):
1. **Nucleotide Prediction**: Self-supervised task (loss only)
2. **Base Prediction** (UniFrac/Taxonomy): Self-supervised task (loss only)
3. **Count Prediction**: Auxiliary task (loss + helps learn ASV importance)
4. **Target Prediction**: Primary task (main objective)

**Shared Representations**:
- All tasks use the same base embeddings from SequenceEncoder
- Learning to predict nucleotides helps learn sequence patterns
- Learning to predict UniFrac helps learn phylogenetic relationships
- Learning to predict counts helps learn ASV importance
- All tasks improve the shared base embeddings through multi-task learning

**Loss Computation**:
- Nucleotide loss: CrossEntropy on nucleotide predictions (self-supervised)
- Base loss: MSE/CrossEntropy on UniFrac/Taxonomy predictions (self-supervised)
- Count loss: MSE on count predictions (auxiliary task)
- Target loss: MSE/CrossEntropy on target predictions (primary task)
- Total loss: Weighted sum of all losses

## Implementation Requirements

**Class Structure**:
- Inherit from `nn.Module`
- Initialize SequenceEncoder as `base_model` (can be passed or created)
- Initialize count and target encoders
- Handle base model freezing

**Key Methods**:
- `__init__()`: Initialize base model and heads
- `forward()`: Forward pass through base model and heads
- Return dictionary of all predictions

**Base Model Initialization**:
- Can accept SequenceEncoder instance as parameter
- Or create SequenceEncoder internally with specified `encoder_type`
- Default: `encoder_type='unifrac'` (UniFracEncoder)
- Other options: 'taxonomy', 'faith_pd', 'combined'
- Combined type: Predicts UniFrac, Faith PD, and Taxonomy simultaneously

**Freezing Base Model**:
- If `freeze_base=True`: Set `requires_grad=False` for all base model parameters
- Do this in `__init__()` after creating base model
- Frozen model still participates in forward pass but gradients don't flow
- Only count and target encoders receive gradients

**Forward Pass Logic**:
1. Call base model: `base_outputs = base_model(tokens, counts, training=training)`
2. Extract base embeddings: `base_embeddings = base_outputs['base_embeddings']`
3. Extract base predictions: `base_predictions = base_outputs['base_prediction']` (if training, for loss only)
4. Extract nucleotide predictions: `nuc_predictions = base_outputs['nuc_predictions']` (if training, for loss only)
5. Process with count encoder: `count_prediction = count_encoder(base_embeddings)` (uses embeddings, NOT predictions)
6. Process with target encoder: `target_prediction = target_encoder(base_embeddings)` (uses embeddings, NOT predictions)
7. Return dictionary with all outputs

**Output Dictionary**:
```python
{
    'target_prediction': [B, out_dim],
    'count_prediction': [B, S, 1],
    'base_embeddings': [B, S, D],
    'base_prediction': [B, base_output_dim],  # if training - for loss only
    'nuc_predictions': [B, S, L, 5],  # if training - for loss only
}
```

## Implementation Checklist

- [x] Create `SequenceRegressor` class inheriting from `nn.Module`
- [x] Accept SequenceEncoder as `base_model` parameter or create internally
- [x] Handle `encoder_type` parameter for base model creation
- [x] Handle `freeze_base` parameter (set `requires_grad=False`)
- [x] Initialize count encoder (transformer + linear)
- [x] Initialize target encoder (transformer + pooling + linear)
- [x] Handle classifier vs regression mode
- [x] Implement forward pass
- [x] Extract base embeddings from base model output
- [x] Use base embeddings (NOT base predictions) for count and target encoders
- [x] Return dictionary of predictions
- [x] Test with dummy data
- [x] Verify output shapes
- [x] Test training and inference modes
- [x] Test with frozen and unfrozen base model
- [x] Verify base predictions are NOT used as input to heads

## Key Considerations

### Composition Pattern
- SequenceRegressor contains SequenceEncoder (composition)
- Not inheriting from SequenceEncoder
- Enables flexible base model swapping

### Shared Base Embeddings
- Base embeddings come from SequenceEncoder
- Used by both count and target encoders
- **NOT** base predictions - those are for loss only
- Enables multi-task learning

### Base Predictions Are For Loss Only
- Base predictions (UniFrac) are NOT used as input to count/target encoders
- They are side outputs used only for loss computation
- This is self-supervised learning, not sequential prediction
- Base embeddings (not predictions) are what feed into regression heads

### Base Model Freezing
- Set `requires_grad=False` for base model parameters
- Base model still participates in forward pass
- Only regression heads receive gradients
- Useful for transfer learning

### Multi-Task Learning
- Predicts multiple targets simultaneously
- Count prediction helps learn ASV importance
- Base prediction (UniFrac) adds phylogenetic signal (via loss, not input)
- Nucleotide prediction adds sequence signal (via loss, not input)
- All tasks share base embeddings

### Loss Computation
- Loss computation is separate (see `10_training_losses.md`)
- Model returns dictionary with all necessary outputs
- Loss function extracts values from dictionary
- Base loss uses base predictions from base model (for self-supervised learning)
- Target loss uses target predictions (primary task)

## Testing Requirements

### Basic Forward Pass
- Input: `[B, S, L]` tokens, `[B, S, 1]` counts
- Output: Dictionary with predictions
- Shapes:
  - `target_prediction`: `[B, out_dim]`
  - `count_prediction`: `[B, S, 1]`
  - `base_embeddings`: `[B, S, D]`

### Training Mode
- Returns additional outputs: `base_prediction`, `nuc_predictions`
- Verify shapes match expected dimensions
- Verify base predictions are NOT used as input to heads

### Base Model Freezing
- Verify gradients don't flow to frozen base model
- Verify gradients flow to count/target encoders
- Verify base model parameters don't change

### Composition
- Verify base model can be swapped
- Verify base model outputs are used correctly
- Test with different encoder types
- Test with combined encoder type

### Multi-Task Learning
- Verify all tasks share base embeddings
- Verify base predictions are side outputs (not inputs)
- Verify loss computation uses all predictions correctly

## Notes

- **Composition over inheritance**: SequenceRegressor composes SequenceEncoder
- **Shared embeddings**: Base embeddings shared between heads
- **Base predictions for loss only**: NOT used as input to regression heads
- **Multi-task learning**: Parallel tasks with shared representations
- **Freezing**: Can freeze base model for transfer learning
- **Loss computation**: Keep separate from model (PyTorch convention)
- **Combined encoder**: Can use combined encoder type for multiple base predictions
