# Testing Strategy

## Objective
Define comprehensive testing strategy for PyTorch implementation.

## Testing Phases

### Phase 1: Unit Tests (Component-Level)

**Components to Test**:
1. BIOMLoader - Table loading and rarefaction
2. UniFracComputer - Distance computation
3. SequenceTokenizer - Tokenization
4. ASVDataset - Data loading
5. AttentionPooling - Attention mechanism
6. PositionEmbedding - Position encoding
7. TransformerEncoder - Self-attention
8. ASVEncoder - Sequence processing
9. BaseSequenceEncoder - Sample processing
10. SequenceEncoder - UniFrac prediction
11. SequenceRegressor - Main model
12. Loss functions - Loss computation
13. Metrics - Metric computation

**Test Requirements**:
- Create dummy inputs with correct shapes
- Call forward pass or function
- Verify output shapes match expected
- Verify no errors or warnings
- Test with edge cases

### Phase 2: Integration Tests

**Data Pipeline**:
- Load BIOM table → Rarefy → Compute UniFrac → Tokenize → Dataset
- Verify end-to-end data flow
- Verify tensor shapes and dtypes

**Model Pipeline**:
- Create model → Forward pass → Loss computation
- Verify all components work together
- Verify output dictionary structure

**Training Pipeline**:
- Create model, optimizer, data loaders
- Run training step
- Run validation step
- Verify training loop works

### Phase 3: End-to-End Tests

**Full Training**:
- Load real BIOM table and tree
- Create model
- Train for 1-2 epochs on small dataset
- Verify:
  - Loss decreases (or stays stable)
  - Model parameters update
  - No errors or crashes
  - Checkpoints save correctly

### Phase 4: Validation Tests

**Numerical Validation**:
- Compare outputs with known values
- Verify UniFrac distances match reference
- Verify loss values are reasonable
- Check for NaN or Inf values

**Shape Validation**:
- Verify all tensor shapes match expected
- Check batch dimension consistency
- Verify mask shapes

## Test Data

### Test Data Location
Test data files are located in the `./data/` folder:
- `./data/fall_train_only_all_outdoor.biom` - BIOM table for testing
- `./data/fall_train_only_all_outdoor.tsv` - TSV version (alternative format)
- `./data/all-outdoors_sepp_tree.nwk` - Phylogenetic tree for UniFrac computation

### Synthetic Data
- Generate small BIOM table (10 samples, 50 ASVs) for unit tests
- Generate simple phylogenetic tree for unit tests
- Generate synthetic sequences for unit tests
- Generate synthetic metadata for unit tests

### Real Data (Integration & End-to-End Tests)
- Use `./data/fall_train_only_all_outdoor.biom` for BIOM loading tests
- Use `./data/all-outdoors_sepp_tree.nwk` for UniFrac computation tests
- Use both files together for integration and end-to-end tests
- These files provide real-world data for validation

## Implementation Checklist

### Unit Tests
- [ ] Test BIOMLoader with sample table
- [ ] Test UniFracComputer with sample table and tree
- [ ] Test SequenceTokenizer with sample sequences
- [x] Test ASVDataset with sample data
- [ ] Test all model components individually
- [ ] Test loss functions
- [ ] Test metrics

### Integration Tests
- [ ] Test data pipeline end-to-end using `./data/fall_train_only_all_outdoor.biom` and `./data/all-outdoors_sepp_tree.nwk`
- [ ] Test model forward pass
- [ ] Test loss computation
- [ ] Test training step
- [ ] Test validation step

### End-to-End Tests
- [ ] Test full training loop
- [ ] Test checkpoint saving/loading
- [ ] Test early stopping
- [ ] Test CLI interface

### Validation Tests
- [ ] Verify tensor shapes throughout
- [ ] Verify no NaN/Inf values
- [ ] Verify UniFrac distances
- [ ] Verify loss values reasonable

## Key Testing Areas

### Data Pipeline
- BIOM loading and rarefaction
- UniFrac computation accuracy
- Tokenization correctness
- Dataset and DataLoader functionality

### Model Components
- Forward pass correctness
- Output shapes
- Mask handling
- Training vs inference modes

### Training
- Loss computation
- Gradient flow
- Optimizer updates
- Early stopping
- Checkpointing

### Edge Cases
- Empty samples
- Single ASV per sample
- Very long sequences
- Very short sequences
- All padding
- Batch size = 1

## Debugging Tools

### Print Statements
- Print tensor shapes at key points
- Print tensor values for debugging
- Print device information

### Gradient Checking
- Use `torch.autograd.detect_anomaly()` for gradient issues
- Check for NaN gradients
- Verify gradients flow correctly

### Memory Profiling
- Monitor GPU memory usage
- Check for memory leaks
- Profile memory usage

## Test Execution

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test
```bash
pytest tests/test_asv_encoder.py -v
```

### Run with Coverage
```bash
pytest tests/ --cov=aam.pytorch --cov-report=html
```

## Success Criteria

### Minimum for Prototype
1. ✅ All components can be instantiated
2. ✅ Forward pass works without errors
3. ✅ Loss computation works
4. ✅ Training loop runs without errors
5. ✅ Model can be saved and loaded
6. ✅ Training on small dataset produces reasonable results

### Full Validation
1. All unit tests pass
2. Integration tests pass
3. End-to-end training works
4. No memory leaks
5. Performance is reasonable
6. Edge cases handled correctly

## Notes

- Test incrementally as you build
- Don't wait until everything is built
- Use small datasets for quick iteration
- Verify shapes at each step
- Keep tests simple and focused
