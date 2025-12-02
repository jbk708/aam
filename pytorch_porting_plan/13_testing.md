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
9. SampleSequenceEncoder - Sample processing
10. SequenceEncoder - UniFrac prediction
11. SequencePredictor - Main model
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

### Unit Tests (✅ Completed - PYT-5.1)
- [x] Test BIOMLoader with sample table (21 tests)
- [x] Test UniFracComputer with sample table and tree (19 tests + 12 error handling tests)
- [x] Test SequenceTokenizer with sample sequences (12 tests)
- [x] Test ASVDataset with sample data (18 tests + 5 edge case tests)
- [x] Test all model components individually (all components have dedicated test files)
  - [x] AttentionPooling (17 tests)
  - [x] PositionEmbedding (8 tests)
  - [x] TransformerEncoder (23 tests)
  - [x] ASVEncoder (28 tests)
  - [x] SampleSequenceEncoder (30 tests)
  - [x] SequenceEncoder (25 tests)
  - [x] SequencePredictor (27 tests)
- [x] Test loss functions (18 tests)
- [x] Test metrics (11 tests)
- [x] Test trainer (21 tests + 6 edge case tests)
- [x] Test CLI interface (28 tests + 10 integration tests)

**Coverage:** 94% (333 tests passing, 4 skipped)

### Integration Tests (In Progress - PYT-5.2)
- [x] Test CLI integration (train/predict commands with mocked components)
- [x] Test dataset integration (DataLoader iteration)
- [x] Test UniFrac integration (with real data files)
- [ ] Test data pipeline end-to-end using `./data/fall_train_only_all_outdoor.biom` and `./data/all-outdoors_sepp_tree.nwk`
- [ ] Test model forward pass integration
- [ ] Test loss computation integration
- [ ] Test training step integration
- [ ] Test validation step integration

### End-to-End Tests (Pending - PYT-5.2)
- [ ] Test full training loop
- [x] Test checkpoint saving/loading (covered in trainer tests)
- [x] Test early stopping (covered in trainer tests)
- [x] Test CLI interface (covered in CLI integration tests)

### Validation Tests (Partially Complete)
- [x] Verify tensor shapes throughout (covered in component tests)
- [x] Verify no NaN/Inf values (covered in component tests)
- [x] Verify UniFrac distances (covered in UniFrac tests)
- [x] Verify loss values reasonable (covered in loss/trainer tests)

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
pytest tests/ --cov=aam --cov-report=html
```

**Current Coverage:** 94% (as of PYT-5.1 completion)

## Success Criteria

### Minimum for Prototype
1. ✅ All components can be instantiated
2. ✅ Forward pass works without errors
3. ✅ Loss computation works
4. ✅ Training loop runs without errors
5. ✅ Model can be saved and loaded
6. ✅ Training on small dataset produces reasonable results

### Full Validation
1. ✅ All unit tests pass (333 tests passing, 4 skipped)
2. ⏳ Integration tests pass (partially complete - see PYT-5.2)
3. ⏳ End-to-end training works (pending - see PYT-5.2)
4. ⏳ No memory leaks (not yet tested)
5. ⏳ Performance is reasonable (not yet tested)
6. ✅ Edge cases handled correctly (comprehensive edge case tests added)

## Notes

- Test incrementally as you build
- Don't wait until everything is built
- Use small datasets for quick iteration
- Verify shapes at each step
- Keep tests simple and focused

## Test File Structure (Updated after PYT-5.1)

All components have dedicated test files:
- `test_biom_loader.py` - BIOM loading and rarefaction
- `test_unifrac.py` - UniFrac computation (includes error handling)
- `test_tokenizer.py` - Sequence tokenization
- `test_dataset.py` - Dataset and collate function (includes edge cases)
- `test_attention_pooling.py` - Attention pooling and mask utilities
- `test_position_embedding.py` - Position embeddings
- `test_transformer.py` - Transformer encoder
- `test_asv_encoder.py` - ASV-level encoder
- `test_sample_sequence_encoder.py` - Sample-level encoder
- `test_sequence_encoder.py` - Sequence encoder with UniFrac head
- `test_sequence_predictor.py` - Main prediction model
- `test_losses.py` - Loss functions
- `test_metrics.py` - Metrics computation
- `test_trainer.py` - Training loop (includes edge cases)
- `test_cli.py` - CLI interface (includes integration tests)

**Note:** `test_models.py` mentioned in some tickets is not needed - all model components have dedicated test files. This is the preferred structure as it keeps tests organized by component.
