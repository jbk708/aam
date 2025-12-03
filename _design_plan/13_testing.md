# Testing Strategy

**Status:** ✅ Completed

## Overview
Comprehensive testing strategy for PyTorch implementation. All components have dedicated test files.

## Test Coverage
- **Unit Tests**: 333 tests passing, 4 skipped (94% coverage)
- **Integration Tests**: 13 comprehensive tests
- **End-to-End Tests**: 3 slow tests with real data

## Test Files
All components have dedicated test files in `tests/`:
- Data pipeline: `test_biom_loader.py`, `test_unifrac.py`, `test_tokenizer.py`, `test_dataset.py`
- Model components: `test_attention_pooling.py`, `test_position_embedding.py`, `test_transformer.py`, `test_asv_encoder.py`, `test_sample_sequence_encoder.py`, `test_sequence_encoder.py`, `test_sequence_predictor.py`
- Training: `test_losses.py`, `test_metrics.py`, `test_trainer.py`
- CLI: `test_cli.py`
- Integration: `test_integration.py`

## Test Execution
```bash
pytest tests/ -v                    # Run all tests
pytest tests/ --cov=aam --cov-report=html  # With coverage
```

## Success Criteria
✅ All components instantiate correctly
✅ Forward passes work without errors
✅ Loss computation works
✅ Training loop runs without errors
✅ Model can be saved and loaded
✅ Edge cases handled correctly
