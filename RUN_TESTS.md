# Running Tests

## Recommended pytest commands for agent interpretability

### Full test suite with verbose output
```bash
pytest tests/ -vv --tb=short -ra --durations=10
```

**Flags explained:**
- `-vv`: Very verbose - shows each test name as it runs
- `--tb=short`: Short traceback format (easier to read)
- `-ra`: Show summary of all test outcomes (passed, failed, skipped, etc.)
- `--durations=10`: Show 10 slowest tests at the end

### Stop on first failure (good for debugging)
```bash
pytest tests/ -vv --tb=short -x
```

### Run specific test file
```bash
pytest tests/test_losses.py -vv --tb=short
```

### Run specific test class
```bash
pytest tests/test_losses.py::TestTargetLoss -vv --tb=short
```

### Run specific test function
```bash
pytest tests/test_losses.py::TestTargetLoss::test_target_loss_regression -vv --tb=short
```

### Run tests matching a pattern
```bash
pytest tests/ -k "loss" -vv --tb=short
```

### With coverage report
```bash
pytest tests/ -vv --tb=short --cov=aam --cov-report=term-missing
```

## Test files available:
- `test_asv_encoder.py` - ASV encoder tests
- `test_attention_pooling.py` - Attention pooling tests
- `test_biom_loader.py` - BIOM data loading tests
- `test_dataset.py` - Dataset tests
- `test_integration.py` - Integration tests
- `test_losses.py` - Loss function tests
- `test_metrics.py` - Metrics tests
- `test_position_embedding.py` - Position embedding tests
- `test_sample_sequence_encoder.py` - Sample encoder tests
- `test_sequence_encoder.py` - Sequence encoder tests
- `test_sequence_predictor.py` - Sequence predictor tests
- `test_tokenizer.py` - Tokenizer tests
- `test_trainer.py` - Trainer tests
- `test_transformer.py` - Transformer tests
- `test_unifrac.py` - UniFrac distance tests
