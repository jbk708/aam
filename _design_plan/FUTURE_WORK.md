# Future Work

## Completed

- All core components (data pipeline, models, training, CLI)
- Phase 8: Feature enhancements
- Phase 9: UniFrac underfitting fixes
- Phase 10: Performance optimizations (mixed precision, compilation, SDPA)
- Phase 11: Critical fixes (target normalization, metadata handling, LR scheduling)

## Outstanding (~80-115 hours)

See `.agents/PYTORCH_PORTING_TICKETS.md` for detailed tickets.

### Performance (Phase 10, 12)
- Multi-GPU training (DDP)
- FSDP for large models
- Batch size optimization
- Caching mechanisms

### Model Improvements (Phase 13)
- Attention visualization
- Feature importance analysis
- Additional encoder types (Bray-Curtis, Jaccard)

### Data Pipeline (Phase 14)
- Streaming data loading
- Data augmentation

### Training (Phase 15)
- Experiment tracking (W&B, MLflow)
- Hyperparameter optimization (Optuna)

### Evaluation (Phase 16)
- Benchmarking suite
- Error analysis tools

### Documentation & Deployment (Phase 17)
- Sphinx API docs
- Tutorial notebooks
- ONNX export
- Docker containerization
