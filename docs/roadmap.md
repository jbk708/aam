# AAM Roadmap

Future enhancements and ideas for AAM development.

## Completed Features

- Core components (data pipeline, models, training, CLI)
- Memory-efficient defaults (gradient checkpointing, ASV chunking)
- Mixed precision training (fp16, bf16)
- Model compilation (torch.compile)
- Categorical features with multiple fusion strategies
- Regressor optimizations (MLP head, quantile/asymmetric loss, residual head)
- Attention-based fusion (GMU, cross-attention, perceiver)
- Multi-GPU support (DDP, FSDP, DataParallel)
- ROCm support for AMD GPUs

## Outstanding Tickets

See `.agents/TICKET_OVERVIEW.md` for current work:

| Ticket | Description | Priority |
|--------|-------------|----------|
| REG-9 | Mixture of Experts | LOW |
| PYT-19.4 | Hierarchical categories | LOW |

## Future Ideas

### Performance Optimizations

**Streaming Validation**
- Compute metrics incrementally instead of accumulating all predictions
- Reduce validation memory from O(dataset) to O(batch)

**Configurable FFN Ratio**
- Add `--ffn-ratio` flag (default: 4, can reduce to 2)
- Trade memory for capacity

### Model Improvements

**Attention Visualization**
- Tools to visualize attention patterns
- Which ASVs drive predictions?

**Additional Distance Metrics**
- Bray-Curtis, Jaccard distance support
- Alternative self-supervised objectives

### Data Pipeline

**Streaming Data Loading**
- Handle datasets larger than memory
- Useful for large-scale studies

**Data Augmentation**
- Sequence augmentation strategies
- Dropout at ASV level

### Experiment Tracking

**W&B / MLflow Integration**
- Experiment logging and comparison
- Hyperparameter tracking

**Hyperparameter Optimization**
- Optuna integration for automated tuning

### Deployment

**ONNX Export**
- Export models for inference optimization
- Deployment to edge devices

**Docker Containerization**
- Reproducible deployment environment

## Contributing

Interested in implementing any of these? See [CONTRIBUTING.md](../CONTRIBUTING.md) for how to get started.
