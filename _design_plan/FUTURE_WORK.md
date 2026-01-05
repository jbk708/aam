# Future Work

## Completed

- All core components (data pipeline, models, training, CLI)
- Memory-efficient defaults (gradient checkpointing, mem_efficient attention, ASV chunking)
- Mixed precision training (fp16, bf16)
- Model compilation (torch.compile)
- Pre-computed UniFrac matrix loading
- Modular CLI package structure

## Planned: Categorical Features (CAT-1 through CAT-7)

Support for categorical metadata features (location, season, site type) that condition target predictions. See [15_categorical_features.md](15_categorical_features.md) for detailed design.

**Components:**
- `CategoricalEncoder`: String → integer index encoding with serialization
- `CategoricalEmbedder`: nn.Module for embedding categorical features
- SequencePredictor integration with fusion strategies (concat/add)
- CLI flags: `--categorical-columns`, `--categorical-embed-dim`, `--categorical-fusion`

**Tickets:** `.agents/CATEGORICAL_FEATURE_TICKETS.md` (7 tickets, ~21-29 hours)

## Outstanding

See `.agents/TICKET_OVERVIEW.md` for all outstanding work.

### Performance Optimizations

**Multi-GPU Training (DDP)** - 8-12 hours
- Distributed data parallel for multi-GPU training
- Synchronization of batch normalization

**FSDP** - 12-16 hours
- Fully sharded data parallel for large models

**Streaming Validation** - 3-4 hours
- Compute metrics incrementally instead of accumulating all predictions
- Reduce validation memory from O(dataset) to O(batch)

### Memory Optimizations

**Configurable FFN Ratio** - 3-4 hours
- Add `--ffn-ratio` flag (default: 4, can reduce to 2)
- Trade memory for capacity

**Lazy Embedding Computation** - 4-6 hours
- Only compute/return sample_embeddings when needed
- Add `return_intermediates` flag

### Model Improvements

**Attention Visualization** - 4-6 hours
- Tools to visualize attention patterns

**Additional Metrics** - 4-6 hours
- Bray-Curtis, Jaccard distance support

### Data Pipeline

**Streaming Data Loading** - 8-12 hours
- Handle datasets larger than memory

**Data Augmentation** - 4-6 hours
- Sequence augmentation strategies

### Experiment Tracking

**W&B / MLflow Integration** - 4-6 hours
- Experiment logging and comparison

**Hyperparameter Optimization** - 6-8 hours
- Optuna integration for automated tuning

### Documentation & Deployment

**Sphinx API Docs** - 4-6 hours
- Auto-generated API documentation

**ONNX Export** - 4-6 hours
- Export models for inference optimization

**Docker Containerization** - 2-4 hours
- Reproducible deployment environment

## Historical Analysis

See `archive/` for completed analysis documents:
- `19_unifrac_underfitting_analysis.md` - UniFrac architecture fixes
- `20_optimization_plan.md` - Performance optimization tickets
- `22_sigmoid_saturation_fix.md` - Distance normalization fix
- `23_memory_optimization_plan.md` - Memory optimization proposals
