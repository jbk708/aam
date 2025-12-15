# Ticket Overview - Outstanding Work

**Last Updated:** 2025-01-XX  
**Status:** Active tracking of all outstanding tickets

## Quick Summary

- **Total Outstanding Tickets:** 18 tickets
- **Total Estimated Time:** ~115-145 hours
- **Completed Phases:** Phase 8, Phase 9, Phase 11 (all complete), Phase 10 (3/7 complete)

## Outstanding Tickets by Priority

### HIGH Priority (None currently)
- ✅ **PYT-11.4**: Refactor CLI/Model to Ingest Pre-Generated UniFrac Matrices - **COMPLETED**

### MEDIUM Priority (4 tickets)

**Phase 10: Performance Optimizations**
1. **PYT-10.3**: Optimize Data Loading (3-4 hours)
   - Increase default `num_workers`, add prefetching, pin memory
   - Medium impact, medium effort

2. **PYT-10.4**: Implement Gradient Checkpointing (3-4 hours)
   - 30-50% memory reduction, enable larger batches
   - High impact, medium effort

3. **PYT-10.5**: Optimize Attention Computation (4-6 hours)
   - Use `scaled_dot_product_attention` for better performance
   - Medium impact, medium-high effort

4. **PYT-10.6**: Implement Multi-GPU Training (DDP) (8-12 hours)
   - Linear scaling with number of GPUs
   - Very high impact, high effort (requires hardware)

### LOW Priority (15 tickets)

**Phase 12: Additional Performance Optimizations (3 tickets)**
- PYT-12.1: Implement FSDP (12-16 hours)
- PYT-12.2: Implement Batch Size Optimization Strategies (4-6 hours)
- PYT-12.3: Implement Caching Mechanisms (3-4 hours)

**Phase 13: Model Improvements (3 tickets)**
- PYT-13.1: Add Attention Visualization Tools (4-6 hours)
- PYT-13.2: Implement Feature Importance Analysis (4-6 hours)
- PYT-13.3: Support Additional Encoder Types (4-6 hours)

**Phase 14: Data Pipeline Enhancements (2 tickets)**
- PYT-14.1: Support Streaming Data Loading (6-8 hours)
- PYT-14.2: Implement Data Augmentation Strategies (4-6 hours)

**Phase 15: Training Improvements (2 tickets)**
- PYT-15.1: Integrate Experiment Tracking (4-6 hours)
- PYT-15.2: Add Hyperparameter Optimization Support (6-8 hours)

**Phase 16: Evaluation and Analysis Tools (2 tickets)**
- PYT-16.1: Create Benchmarking Suite (4-6 hours)
- PYT-16.2: Implement Error Analysis Tools (4-6 hours)

**Phase 17: Documentation and Deployment (4 tickets)**
- PYT-17.1: Generate API Documentation (Sphinx) (4-6 hours)
- PYT-17.2: Create Tutorial Notebooks (4-6 hours)
- PYT-17.3: Add ONNX Export Support (3-4 hours)
- PYT-17.4: Create Docker Containerization (2-3 hours)

## Recently Completed

### Phase 11: Critical Fixes
- ✅ **PYT-11.1**: Fix UniFrac Distance Predictions Exceeding 1.0
  - Implemented sigmoid normalization in `compute_pairwise_distances()`
  - All distances now bounded to [0, 1] range
  - Comprehensive test coverage added

- ✅ **PYT-11.4**: Refactor CLI/Model to Ingest Pre-Generated UniFrac Matrices
  - Created `UniFracLoader` class for loading pre-computed matrices
  - Updated CLI to use `--unifrac-matrix` instead of `--tree`
  - Removed all computation-related flags and logic
  - Deprecated computation methods with warnings
  - Comprehensive test suite (21 tests, all passing)
  - All existing tests updated to use pre-computed matrices

### Phase 10: Performance Optimizations
- ✅ **PYT-10.1**: Implement Mixed Precision Training (FP16/BF16)
- ✅ **PYT-10.2**: Implement Model Compilation (`torch.compile()`)
- ✅ **PYT-10.2.1**: Fix Dependencies to Enable Model Compilation

## Recommended Next Steps

### Immediate (Next Sprint)
1. **PYT-10.3**: Optimize Data Loading - Quick win for training speed
2. **PYT-10.4**: Implement Gradient Checkpointing - High impact for memory

### Short-Term (Next 2-3 Sprints)
3. **PYT-10.5**: Optimize Attention Computation
4. **PYT-12.3**: Implement Caching Mechanisms - Speed up repeated computations

### Medium-Term (Future Sprints)
5. **PYT-10.6**: Implement Multi-GPU Training (if hardware available)
6. **PYT-13.1**: Add Attention Visualization Tools - Improve interpretability
7. **PYT-15.1**: Integrate Experiment Tracking - Better experiment management

### Long-Term (Research/Exploration)
8. **PYT-12.1**: Implement FSDP - For very large models
9. **PYT-17.1-17.4**: Documentation and Deployment - Production readiness

## Notes

- All Phase 8 and Phase 9 tickets are complete
- Phase 10 focuses on performance optimizations (see `_design_plan/20_optimization_plan.md`)
- Phase 11 critical fixes are complete
- Phases 12-17 are enhancements from `_design_plan/FUTURE_WORK.md`
- See `.agents/PYTORCH_PORTING_TICKETS.md` for detailed ticket descriptions
