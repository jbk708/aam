# Performance Optimization Plan

**Status:** 📋 Planning  
**Priority:** MEDIUM  
**Created:** 2025  
**Related Tickets:** Future Phase 10 tickets

## Executive Summary

This document outlines a comprehensive performance optimization plan for the AAM PyTorch implementation. The plan focuses on training speed, memory efficiency, and scalability improvements to enable larger-scale training and faster experimentation.

## Current Performance Baseline

### Training Configuration
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)
- **Scheduler**: WarmupCosineScheduler (10k warmup steps)
- **Batch Size**: 8 (must be even for UniFrac)
- **Gradient Accumulation**: Supported (default: 1)
- **Memory Optimizations**: 
  - ✅ Gradient accumulation
  - ✅ Chunked ASV processing
  - ✅ CUDA expandable segments
  - ✅ Gradient clipping

### Known Bottlenecks
1. **Memory**: Large batches require significant GPU memory
2. **Speed**: Training can be slow for large datasets
3. **Scalability**: Single GPU training limits throughput

## Optimization Goals

### Primary Goals
1. **Training Speed**: 2-3x faster training without accuracy loss
2. **Memory Efficiency**: Support 2x larger batch sizes on same hardware
3. **Scalability**: Enable multi-GPU training for large-scale experiments

### Secondary Goals
1. **Model Compilation**: Leverage PyTorch 2.0+ optimizations
2. **Mixed Precision**: Reduce memory and increase speed
3. **Data Loading**: Optimize data pipeline throughput

## Optimization Phases

### Phase 1: Low-Hanging Fruit (Quick Wins)
**Priority:** HIGH | **Effort:** Low-Medium | **Impact:** Medium-High

#### 1.1 Mixed Precision Training (FP16/BF16)
**Priority:** HIGH | **Effort:** Low (2-3 hours) | **Impact:** High

**Benefits:**
- ~2x memory reduction
- ~1.5-2x training speedup (on modern GPUs)
- Minimal code changes

**Implementation:**
- Use `torch.cuda.amp.autocast()` for forward pass
- Use `GradScaler` for gradient scaling
- Add `--mixed-precision` CLI flag (choices: fp16, bf16, none)

**Files to Modify:**
- `aam/training/trainer.py` - Add autocast context managers
- `aam/cli.py` - Add mixed precision option

**Testing:**
- Verify numerical stability (no NaN/Inf)
- Compare training metrics with/without mixed precision
- Test gradient scaling behavior

**Dependencies:** None

#### 1.2 Model Compilation (`torch.compile()`)
**Priority:** MEDIUM | **Effort:** Low (1-2 hours) | **Impact:** Medium

**Benefits:**
- 10-30% speedup on PyTorch 2.0+
- Automatic kernel fusion and optimization
- Minimal code changes

**Implementation:**
- Wrap model with `torch.compile()` after initialization
- Add `--compile-model` CLI flag
- Support both eager and compiled modes

**Files to Modify:**
- `aam/training/trainer.py` - Compile model if flag set
- `aam/cli.py` - Add compile option

**Testing:**
- Verify compiled model produces same outputs
- Benchmark speedup on different hardware
- Test with different backends (inductor, nvfuser)

**Dependencies:** PyTorch 2.0+

#### 1.3 Optimize Data Loading
**Priority:** MEDIUM | **Effort:** Medium (3-4 hours) | **Impact:** Medium

**Current Issues:**
- `num_workers=0` by default (single-threaded)
- No prefetching
- Potential I/O bottlenecks

**Optimizations:**
- Increase `num_workers` (default: 4-8)
- Add `prefetch_factor` for DataLoader
- Pin memory for faster GPU transfer
- Profile data loading time

**Files to Modify:**
- `aam/cli.py` - Update default `num_workers`
- `aam/data/dataset.py` - Optimize data access patterns

**Testing:**
- Benchmark data loading throughput
- Verify no data corruption with multiple workers
- Test on different storage systems

**Dependencies:** None

### Phase 2: Memory Optimizations
**Priority:** MEDIUM | **Effort:** Medium | **Impact:** High

#### 2.1 Gradient Checkpointing
**Priority:** MEDIUM | **Effort:** Medium (3-4 hours) | **Impact:** High

**Benefits:**
- 30-50% memory reduction
- Enable larger models/batches
- Trade compute for memory

**Implementation:**
- Use `torch.utils.checkpoint.checkpoint()` for transformer layers
- Add `--gradient-checkpointing` flag
- Apply to ASVEncoder and transformer layers

**Files to Modify:**
- `aam/models/transformer.py` - Add checkpointing
- `aam/models/asv_encoder.py` - Add checkpointing option
- `aam/cli.py` - Add flag

**Testing:**
- Verify memory reduction
- Compare training speed (should be slower)
- Test gradient correctness

**Dependencies:** None

#### 2.2 Optimize Attention Computation
**Priority:** MEDIUM | **Effort:** Medium-High (4-6 hours) | **Impact:** Medium

**Current Issues:**
- Standard attention may be inefficient for large sequences
- No flash attention support

**Optimizations:**
- Implement Flash Attention 2 (if available)
- Optimize attention mask handling
- Consider sparse attention for very long sequences

**Files to Modify:**
- `aam/models/transformer.py` - Optimize attention
- Consider using `torch.nn.functional.scaled_dot_product_attention`

**Testing:**
- Benchmark attention computation time
- Verify numerical equivalence
- Test on different sequence lengths

**Dependencies:** PyTorch 2.0+ (for scaled_dot_product_attention)

### Phase 3: Distributed Training
**Priority:** LOW | **Effort:** High | **Impact:** Very High

#### 3.1 Multi-GPU Training (DDP)
**Priority:** LOW | **Effort:** High (8-12 hours) | **Impact:** Very High

**Benefits:**
- Linear scaling with number of GPUs
- Enable training on very large datasets
- Faster experimentation

**Implementation:**
- Use `torch.nn.parallel.DistributedDataParallel`
- Add distributed training setup
- Handle data splitting across GPUs
- Sync metrics across processes

**Files to Modify:**
- `aam/training/trainer.py` - Add DDP support
- `aam/cli.py` - Add distributed training options
- Create distributed training script

**Testing:**
- Test on 2+ GPUs
- Verify same results as single GPU
- Benchmark scaling efficiency

**Dependencies:** Multi-GPU hardware

#### 3.2 FSDP (Fully Sharded Data Parallel)
**Priority:** LOW | **Effort:** Very High (12-16 hours) | **Impact:** Very High

**Benefits:**
- Memory-efficient distributed training
- Enable very large models
- Better scaling than DDP for large models

**Implementation:**
- Use `torch.distributed.fsdp.FullyShardedDataParallel`
- Configure sharding strategy
- Handle optimizer state sharding

**Files to Modify:**
- `aam/training/trainer.py` - Add FSDP support
- `aam/cli.py` - Add FSDP options

**Testing:**
- Test on multiple GPUs
- Verify memory efficiency
- Benchmark performance

**Dependencies:** Multi-GPU hardware, PyTorch 2.0+

### Phase 4: Advanced Optimizations
**Priority:** LOW | **Effort:** Medium-High | **Impact:** Medium

#### 4.1 Batch Size Optimization Strategies
**Priority:** LOW | **Effort:** Medium (4-6 hours) | **Impact:** Medium

**Strategies:**
- Dynamic batch sizing based on memory
- Gradient accumulation optimization
- Automatic batch size finder

**Implementation:**
- Add batch size finder utility
- Implement dynamic batch sizing
- Optimize gradient accumulation

**Files to Modify:**
- `aam/training/trainer.py` - Add batch size utilities
- `aam/cli.py` - Add batch size options

**Dependencies:** None

#### 4.2 Caching Mechanisms
**Priority:** LOW | **Effort:** Medium (3-4 hours) | **Impact:** Low-Medium

**Targets:**
- UniFrac distance computation (expensive)
- Tokenized sequences (if static)
- Rarefied tables (if not regenerating)

**Implementation:**
- Add caching layer for UniFrac distances
- Cache tokenized sequences
- Configurable cache size and eviction

**Files to Modify:**
- `aam/data/unifrac.py` - Add caching
- `aam/data/dataset.py` - Add sequence caching

**Dependencies:** None

## Implementation Priority

### Immediate (Next Sprint)
1. **Mixed Precision Training** (Phase 1.1) - High impact, low effort
2. **Model Compilation** (Phase 1.2) - Medium impact, low effort
3. **Data Loading Optimization** (Phase 1.3) - Medium impact, medium effort

### Short-Term (Next 2-3 Sprints)
4. **Gradient Checkpointing** (Phase 2.1) - High impact, medium effort
5. **Attention Optimization** (Phase 2.2) - Medium impact, medium-high effort

### Medium-Term (Future Sprints)
6. **Multi-GPU Training** (Phase 3.1) - Very high impact, high effort
7. **Batch Size Optimization** (Phase 4.1) - Medium impact, medium effort

### Long-Term (Research/Exploration)
8. **FSDP** (Phase 3.2) - Very high impact, very high effort
9. **Caching Mechanisms** (Phase 4.2) - Low-medium impact, medium effort

## Success Metrics

### Performance Targets
- **Training Speed**: 2-3x faster with mixed precision + compilation
- **Memory Usage**: 30-50% reduction with gradient checkpointing
- **Scalability**: Linear scaling with number of GPUs (DDP)
- **Batch Size**: Support 2x larger batches on same hardware

### Measurement
- Benchmark training time per epoch
- Measure peak GPU memory usage
- Track throughput (samples/second)
- Compare accuracy before/after optimizations

## Risk Assessment

### Low Risk
- Mixed precision training: Well-established, minimal risk
- Model compilation: PyTorch built-in, reversible
- Data loading optimization: Low risk, easy to revert

### Medium Risk
- Gradient checkpointing: May affect training dynamics slightly
- Attention optimization: Need to verify numerical equivalence

### High Risk
- Distributed training: Complex setup, potential synchronization issues
- FSDP: Very complex, requires careful testing

## Dependencies

### External Dependencies
- PyTorch 2.0+ for `torch.compile()` and `scaled_dot_product_attention`
- Multi-GPU hardware for distributed training
- Modern GPUs (Ampere+) for best mixed precision performance

### Internal Dependencies
- Stable training pipeline (already achieved)
- Comprehensive test suite (already achieved)
- Good understanding of memory usage patterns

## Testing Strategy

### Unit Tests
- Test mixed precision training produces same results (within tolerance)
- Verify model compilation maintains correctness
- Test gradient checkpointing produces correct gradients

### Integration Tests
- End-to-end training with optimizations enabled
- Compare metrics with/without optimizations
- Test distributed training correctness

### Benchmarking
- Create benchmark suite for performance measurement
- Track performance improvements over time
- Compare against baseline

## Documentation

### Required Documentation
- Optimization guide (how to use each optimization)
- Performance benchmarks (before/after)
- Hardware requirements for each optimization
- Troubleshooting guide

## Next Steps

1. **Create tickets** for Phase 1 optimizations (mixed precision, compilation, data loading)
2. **Benchmark baseline** performance on target hardware
3. **Implement Phase 1** optimizations
4. **Measure improvements** and document results
5. **Plan Phase 2** based on Phase 1 results

## Notes

- Focus on optimizations that provide the most value for users
- Prioritize optimizations that don't require hardware changes
- Maintain backward compatibility when possible
- Document performance improvements for users
- Consider trade-offs (speed vs. memory, accuracy vs. speed)

---

**Document Status**: Ready for implementation  
**Next Action**: Create Phase 1 tickets and begin benchmarking
