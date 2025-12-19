# Ticket Overview

**Status:** 32 outstanding tickets (~138-197 hours)

## Quick Links
- **PyTorch work:** `PYTORCH_PORTING_TICKETS.md`
- **Cosmos onboarding:** `COSMOS_ONBOARDING_TICKETS.md`
- **Completed work:** `ARCHIVED_TICKETS.md`
- **Workflow:** `WORKFLOW.md`

## Outstanding by Priority

### HIGH (7) - Cosmos Onboarding
- COS-1.1: Create ROCm Singularity Container Definition
- COS-1.2: Create Cosmos Environment Setup Script
- COS-2.1: Audit and Update CUDA-Specific Code
- COS-3.1: Create SLURM Job Scripts
- COS-4.1: Implement DDP for ROCm/RCCL
- COS-5.2: Numerical Validation (CUDA vs ROCm)
- COS-7.1: Cosmos Quick Start Guide

### MEDIUM (7)
- PYT-10.6: Multi-GPU Training (DDP) - superseded by COS-4.1 for Cosmos
- PYT-21.3: Regressor Head Optimization
- COS-2.2: Unified Memory Optimization for MI300A
- COS-3.2: Data Management Scripts
- COS-5.1: ROCm CI/CD Pipeline
- COS-6.1: MI300A Performance Profiling
- COS-7.2: Cosmos Best Practices Guide

### LOW (18)
- COS-4.2: FSDP Support for Large Models
- COS-6.2: Flash Attention for ROCm
- PYT-18.5: Lazy Sample Embedding Computation
- PYT-18.6: Memory-Aware Dynamic Batching
- Phase 12: FSDP, Batch Size Optimization
- Phase 13: Attention Visualization, Feature Importance, Encoder Types
- Phase 14: Streaming Data, Augmentation
- Phase 15: Experiment Tracking, Hyperparameter Optimization
- Phase 16: Benchmarking, Error Analysis
- Phase 17: Docs, Tutorials, ONNX, Docker

## Recently Completed/Cancelled
- PYT-21.4: Update Training Progress Bar for Fine-Tuning (hide NL/NA, show RL/CL)
- PYT-18.4: Configurable FFN Intermediate Size (cancelled - memory not an issue)
- PYT-18.3: Skip Nucleotide Predictions During Inference (already implemented)
- PYT-12.3: Caching Mechanisms (sequence tokenization cache)
- PYT-18.2: Streaming Validation Metrics (O(batch) memory)
- PYT-19.1: Fix Failing Unit Tests (test fixes)
- PYT-20.1: Masked Autoencoder (MAE for nucleotide prediction)
- PYT-18.1: Memory-Efficient Defaults

## Recommended Next Steps

### Cosmos Onboarding (Priority)
1. **COS-1.1** - Container definition (blocks everything)
2. **COS-1.2** - Environment setup script
3. **COS-2.1** - ROCm compatibility audit
4. **COS-3.1** - SLURM job scripts
5. **COS-5.2** - Numerical validation

### PyTorch Improvements (Secondary)
1. **PYT-21.3** - Regressor head optimization
