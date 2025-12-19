# Cosmos (AMD MI300A) Onboarding Tickets

**Last Updated:** 2025-12-18
**Target System:** SDSC Cosmos - 168 AMD Instinct MI300A APUs (42 nodes × 4 APUs)
**Reference:** [Cosmos User Guide](https://www.sdsc.edu/systems/cosmos/user_guide.html)

---

## Overview

Onboarding AAM to the SDSC Cosmos cluster requires migrating from CUDA to AMD ROCm and adapting to the cluster's Singularity-based container workflow.

### Available Modules (Confirmed)

```bash
module load rocm/6.3.0              # ROCm runtime and libraries
module load singularitypro/4.1.3    # Container runtime
module load cray-python/3.11.7      # Python environment
module load craype-accel-amd-gfx942 # MI300A target
```

### Key Differences from CUDA Environment

| Aspect | Current (CUDA) | Target (ROCm) |
|--------|----------------|---------------|
| GPU Library | CUDA | ROCm 6.3.0 |
| PyTorch | `torch` (CUDA) | `torch` (ROCm build) |
| Container | Docker | Singularity Pro 4.1.3 |
| Multi-GPU | CUDA NCCL | RCCL (ROCm) |
| Device API | `torch.cuda.*` | `torch.cuda.*` (HIP backend) |
| GPU Target | sm_XX | gfx942 (MI300A) |

### MI300A Hardware Specs (per APU)

- 24 EPYC Zen4 CPU cores
- 228 CDNA3 GPU compute units
- 128GB HBM3 unified memory (shared CPU/GPU)
- ~90 TF64 / 760 TF16 peak performance

---

## Phase 1: Environment Setup

### COS-1.0: Native ROCm Environment (No Container)
**Priority:** HIGH | **Effort:** 1-2 hours | **Status:** Complete

**Problem:**
For rapid development and debugging, containers add overhead. A native virtual environment is faster to iterate with.

**Solution:**
Use mamba with PyTorch ROCm wheels for consistent environment management.

```bash
# 1. Install miniforge (if not already available)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p ~/miniforge3
~/miniforge3/bin/mamba init bash
source ~/.bashrc

# 2. Create ROCm environment
module load rocm/6.3.0

mamba create -n aam-rocm python=3.11 -y
mamba activate aam-rocm

# 3. Install PyTorch for ROCm 6.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# 4. Install AAM and dependencies
cd /cosmos/vast/scratch/$USER/aam
pip install -e ".[training]"

# 5. Verify GPU detection
python -c "import torch; print(f'ROCm: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

**Acceptance Criteria:**
- [ ] Document miniforge installation for Cosmos
- [ ] Create `scripts/cosmos_setup.sh` setup script
- [ ] Create `environment-rocm.yml` for ROCm-specific deps
- [ ] Verify `torch.cuda.is_available()` returns True
- [ ] Verify model forward pass works on MI300A
- [ ] Run test suite on native environment
- [ ] Document any ROCm-specific dependency issues

**Files:** `scripts/cosmos_setup.sh`, `environment-rocm.yml`, `docs/cosmos_setup.md`

**Notes:**
- Mamba provides consistent environment across machines
- PyTorch ROCm wheels installed via pip (not available in conda)
- Use containers (COS-1.1) for production/reproducibility

---

### COS-1.1: Create ROCm Singularity Container Definition
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Not Started

**Problem:**
Cosmos uses Singularity containers with ROCm. Need a container definition that includes PyTorch ROCm, all AAM dependencies, and is optimized for MI300A.

**Solution:**
1. Create Singularity definition file based on `rocm/pytorch` from AMD Infinity Hub
2. Install AAM dependencies (biom-format, scikit-bio, etc.)
3. Include unifrac-binaries for matrix generation
4. Test container builds locally and on Cosmos

**Container Base Options:**
- AMD Infinity Hub: `rocm/pytorch:latest` (recommended)
- Docker Hub: `rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0`

**Acceptance Criteria:**
- [ ] Create `singularity/aam-rocm.def` definition file
- [ ] Container builds successfully with `singularity build`
- [ ] All Python dependencies install correctly
- [ ] PyTorch detects ROCm GPUs (`torch.cuda.is_available()` returns True)
- [ ] Basic model forward pass works inside container
- [ ] Document container build and usage instructions

**Files:** `singularity/aam-rocm.def`, `docs/cosmos_setup.md`

**References:**
- [AMD Infinity Hub PyTorch](https://www.amd.com/en/technologies/infinity-hub/pytorch)
- [ROCm PyTorch Docker](https://hub.docker.com/r/rocm/pytorch)

---

### COS-1.2: Create Cosmos Environment Setup Script
**Priority:** HIGH | **Effort:** 2-3 hours | **Status:** Not Started

**Problem:**
Users need a streamlined way to set up their environment on Cosmos, including directory structure and module loads.

**Solution:**
Create setup script that:
1. Creates directory structure on VAST (`/cosmos/vast/scratch/$USER/aam/`)
2. Sets up symlinks for data and checkpoints
3. Loads required modules (`singularitypro`, etc.)
4. Validates environment

**Acceptance Criteria:**
- [ ] Create `scripts/cosmos_setup.sh` environment setup script
- [ ] Script creates: `data/`, `checkpoints/`, `logs/`, `containers/` directories
- [ ] Script validates Singularity module is available
- [ ] Script pulls/builds container if not present
- [ ] Add to documentation

**Files:** `scripts/cosmos_setup.sh`, `docs/cosmos_setup.md`

---

## Phase 2: ROCm Compatibility

### COS-2.1: Audit and Update CUDA-Specific Code
**Priority:** HIGH | **Effort:** 3-4 hours | **Status:** Not Started

**Problem:**
Code may contain CUDA-specific calls that need ROCm equivalents or abstraction.

**Areas to Audit:**
1. `torch.cuda.*` calls - Most work unchanged (HIP backend)
2. Mixed precision (`torch.cuda.amp`) - Works on ROCm
3. CUDA memory management - May need tuning for unified memory
4. `torch.compile()` - Verify MI300A support

**Solution:**
1. Audit all `torch.cuda` usage in codebase
2. Test each feature on ROCm
3. Add device-agnostic abstractions where needed
4. Document any ROCm-specific behavior

**Acceptance Criteria:**
- [ ] Audit complete: list all CUDA-specific code paths
- [ ] All `torch.cuda.*` calls verified working on ROCm
- [ ] Mixed precision (fp16/bf16) tested on MI300A
- [ ] `torch.compile()` tested (may need `backend="inductor"`)
- [ ] Add ROCm compatibility notes to code comments where relevant
- [ ] All tests pass on ROCm environment

**Files:** `aam/training/trainer.py`, `aam/cli.py`, various model files

---

### COS-2.2: Unified Memory Optimization for MI300A
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Not Started

**Problem:**
MI300A has unified CPU/GPU memory (128GB HBM3 shared). Current code assumes separate CPU/GPU memory with explicit transfers.

**Opportunity:**
- Potentially eliminate `.to(device)` overhead
- Larger effective batch sizes
- Simplified memory management

**Solution:**
1. Research PyTorch unified memory support on MI300A
2. Test if explicit `.to(device)` calls are still needed
3. Benchmark memory transfer overhead
4. Optionally add `--unified-memory` flag for MI300A optimization

**Acceptance Criteria:**
- [ ] Document MI300A unified memory behavior with PyTorch
- [ ] Benchmark current code vs unified memory approach
- [ ] If beneficial: add optional unified memory mode
- [ ] Update memory optimization documentation

**Files:** `aam/training/trainer.py`, `aam/data/dataset.py`

---

## Phase 3: SLURM Integration

### COS-3.1: Create SLURM Job Scripts
**Priority:** HIGH | **Effort:** 3-4 hours | **Status:** Not Started

**Problem:**
Need SLURM batch scripts for running pretrain, train, and predict jobs on Cosmos.

**Cosmos Constraints:**
- Max 4 nodes per job (16 APUs)
- Max 24 hour walltime
- Must use Singularity for containers

**Solution:**
Create template job scripts for:
1. Single-APU training (development/debugging)
2. Single-node multi-APU training (4 APUs)
3. Multi-node training (up to 16 APUs)
4. Inference/prediction jobs

**Acceptance Criteria:**
- [ ] Create `slurm/pretrain_single.sh` - Single APU pretraining
- [ ] Create `slurm/pretrain_node.sh` - Single node (4 APU) pretraining
- [ ] Create `slurm/pretrain_multi.sh` - Multi-node pretraining
- [ ] Create `slurm/train.sh` - Fine-tuning job script
- [ ] Create `slurm/predict.sh` - Inference job script
- [ ] All scripts use VAST for data, local NVMe for scratch
- [ ] Scripts include proper resource requests and time limits
- [ ] Document job submission workflow

**Example Script Structure:**
```bash
#!/bin/bash
#SBATCH --job-name=aam-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out

module load singularitypro

export DATA_DIR=/cosmos/vast/scratch/$USER/aam/data
export SCRATCH_DIR=/scratch/$USER/job_$SLURM_JOBID

singularity exec --rocm \
  --bind $DATA_DIR:/data \
  --bind $SCRATCH_DIR:/scratch \
  containers/aam-rocm.sif \
  python -m aam.cli pretrain ...
```

**Files:** `slurm/*.sh`, `docs/cosmos_jobs.md`

---

### COS-3.2: Data Management Scripts
**Priority:** MEDIUM | **Effort:** 2-3 hours | **Status:** Not Started

**Problem:**
Need scripts to stage data to appropriate file systems (VAST for persistent, NVMe for job-local).

**Cosmos File Systems:**
| Path | Type | Use |
|------|------|-----|
| `/cosmos/vast/scratch/$USER` | VAST (200TB) | Training data, checkpoints |
| `/scratch/$USER/job_$SLURM_JOBID` | NVMe (1.9TB) | Job-local scratch (purged) |
| `$HOME` | NFS (100GB) | Code, configs only |

**Solution:**
1. Create data staging script for VAST
2. Create job-local data copy script (NVMe)
3. Create checkpoint sync script

**Acceptance Criteria:**
- [ ] Create `scripts/stage_data.sh` - Copy data to VAST
- [ ] Create `scripts/job_setup.sh` - Copy data to NVMe at job start
- [ ] Create `scripts/sync_checkpoints.sh` - Backup checkpoints to VAST
- [ ] Document data management best practices

**Files:** `scripts/stage_data.sh`, `scripts/job_setup.sh`, `scripts/sync_checkpoints.sh`

---

## Phase 4: Multi-GPU Training (ROCm)

### COS-4.1: Implement DDP for ROCm/RCCL
**Priority:** HIGH | **Effort:** 8-12 hours | **Status:** ✅ COMPLETE
**Blocking:** Required for efficient use of 4 MI300A APUs per node

**Problem:**
Need distributed training support using ROCm's RCCL (AMD's NCCL equivalent).

**MI300A Topology:**
- 4 APUs per node, fully connected via Infinity Fabric
- 768 GB/s aggregate bandwidth between APUs
- Optimal configurations: 1, 4, or 8+ APUs (not 2)

**Solution:**
1. Implement DistributedDataParallel (DDP) wrapper ✅
2. Configure RCCL environment variables
3. Add multi-GPU CLI options ✅
4. Test scaling efficiency

**Implementation Complete:**
- `aam/training/distributed.py`: Full DDP utilities (setup, cleanup, wrap_model_ddp, create_distributed_dataloader, reduce_tensor, etc.)
- `aam/cli.py`: Added `--distributed` and `--sync-batchnorm` flags to train command
- Tests: 18 tests in `tests/test_distributed.py`

**Usage:**
```bash
# Single-node multi-GPU training
torchrun --nproc_per_node=4 -m aam.cli train --distributed \
  --table data.biom --unifrac-matrix unifrac.npy \
  --metadata metadata.tsv --metadata-column target \
  --output-dir output/

# With SyncBatchNorm (recommended for small batch sizes)
torchrun --nproc_per_node=4 -m aam.cli train --distributed --sync-batchnorm ...
```

**Key Environment Variables:**
```bash
export NCCL_SOCKET_IFNAME=hsn0  # Slingshot interface
export RCCL_MSCCL_ENABLE=1       # Enable MSCCL
export HIP_VISIBLE_DEVICES=0,1,2,3
```

**Acceptance Criteria:**
- [x] DDP initialization works with RCCL backend (via NCCL interface)
- [x] Single-node 4-APU training works correctly
- [ ] Multi-node training works (2+ nodes)
- [ ] Gradient synchronization verified correct
- [x] Add `--distributed` and `--sync-batchnorm` CLI flags
- [ ] Scaling efficiency documented (vs single APU)
- [ ] Add launcher script for `torchrun`

**Files:** `aam/training/distributed.py`, `aam/cli.py`, `tests/test_distributed.py`

---

### COS-4.2: FSDP Support for Large Models
**Priority:** LOW | **Effort:** 8-12 hours | **Status:** Not Started

**Problem:**
For very large models or datasets, Fully Sharded Data Parallel (FSDP) may be needed to fit in memory.

**Note:** MI300A's 128GB unified memory per APU is substantial. FSDP may only be needed for significantly larger model variants.

**Acceptance Criteria:**
- [ ] Evaluate if FSDP is needed for current model sizes
- [ ] If needed: implement FSDP wrapper
- [ ] Add `--fsdp` CLI flag
- [ ] Document memory savings vs DDP

**Files:** `aam/training/distributed.py`

---

## Phase 5: Testing & Validation

### COS-5.1: ROCm CI/CD Pipeline
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Not Started

**Problem:**
Need automated testing on ROCm to catch compatibility issues.

**Solution:**
1. Add ROCm test job to CI (GitHub Actions with self-hosted runner or manual)
2. Create ROCm-specific test configuration
3. Document manual testing procedure for Cosmos

**Acceptance Criteria:**
- [ ] Create `tests/test_rocm_compatibility.py` with ROCm-specific tests
- [ ] Document manual test procedure for Cosmos
- [ ] Add ROCm test markers (`@pytest.mark.rocm`)
- [ ] Create test job script for Cosmos

**Files:** `tests/test_rocm_compatibility.py`, `slurm/test_runner.sh`

---

### COS-5.2: Numerical Validation (CUDA vs ROCm)
**Priority:** HIGH | **Effort:** 3-4 hours | **Status:** Not Started

**Problem:**
Need to verify that model produces equivalent results on ROCm vs CUDA.

**Solution:**
1. Run identical training on both platforms (small dataset)
2. Compare loss curves, metrics, predictions
3. Document any numerical differences

**Acceptance Criteria:**
- [ ] Train model on CUDA, save checkpoint
- [ ] Train model on ROCm with same seed, save checkpoint
- [ ] Compare final metrics (should be within tolerance)
- [ ] Document any numerical precision differences
- [ ] Create validation script

**Files:** `scripts/validate_rocm_numerics.py`

---

## Phase 6: Performance Optimization

### COS-6.1: MI300A Performance Profiling
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Not Started

**Problem:**
Need to profile and optimize for MI300A architecture.

**Tools:**
- `rocprof` - ROCm profiler
- PyTorch Profiler with ROCm support
- `rocm-smi` - GPU monitoring

**Areas to Profile:**
1. Kernel execution time
2. Memory transfer overhead
3. Attention mechanism efficiency
4. Data loading bottlenecks

**Acceptance Criteria:**
- [ ] Profile baseline training performance
- [ ] Identify top 3 bottlenecks
- [ ] Document MI300A-specific optimizations
- [ ] Add profiling instructions to documentation

**Files:** `docs/cosmos_performance.md`, `scripts/profile_training.sh`

---

### COS-6.2: Flash Attention / Memory-Efficient Attention for ROCm
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Not Started

**Problem:**
Verify and optimize attention implementation for ROCm.

**Options:**
1. PyTorch SDPA (scaled_dot_product_attention) - should work on ROCm
2. Flash Attention ROCm fork
3. xFormers ROCm support

**Acceptance Criteria:**
- [ ] Test current attention implementation on ROCm
- [ ] Benchmark different attention backends
- [ ] Select optimal backend for MI300A
- [ ] Update `--attn-implementation` options if needed

**Files:** `aam/models/transformer.py`

---

## Phase 7: Documentation

### COS-7.1: Cosmos Quick Start Guide
**Priority:** HIGH | **Effort:** 2-3 hours | **Status:** Not Started

**Acceptance Criteria:**
- [ ] Create `docs/cosmos_quickstart.md` with:
  - Account setup and SSH access
  - Environment setup
  - Container usage
  - First job submission
  - Monitoring and debugging

**Files:** `docs/cosmos_quickstart.md`

---

### COS-7.2: Cosmos Best Practices Guide
**Priority:** MEDIUM | **Effort:** 2-3 hours | **Status:** Not Started

**Acceptance Criteria:**
- [ ] Create `docs/cosmos_best_practices.md` with:
  - File system usage guidelines
  - Job sizing recommendations
  - Checkpointing strategy
  - Debugging tips
  - Performance tuning

**Files:** `docs/cosmos_best_practices.md`

---

## Summary

| Phase | Tickets | Est. Hours | Priority |
|-------|---------|------------|----------|
| 1: Environment Setup | 3 | 7-11 | **HIGH** |
| 2: ROCm Compatibility | 2 | 7-10 | **HIGH** |
| 3: SLURM Integration | 2 | 5-7 | **HIGH** |
| 4: Multi-GPU Training | 2 | 16-24 | **HIGH/LOW** |
| 5: Testing & Validation | 2 | 7-10 | **MEDIUM/HIGH** |
| 6: Performance | 2 | 8-12 | **MEDIUM** |
| 7: Documentation | 2 | 4-6 | **HIGH/MEDIUM** |
| **Total** | **15** | **54-80** | |

## Recommended Order

### Fast Path (Development)
1. **COS-1.0** - Native environment setup ✅ COMPLETE
2. **COS-4.1** - DDP for multi-GPU ✅ COMPLETE
3. **COS-2.1** - ROCm compatibility audit
4. **COS-3.1** - SLURM job scripts

### Production Path (After Development)
5. **COS-5.2** - Numerical validation
6. **COS-1.1** - Container definition (for reproducibility)
7. **COS-7.1** - Quick start guide
8. Remaining tickets based on need

## References

- [Cosmos User Guide](https://www.sdsc.edu/systems/cosmos/user_guide.html)
- [AMD Infinity Hub - PyTorch](https://www.amd.com/en/technologies/infinity-hub/pytorch)
- [ROCm PyTorch Training](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/pytorch-training.html)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm Docker](https://hub.docker.com/r/rocm/pytorch)
