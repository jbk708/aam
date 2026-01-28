# ROCm/Cosmos Tickets

**Last Updated:** 2026-01-28
**Status:** 2 tickets remaining (~7-10 hours) | LOW priority

**Current Config (ROCm 6.3 + PyTorch 2.7+):**
```bash
# No special flags needed - defaults work correctly
aam pretrain --data-parallel --batch-size 32 ...
```

**Legacy (ROCm 6.2):** Requires `--attn-implementation math --no-gradient-checkpointing`

---

## Outstanding Tickets

### COS-9.6: SLURM Job Templates
**Priority:** LOW | **Effort:** 3-4 hours | **Status:** Not Started

Create optimized SLURM scripts for Cosmos:
- `slurm/pretrain_single.sh` - Single APU
- `slurm/pretrain_node.sh` - 4 APU with DataParallel
- `slurm/pretrain_multi.sh` - Multi-node DDP

---

### COS-9.7: ROCm Singularity Container
**Priority:** LOW | **Effort:** 4-6 hours | **Status:** Not Started

Reproducible container for production runs.

---

## Summary

| Ticket | Description | Effort | Priority | Status |
|--------|-------------|--------|----------|--------|
| **COS-9.6** | SLURM templates | 3-4h | LOW | Not Started |
| **COS-9.7** | Singularity container | 4-6h | LOW | Not Started |
| **Total** | | **7-10h** | |
