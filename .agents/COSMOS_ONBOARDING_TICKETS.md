# ROCm/Cosmos Tickets

**Last Updated:** 2026-01-20
**Status:** 4 tickets remaining (~13-19 hours)

**Completed:** COS-9.1 to COS-9.3, COS-9.9 (see `ARCHIVED_TICKETS.md`)

**Current Config (ROCm 6.3 + PyTorch 2.7+):**
```bash
# No special flags needed - defaults work correctly
aam pretrain --data-parallel --batch-size 32 ...
```

**Legacy (ROCm 6.2):** Requires `--attn-implementation math --no-gradient-checkpointing`

---

## Outstanding Tickets

### COS-9.5: Kernel Profiling (rocprof)
**Priority:** LOW | **Effort:** 4-6 hours | **Status:** Not Started

Profile training with `rocprof --stats` to identify bottlenecks.

---

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

### COS-9.8: Documentation
**Priority:** LOW | **Effort:** 2-3 hours | **Status:** Not Started

Consolidate ROCm setup and best practices in README.

---

## Summary

| Ticket | Description | Effort | Priority |
|--------|-------------|--------|----------|
| **COS-9.5** | Kernel profiling | 4-6h | LOW |
| **COS-9.6** | SLURM templates | 3-4h | LOW |
| **COS-9.7** | Singularity container | 4-6h | LOW |
| **COS-9.8** | Documentation | 2-3h | LOW |
| **Total** | | **13-19h** | |
