# UniFrac Distance Computer

## Objective
Implement computation of phylogenetic distances (UniFrac) from BIOM table and reference phylogenetic tree.

## Requirements

### Input
- Rarefied BIOM table (from `01_biom_loader.md`)
- Reference phylogenetic tree (`.nwk` Newick format)
- ASV IDs must match between table and tree

### UniFrac Metrics

**1. Unweighted UniFrac**
- Computes pairwise distances between samples
- Based on presence/absence of ASVs in phylogenetic context
- Output: `skbio.DistanceMatrix` object (not numpy array)
- Shape: `[N_samples, N_samples]` distance matrix
- Uses `unifrac-binaries` library (https://github.com/biocore/unifrac-binaries)
- API: `unifracbinaries.unweighted()` or similar function

**2. Faith's Phylogenetic Diversity (Faith PD)**
- Computes total branch length for each sample
- Single value per sample
- Output: `skbio.DistanceMatrix` or pandas Series
- Shape: `[N_samples, 1]` vector
- Uses `unifrac-binaries` library
- API: `unifracbinaries.faith_pd()` or similar function

### Implementation Requirements

**Class Structure**:
- Create `UniFracComputer` class
- Methods:
  - `compute_unweighted(table, tree_path)`: Compute unweighted UniFrac → returns `DistanceMatrix`
  - `compute_faith_pd(table, tree_path)`: Compute Faith PD → returns `DistanceMatrix` or Series
  - `extract_batch_distances(distances, sample_ids, metric)`: Extract distances for batch samples

**Key Considerations**:
- ASV IDs must match between table and tree
- Use `unifrac-binaries` library (install via: `pip install unifrac-binaries` or `conda install -c biocore unifrac-binaries`)
- UniFrac computation may require temporary BIOM file (check library API requirements)
- Store as `DistanceMatrix` object (not numpy array) for filtering capabilities
- Handle epoch regeneration: recompute if rarefaction changes
- Check library documentation for exact API: https://github.com/biocore/unifrac-binaries/tree/main

### Batch Processing

**For Training Batches**:
- Pre-compute full distance matrix for all samples (stored as `DistanceMatrix`)
- For each batch, extract relevant rows/columns using `DistanceMatrix.filter(sample_ids)`
- Unweighted UniFrac: Extract `[batch_size, batch_size]` submatrix via `.filter().data`
- Faith PD: Extract `[batch_size, 1]` vector via `.loc[sample_ids].to_numpy().reshape((-1, 1))`

**Implementation**:
- Compute full distance matrix once per rarefied table
- Store as `skbio.DistanceMatrix` object (enables filtering by sample IDs)
- Extract batch-level distances during data loading using `.filter()` method
- If `gen_new_tables=True`: Recompute UniFrac each epoch when new rarefied table is created

### Epoch Regeneration

**Behavior**:
- If `gen_new_tables=True`: Create new rarefied table each epoch → recompute UniFrac
- If `gen_new_tables=False`: Reuse same rarefied table → reuse same UniFrac distances
- UniFrac computation happens in `_create_encoder_target()` method
- Called during initialization and during epoch regeneration

**Implementation**:
- Store `encoder_target` (DistanceMatrix) as instance variable
- Regenerate when new rarefied table is created
- Cache if rarefaction doesn't change

### Batch Size Constraint

**Important**: Batch size must be even (multiple of 2)
- Check: `if batch_size % 2 != 0: raise Exception("Batch size must be multiple of 2")`
- This constraint exists in the base implementation
- Reason: Likely related to pairwise distance computation or loss function

## Implementation Checklist

- [ ] Install `unifrac-binaries` library (`pip install unifrac-binaries` or `conda install -c biocore unifrac-binaries`)
- [ ] Review library API documentation: https://github.com/biocore/unifrac-binaries/tree/main
- [ ] Create `UniFracComputer` class
- [ ] Implement `compute_unweighted()` using `unifrac-binaries` → returns `DistanceMatrix`
- [ ] Implement `compute_faith_pd()` using `unifrac-binaries` → returns `DistanceMatrix` or Series
- [ ] Handle temporary file creation/deletion if required by library (use `tempfile` module)
- [ ] Implement batch-level distance extraction using `.filter()` method
- [ ] Handle ASV ID matching
- [ ] Store as `DistanceMatrix` object (not numpy array)
- [ ] Implement epoch regeneration logic (recompute if rarefaction changes)
- [ ] Add batch size validation (must be even)
- [ ] Test with sample BIOM table and tree
- [ ] Verify distance matrix properties (symmetric, non-negative)
- [ ] Test batch extraction with different batch sizes
- [ ] Handle edge cases (single sample, no shared ASVs, etc.)

## Key Considerations

### DistanceMatrix vs Numpy Array
- **Store as DistanceMatrix**: Enables filtering by sample IDs
- **Extract as numpy**: Use `.data` property or `.to_numpy()` for batch extraction
- **Filtering**: Use `distance_matrix.filter(sample_ids)` to get submatrix for batch

### Library Usage
- Install: `pip install unifrac-binaries` or `conda install -c biocore unifrac-binaries`
- Check library API documentation: https://github.com/biocore/unifrac-binaries/tree/main
- May require BIOM file path (not Table object) - check library API
- If file path required: Create temporary file, compute distances, delete file
- Use `tempfile` module for safe temporary file handling
- Handle errors to ensure cleanup (use context manager or try/finally)
- Verify exact function names and signatures from library documentation

### ASV ID Matching
- Ensure ASV IDs in BIOM table match tree tip labels
- Handle mismatches gracefully (filter or raise error)
- Verify all ASVs in table are in tree (or handle missing gracefully)

### Memory Management
- Distance matrices scale as O(N²) for N samples
- For large datasets, consider:
  - Computing on-demand per batch (slower but less memory)
  - Storing sparse representation if possible
  - Caching computed distances if rarefaction is fixed

### Reproducibility
- UniFrac computation should be deterministic
- If rarefaction changes per epoch, recompute distances
- Cache distances if rarefaction is fixed

## Integration with Data Pipeline

**During Initialization**:
1. Load BIOM table
2. Rarefy table
3. Compute UniFrac distances (full matrix) → store as `DistanceMatrix`
4. Store `encoder_target` for batch extraction

**During Epoch Regeneration** (if `gen_new_tables=True`):
1. Create new rarefied table
2. Recompute UniFrac distances → new `DistanceMatrix`
3. Update `encoder_target`

**During Batch Creation**:
1. Get batch sample IDs
2. Extract distances: `encoder_target.filter(sample_ids).data` → `[batch_size, batch_size]`
3. Include in batch dictionary as `unifrac_target`

## Loss Computation Integration

**Unweighted UniFrac Loss**:
- Model predicts pairwise distances: `[batch_size, batch_size]`
- Target is extracted pairwise distances: `[batch_size, batch_size]`
- Loss computed on upper triangle (symmetric matrix)
- Use `torch.triu()` to extract upper triangle
- Normalize by number of pairs: `batch_size * (batch_size - 1) / 2`

**Faith PD Loss**:
- Model predicts per-sample diversity: `[batch_size, 1]`
- Target is extracted Faith PD: `[batch_size, 1]`
- Standard MSE loss

## Testing Requirements

- Test with small BIOM table and tree (10 samples) for unit tests
- Test with real data for integration tests:
  - BIOM table: `./data/fall_train_only_all_outdoor.biom`
  - Tree: `./data/all-outdoors_sepp_tree.nwk`
- Verify unweighted UniFrac produces symmetric `DistanceMatrix`
- Verify Faith PD produces correct shape
- Test batch-level extraction using `.filter()` method
- Test with `gen_new_tables=True` (recompute each epoch)
- Test with `gen_new_tables=False` (reuse distances)
- Test batch size validation (must be even)
- Test with mismatched ASV IDs (error handling)
- Verify distances are non-negative
- Test upper triangle extraction for loss computation

## Test Data

- Unit tests: Generate synthetic small BIOM tables and trees
- Integration tests: Use `./data/fall_train_only_all_outdoor.biom` and `./data/all-outdoors_sepp_tree.nwk`

## Notes

- **Library**: Use `unifrac-binaries` from https://github.com/biocore/unifrac-binaries/tree/main
- **Installation**: `pip install unifrac-binaries` or `conda install -c biocore unifrac-binaries`
- **API Documentation**: Check library repository for exact function names and signatures
- **DistanceMatrix is key**: Store as `skbio.DistanceMatrix` for filtering capabilities
- **Epoch regeneration**: Recompute if rarefaction changes, cache otherwise
- **Batch extraction**: Use `.filter()` method for efficient extraction
- **Batch size**: Must be even (constraint from base implementation)
- **Loss computation**: Use upper triangle for pairwise distances
- **UniFrac is computationally expensive**: Optimize where possible, cache when safe
