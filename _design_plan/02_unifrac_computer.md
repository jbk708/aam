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
- Uses `unifrac` library (package name is `unifrac`, repo: https://github.com/biocore/unifrac-binaries)
- API: `unifrac.unweighted(table, tree)` - accepts `biom.Table` and `skbio.TreeNode` objects directly

**2. Faith's Phylogenetic Diversity (Faith PD)**
- Computes total branch length for each sample
- Single value per sample
- Output: `pandas.Series` (as returned by library)
- Shape: `[N_samples]` (indexed by sample IDs)
- Uses `unifrac` library
- API: `unifrac.faith_pd(table, tree)` - accepts `biom.Table` and `skbio.TreeNode` objects directly

### Implementation Requirements

**Class Structure**:
- Create `UniFracComputer` class
- Methods:
  - `compute_unweighted(table, tree_path)`: Compute unweighted UniFrac → returns `DistanceMatrix`
  - `compute_faith_pd(table, tree_path)`: Compute Faith PD → returns `DistanceMatrix` or Series
  - `extract_batch_distances(distances, sample_ids, metric)`: Extract distances for batch samples

**Key Considerations**:
- ASV IDs must match between table and tree
- Use `unifrac` library (package name is `unifrac`, install via: `pip install unifrac` or `conda install -c biocore unifrac`)
- Library accepts `biom.Table` and `skbio.TreeNode` objects directly (no temporary files needed)
- Store as `DistanceMatrix` object (not numpy array) for filtering capabilities
- Handle epoch regeneration: recompute if rarefaction changes
- Library API: https://github.com/biocore/unifrac-binaries/tree/main

### Batch Processing

**For Training Batches**:
- Pre-compute full distance matrix for all samples (stored as `DistanceMatrix` for unweighted, `Series` for Faith PD)
- For each batch, extract relevant rows/columns:
  - Unweighted UniFrac: Use `DistanceMatrix.filter(sample_ids).data` → `[batch_size, batch_size]`
  - Faith PD: Use `Series.loc[sample_ids].to_numpy().reshape(-1, 1)` → `[batch_size, 1]`

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

- [x] Install `unifrac` library (`pip install unifrac` or `conda install -c biocore unifrac`)
- [x] Review library API documentation: https://github.com/biocore/unifrac-binaries/tree/main
- [x] Create `UniFracComputer` class
- [x] Implement `compute_unweighted()` using `unifrac` → returns `DistanceMatrix`
- [x] Implement `compute_faith_pd()` using `unifrac` → returns `pandas.Series`
- [x] Library accepts Table and TreeNode objects directly (no temporary files needed)
- [x] Implement batch-level distance extraction using `.filter()` for DistanceMatrix and `.loc[]` for Series
- [x] Handle ASV ID matching (error handling for mismatches)
- [x] Store as `DistanceMatrix` object for unweighted (not numpy array)
- [x] Store as `pandas.Series` for Faith PD
- [x] Implement epoch regeneration logic (ready for integration)
- [x] Add batch size validation (must be even) - integrated into `extract_batch_distances()`
- [x] Test with sample BIOM table and tree (19 tests, all passing)
- [x] Verify distance matrix properties (symmetric, non-negative, zero diagonal)
- [x] Test batch extraction with different batch sizes
- [x] Handle edge cases (empty lists, missing IDs, invalid metrics, etc.)

## Key Considerations

### DistanceMatrix vs Numpy Array
- **Store as DistanceMatrix**: Enables filtering by sample IDs
- **Extract as numpy**: Use `.data` property or `.to_numpy()` for batch extraction
- **Filtering**: Use `distance_matrix.filter(sample_ids)` to get submatrix for batch

### Library Usage
- Install: `pip install unifrac` or `conda install -c biocore unifrac` (package name is `unifrac`)
- Library API documentation: https://github.com/biocore/unifrac-binaries/tree/main
- **Library accepts objects directly**: `unifrac.unweighted(table, tree)` and `unifrac.faith_pd(table, tree)` accept:
  - `table`: `biom.Table` object (or file path string)
  - `tree`: `skbio.TreeNode` object (or file path string)
- No temporary files needed when passing objects directly
- Load tree using `skbio.read(tree_path, format="newick", into=TreeNode)`
- Error handling: FileNotFoundError for missing files, ValueError for ASV ID mismatches

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

## Implementation Notes

### Actual Implementation
- **File Created**: `aam/data/unifrac.py` (not `unifrac_computer.py`)
- **Library**: Package name is `unifrac` (repo: https://github.com/biocore/unifrac-binaries)
- **Installation**: `pip install unifrac` or `conda install -c biocore unifrac`
- **API**: Library accepts `biom.Table` and `skbio.TreeNode` objects directly (no file I/O needed)
- **Return Types**:
  - `compute_unweighted()`: Returns `skbio.DistanceMatrix`
  - `compute_faith_pd()`: Returns `pandas.Series` (not DistanceMatrix - library default)
- **Batch Extraction**: `extract_batch_distances()` handles both `DistanceMatrix` (unweighted) and `Series` (faith_pd)
- **Error Handling**: FileNotFoundError for missing files, ValueError for ASV mismatches and invalid inputs
- **Batch Size Validation**: Integrated into `extract_batch_distances()` for unweighted metric
- **Tests**: 19 tests implemented, all passing (unit + integration)

### Key Design Decisions
- **DistanceMatrix for unweighted**: Enables filtering by sample IDs using `.filter()` method
- **Series for Faith PD**: Library returns Series, and it's more natural for per-sample values
- **Type checking in extract_batch_distances**: Validates input type matches metric type
- **No temporary files**: Library API allows direct object passing, cleaner implementation

### Notes
- **DistanceMatrix is key**: Store as `skbio.DistanceMatrix` for filtering capabilities
- **Epoch regeneration**: Ready for integration - recompute if rarefaction changes, cache otherwise
- **Batch extraction**: Use `.filter()` for DistanceMatrix, `.loc[]` for Series
- **Batch size**: Must be even (validated in `extract_batch_distances()` for unweighted)
- **Loss computation**: Use upper triangle for pairwise distances
- **UniFrac is computationally expensive**: Optimize where possible, cache when safe
