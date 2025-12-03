# BIOM Table Loader

## Objective
Implement functionality to load, process, and rarefy BIOM tables for microbial sequencing data.

## Requirements

### Input
- BIOM table file (`.biom` format)
- Contains sample × ASV abundance matrix
- May include sample and observation metadata

### Processing Steps

**1. Load BIOM Table**
- Use `biom-format` library to read `.biom` file
- Extract abundance matrix (samples × ASVs)
- Extract sample IDs and ASV IDs
- Handle metadata if present

**2. Rarefaction**
- Subsample each sample to consistent depth
- Random sampling without replacement
- Ensures samples are comparable
- Depth parameter: typically 5000 reads per sample

**3. Sequence Extraction**
- Extract sequences from observation IDs (ASV IDs)
- Observation IDs are assumed to be 150bp DNA sequences
- No metadata handling required

### Implementation Requirements

**Class Structure**:
- Create `BIOMLoader` class
- Methods:
  - `load_table(path)`: Load BIOM file
  - `rarefy(table, depth, with_replacement, random_seed, inplace)`: Rarefy to specified depth using biom-format's `subsample()`
  - `get_sequences(table)`: Extract sequences from observation IDs

**Note:** `filter_and_sort()` was removed - all ASVs are used, no filtering needed since sequence length is capped at 150bp.

**Key Considerations**:
- Handle different BIOM table formats
- Efficient rarefaction (can be slow for large tables)
- Reproducibility (set random seed)
- Memory efficiency for large tables

**Output Format**:
- Rarefied table: `biom.Table` object
- Sample IDs: list of strings
- ASV IDs: list of strings
- Counts matrix: can be extracted as needed

## Implementation Checklist

- [x] Create `BIOMLoader` class
- [x] Implement `load_table()` method
- [x] Implement `rarefy()` method (uses biom-format's `subsample()`)
- [x] Implement `get_sequences()` method (extracts from observation IDs)
- [x] Handle empty samples/ASVs (drops samples below depth)
- [x] Support reproducible rarefaction (random seed)
- [x] Test with sample BIOM table
- [x] Handle edge cases (empty table, very small table, etc.)
- [x] Unit tests with 150bp sequences and no metadata

## Key Considerations

### Rarefaction Algorithm
- For each sample, randomly sample `depth` reads
- Use multinomial sampling based on ASV proportions
- Ensure reproducibility with random seed
- Handle samples with fewer than `depth` reads (skip or use all)

### Memory Management
- BIOM tables can be large (thousands of samples × thousands of ASVs)
- Use sparse matrix representations where possible
- Consider chunking for very large tables

### Sequence Extraction
- Observation IDs are assumed to be 150bp DNA sequences
- No metadata is used - sequences extracted directly from observation IDs
- All ASVs are used - no filtering needed since sequence length is capped at 150bp

## Testing Requirements

- Test with small BIOM table (10 samples, 100 ASVs) for unit tests
- Test with real data: `./data/fall_train_only_all_outdoor.biom` for integration tests
- Test rarefaction produces correct depth
- Test filtering keeps correct number of ASVs
- Test with edge cases (empty table, single sample, etc.)
- Verify reproducibility with same random seed

## Test Data

- Unit tests: Generate synthetic small BIOM tables
- Integration tests: Use `./data/fall_train_only_all_outdoor.biom`

## Notes

- BIOM format is standard for microbial data
- Rarefaction is critical for comparing samples
- Filtering reduces computational requirements
- Can cache rarefied tables if recomputing is expensive
