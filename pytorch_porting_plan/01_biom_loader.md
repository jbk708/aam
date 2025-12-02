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

**3. Filter and Sort**
- Remove empty samples/ASVs
- Sort ASVs by abundance (for each sample)
- Limit to top N ASVs per sample (token_limit)
- Prepare for tokenization

### Implementation Requirements

**Class Structure**:
- Create `BIOMLoader` class
- Methods:
  - `load_table(path)`: Load BIOM file
  - `rarefy(table, depth)`: Rarefy to specified depth
  - `filter_and_sort(table, token_limit)`: Filter and sort ASVs
  - `get_sequences(table)`: Extract sequences for ASVs

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

- [ ] Create `BIOMLoader` class
- [ ] Implement `load_table()` method
- [ ] Implement `rarefy()` method
- [ ] Implement `filter_and_sort()` method
- [ ] Handle empty samples/ASVs
- [ ] Support reproducible rarefaction (random seed)
- [ ] Test with sample BIOM table
- [ ] Handle edge cases (empty table, very small table, etc.)

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

### ASV Filtering
- Sort ASVs by abundance (descending) for each sample
- Keep top `token_limit` ASVs per sample
- This reduces computational load while keeping most abundant ASVs

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
