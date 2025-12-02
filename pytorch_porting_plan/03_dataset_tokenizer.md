# Dataset and Tokenizer

**Status:** ✅ Completed

## Objective
Implement PyTorch Dataset class and sequence tokenization for microbial sequencing data.

## Requirements

### Tokenization

**Nucleotide to Token Mapping**:
- A (Adenine) → 1
- C (Cytosine) → 2
- G (Guanine) → 3
- T (Thymine) → 4
- Padding → 0
- Special tokens (if needed) → 5+

**Sequence Processing**:
- Extract sequences for each ASV from BIOM table metadata or separate file
- Convert nucleotide strings to token sequences
- Pad sequences to `max_bp` length
- Handle sequences longer than `max_bp` (truncate or error)

### Dataset Class

**Inherit from `torch.utils.data.Dataset`**:
- Implement `__len__()`: Return number of samples
- Implement `__getitem__(idx)`: Return single sample dictionary

**Sample Dictionary Structure**:
```python
{
    'tokens': torch.LongTensor,      # [num_asvs, max_bp]
    'counts': torch.FloatTensor,   # [num_asvs, 1]
    'y_target': torch.FloatTensor,  # [1] or [out_dim]
    'unifrac_target': torch.FloatTensor,  # [batch_size] or [batch_size, batch_size]
    'sample_id': str,              # Sample identifier
}
```

**Batch Collation**:
- Custom `collate_fn` to handle variable ASV counts
- Pad to `token_limit` ASVs per sample
- Stack into batch tensors

### Implementation Requirements

**Tokenizer Class**:
- Create `SequenceTokenizer` class
- Methods:
  - `tokenize(sequence)`: Convert string to tokens
  - `tokenize_batch(sequences)`: Batch tokenization
  - `pad_sequences(sequences, max_length)`: Pad to max length
- Handle different sequence formats (strings, bytes, etc.)

**Dataset Class**:
- Create `ASVDataset` class inheriting from `Dataset`
- Initialize with:
  - BIOM table (rarefied)
  - Metadata DataFrame
  - UniFrac distances
  - Tokenizer instance
  - Configuration (max_bp, token_limit, etc.)
- Handle epoch regeneration if rarefaction changes

**DataLoader Setup**:
- Create `DataLoader` with custom `collate_fn`
- Handle batching, shuffling, num_workers
- Ensure proper device placement

## Implementation Checklist

- [x] Create `SequenceTokenizer` class
- [x] Implement nucleotide to token mapping
- [x] Implement sequence padding
- [x] Handle sequence truncation
- [x] Create `ASVDataset` class
- [x] Implement `__len__()` method
- [x] Implement `__getitem__()` method
- [x] Handle ASV sequence extraction
- [x] Integrate UniFrac target extraction
- [x] Create custom `collate_fn` for batching
- [x] Create DataLoader setup function
- [x] Test with sample data
- [x] Verify tensor shapes and dtypes
- [x] Handle edge cases (no ASVs, very long sequences, etc.)

## Key Considerations

### Sequence Extraction
- Sequences may be in BIOM table metadata
- Or in separate taxonomy/sequence file
- Handle missing sequences gracefully
- Support different sequence formats

### Padding Strategy
- Pad sequences to `max_bp` with 0s
- Pad ASVs to `token_limit` per sample
- Create masks to identify valid tokens/ASVs
- Masks: `1` for valid, `0` for padding

### Batch Collation
- Samples may have different numbers of ASVs
- Pad to `token_limit` ASVs per sample
- Create batch-level masks
- Extract UniFrac distances for batch samples

### Epoch Regeneration
- If rarefaction changes per epoch, regenerate data
- Otherwise, reuse cached data
- Use `IterableDataset` for true lazy loading if needed

### Memory Efficiency
- Don't pre-tokenize all sequences if memory limited
- Tokenize on-the-fly in `__getitem__`
- Or pre-tokenize and cache if memory allows

## Testing Requirements

- Test tokenization with sample sequences
- Test padding and truncation
- Test Dataset with small BIOM table for unit tests
- Test Dataset with real data for integration tests:
  - Use `./data/fall_train_only_all_outdoor.biom` and `./data/all-outdoors_sepp_tree.nwk`
- Test batch collation produces correct shapes
- Test DataLoader iteration
- Verify masks are correct
- Test with edge cases (empty samples, single ASV, etc.)

## Test Data

- Unit tests: Generate synthetic small BIOM tables
- Integration tests: Use `./data/fall_train_only_all_outdoor.biom` and `./data/all-outdoors_sepp_tree.nwk`

## Notes

- Tokenization is straightforward (A/C/G/T mapping)
- Padding is critical for batching
- Masks enable proper attention computation
- UniFrac targets need batch-level extraction
