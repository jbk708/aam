# Investigation: Why Many Sequences Have All Padding

## Problem
809 out of 1536 sequences (52.67%) have all padding after attention pooling, causing NaN in final nucleotide predictions.

## Potential Causes

### 1. **Truncation in collate_fn removes all valid ASVs**
**Hypothesis**: When `token_limit=256` is applied, if a sample has >256 ASVs, we take `tokens[:token_limit]`. If the first 256 ASVs happen to be all padding (unlikely but possible), or if truncation somehow removes all valid sequences.

**Investigation Plan**:
- Check how many ASVs each sample has before truncation
- Verify that samples have valid ASVs after truncation
- Check if truncation is preserving the correct ASVs (should be first N ASVs)

### 2. **START_TOKEN not properly added or preserved**
**Hypothesis**: START_TOKEN (value 5) should be at position 0 of each valid sequence. If START_TOKEN is missing or overwritten, sequences appear as all padding.

**Investigation Plan**:
- Verify START_TOKEN is added in tokenizer
- Check if START_TOKEN survives padding/truncation
- Count how many sequences have START_TOKEN at position 0

### 3. **Mask creation logic is incorrect**
**Hypothesis**: Mask `(tokens_flat > 0).long()` treats START_TOKEN=5 as valid, but if sequences are all zeros except START_TOKEN, they might be incorrectly identified as all-padding.

**Investigation Plan**:
- Check mask creation: `mask = (tokens_flat > 0).long()`
- Verify START_TOKEN positions are included in mask
- Check if sequences with only START_TOKEN are being treated as all-padding

### 4. **Padding overwrites valid tokens**
**Hypothesis**: Padding logic might be overwriting valid sequence data, or sequences might be getting padded incorrectly.

**Investigation Plan**:
- Check padding logic in tokenizer
- Verify sequences maintain valid tokens after padding
- Check if padding is applied correctly in collate_fn

### 5. **Empty or invalid sequences in source data**
**Hypothesis**: Some ASVs in the BIOM table might have empty sequences or sequences that tokenize to all zeros.

**Investigation Plan**:
- Check source sequences before tokenization
- Verify sequences have valid length after tokenization
- Check if any sequences tokenize to all zeros

### 6. **Truncation removes START_TOKEN**
**Hypothesis**: If sequences are truncated at the sequence level (not just ASV level), START_TOKEN might be removed.

**Investigation Plan**:
- Check if sequence-level truncation is happening
- Verify START_TOKEN is preserved after all operations
- Check sequence lengths before/after truncation

### 7. **Collate_fn padding creates all-zero sequences**
**Hypothesis**: When padding to `token_limit`, if `num_asvs < token_limit`, the padding might be creating sequences that are all zeros.

**Investigation Plan**:
- Check padding logic: `padded_tokens[:num_asvs] = tokens`
- Verify that padded positions (after num_asvs) are correctly set to zeros
- Check if padding is accidentally overwriting valid data

## Investigation Script Plan

Create a script that:
1. Loads a batch and checks each sample
2. For each sample, checks:
   - Number of ASVs before truncation
   - Number of ASVs after truncation
   - How many sequences have START_TOKEN at position 0
   - How many sequences are all zeros
   - Mask values for each sequence
3. Traces a few specific sequences through the pipeline:
   - One that should be valid but appears as all-padding
   - One that is correctly identified as all-padding
   - One that is correctly identified as valid

## Key Questions to Answer

1. **Are the all-padding sequences actually all zeros, or do they have START_TOKEN?**
   - If they have START_TOKEN, mask logic might be wrong
   - If they're all zeros, something is removing/overwriting valid tokens

2. **Do samples have valid ASVs after truncation?**
   - Validation should catch this, but let's verify

3. **Is START_TOKEN being preserved through the pipeline?**
   - Check at each stage: tokenization, padding, truncation, batching

4. **Are the all-padding sequences from padding positions or from actual ASVs?**
   - If from padding positions, this is expected
   - If from actual ASVs, this is a bug
