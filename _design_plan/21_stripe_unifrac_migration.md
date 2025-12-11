# Stripe-Based UniFrac Migration Plan

**Status:** 📋 Planning  
**Priority:** HIGH  
**Created:** 2025  
**Related Branch:** `pyt-10.3.1-optimize-tree-pruning`  
**Reference:** https://github.com/biocore/unifrac-binaries/blob/4c9644865f34d792e5987b690a10790ed9dafb61/src/api.hpp#L903

## Executive Summary

This document outlines the migration plan from pairwise UniFrac distance computation to stripe-based UniFrac computation. Stripe-based UniFrac computes distances for each sample against a reference set (stripe) rather than all pairwise combinations, which is more memory-efficient and potentially faster for large datasets.

## Current State (Pairwise UniFrac)

### Architecture
- **Computation**: Full pairwise distance matrices `[N_samples, N_samples]`
- **Batch Extraction**: Extracts `[batch_size, batch_size]` pairwise matrices
- **Loss Function**: Computes pairwise distances from embeddings `[batch_size, batch_size]`, masks diagonal
- **Memory**: O(N²) for full matrix, O(batch_size²) per batch
- **Training**: Batch size must be even (for pairwise distances)

### Key Files
- `aam/data/unifrac.py`: `UniFracComputer` class with `compute_unweighted()`, `extract_batch_distances()`, `compute_batch_unweighted()`
- `aam/data/dataset.py`: `collate_fn()` extracts pairwise distances
- `aam/training/losses.py`: `compute_pairwise_distances()`, `compute_base_loss()` with diagonal masking
- `aam/training/trainer.py`: Metrics computation expects pairwise matrices

## Target State (Stripe-Based UniFrac)

### Architecture
- **Computation**: Stripe-based distances `[N_samples, N_reference_samples]` where each sample is compared against a reference set
- **Batch Extraction**: Extracts `[batch_size, N_reference_samples]` stripe matrices
- **Loss Function**: Computes stripe distances from embeddings `[batch_size, N_reference_samples]`
- **Memory**: O(N × R) where R is reference set size (typically R << N)
- **Training**: No batch size restriction (stripe-based doesn't require even batch sizes)

### Benefits
1. **Memory Efficiency**: O(N × R) vs O(N²) for pairwise
2. **Scalability**: Can handle larger datasets with fixed reference set
3. **Flexibility**: No batch size restrictions
4. **Speed**: Potentially faster computation (fewer distance calculations)

## Migration Tickets

### Phase 1: Research and API Investigation

#### PYT-11.1: Investigate Stripe-Based UniFrac API ✅
**Priority:** HIGH | **Effort:** Medium (4-6 hours) | **Dependencies:** None | **Status:** ✅ Completed

**Tasks:**
1. Review unifrac-binaries API documentation for stripe-based computation
2. Check if `unifrac` Python package supports stripe-based computation
3. If not available in Python package, investigate:
   - **scikit-bio unifrac implementation** (check `skbio.diversity.beta.unweighted_unifrac` and related functions)
   - Direct C++ API bindings
   - Alternative libraries (e.g., `fastunifrac`)
   - Custom implementation using unifrac internals
4. If scikit-bio is investigated:
   - Check if `skbio.diversity.beta` supports stripe-based computation
   - Review `skbio.diversity.beta.unweighted_unifrac` API for batch/stripe capabilities
   - Test if scikit-bio can compute distances for samples against a reference set
   - Document scikit-bio API patterns and limitations
5. Determine reference set selection strategy:
   - Fixed reference samples (e.g., first N samples)
   - Random reference samples
   - Representative samples (e.g., k-means centroids)
   - All samples (degenerates to pairwise, but with different API)
6. Document API signature and usage patterns
7. Create proof-of-concept script demonstrating stripe-based computation

**Deliverables:**
- API investigation report (covering both `unifrac` package and scikit-bio)
- Proof-of-concept script (using whichever library supports stripe computation)
- Reference set selection strategy recommendation
- Library recommendation (unifrac package vs scikit-bio vs custom implementation)

**Files Created:**
- ✅ `_design_plan/21_stripe_unifrac_api_investigation.md` - Complete investigation report
- ✅ `scripts/investigate_stripe_unifrac.py` - Investigation script
- ✅ `scripts/test_dense_pair.py` - Proof-of-concept test script

**Key Findings:**
- ✅ `unifrac` package provides `*_dense_pair` functions suitable for stripe computation
- ✅ `unweighted_dense_pair` can compute distances between specific sample pairs
- ✅ Verified numerically equivalent to full matrix extraction (max diff < 1e-9)
- ⚠️ scikit-bio has limited support (pairwise only, less optimized)
- ✅ **Recommendation:** Use `unifrac.unweighted_dense_pair` for stripe implementation

---

### Phase 2: Core UniFracComputer Updates

#### PYT-11.2: Add Stripe-Based Computation to UniFracComputer ✅
**Priority:** HIGH | **Effort:** High (8-12 hours) | **Dependencies:** PYT-11.1 | **Status:** ✅ Completed

**Tasks:**
1. Add `compute_unweighted_stripe()` method to `UniFracComputer`:
   - Accepts reference sample IDs
   - Returns stripe distance matrix `[N_samples, N_reference_samples]`
   - Handles table filtering and tree loading
2. Add `compute_batch_unweighted_stripe()` method:
   - Computes stripe distances for batch samples
   - Uses cached reference set
   - Returns `[batch_size, N_reference_samples]`
3. Add `extract_batch_stripe_distances()` method:
   - Extracts stripe distances from pre-computed stripe matrix
   - Handles sample ID ordering
4. Add reference set management:
   - `set_reference_samples()` method
   - Reference set caching
   - Validation of reference samples
5. Update `setup_lazy_computation()` to support stripe mode
6. Maintain backward compatibility with pairwise methods (deprecate later)

**Implementation Details:**
```python
def compute_unweighted_stripe(
    self,
    table: Table,
    tree_path: str,
    reference_sample_ids: List[str],
    filter_table: bool = True
) -> np.ndarray:
    """Compute unweighted UniFrac distances in stripe format.
    
    Args:
        table: Rarefied biom.Table object
        tree_path: Path to phylogenetic tree file
        reference_sample_ids: List of reference sample IDs
        filter_table: If True, filter table to only include ASVs present in tree
    
    Returns:
        numpy array [N_samples, N_reference_samples] containing stripe distances
    """
    # Implementation using stripe-based API
    pass

def extract_batch_stripe_distances(
    self,
    stripe_distances: np.ndarray,
    sample_ids: List[str],
    reference_sample_ids: List[str],
    all_sample_ids: List[str]
) -> np.ndarray:
    """Extract stripe distances for a batch of samples.
    
    Args:
        stripe_distances: Pre-computed stripe matrix [N_samples, N_reference_samples]
        sample_ids: List of sample IDs for the batch
        reference_sample_ids: List of reference sample IDs (columns)
        all_sample_ids: List of all sample IDs (rows) matching stripe_distances
    
    Returns:
        numpy array [batch_size, N_reference_samples]
    """
    # Extract rows corresponding to sample_ids
    pass
```

**Files Modified:**
- ✅ `aam/data/unifrac.py` - Added stripe computation methods

**Files Created:**
- ✅ `tests/test_unifrac_stripe.py` - Comprehensive test suite (14 tests, all passing)

**Implementation Summary:**
- ✅ Added `set_reference_samples()` method for reference set management
- ✅ Added `compute_unweighted_stripe()` method using `unifrac.unweighted_dense_pair`
- ✅ Added `compute_batch_unweighted_stripe()` method with caching support
- ✅ Added `extract_batch_stripe_distances()` method for pre-computed stripe extraction
- ✅ All methods include proper validation, error handling, and caching
- ✅ Verified numerical equivalence with pairwise computation (max diff < 1e-6)
- ✅ No batch size restrictions (unlike pairwise mode)
- ✅ All 14 tests passing

---

### Phase 3: Dataset and Collation Updates

#### PYT-11.3: Update Dataset Collation for Stripe-Based Distances
**Priority:** HIGH | **Effort:** Medium (4-6 hours) | **Dependencies:** PYT-11.2

**Tasks:**
1. Update `collate_fn()` in `aam/data/dataset.py`:
   - Add `stripe_mode` parameter
   - Add `reference_sample_ids` parameter
   - Update distance extraction logic:
     - If `stripe_mode=True`: use `extract_batch_stripe_distances()` or `compute_batch_unweighted_stripe()`
     - If `stripe_mode=False`: use existing pairwise methods (backward compatibility)
   - Update shape validation: expect `[batch_size, N_reference_samples]` instead of `[batch_size, batch_size]`
2. Update `MicrobiomeDataset` class:
   - Add `stripe_mode` parameter to `__init__()`
   - Add `reference_sample_ids` parameter
   - Store reference sample IDs
   - Pass to `collate_fn()`
3. Update lazy computation path:
   - `compute_batch_unweighted_stripe()` for lazy stripe computation
   - Cache reference set in `UniFracComputer`
4. Remove batch size validation for stripe mode (no longer requires even batch size)

**Implementation Details:**
```python
def collate_fn(
    batch: List[Dict[str, Union[torch.Tensor, str]]],
    token_limit: int,
    unifrac_distances: Optional[Union[DistanceMatrix, np.ndarray]] = None,
    unifrac_metric: str = "unweighted",
    unifrac_computer: Optional["UniFracComputer"] = None,
    lazy_unifrac: bool = False,
    stripe_mode: bool = False,  # NEW
    reference_sample_ids: Optional[List[str]] = None,  # NEW
    all_sample_ids: Optional[List[str]] = None,  # NEW (for stripe extraction)
) -> Dict[str, torch.Tensor]:
    # ... existing code ...
    
    if unifrac_distances is not None or (lazy_unifrac and unifrac_computer is not None):
        if stripe_mode:
            if lazy_unifrac and unifrac_computer is not None:
                batch_distances = unifrac_computer.compute_batch_unweighted_stripe(
                    sample_ids, reference_sample_ids=reference_sample_ids
                )
            else:
                batch_distances = unifrac_computer.extract_batch_stripe_distances(
                    unifrac_distances, sample_ids, 
                    reference_sample_ids=reference_sample_ids,
                    all_sample_ids=all_sample_ids
                )
        else:
            # Existing pairwise logic
            # ...
```

**Files to Modify:**
- `aam/data/dataset.py`

**Files to Update:**
- `tests/test_dataset.py` (add stripe mode tests)

---

### Phase 4: Loss Function Updates

#### PYT-11.4: Update Loss Functions for Stripe-Based Distances
**Priority:** HIGH | **Effort:** Medium (4-6 hours) | **Dependencies:** PYT-11.3

**Tasks:**
1. Add `compute_stripe_distances()` function in `aam/training/losses.py`:
   - Computes stripe distances from embeddings `[batch_size, embedding_dim]`
   - Compares against reference embeddings `[N_reference_samples, embedding_dim]`
   - Returns `[batch_size, N_reference_samples]`
   - Uses Euclidean distance (same as pairwise)
2. Update `compute_base_loss()`:
   - Add `stripe_mode` parameter
   - Add `reference_embeddings` parameter (for stripe mode)
   - Update logic:
     - If `stripe_mode=True`: use `compute_stripe_distances()` instead of `compute_pairwise_distances()`
     - No diagonal masking needed (stripe distances don't have diagonal)
     - Shape validation: `[batch_size, N_reference_samples]`
   - Maintain backward compatibility with pairwise mode
3. Update `BaseLoss` class:
   - Add `stripe_mode` parameter to `__init__()`
   - Store reference embeddings if provided
   - Pass to `compute_base_loss()`

**Implementation Details:**
```python
def compute_stripe_distances(
    embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    normalize: bool = True,
    scale: float = 1.0,
) -> torch.Tensor:
    """Compute stripe distances from embeddings to reference embeddings.
    
    Args:
        embeddings: Sample embeddings [batch_size, embedding_dim]
        reference_embeddings: Reference sample embeddings [N_reference_samples, embedding_dim]
        normalize: If True, normalize distances to [0, 1] range
        scale: Scaling factor for normalization
    
    Returns:
        Stripe distance matrix [batch_size, N_reference_samples]
    """
    # Compute Euclidean distances: ||embeddings[i] - reference_embeddings[j]||
    # Shape: [batch_size, N_reference_samples]
    distances = torch.cdist(embeddings, reference_embeddings, p=2)
    
    if normalize:
        # Normalize to [0, 1] range (same as pairwise)
        max_dist = distances.max()
        if max_dist > 0:
            distances = distances / (max_dist * scale)
    
    return distances

def compute_base_loss(
    self,
    base_pred: torch.Tensor,
    base_true: torch.Tensor,
    encoder_type: str,
    embeddings: Optional[torch.Tensor] = None,
    stripe_mode: bool = False,  # NEW
    reference_embeddings: Optional[torch.Tensor] = None,  # NEW
) -> torch.Tensor:
    # For UniFrac, compute distances from embeddings if provided
    if encoder_type == "unifrac" and embeddings is not None:
        if stripe_mode:
            if reference_embeddings is None:
                raise ValueError("reference_embeddings required for stripe mode")
            base_pred = compute_stripe_distances(embeddings, reference_embeddings)
        else:
            base_pred = compute_pairwise_distances(embeddings)
    
    # No diagonal masking for stripe mode (no diagonal exists)
    if stripe_mode:
        return nn.functional.mse_loss(base_pred, base_true)
    else:
        # Existing pairwise logic with diagonal masking
        # ...
```

**Files to Modify:**
- `aam/training/losses.py`

**Files to Update:**
- `tests/test_losses.py` (add stripe mode tests)

---

### Phase 5: Model and Training Updates

#### PYT-11.5: Update Sequence Encoder for Stripe Mode
**Priority:** MEDIUM | **Effort:** Low (2-3 hours) | **Dependencies:** PYT-11.4

**Tasks:**
1. Update `SequenceEncoder` forward pass:
   - No changes needed (still returns embeddings)
   - Stripe distances computed in loss function
2. Update `SequencePredictor`:
   - No changes needed (embeddings are sufficient)
3. Document that embeddings are used for both pairwise and stripe modes

**Files to Review:**
- `aam/models/sequence_encoder.py`
- `aam/models/sequence_predictor.py`

---

#### PYT-11.6: Update Trainer for Stripe-Based Training
**Priority:** HIGH | **Effort:** Medium (4-6 hours) | **Dependencies:** PYT-11.4, PYT-11.5

**Tasks:**
1. Update `Trainer` class:
   - Add `stripe_mode` parameter to `__init__()`
   - Add `reference_sample_ids` parameter
   - Store reference sample IDs
   - Pass to dataset and loss function
2. Update reference embeddings handling:
   - Compute reference embeddings once at start of epoch
   - Cache reference embeddings
   - Pass to loss function
3. Update metrics computation:
   - Handle stripe distance matrices `[batch_size, N_reference_samples]`
   - Update validation plotting (if applicable)
   - Update R² and correlation metrics for stripe format
4. Update progress bar and logging:
   - Remove batch size validation for stripe mode
   - Update metrics display

**Implementation Details:**
```python
class Trainer:
    def __init__(
        self,
        # ... existing parameters ...
        stripe_mode: bool = False,
        reference_sample_ids: Optional[List[str]] = None,
    ):
        self.stripe_mode = stripe_mode
        self.reference_sample_ids = reference_sample_ids
        # ...
    
    def _get_reference_embeddings(self, model, dataloader):
        """Compute reference embeddings from reference samples."""
        if not self.stripe_mode or self.reference_sample_ids is None:
            return None
        
        # Get reference samples from dataset
        # Run through model to get embeddings
        # Return [N_reference_samples, embedding_dim]
        pass
```

**Files to Modify:**
- `aam/training/trainer.py`

**Files to Update:**
- `tests/test_trainer.py` (add stripe mode tests)

---

#### PYT-11.7: Update CLI for Stripe Mode
**Priority:** MEDIUM | **Effort:** Low (2-3 hours) | **Dependencies:** PYT-11.6

**Tasks:**
1. Add CLI arguments:
   - `--stripe-mode`: Enable stripe-based UniFrac
   - `--reference-samples`: Path to file with reference sample IDs, or number of reference samples
   - `--reference-selection`: Strategy for selecting reference samples (first, random, representative)
2. Update argument parsing:
   - Validate stripe mode compatibility
   - Load or generate reference sample IDs
   - Pass to dataset and trainer
3. Update help text and documentation

**Implementation Details:**
```python
@click.option(
    "--stripe-mode",
    is_flag=True,
    default=False,
    help="Use stripe-based UniFrac instead of pairwise (more memory-efficient)"
)
@click.option(
    "--reference-samples",
    type=str,
    default=None,
    help="Reference samples: path to file with sample IDs, or number (e.g., '100' for first 100 samples)"
)
@click.option(
    "--reference-selection",
    type=click.Choice(["first", "random", "representative"]),
    default="first",
    help="Strategy for selecting reference samples"
)
```

**Files to Modify:**
- `aam/cli.py`

**Files to Update:**
- `tests/test_cli.py` (add stripe mode tests)

---

### Phase 6: Testing and Validation

#### PYT-11.8: Comprehensive Testing for Stripe Mode
**Priority:** HIGH | **Effort:** High (8-12 hours) | **Dependencies:** PYT-11.2-PYT-11.7

**Tasks:**
1. Unit tests for stripe computation:
   - `test_unifrac_stripe.py`: Test `compute_unweighted_stripe()`, `extract_batch_stripe_distances()`, etc.
   - Verify correctness against pairwise computation (subset)
   - Test reference set management
2. Dataset tests:
   - Test `collate_fn()` with stripe mode
   - Test lazy stripe computation
   - Test batch size flexibility (odd batch sizes)
3. Loss function tests:
   - Test `compute_stripe_distances()`
   - Test `compute_base_loss()` with stripe mode
   - Verify gradient flow
4. Integration tests:
   - End-to-end training with stripe mode
   - Compare results with pairwise mode (should be similar)
   - Test reference embedding computation
5. Performance tests:
   - Memory usage comparison (stripe vs pairwise)
   - Speed comparison
   - Scalability tests

**Files to Create:**
- `tests/test_unifrac_stripe.py`
- `tests/test_losses_stripe.py` (or add to existing)

**Files to Update:**
- `tests/test_dataset.py`
- `tests/test_trainer.py`
- `tests/test_integration.py`

---

### Phase 7: Documentation and Migration

#### PYT-11.9: Update Documentation
**Priority:** MEDIUM | **Effort:** Medium (3-4 hours) | **Dependencies:** PYT-11.8

**Tasks:**
1. Update design plan documents:
   - `_design_plan/02_unifrac_computer.md`: Document stripe-based methods
   - `_design_plan/10_training_losses.md`: Document stripe loss computation
   - `_design_plan/08_sequence_encoder.md`: Note stripe mode support
2. Update architecture documentation:
   - `ARCHITECTURE.md`: Document stripe-based architecture
   - Explain memory benefits
3. Update user documentation:
   - `README.md`: Add stripe mode usage examples
   - CLI help text updates
4. Create migration guide:
   - How to migrate from pairwise to stripe
   - When to use stripe vs pairwise
   - Reference set selection guidelines

**Files to Modify:**
- `_design_plan/02_unifrac_computer.md`
- `_design_plan/10_training_losses.md`
- `_design_plan/08_sequence_encoder.md`
- `ARCHITECTURE.md`
- `README.md`

**Files to Create:**
- `_design_plan/STRIPE_MIGRATION_GUIDE.md`

---

#### PYT-11.10: Deprecation Strategy for Pairwise Mode
**Priority:** LOW | **Effort:** Low (2-3 hours) | **Dependencies:** PYT-11.9

**Tasks:**
1. Add deprecation warnings to pairwise methods:
   - `compute_unweighted()`: Warn that pairwise mode is deprecated
   - `extract_batch_distances()`: Warn for pairwise mode
   - `compute_pairwise_distances()`: Warn in loss function
2. Update documentation:
   - Mark pairwise methods as deprecated
   - Recommend stripe mode for new code
3. Plan removal timeline:
   - Keep pairwise mode for 1-2 releases
   - Remove after stripe mode is proven stable

**Files to Modify:**
- `aam/data/unifrac.py` (add deprecation warnings)
- `aam/training/losses.py` (add deprecation warnings)

---

## Implementation Priority

### Phase 1: Foundation (Weeks 1-2)
1. **PYT-11.1**: API Investigation - **CRITICAL** (must be done first)
2. **PYT-11.2**: Core UniFracComputer Updates - **HIGH** (blocks everything else)

### Phase 2: Integration (Weeks 2-3)
3. **PYT-11.3**: Dataset Updates - **HIGH**
4. **PYT-11.4**: Loss Function Updates - **HIGH**
5. **PYT-11.5**: Model Updates - **MEDIUM** (may be minimal)

### Phase 3: Training and CLI (Week 3-4)
6. **PYT-11.6**: Trainer Updates - **HIGH**
7. **PYT-11.7**: CLI Updates - **MEDIUM**

### Phase 4: Testing and Documentation (Week 4-5)
8. **PYT-11.8**: Comprehensive Testing - **HIGH**
9. **PYT-11.9**: Documentation - **MEDIUM**
10. **PYT-11.10**: Deprecation Strategy - **LOW** (can be done later)

## Success Metrics

### Functional Requirements
- ✅ Stripe-based UniFrac computation works correctly
- ✅ Training with stripe mode produces similar results to pairwise mode
- ✅ Memory usage reduced (measure O(N×R) vs O(N²))
- ✅ No batch size restrictions in stripe mode
- ✅ All tests pass

### Performance Requirements
- Memory: 50-90% reduction for large datasets (depends on R/N ratio)
- Speed: Comparable or faster than pairwise (fewer distance calculations)
- Scalability: Can handle larger datasets with fixed reference set size

### Quality Requirements
- Code coverage: Maintain >80% coverage
- Documentation: All public APIs documented
- Backward compatibility: Pairwise mode still works (with deprecation warnings)

## Risk Assessment

### High Risk
- **API Availability**: Stripe-based API may not be available in Python `unifrac` package
  - **Mitigation**: Investigate alternatives (C++ bindings, custom implementation)
- **Reference Set Selection**: Poor reference set may degrade model performance
  - **Mitigation**: Test multiple strategies, allow user to provide reference set

### Medium Risk
- **Loss Function Changes**: Stripe distances may require different loss formulation
  - **Mitigation**: Extensive testing, compare with pairwise results
- **Migration Complexity**: Large codebase changes across multiple files
  - **Mitigation**: Phased approach, comprehensive testing

### Low Risk
- **Backward Compatibility**: Maintaining pairwise mode adds complexity
  - **Mitigation**: Deprecation strategy, clear migration path

## Dependencies

### External Dependencies
- `unifrac` Python package (or alternative) must support stripe-based computation
- May require C++ bindings or custom implementation

### Internal Dependencies
- Stable pairwise implementation (for comparison and fallback)
- Comprehensive test suite (for validation)

## Open Questions

1. **Reference Set Size**: What is optimal R? (e.g., R=100, R=1000, R=0.1*N)
2. **Reference Set Selection**: Which strategy works best? (first, random, representative)
3. **API Availability**: Does Python `unifrac` package support stripe-based computation?
4. **Performance Trade-offs**: How does stripe mode affect training dynamics?
5. **Compatibility**: Should we support both modes simultaneously or force migration?

## Next Steps

1. **Start with PYT-11.1**: Investigate stripe-based API availability
2. **Create proof-of-concept**: Demonstrate stripe computation works
3. **Review with team**: Validate approach before full implementation
4. **Begin Phase 1**: Implement core UniFracComputer updates

---

**Document Status**: Ready for review  
**Next Action**: Review and approve ticket plan, then begin PYT-11.1
