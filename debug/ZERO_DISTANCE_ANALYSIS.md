# Zero-Distance UniFrac Pairs Analysis

**Date:** 2024  
**Ticket:** PYT-8.13  
**Status:** ✅ Completed

## Executive Summary

Analysis of zero-distance UniFrac pairs in the training dataset reveals that **zero distances are extremely rare** (0.00% of pairwise comparisons). This finding contradicts the initial hypothesis that a large cluster of zero-distance pairs was causing bimodal distribution issues. The actual distribution shows a unimodal distribution centered around 0.72, with no significant zero-distance cluster.

## Dataset Information

- **BIOM Table:** `data/fall_train_only_all_outdoor.biom`
- **Phylogenetic Tree:** `data/all-outdoors_sepp_tree.nwk`
- **Rarefaction Depth:** 1,000 reads per sample
- **Original Samples:** 781
- **After Rarefaction:** 678 samples
- **ASVs (Original):** 172,536
- **ASVs (After Rarefaction):** 17,232

## Key Findings

### 1. Zero-Distance Prevalence

- **Total Pairwise Comparisons:** 229,503
- **Zero-Distance Pairs:** 1 (0.00%)
- **Non-Zero Distances:** 229,502 (100.00%)
- **Samples Involved in Zero-Distance Pairs:** 2 / 678 (0.29%)

**Conclusion:** Zero distances are extremely rare in this dataset. Only 1 pair out of 229,503 comparisons has a zero distance, involving just 2 samples.

### 2. Zero-Distance Distribution per Sample

- **Mean Zero Pairs per Sample:** 0.00
- **Median Zero Pairs per Sample:** 0
- **Maximum Zero Pairs per Sample:** 1
- **Samples with 0 Zero Pairs:** 676 (99.71%)
- **Samples with 1+ Zero Pairs:** 2 (0.29%)
- **Samples with 10+ Zero Pairs:** 0
- **Samples with 50+ Zero Pairs:** 0

**Conclusion:** The single zero-distance pair involves only 2 samples, and no sample is involved in multiple zero-distance pairs. This suggests the zero distance is likely due to:
- Identical samples (duplicates)
- Rarefaction artifacts creating identical community profiles
- Truly identical microbial communities (rare but possible)

### 3. Non-Zero Distance Statistics

- **Count:** 229,502
- **Mean:** 0.723279
- **Median:** 0.750563
- **Standard Deviation:** 0.146757
- **Minimum:** 0.007962
- **Maximum:** 0.985179
- **25th Percentile:** 0.633008
- **75th Percentile:** 0.832533

**Key Observations:**
- **Unimodal Distribution:** The distribution is unimodal, centered around 0.72-0.75
- **No Bimodal Pattern:** There is no evidence of a bimodal distribution with a zero-distance cluster
- **High Mean Distance:** The mean distance (0.72) is relatively high, indicating samples are generally dissimilar
- **Wide Range:** Distances span from 0.008 to 0.985, covering nearly the full [0, 1] range

### 4. Near-Zero Distances

- **Near-Zero Distances (< 1e-6):** 0 (0.0000%)

**Conclusion:** There are no near-zero distances, confirming that zero distances are truly rare and not part of a continuous distribution near zero.

### 5. Sample Abundance Analysis

- **Samples with Zero Distances:**
  - Mean: 1,000.00
  - Median: 1,000
  - Std: 0.00

- **Samples without Zero Distances:**
  - Mean: 1,000.00
  - Median: 1,000
  - Std: 0.00

**Conclusion:** All samples have identical total counts (1,000) due to rarefaction, so abundance differences cannot explain zero distances. The zero-distance pair likely represents:
- Identical ASV composition after rarefaction
- Duplicate samples in the original dataset
- Truly identical microbial communities

## Implications for Model Training

### 1. Zero-Distance Cluster Hypothesis Disproven

The initial hypothesis that a large cluster of zero-distance pairs was causing bimodal distribution issues is **not supported by the data**. The actual distribution is unimodal with only 1 zero-distance pair (0.00%).

### 2. Model Underfitting Root Cause

Given that zero distances are extremely rare, the model's underfitting (R² = 0.0455) is **not caused by a zero-distance cluster**. The root causes are likely:

1. **Loss Function Issues:**
   - Diagonal elements included in loss (addressed in PYT-8.12)
   - No bounded constraint for [0, 1] range
   - MSE loss may not be optimal for this distribution

2. **Distribution Characteristics:**
   - High mean distance (0.72) indicates samples are generally dissimilar
   - Model may struggle with high-distance predictions
   - No zero-distance handling needed (too rare to matter)

3. **Training Configuration:**
   - Learning rate may be suboptimal
   - Model capacity may be insufficient
   - Loss weighting may need adjustment

### 3. Handling Strategy Recommendations

**Recommendation: DO NOT implement zero-distance weighting or removal.**

**Rationale:**
- Zero distances are too rare (0.00%) to significantly impact training
- Removing or down-weighting 1 pair out of 229,503 will have negligible effect
- Focus should be on other loss function improvements (bounded loss, diagonal masking)

**Alternative Focus Areas:**
1. ✅ **Diagonal Masking (PYT-8.12):** Already implemented - excludes diagonal from loss
2. **Bounded Regression Loss (PYT-8.14):** Implement clipped MSE to enforce [0, 1] constraint
3. **Learning Rate Tuning (PYT-8.16):** Optimize learning rate for this distribution
4. **Model Architecture:** Consider if model capacity is sufficient for high-distance predictions

## Visualizations

The following visualizations were generated:

1. **distance_histogram.png:** Histogram of all UniFrac distances (highlighting zero cluster)
2. **non_zero_distance_histogram.png:** Histogram of non-zero distances only
3. **non_zero_distance_histogram_log.png:** Log-scale histogram of non-zero distances
4. **sample_zero_count_histogram.png:** Distribution of zero-distance pair counts per sample
5. **sample_abundance_comparison.png:** Comparison of sample abundances for samples with/without zero distances

All visualizations are saved in `debug/zero_distance_analysis/`.

## Conclusions

1. **Zero distances are extremely rare** (0.00% of pairwise comparisons)
2. **No bimodal distribution** - the distribution is unimodal centered around 0.72
3. **Zero-distance handling is not needed** - too rare to impact training
4. **Focus should shift** to other loss function improvements and training configuration

## Next Steps

1. ✅ **Complete PYT-8.12:** Mask diagonal in UniFrac loss computation (already done)
2. **Implement PYT-8.14:** Bounded regression loss (clipped MSE) to enforce [0, 1] constraint
3. **Implement PYT-8.16:** Tune learning rate for UniFrac model training
4. **Cancel PYT-8.15:** Weighted loss for zero-distance pairs (not needed - zero distances too rare)

## Files Generated

- `debug/investigate_zero_distance_samples.py` - Analysis script
- `debug/zero_distance_analysis/` - Directory with all visualizations
- `debug/ZERO_DISTANCE_ANALYSIS.md` - This analysis report

---

**Analysis completed successfully.**  
**Recommendation:** Proceed with bounded regression loss and learning rate tuning, skip zero-distance weighting.
