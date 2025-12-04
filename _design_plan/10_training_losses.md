# Training Losses and Metrics

**Status:** ✅ Completed

## Overview
Multi-task loss functions and metrics. Implemented in `aam/training/losses.py` and `aam/training/metrics.py`.

## Loss Functions
- **Target Loss**: MSE (regression) or NLL (classification) with optional class weights
- **Count Loss**: Masked MSE for ASV counts
- **Base Loss**: MSE for UniFrac distances (weighted by `penalty`)
  - Also available as `unifrac_loss` (when encoder_type="unifrac"), `faith_loss` (when encoder_type="faith_pd"), or `taxonomy_loss` (when encoder_type="taxonomy")
  - These are aliases to `base_loss` for clarity in logging and progress bars
- **Nucleotide Loss**: Masked CrossEntropy (weighted by `nuc_penalty`)
- **Total Loss**: Weighted sum of all losses

## Metrics
- **Regression**: MAE, MSE, R-squared
- **Classification**: Accuracy, Confusion Matrix, Precision/Recall/F1

## Implementation
- **Losses**: `MultiTaskLoss` in `aam/training/losses.py`
- **Metrics**: `compute_regression_metrics()`, `compute_classification_metrics()`, `compute_count_metrics()` in `aam/training/metrics.py`
- **Testing**: Comprehensive unit tests (18 loss tests + 11 metrics tests passing)
