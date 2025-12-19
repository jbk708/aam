# UniFrac Distance Loading

**Status:** Complete

## Overview
Loads pre-computed UniFrac distance matrices from disk. Users should generate UniFrac matrices using `ssu` from unifrac-binaries, then load them for training. Implemented in `aam/data/unifrac_loader.py`.

## Key Features
- **Matrix Loading**: Load pre-computed distance matrices from `.npy`, `.npz`, `.h5`, or `.csv` files
- **Format Support**: Pairwise (N×N), stripe (N×M), and Faith PD (1D) formats
- **Batch Extraction**: Extract batch-level distances from pre-computed matrices
- **Validation**: Ensures matrix dimensions match sample IDs
- **Automatic Format Detection**: Infers format from file extension and matrix shape

## Implementation
- **Class**: `UniFracLoader` in `aam/data/unifrac_loader.py`
- **Methods**: 
  - `load_matrix()` - Load matrix from disk with format detection
  - `extract_batch_distances()` - Extract batch distances for pairwise or Faith PD
  - `extract_batch_stripe_distances()` - Extract batch distances for stripe format
  - `validate_matrix_dimensions()` - Validate matrix dimensions match sample IDs
- **Testing**: Comprehensive unit tests (21 tests, all passing)

## File Formats Supported
- **`.npy`**: NumPy array format (supports both single arrays and `.npz` archives)
- **`.h5` / `.hdf5`**: HDF5 format (requires `h5py`)
- **`.csv`**: CSV format with sample IDs as index

## Example Usage
```python
from aam.data.unifrac_loader import UniFracLoader

loader = UniFracLoader()
# Load pairwise matrix
distances = loader.load_matrix("distances.npy", sample_ids=sample_ids, matrix_format="pairwise")
# Extract batch distances
batch_distances = loader.extract_batch_distances(distances, batch_sample_ids, metric="unweighted")
```
