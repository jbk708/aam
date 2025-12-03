# BIOM Table Loader

**Status:** ✅ Completed

## Overview
Loads, processes, and rarefies BIOM tables for microbial sequencing data. Implemented in `aam/data/biom_loader.py`.

## Key Features
- Loads BIOM table files (`.biom` format)
- Rarefies samples to consistent depth (default: 5000 reads)
- Extracts 150bp DNA sequences from observation IDs (ASV IDs)
- Supports reproducible rarefaction with random seed

## Implementation
- **Class**: `BIOMLoader` in `aam/data/biom_loader.py`
- **Methods**: `load_table()`, `rarefy()`, `get_sequences()`
- **Testing**: Comprehensive unit tests (21 tests passing)
