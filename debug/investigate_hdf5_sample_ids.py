#!/usr/bin/env python
"""Debug script to investigate sample ID mismatch between HDF5 and BIOM files."""

import sys
from pathlib import Path
import h5py
import biom
from biom import Table

# File paths
h5_path = "/Users/jbk/repos/aam/data/fall_train_only_all_outdoor.h5"
biom_path = "/Users/jbk/repos/aam/data/fall_train_only_all_outdoor.biom"

print("=" * 80)
print("Investigating sample ID mismatch between HDF5 and BIOM files")
print("=" * 80)

# Load HDF5 file
print("\n1. Loading HDF5 file...")
with h5py.File(h5_path, 'r') as f:
    print(f"   Keys in HDF5: {list(f.keys())}")
    
    if 'matrix' in f:
        matrix = f['matrix']
        print(f"   Matrix shape: {matrix.shape}")
        print(f"   Matrix dtype: {matrix.dtype}")
    
    if 'order' in f:
        order = f['order']
        print(f"   Order shape: {order.shape}")
        print(f"   Order dtype: {order.dtype}")
        print(f"   Order dtype kind: {order.dtype.kind}")
        
        # Get sample IDs
        if order.dtype.kind == 'S':  # String/bytes
            h5_sample_ids = [sid.decode('utf-8') for sid in order]
        else:
            h5_sample_ids = [str(sid) for sid in order]
        
        print(f"   Number of HDF5 samples: {len(h5_sample_ids)}")
        print(f"   First 5 HDF5 sample IDs:")
        for i, sid in enumerate(h5_sample_ids[:5]):
            print(f"     [{i}] {repr(sid)} (type: {type(sid).__name__})")
        print(f"   Last 5 HDF5 sample IDs:")
        for i, sid in enumerate(h5_sample_ids[-5:], len(h5_sample_ids)-5):
            print(f"     [{i}] {repr(sid)} (type: {type(sid).__name__})")

# Load BIOM table
print("\n2. Loading BIOM table...")
table = biom.load_table(biom_path)
biom_sample_ids = list(table.ids(axis="sample"))
print(f"   Number of BIOM samples: {len(biom_sample_ids)}")
print(f"   First 5 BIOM sample IDs:")
for i, sid in enumerate(biom_sample_ids[:5]):
    print(f"     [{i}] {repr(sid)} (type: {type(sid).__name__})")
print(f"   Last 5 BIOM sample IDs:")
for i, sid in enumerate(biom_sample_ids[-5:], len(biom_sample_ids)-5):
    print(f"     [{i}] {repr(sid)} (type: {type(sid).__name__})")

# Compare sample IDs
print("\n3. Comparing sample IDs...")
h5_set = set(h5_sample_ids)
biom_set = set(biom_sample_ids)

print(f"   HDF5 unique samples: {len(h5_set)}")
print(f"   BIOM unique samples: {len(biom_set)}")
print(f"   Intersection: {len(h5_set & biom_set)}")
print(f"   Only in HDF5: {len(h5_set - biom_set)}")
print(f"   Only in BIOM: {len(biom_set - h5_set)}")

if h5_set & biom_set:
    print(f"\n   Common samples (first 10):")
    for sid in sorted(list(h5_set & biom_set))[:10]:
        print(f"     {repr(sid)}")
else:
    print("\n   No common samples found!")
    print("\n   Checking for similar IDs (first 10 from each)...")
    print("   First 10 HDF5 IDs:")
    for sid in h5_sample_ids[:10]:
        print(f"     {repr(sid)}")
    print("   First 10 BIOM IDs:")
    for sid in biom_sample_ids[:10]:
        print(f"     {repr(sid)}")
    
    # Try to find matches with different string representations
    print("\n   Checking for matches with different representations...")
    h5_str_set = {str(sid).strip() for sid in h5_sample_ids}
    biom_str_set = {str(sid).strip() for sid in biom_sample_ids}
    str_intersection = h5_str_set & biom_str_set
    print(f"   String intersection (after str() and strip()): {len(str_intersection)}")
    
    if str_intersection:
        print(f"   Found matches after string conversion (first 10):")
        for sid in sorted(list(str_intersection))[:10]:
            print(f"     {repr(sid)}")

print("\n" + "=" * 80)
