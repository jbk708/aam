#!/usr/bin/env python
"""Diagnostic script to check tree_pruner module availability."""

import sys
import os
from pathlib import Path

print("=" * 60)
print("Tree Pruner Module Diagnostic")
print("=" * 60)

# Check 1: File exists
repo_root = Path(__file__).parent
tree_pruner_path = repo_root / "aam" / "data" / "tree_pruner.py"
print(f"\n1. File existence check:")
print(f"   Repo root: {repo_root}")
print(f"   tree_pruner.py path: {tree_pruner_path}")
print(f"   File exists: {tree_pruner_path.exists()}")

if tree_pruner_path.exists():
    print(f"   File size: {tree_pruner_path.stat().st_size} bytes")
    print(f"   File readable: {os.access(tree_pruner_path, os.R_OK)}")

# Check 2: Python path
print(f"\n2. Python path check:")
print(f"   Current directory: {os.getcwd()}")
print(f"   Script directory: {repo_root}")
print(f"   Python executable: {sys.executable}")
print(f"   Python version: {sys.version}")

# Check 3: aam.data package
print(f"\n3. aam.data package check:")
try:
    import aam.data
    print(f"   ✓ aam.data imported successfully")
    print(f"   aam.data path: {aam.data.__path__}")
    
    # Check if tree_pruner.py is in the path
    data_dir = Path(aam.data.__path__[0])
    tree_pruner_in_package = data_dir / "tree_pruner.py"
    print(f"   tree_pruner.py in package: {tree_pruner_in_package.exists()}")
    
    # List all modules
    import pkgutil
    modules = [name for _, name, _ in pkgutil.iter_modules(aam.data.__path__)]
    print(f"   Modules in aam.data: {modules}")
    print(f"   tree_pruner in modules: {'tree_pruner' in modules}")
    
except ImportError as e:
    print(f"   ✗ Failed to import aam.data: {e}")

# Check 4: Direct import attempt
print(f"\n4. Direct import attempt:")
try:
    from aam.data.tree_pruner import load_or_prune_tree, get_pruning_stats
    print(f"   ✓ Successfully imported load_or_prune_tree, get_pruning_stats")
    print(f"   load_or_prune_tree: {load_or_prune_tree}")
    print(f"   get_pruning_stats: {get_pruning_stats}")
except ImportError as e:
    print(f"   ✗ Failed to import: {e}")
    import traceback
    traceback.print_exc()

# Check 5: sys.path
print(f"\n5. sys.path check:")
aam_paths = [p for p in sys.path if 'aam' in p.lower() or os.getcwd() in p]
if aam_paths:
    print(f"   Relevant paths containing 'aam':")
    for p in aam_paths:
        print(f"     - {p}")
else:
    print(f"   No paths containing 'aam' found")
    print(f"   First 5 sys.path entries:")
    for p in sys.path[:5]:
        print(f"     - {p}")

# Check 6: Editable installation
print(f"\n6. Editable installation check:")
try:
    import importlib.metadata
    dist = importlib.metadata.distribution('aam')
    print(f"   Package name: {dist.metadata['Name']}")
    print(f"   Package version: {dist.version}")
    
    # Check if it's editable
    editable_file = Path(dist.files[0].locate()) if dist.files else None
    if editable_file:
        print(f"   First file location: {editable_file}")
        # Check for .pth file or editable marker
        site_packages = Path(sys.executable).parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        editable_marker = site_packages / f"__editable__.{dist.metadata['Name']}-{dist.version}.pth"
        if editable_marker.exists():
            print(f"   ✓ Editable installation detected: {editable_marker}")
            print(f"   Editable marker contents:")
            with open(editable_marker) as f:
                print(f"     {f.read().strip()}")
        else:
            print(f"   ✗ No editable marker found")
except Exception as e:
    print(f"   Could not check editable installation: {e}")

print("\n" + "=" * 60)
print("Recommendations:")
print("=" * 60)

if not tree_pruner_path.exists():
    print("1. ✗ tree_pruner.py file is missing!")
    print("   → Run: git pull origin pyt-10.3.1-optimize-tree-pruning")
else:
    print("1. ✓ tree_pruner.py file exists")

try:
    from aam.data.tree_pruner import load_or_prune_tree
    print("2. ✓ Module can be imported")
    print("   → The issue may be resolved. Try running your command again.")
except ImportError:
    print("2. ✗ Module cannot be imported")
    print("   → Try: pip install -e . --force-reinstall --no-deps")
    print("   → Or: python -c 'import sys; sys.path.insert(0, \".\"); from aam.data.tree_pruner import load_or_prune_tree'")

print("=" * 60)
