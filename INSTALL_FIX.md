# Fix for ModuleNotFoundError: aam.data.tree_pruner

The new `tree_pruner.py` module needs to be included in the package installation.

## Diagnostic

First, run the diagnostic script to see what's wrong:

```bash
conda activate aam-rebuild
cd /home/jokirkland/repos/aam
python diagnose_tree_pruner.py
```

## Quick Fix

1. **Ensure you're on the correct branch and file exists:**
```bash
conda activate aam-rebuild
cd /home/jokirkland/repos/aam
git pull origin pyt-10.3.1-optimize-tree-pruning
ls -la aam/data/tree_pruner.py  # Should show the file
```

2. **Force reinstall the package:**
```bash
pip install -e . --force-reinstall --no-deps
```

3. **Verify the import works:**
```bash
python -c "from aam.data.tree_pruner import load_or_prune_tree; print('Success!')"
```

## Alternative: Manual import test

If the above doesn't work, test if the file can be imported directly:

```bash
cd /home/jokirkland/repos/aam
python -c "import sys; sys.path.insert(0, '.'); from aam.data.tree_pruner import load_or_prune_tree; print('Direct import works')"
```

If this works but the package import doesn't, there may be an issue with the editable installation. Try:

```bash
pip uninstall aam -y
pip install -e .
```
