# Fix for ModuleNotFoundError: aam.data.tree_pruner

The new `tree_pruner.py` module needs to be included in the package installation.

## Quick Fix

On the remote machine where you're running the command, reinstall the package:

```bash
cd /home/jokirkland/repos/aam
pip install -e .
```

This will reinstall the package in editable mode and include the new `tree_pruner.py` module.

## Alternative: Ensure file is synced

If you haven't pulled the latest changes yet:

```bash
cd /home/jokirkland/repos/aam
git pull origin pyt-10.3.1-optimize-tree-pruning
pip install -e .
```

The `tree_pruner.py` file should be at `aam/data/tree_pruner.py` after pulling.
