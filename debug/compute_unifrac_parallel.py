#!/usr/bin/env python
"""Parallel UniFrac distance matrix computation script.

This is a standalone script that can be run directly.
Usage: python compute_unifrac_parallel.py --help
"""

import sys
from pathlib import Path

# Add parent directory to path to import aam modules
sys.path.insert(0, str(Path(__file__).parent))

# Now import and run the actual script
from aam.scripts.compute_unifrac_parallel import main

if __name__ == "__main__":
    main()
