#!/usr/bin/env python3
"""Root execution runner for all paper experiments."""

import sys
import os

# Set MPLCONFIGDIR to avoid warnings
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
os.makedirs("/tmp/matplotlib_cache", exist_ok=True)

# Add current workspace to pythonpath
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from mec_into.main import main

if __name__ == "__main__":
    main()
