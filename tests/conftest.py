"""
Include src/atom_chip in the PYTHONPATH environment variable.
"""

import os
import sys
import jax


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "../src")
sys.path.insert(0, SRC_DIR)


# Configure JAX for 64-bit precision
jax.config.update("jax_enable_x64", True)
