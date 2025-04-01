"""
Include src/atom_chip in the PYTHONPATH environment variable.
"""

import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "../src")
sys.path.insert(0, SRC_DIR)
