#!/usr/bin/env python
"""
PAF Evaluation Entry Point - Run from project root

This script properly sets up Python paths and runs paf_evaluation.py.
Use this instead of running paf_evaluation.py directly.

Usage:
    python paf_eval.py --test-type perturbation --steps 50
    python paf_eval.py --test-type qualitative --visualize-layers
"""

import sys
from pathlib import Path

# Add src/ to Python path
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Now run the actual evaluation script
from Evaluation.paf_evaluation import main

if __name__ == '__main__':
    main()
