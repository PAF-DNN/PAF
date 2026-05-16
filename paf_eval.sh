#!/bin/bash
# paf_eval.sh - Convenience script to run PAF evaluation from project root
#
# Usage:
#   ./paf_eval.sh --test-type perturbation --steps 50
#   ./paf_eval.sh --test-type qualitative --visualize-layers

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run paf_evaluation.py with arguments passed through
python "${SCRIPT_DIR}/src/Evaluation/paf_evaluation.py" "$@"
