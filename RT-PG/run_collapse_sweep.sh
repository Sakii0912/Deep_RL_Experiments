#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

# "$PYTHON" collapse_emperical.py --a 0.8 --a-prime 0.8 --plot-file no_collapse.png
"$PYTHON" collapse_emperical.py --a 0.0 --a-prime 0.8 --plot-file model_collapse.png
"$PYTHON" collapse_emperical.py --a 0.8 --a-prime 0.0 --plot-file control_collapse.png
"$PYTHON" collapse_emperical.py --a 0.0 --a-prime 0.0 --plot-file full_collapse.png