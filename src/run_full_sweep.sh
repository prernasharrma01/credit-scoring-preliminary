#!/bin/bash
# Master orchestrator: runs every model as an ISOLATED subprocess to avoid
# any state leakage between AIF360/TensorFlow calls (a real issue hit and
# fixed earlier in this project).
set -e
cd "$(dirname "$0")"

echo "=== Baseline (NONE) ==="
python3 run_single.py NONE 2>&1 | grep -v "WARNING\|oneDNN\|cpu_feature\|To enable\|I0000\|Instructions\|rate = 1\|E0000\|cuda drivers"

echo ""
echo "=== PI Models (4) ==="
for pre in Re DIR; do
  for inp in EGR AL; do
    echo "--- $pre + $inp ---"
    python3 run_single.py $pre $inp 2>&1 | grep -v "WARNING\|oneDNN\|cpu_feature\|To enable\|I0000\|Instructions\|rate = 1\|E0000\|cuda drivers"
  done
done

echo ""
echo "=== PIP Models (8) ==="
for pre in Re DIR; do
  for inp in EGR AL; do
    for post in ROC EO; do
      echo "--- $pre + $inp + $post ---"
      python3 run_post.py ${pre}_${inp} $post 2>&1 | grep -v "WARNING\|oneDNN\|cpu_feature\|To enable\|I0000\|Instructions\|rate = 1\|E0000\|cuda drivers"
    done
  done
done

echo ""
echo "=== Aggregating results ==="
python3 aggregate_full_sweep.py