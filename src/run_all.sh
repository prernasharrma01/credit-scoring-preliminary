#!/bin/bash
# FILE 2 of 3: Runs every model across every seed, calling run_model.py
# fresh for each one (that isolation is what keeps AIF360/TensorFlow from
# leaking state between runs).
#
# Usage:
#   bash run_all.sh            -> runs seeds 1-15 (the full thing)
#   bash run_all.sh 1 5        -> runs only seeds 1 through 5
#   bash run_all.sh 6 10       -> runs only seeds 6 through 10 (continue later)
set -e
cd "$(dirname "$0")"

START=${1:-1}
END=${2:-15}

MODELS="NONE Re+EGR Re+AL DIR+EGR DIR+AL Re+EGR+ROC Re+EGR+EO Re+AL+ROC Re+AL+EO DIR+EGR+ROC DIR+EGR+EO DIR+AL+ROC DIR+AL+EO"

for seed in $(seq $START $END); do
  echo "================================================"
  echo "SEED $seed"
  echo "================================================"
  for model in $MODELS; do
    python3 run_model.py $model $seed 2>&1 | grep -v "WARNING\|oneDNN\|cpu_feature\|To enable\|I0000\|Instructions\|rate = 1\|E0000\|cuda drivers\|pip install"
  done
done

echo ""
echo "================================================"
echo "Analyzing results..."
echo "================================================"
python3 analyze_results.py