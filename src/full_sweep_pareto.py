import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import find_non_dominated

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")

with open(f"{OUT_DIR}/full_sweep_results.json") as f:
    results = json.load(f)

non_dominated_all = find_non_dominated(results, objective_keys=["SPD", "DI", "EOD", "PED", "Acc"])

print("=" * 70)
print("MANY-OBJECTIVE PARETO ANALYSIS (all 13 models, SPD+DI+EOD+PED+Acc)")
print("=" * 70)
print("Non-dominated (Pareto-optimal) models:")
for m in non_dominated_all:
    print(f"  - {m}")
print(f"\nIs baseline (NONE) non-dominated? {'NONE' in non_dominated_all}")

with open(f"{OUT_DIR}/full_sweep_non_dominated.json", "w") as f:
    json.dump({"non_dominated": non_dominated_all}, f, indent=2)

print(f"\nSaved to {OUT_DIR}/full_sweep_non_dominated.json")