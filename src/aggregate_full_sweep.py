"""Aggregates all individual _metrics.json files into full_sweep_results.json / .csv"""

import json
import glob
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PICKLE_DIR = os.path.join(BASE_DIR, "outputs", "full_sweep_pickles")
OUT_DIR = os.path.join(BASE_DIR, "outputs")

NAME_MAP = {
    "NONE": "NONE",
    "Re_EGR": "Re+EGR", "Re_AL": "Re+AL", "DIR_EGR": "DIR+EGR", "DIR_AL": "DIR+AL",
    "Re_EGR_ROC": "Re+EGR+ROC", "Re_EGR_EO": "Re+EGR+EO",
    "Re_AL_ROC": "Re+AL+ROC", "Re_AL_EO": "Re+AL+EO",
    "DIR_EGR_ROC": "DIR+EGR+ROC", "DIR_EGR_EO": "DIR+EGR+EO",
    "DIR_AL_ROC": "DIR+AL+ROC", "DIR_AL_EO": "DIR+AL+EO",
}

results = {}
for path in glob.glob(f"{PICKLE_DIR}/*_metrics.json"):
    key = os.path.basename(path).replace("_metrics.json", "")
    display_name = NAME_MAP.get(key, key)
    with open(path) as f:
        results[display_name] = json.load(f)

order = ["NONE", "Re+EGR", "Re+AL", "DIR+EGR", "DIR+AL",
         "Re+EGR+ROC", "Re+EGR+EO", "Re+AL+ROC", "Re+AL+EO",
         "DIR+EGR+ROC", "DIR+EGR+EO", "DIR+AL+ROC", "DIR+AL+EO"]
results_ordered = {k: results[k] for k in order if k in results}

with open(f"{OUT_DIR}/full_sweep_results.json", "w") as f:
    json.dump(results_ordered, f, indent=2)

df = pd.DataFrame(results_ordered).T
df.to_csv(f"{OUT_DIR}/full_sweep_results.csv")
print(df.round(4))
print(f"\n{len(results_ordered)}/13 models completed.")