"""
FILE 3 of 3: Combines everything after the runs finish --
  1. Aggregates all individual metrics files into one summary table (mean/std per model)
  2. Runs paired t-tests comparing each model against the NONE baseline (matched by seed)
  3. Builds one chart showing means, error bars, and significance stars
"""

import os
import re
import json
import glob
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "multiseed_results")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_ORDER = ["NONE", "Re+EGR", "Re+AL", "DIR+EGR", "DIR+AL",
               "Re+EGR+ROC", "Re+EGR+EO", "Re+AL+ROC", "Re+AL+EO",
               "DIR+EGR+ROC", "DIR+EGR+EO", "DIR+AL+ROC", "DIR+AL+EO"]
METRICS = ["SPD", "DI", "EOD", "PED", "Acc"]

pattern = re.compile(r"^(.*)_seed(\d+)_metrics\.json$")
raw_rows = []
for path in glob.glob(f"{RESULTS_DIR}/*_metrics.json"):
    fname = os.path.basename(path)
    match = pattern.match(fname)
    if not match:
        continue
    key, seed = match.group(1), int(match.group(2))
    model = key.replace("_", "+") if key != "NONE" else "NONE"
    with open(path) as f:
        metrics = json.load(f)
    raw_rows.append({"model": model, "seed": seed, **metrics})

raw_df = pd.DataFrame(raw_rows)
raw_df.to_csv(f"{OUT_DIR}/multiseed_raw.csv", index=False)

summary_rows = {}
for model in MODEL_ORDER:
    sub = raw_df[raw_df["model"] == model]
    if len(sub) == 0:
        continue
    summary_rows[model] = {"n_seeds": len(sub)}
    for m in METRICS:
        summary_rows[model][f"{m}_mean"] = round(sub[m].mean(), 4)
        summary_rows[model][f"{m}_std"] = round(sub[m].std(), 4)

summary_df = pd.DataFrame(summary_rows).T
summary_df.to_csv(f"{OUT_DIR}/multiseed_summary.csv")

baseline = raw_df[raw_df["model"] == "NONE"].set_index("seed")
sig_rows = {}
for model in MODEL_ORDER:
    if model == "NONE":
        continue
    sub = raw_df[raw_df["model"] == model].set_index("seed")
    common_seeds = sorted(set(sub.index) & set(baseline.index))
    if len(common_seeds) < 2:
        continue
    row = {"n_paired_seeds": len(common_seeds)}
    for m in METRICS:
        model_vals = sub.loc[common_seeds, m].values
        base_vals = baseline.loc[common_seeds, m].values
        t_stat, p_val = stats.ttest_rel(model_vals, base_vals)
        ideal = 1.0 if m == "DI" else 0.0
        improved = np.nanmean(np.abs(model_vals - ideal)) < np.nanmean(np.abs(base_vals - ideal))
        row[f"{m}_p"] = round(p_val, 4) if not np.isnan(p_val) else None
        row[f"{m}_sig"] = "**" if (not np.isnan(p_val) and p_val < 0.01) else ("*" if (not np.isnan(p_val) and p_val < 0.05) else "")
        row[f"{m}_improved"] = bool(improved)
    sig_rows[model] = row

sig_df = pd.DataFrame(sig_rows).T
sig_df.to_csv(f"{OUT_DIR}/multiseed_significance.csv")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print("=" * 100)
print("SUMMARY (mean +/- std across seeds)")
print("=" * 100)
print(summary_df)

print("\n" + "=" * 100)
print("SIGNIFICANCE vs. NONE baseline (paired t-test, * p<0.05, ** p<0.01)")
print("=" * 100)
for model, row in sig_df.iterrows():
    sig_and_improved = [m for m in METRICS if row.get(f"{m}_sig") and row.get(f"{m}_improved")]
    print(f"{model}: significantly better on {', '.join(sig_and_improved) if sig_and_improved else '(none)'}")

models = [m for m in MODEL_ORDER if m in summary_df.index]
colors = {"SPD": "#4C72B0", "DI": "#DD8452", "EOD": "#55A868", "PED": "#C44E52", "Acc": "#8172B2"}
x = np.arange(len(models))
width = 0.15
fig, ax = plt.subplots(figsize=(16, 7))

for i, m in enumerate(METRICS):
    means = summary_df.loc[models, f"{m}_mean"].values
    stds = summary_df.loc[models, f"{m}_std"].values
    offset = (i - 2) * width
    ax.bar(x + offset, means, width, yerr=stds, capsize=2, label=m, color=colors[m])
    for j, model in enumerate(models):
        if model == "NONE" or model not in sig_df.index:
            continue
        mark = sig_df.loc[model, f"{m}_sig"]
        if mark:
            ax.annotate(mark, (x[j] + offset, means[j] + stds[j]), textcoords="offset points",
                        xytext=(0, 2), ha="center", fontsize=8, color=colors[m])

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=25, ha="right")
ax.axhline(0, color="grey", linewidth=0.8)
ax.set_ylabel("Metric value (mean ± std across seeds)")
ax.set_title("Full 13-Model Sweep: Multi-Seed Results with Significance vs. Baseline\n(* p<0.05, ** p<0.01, paired t-test matched by seed)")
ax.legend(title="Metric", loc="upper right", ncol=5, fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/multiseed_significance.png", dpi=150)

print(f"\nSaved: {OUT_DIR}/multiseed_summary.csv")
print(f"Saved: {OUT_DIR}/multiseed_significance.csv")
print(f"Saved: {OUT_DIR}/multiseed_raw.csv")
print(f"Saved: {FIG_DIR}/multiseed_significance.png")