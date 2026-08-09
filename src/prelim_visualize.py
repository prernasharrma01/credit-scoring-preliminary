import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
with open(f"{BASE_DIR}/outputs/prelim_results.json") as f:
    results = json.load(f)

names = ["NONE", "DIR only", "AL only", "EO only", "DIR+AL+EO"]
metrics = ["SPD", "DI", "EOD", "PED", "Acc"]
colors = {"SPD": "#4C72B0", "DI": "#DD8452", "EOD": "#55A868", "PED": "#C44E52", "Acc": "#8172B2"}

x = np.arange(len(names))
width = 0.15
fig, ax = plt.subplots(figsize=(11, 6))

for i, m in enumerate(metrics):
    vals = [results[n][m] for n in names]
    offset = (i - 2) * width
    bars = ax.bar(x + offset, vals, width, label=m, color=colors[m])
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3 if v >= 0 else -12),
                    ha="center", fontsize=7, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(names, rotation=10, ha="center", fontsize=10)
ax.axhline(0, color="grey", linewidth=0.8)
ax.set_ylabel("Metric Value")
ax.set_title("Preliminary Results: Baseline vs. Each Technique Alone vs. Full Combination\n(German Credit dataset)")
ax.legend(title="Metric", loc="upper right", ncol=5, fontsize=8)
plt.tight_layout()
plt.savefig(f"{BASE_DIR}/figures/prelim_comparison.png", dpi=150)
print("Saved prelim_comparison.png")
