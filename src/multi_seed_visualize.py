import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
df = pd.read_csv(f"{BASE_DIR}/outputs/multi_seed_summary.csv", index_col=0)
models = df.index.tolist()
metrics = ["SPD", "DI", "EOD", "PED", "Acc"]
colors = {"SPD": "#4C72B0", "DI": "#DD8452", "EOD": "#55A868", "PED": "#C44E52", "Acc": "#8172B2"}

x = np.arange(len(models))
width = 0.15
fig, ax = plt.subplots(figsize=(10, 6))

for i, m in enumerate(metrics):
    means = df[f"{m}_mean"].values
    stds = df[f"{m}_std"].values
    offset = (i - 2) * width
    bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=m, color=colors[m])

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.axhline(0, color="grey", linewidth=0.8)
ax.set_ylabel("Metric Value (mean ± std across 15 random splits)")
ax.set_title("Robustness Check: Same 3 Models Across 15 Different Train/Test Splits\n(error bars = variation across splits)")
ax.legend(title="Metric", loc="upper right", ncol=5, fontsize=8)
plt.tight_layout()
plt.savefig(f"{BASE_DIR}/figures/multi_seed_robustness.png", dpi=150)
print("Saved multi_seed_robustness.png")