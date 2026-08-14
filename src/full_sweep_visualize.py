import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
with open(f"{BASE_DIR}/outputs/full_sweep_results.json") as f:
    results = json.load(f)

FIG_DIR = os.path.join(BASE_DIR, "figures")


def plot_group(names, title, fname):
    metrics = ["SPD", "DI", "EOD", "PED", "Acc"]
    colors = {"SPD": "#4C72B0", "DI": "#DD8452", "EOD": "#55A868", "PED": "#C44E52", "Acc": "#8172B2"}
    x = np.arange(len(names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.6), 6))
    for i, m in enumerate(metrics):
        vals = [results[n][m] for n in names]
        offset = (i - 2) * width
        bars = ax.bar(x + offset, vals, width, label=m, color=colors[m])
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 3 if v >= 0 else -12),
                        ha="center", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylabel("Metric Value")
    ax.set_title(title)
    ax.legend(title="Metric", loc="upper right", ncol=5, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{fname}", dpi=150)
    plt.close()
    print(f"Saved {fname}")


pi_names = ["Re+EGR", "Re+AL", "DIR+EGR", "DIR+AL"]
plot_group(pi_names, "Full Sweep: Pre + In-processing (PI) models — German Credit", "full_sweep_pi.png")

pip_names = ["Re+EGR+ROC", "Re+EGR+EO", "Re+AL+ROC", "Re+AL+EO",
             "DIR+EGR+ROC", "DIR+EGR+EO", "DIR+AL+ROC", "DIR+AL+EO"]
plot_group(pip_names, "Full Sweep: Pre + In + Post-processing (PIP) models — German Credit", "full_sweep_pip.png")

print("All figures saved.")