import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
df = pd.read_csv(f"{BASE_DIR}/outputs/al_hyperparameter_sweep.csv")

fig, ax1 = plt.subplots(figsize=(8, 5.5))

x = df["adversary_loss_weight"].astype(str)
ax1.plot(x, df["accuracy"], marker="o", color="#4C72B0", linewidth=2, label="Accuracy")
ax1.set_xlabel("adversary_loss_weight (AIF360 default = 0.1)")
ax1.set_ylabel("Accuracy", color="#4C72B0")
ax1.tick_params(axis="y", labelcolor="#4C72B0")
ax1.set_ylim(0, 1)

for xi, yi in zip(x, df["accuracy"]):
    ax1.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

ax1.axvspan(-0.5, 0.5, color="#C44E52", alpha=0.08)
ax1.annotate("Default setting\ncollapses here", xy=(0, 0.33), xytext=(0.6, 0.15),
             fontsize=9, color="#C44E52", ha="center")

ax1.set_title("Adversarial Learning: hyperparameter sweep\n(finding a stable adversary_loss_weight)")
plt.tight_layout()
plt.savefig(f"{BASE_DIR}/figures/al_sweep_chart.png", dpi=150)
print("Saved al_sweep_chart.png")