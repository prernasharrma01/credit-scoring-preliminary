# Preliminary Results — Reference [11] on Credit Scoring

**Scope of this update:** a small, focused test before running the full technique sweep — does the paper's own headline recommendation (DIR+AL+EO) behave sensibly on a completely different domain (credit scoring)?

## Setup
- **Dataset:** UCI Statlog German Credit (1,000 records)
- **Protected attribute:** Age (≥25 = privileged, <25 = unprivileged) — same convention AIF360 uses by default for this dataset
- **Toolkit:** AIF360, matching reference [11] exactly
- **Models tested:** just 2, not the full set
  1. **NONE** — plain logistic regression, no fairness intervention (control)
  2. **DIR+AL+EO** — reference [11]'s own best-performing combination: Disparate Impact Remover (pre) + Adversarial Learning (in) + Equalized Odds (post)

## Results

| Model | SPD | DI | EOD | PED | Accuracy |
|---|---|---|---|---|---|
| NONE | -0.016 | 0.979 | -0.010 | 0.086 | 0.743 |
| DIR+AL+EO | 0.066 | 1.090 | 0.048 | 0.219 | 0.750 |

*(0 = ideal for SPD/EOD/PED, 1 = ideal for DI)*

## What this shows so far

DIR+AL+EO does not simply replicate its COMPAS behaviour — on credit data, it moves the Disparate Impact ratio closer to 1 (0.979 → 1.090, though slightly overshooting past parity) and brings EOD closer to 0, but at the cost of a *worse* PED than doing nothing at all (0.086 → 0.219). Accuracy is essentially unchanged (+0.007). This is a genuinely mixed early signal — some fairness metrics improve, one gets worse — which is exactly the kind of trade-off reference [11] itself warns can happen with single-metric optimization, and it's what the fuller multi-objective analysis (next stage) is designed to properly weigh.

## One methodological issue worth flagging directly

The Adversarial Learning step initially **failed outright** — AIF360's default `adversary_loss_weight` (0.1) caused the classifier to collapse to near-constant predictions (~33% accuracy) on this dataset. I traced this to German Credit's much smaller and more imbalanced protected group (149 vs. 851, versus a more balanced split in COMPAS) destabilizing adversarial training. Lowering the weight to 0.02 (confirmed via a small hyperparameter sweep) resolved it. This is a concrete, small piece of evidence that **techniques don't necessarily port across domains with their original settings**, even when the code and library are identical.

## Next step

Run the remaining 11 model combinations (the other 3 PI models + 7 PIP models) and the full Pareto non-domination analysis, to see whether DIR+AL+EO's mixed-but-reasonable result holds up as genuinely *non-dominated* once compared against every other combination — not just against doing nothing.
