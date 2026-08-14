"""
Applies a post-processing technique (ROC or EO) on top of a saved PI
combo's predictions, in its own fresh process.

Usage:
    python3 run_post.py Re_EGR ROC
    python3 run_post.py DIR_AL EO
"""

import sys
import os
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from aif360.algorithms.postprocessing import RejectOptionClassification, EqOddsPostprocessing
from data_loader import PRIVILEGED_GROUPS, UNPRIVILEGED_GROUPS
from metrics import compute_metrics

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs", "full_sweep_pickles")


def apply_roc(train_true, train_pred, test_pred):
    roc = RejectOptionClassification(
        unprivileged_groups=UNPRIVILEGED_GROUPS, privileged_groups=PRIVILEGED_GROUPS,
        low_class_thresh=0.01, high_class_thresh=0.99,
        num_class_thresh=100, num_ROC_margin=50,
        metric_name="Equal opportunity difference", metric_ub=0.05, metric_lb=-0.05
    )
    roc = roc.fit(train_true, train_pred)
    return roc.predict(test_pred)


def apply_eo(train_true, train_pred, test_pred):
    eo = EqOddsPostprocessing(
        unprivileged_groups=UNPRIVILEGED_GROUPS, privileged_groups=PRIVILEGED_GROUPS, seed=42
    )
    eo = eo.fit(train_true, train_pred)
    return eo.predict(test_pred)


def main():
    pi_name, post = sys.argv[1], sys.argv[2]

    with open(f"{OUT_DIR}/{pi_name}.pkl", "rb") as f:
        d = pickle.load(f)

    train_true, train_pred = d["train_true"], d["train_pred"]
    test_true, test_pred = d["test_true"], d["test_pred"]

    if post == "ROC":
        test_final = apply_roc(train_true, train_pred, test_pred)
    elif post == "EO":
        test_final = apply_eo(train_true, train_pred, test_pred)
    else:
        raise ValueError(post)

    metrics = compute_metrics(test_true, test_final, UNPRIVILEGED_GROUPS, PRIVILEGED_GROUPS)
    metrics = {k: float(v) for k, v in metrics.items()}

    name = f"{pi_name}_{post}"
    with open(f"{OUT_DIR}/{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"DONE {name}: {metrics}")


if __name__ == "__main__":
    main()