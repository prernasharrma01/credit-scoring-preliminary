"""Shared utilities for the fairness experiment pipeline."""

import copy
import numpy as np
from sklearn.preprocessing import StandardScaler

NUMERIC_COLS = [
    "duration", "credit_amount", "installment_rate", "present_residence",
    "existing_credits", "num_dependents"
]


def scale_datasets(train, test):
    """
    Fit a StandardScaler on the training set's numeric columns and apply
    it to both train and test (avoids leakage). Returns new deep-copied
    datasets with scaled numeric features; one-hot/binary columns untouched.
    """
    train_s = train.copy(deepcopy=True)
    test_s = test.copy(deepcopy=True)

    feature_names = train.feature_names
    numeric_idx = [feature_names.index(c) for c in NUMERIC_COLS if c in feature_names]

    scaler = StandardScaler()
    train_s.features[:, numeric_idx] = scaler.fit_transform(train.features[:, numeric_idx])
    test_s.features[:, numeric_idx] = scaler.transform(test.features[:, numeric_idx])

    return train_s, test_s


def dominates(o1, o2):
    """
    o1, o2 are dicts of objective_name -> value, ALL ALREADY STANDARDIZED
    such that LOWER IS BETTER for every objective (as in the paper's MOO
    section 4.1: fairness metrics -> abs value / inverse DI; accuracy -> negated).
    Returns True if o1 dominates o2.
    """
    keys = list(o1.keys())
    better_or_equal = all(o1[k] <= o2[k] + 1e-9 for k in keys)
    strictly_better = any(o1[k] < o2[k] - 1e-9 for k in keys)
    return better_or_equal and strictly_better


def standardize_objectives(row):
    """
    Convert a raw metrics row {SPD, DI, EOD, PED, Acc} into a minimization-
    oriented objective vector, following the paper's Section 4.1:
      - SPD, EOD, PED -> abs value (converge to 0)
      - DI -> 1/DI if DI < 1, else DI itself (converge to 1 from both sides)
      - Acc -> negated (since higher accuracy is better, minimize -Acc)
    """
    di = row["DI"]
    di_std = (1.0 / di) if di < 1 and di != 0 else di
    return {
        "SPD": abs(row["SPD"]),
        "DI": di_std,
        "EOD": abs(row["EOD"]),
        "PED": abs(row["PED"]),
        "Acc": -row["Acc"],
    }


def find_non_dominated(results_dict, objective_keys=None):
    """
    results_dict: {technique_name: {SPD, DI, EOD, PED, Acc}}
    Returns list of non-dominated technique names based on standardized objectives.
    """
    std_objs = {name: standardize_objectives(vals) for name, vals in results_dict.items()}
    if objective_keys is not None:
        std_objs = {name: {k: v[k] for k in objective_keys} for name, v in std_objs.items()}

    non_dominated = []
    for name, obj in std_objs.items():
        dominated_by_other = False
        for other_name, other_obj in std_objs.items():
            if name == other_name:
                continue
            if dominates(other_obj, obj):
                dominated_by_other = True
                break
        if not dominated_by_other:
            non_dominated.append(name)
    return non_dominated
