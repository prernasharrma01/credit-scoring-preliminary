"""
PRELIMINARY RESULTS — now testing 5 models on German Credit:
  1. NONE       — no fairness intervention (baseline)
  2. DIR only   — pre-processing alone, isolating its own effect
  3. AL only    — in-processing alone (no DIR first), isolating its own effect
  4. EO only    — post-processing alone, correcting the plain baseline's
                  predictions directly (no DIR, no AL)
  5. DIR+AL+EO  — Farayola et al. (2024)'s own headline "best" combination

Models 2-4 isolate each single technique on its own -- building blocks
toward eventually testing the full 13-combination sweep.
"""

import sys
import os
import io
import contextlib
import json
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "42"

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.postprocessing import EqOddsPostprocessing

from data_loader import preprocess, PRIVILEGED_GROUPS, UNPRIVILEGED_GROUPS
from utils import scale_datasets
from metrics import compute_metrics

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def silent(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


def get_train_test():
    dataset, df = preprocess()
    train, test = dataset.split([0.7], shuffle=True, seed=42)
    train, test = scale_datasets(train, test)
    return train, test


def run_baseline(train, test):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(train.features, train.labels.ravel())
    test_pred = test.copy(deepcopy=True)
    test_pred.labels = lr.predict(test.features).reshape(-1, 1)
    test_pred.scores = lr.predict_proba(test.features)[:, 1].reshape(-1, 1)
    train_pred = train.copy(deepcopy=True)
    train_pred.labels = lr.predict(train.features).reshape(-1, 1)
    train_pred.scores = lr.predict_proba(train.features)[:, 1].reshape(-1, 1)
    return train_pred, test_pred


def run_dir_only(train, test):
    """Pre-processing alone: DIR transform, then a plain classifier — no
    in-processing (AL) or post-processing (EO) on top."""
    dir_ = DisparateImpactRemover(repair_level=1.0, sensitive_attribute="age")
    train_dir = dir_.fit_transform(train)
    test_dir = dir_.fit_transform(test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(train_dir.features, train_dir.labels.ravel())
    test_pred = test_dir.copy(deepcopy=True)
    test_pred.labels = lr.predict(test_dir.features).reshape(-1, 1)
    test_pred.scores = lr.predict_proba(test_dir.features)[:, 1].reshape(-1, 1)

    return train_dir, test_dir, test_pred


def run_al_only(train, test):
    """In-processing alone: Adversarial Learning trained directly on the
    plain (un-transformed) data — no DIR first, no EO after. Isolates
    AL's own effect before it's chained with anything else."""
    np.random.seed(42)
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(42)
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(config=config)
    ad = AdversarialDebiasing(
        privileged_groups=PRIVILEGED_GROUPS, unprivileged_groups=UNPRIVILEGED_GROUPS,
        scope_name="al_only_scope", sess=sess, num_epochs=50, batch_size=128,
        debias=True, adversary_loss_weight=0.02,
    )
    silent(ad.fit, train)
    test_pred = ad.predict(test)
    sess.close()
    return test_pred


def run_eo_only(train, test):
    """Post-processing alone: take the plain baseline classifier's own
    predictions (no DIR, no AL) and correct them directly with Equalized
    Odds. Isolates EO's own effect on an otherwise untouched model."""
    train_pred, test_pred = run_baseline(train, test)

    eo = EqOddsPostprocessing(unprivileged_groups=UNPRIVILEGED_GROUPS, privileged_groups=PRIVILEGED_GROUPS, seed=42)
    eo = eo.fit(train, train_pred)
    test_final = eo.predict(test_pred)
    return test_final


def run_dir_al_eo(train, test):
    # --- Pre-processing: Disparate Impact Remover ---
    dir_ = DisparateImpactRemover(repair_level=1.0, sensitive_attribute="age")
    train_dir = dir_.fit_transform(train)
    test_dir = dir_.fit_transform(test)

    # --- In-processing: Adversarial Learning ---
    # NOTE: default adversary_loss_weight=0.1 causes a degenerate collapse on
    # this dataset (small, imbalanced protected group: 149 vs 851). Lowering
    # to 0.02 (confirmed via a small sweep) gives stable, sensible training.
    np.random.seed(42)
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(42)
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(config=config)
    ad = AdversarialDebiasing(
        privileged_groups=PRIVILEGED_GROUPS, unprivileged_groups=UNPRIVILEGED_GROUPS,
        scope_name="al_scope", sess=sess, num_epochs=50, batch_size=128,
        debias=True, adversary_loss_weight=0.02,
    )
    silent(ad.fit, train_dir)
    train_pred = ad.predict(train_dir)
    test_pred = ad.predict(test_dir)
    sess.close()

    # --- Post-processing: Equalized Odds ---
    eo = EqOddsPostprocessing(unprivileged_groups=UNPRIVILEGED_GROUPS, privileged_groups=PRIVILEGED_GROUPS, seed=42)
    eo = eo.fit(train_dir, train_pred)
    test_final = eo.predict(test_pred)

    return train_dir, test_dir, test_final


def main():
    train, test = get_train_test()

    print("Running baseline (NONE)...")
    _, test_pred_none = run_baseline(train, test)
    m_none = compute_metrics(test, test_pred_none, UNPRIVILEGED_GROUPS, PRIVILEGED_GROUPS)

    print("Running DIR only (pre-processing alone)...")
    train_dir_true, test_dir_true, test_dir_pred = run_dir_only(train, test)
    m_dir = compute_metrics(test_dir_true, test_dir_pred, UNPRIVILEGED_GROUPS, PRIVILEGED_GROUPS)

    print("Running AL only (in-processing alone)...")
    test_al_pred = run_al_only(train, test)
    m_al = compute_metrics(test, test_al_pred, UNPRIVILEGED_GROUPS, PRIVILEGED_GROUPS)

    print("Running EO only (post-processing alone)...")
    test_eo_pred = run_eo_only(train, test)
    m_eo = compute_metrics(test, test_eo_pred, UNPRIVILEGED_GROUPS, PRIVILEGED_GROUPS)

    print("Running DIR+AL+EO (paper's headline recommendation)...")
    train_true, test_true, test_final = run_dir_al_eo(train, test)
    m_diralEo = compute_metrics(test_true, test_final, UNPRIVILEGED_GROUPS, PRIVILEGED_GROUPS)

    results = {
        "NONE": m_none, "DIR only": m_dir, "AL only": m_al, "EO only": m_eo,
        "DIR+AL+EO": m_diralEo,
    }
    results = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}

    with open(f"{OUT_DIR}/prelim_results.json", "w") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results).T
    df.to_csv(f"{OUT_DIR}/prelim_results.csv")

    print("\n" + "=" * 70)
    print("PRELIMINARY RESULTS: 5 models (German Credit)")
    print("=" * 70)
    print(df.round(4))
    print("\nSaved to outputs/prelim_results.csv")


if __name__ == "__main__":
    main()