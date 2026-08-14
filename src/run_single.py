"""
Runs exactly ONE model (baseline, or one PI combo) in a completely fresh
Python process, and pickles the resulting (train_true, train_pred, test_true,
test_pred) datasets plus its metrics to disk.

Isolation is necessary: running multiple AIF360/TensorFlow techniques back
to back in the SAME process can leave residual state that occasionally
causes a later technique to collapse. A fresh process per combo avoids this
entirely (confirmed during earlier debugging of this exact project).

Usage:
    python3 run_single.py NONE
    python3 run_single.py Re EGR
    python3 run_single.py DIR AL
"""

import sys
import os
import io
import contextlib
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "42"

import numpy as np
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.inprocessing import ExponentiatedGradientReduction, AdversarialDebiasing

from data_loader import preprocess, PRIVILEGED_GROUPS, UNPRIVILEGED_GROUPS
from utils import scale_datasets
from metrics import compute_metrics

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs", "full_sweep_pickles")
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


def apply_reweighing(train):
    rw = Reweighing(unprivileged_groups=UNPRIVILEGED_GROUPS, privileged_groups=PRIVILEGED_GROUPS)
    return rw.fit_transform(train)


def apply_dir(train, test, repair_level=1.0):
    dir_ = DisparateImpactRemover(repair_level=repair_level, sensitive_attribute="age")
    train_dir = dir_.fit_transform(train)
    test_dir = dir_.fit_transform(test)
    return train_dir, test_dir


def run_egr(train, test):
    estimator = LogisticRegression(max_iter=1000)
    egr = ExponentiatedGradientReduction(
        estimator=estimator, constraints="DemographicParity", eps=0.01, max_iter=50, drop_prot_attr=False
    )
    egr.fit(train)
    return egr.predict(train), egr.predict(test)


def run_al(train, test):
    np.random.seed(42)
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(42)
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(config=config)
    ad = AdversarialDebiasing(
        privileged_groups=PRIVILEGED_GROUPS, unprivileged_groups=UNPRIVILEGED_GROUPS,
        scope_name="al_scope", sess=sess, num_epochs=50, batch_size=128,
        debias=True, adversary_loss_weight=0.02,  # fixed value -- default 0.1 collapses on this dataset
    )
    silent(ad.fit, train)
    train_pred = ad.predict(train)
    test_pred = ad.predict(test)
    sess.close()
    return train_pred, test_pred


def main():
    args = sys.argv[1:]
    train, test = get_train_test()

    if args[0] == "NONE":
        train_pred, test_pred = run_baseline(train, test)
        train_true, test_true = train, test
        name = "NONE"
    else:
        pre, inp = args[0], args[1]
        name = f"{pre}_{inp}"

        if pre == "Re":
            train_pre = apply_reweighing(train)
            test_pre = test.copy(deepcopy=True)
        elif pre == "DIR":
            train_pre, test_pre = apply_dir(train, test, repair_level=1.0)
        else:
            raise ValueError(pre)

        if inp == "EGR":
            train_pred, test_pred = run_egr(train_pre, test_pre)
        elif inp == "AL":
            train_pred, test_pred = run_al(train_pre, test_pre)
        else:
            raise ValueError(inp)

        train_true, test_true = train_pre, test_pre

    metrics = compute_metrics(test_true, test_pred, UNPRIVILEGED_GROUPS, PRIVILEGED_GROUPS)
    metrics = {k: float(v) for k, v in metrics.items()}

    with open(f"{OUT_DIR}/{name}.pkl", "wb") as f:
        pickle.dump({
            "train_true": train_true, "train_pred": train_pred,
            "test_true": test_true, "test_pred": test_pred,
        }, f)

    with open(f"{OUT_DIR}/{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"DONE {name}: {metrics}")


if __name__ == "__main__":
    main()