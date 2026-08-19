"""
FILE 1 of 3: Runs ONE full model (baseline, or any pre+in, or pre+in+post
combination) for ONE seed, entirely within a single process, and saves its
metrics to disk.

Each process computes exactly one complete model end-to-end, so there's no
need to hand off between separate pre-processing/post-processing scripts
via pickle files anymore -- everything for one model happens in one place.

Isolation between DIFFERENT models is still handled by running this script
fresh via subprocess for every model+seed combination (see run_all.sh) --
that separation is what avoids state leakage between AIF360/TensorFlow
calls, which caused a real training collapse earlier in this project.

Usage:
    python3 run_model.py NONE 1
    python3 run_model.py Re+EGR 1
    python3 run_model.py DIR+AL+EO 1
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
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.inprocessing import ExponentiatedGradientReduction, AdversarialDebiasing
from aif360.algorithms.postprocessing import RejectOptionClassification, EqOddsPostprocessing

from data_loader import preprocess, PRIVILEGED_GROUPS, UNPRIVILEGED_GROUPS
from utils import scale_datasets
from metrics import compute_metrics

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs", "multiseed_results")
os.makedirs(OUT_DIR, exist_ok=True)


def silent(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


def get_train_test(seed):
    dataset, df = preprocess()
    train, test = dataset.split([0.7], shuffle=True, seed=seed)
    return scale_datasets(train, test)


def run_baseline(train, test):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(train.features, train.labels.ravel())
    train_pred = train.copy(deepcopy=True)
    train_pred.labels = lr.predict(train.features).reshape(-1, 1)
    train_pred.scores = lr.predict_proba(train.features)[:, 1].reshape(-1, 1)
    test_pred = test.copy(deepcopy=True)
    test_pred.labels = lr.predict(test.features).reshape(-1, 1)
    test_pred.scores = lr.predict_proba(test.features)[:, 1].reshape(-1, 1)
    return train_pred, test_pred


def apply_pre(pre_name, train, test):
    if pre_name == "Re":
        rw = Reweighing(unprivileged_groups=UNPRIVILEGED_GROUPS, privileged_groups=PRIVILEGED_GROUPS)
        train_pre = rw.fit_transform(train)
        test_pre = test.copy(deepcopy=True)
    elif pre_name == "DIR":
        dir_ = DisparateImpactRemover(repair_level=1.0, sensitive_attribute="age")
        train_pre = dir_.fit_transform(train)
        test_pre = dir_.fit_transform(test)
    else:
        raise ValueError(pre_name)
    return train_pre, test_pre


def run_egr(train, test):
    estimator = LogisticRegression(max_iter=1000)
    egr = ExponentiatedGradientReduction(
        estimator=estimator, constraints="DemographicParity", eps=0.01, max_iter=50, drop_prot_attr=False
    )
    egr.fit(train)
    return egr.predict(train), egr.predict(test)


def run_al(train, test, seed):
    np.random.seed(seed)
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(config=config)
    ad = AdversarialDebiasing(
        privileged_groups=PRIVILEGED_GROUPS, unprivileged_groups=UNPRIVILEGED_GROUPS,
        scope_name="al_scope", sess=sess, num_epochs=50, batch_size=128,
        debias=True, adversary_loss_weight=0.02,
    )
    silent(ad.fit, train)
    train_pred = ad.predict(train)
    test_pred = ad.predict(test)
    sess.close()
    return train_pred, test_pred


def apply_post(post_name, train_true, train_pred, test_pred, seed):
    if post_name == "ROC":
        roc = RejectOptionClassification(
            unprivileged_groups=UNPRIVILEGED_GROUPS, privileged_groups=PRIVILEGED_GROUPS,
            low_class_thresh=0.01, high_class_thresh=0.99,
            num_class_thresh=100, num_ROC_margin=50,
            metric_name="Equal opportunity difference", metric_ub=0.05, metric_lb=-0.05,
        )
        roc = roc.fit(train_true, train_pred)
        return roc.predict(test_pred)
    elif post_name == "EO":
        eo = EqOddsPostprocessing(unprivileged_groups=UNPRIVILEGED_GROUPS, privileged_groups=PRIVILEGED_GROUPS, seed=seed)
        eo = eo.fit(train_true, train_pred)
        return eo.predict(test_pred)
    else:
        raise ValueError(post_name)


def main():
    model_spec = sys.argv[1]
    seed = int(sys.argv[2])

    train, test = get_train_test(seed)

    if model_spec == "NONE":
        train_pred, test_pred = run_baseline(train, test)
        test_true = test
    else:
        parts = model_spec.split("+")
        pre_name, in_name = parts[0], parts[1]
        post_name = parts[2] if len(parts) == 3 else None

        train_pre, test_pre = apply_pre(pre_name, train, test)

        if in_name == "EGR":
            train_pred, test_pred = run_egr(train_pre, test_pre)
        elif in_name == "AL":
            train_pred, test_pred = run_al(train_pre, test_pre, seed)
        else:
            raise ValueError(in_name)

        test_true = test_pre

        if post_name:
            test_final = apply_post(post_name, train_pre, train_pred, test_pred, seed)
            test_pred = test_final

    metrics = compute_metrics(test_true, test_pred, UNPRIVILEGED_GROUPS, PRIVILEGED_GROUPS)
    metrics = {k: float(v) for k, v in metrics.items()}

    safe_name = model_spec.replace("+", "_")
    with open(f"{OUT_DIR}/{safe_name}_seed{seed}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"DONE {model_spec} seed={seed}: Acc={metrics['Acc']:.3f} SPD={metrics['SPD']:.3f}")


if __name__ == "__main__":
    main()