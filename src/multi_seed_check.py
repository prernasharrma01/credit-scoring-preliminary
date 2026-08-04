"""
Multi-seed robustness check.

The single-split preliminary results are vulnerable to a real criticism:
with only ~45 unprivileged people in the 30% test set, a metric like SPD
can shift noticeably just from WHICH people happened to land in the test
set on one particular split -- not from real model behavior.

This script re-runs all 3 models (NONE, DIR only, DIR+AL+EO) across many
different random train/test splits, and reports the MEAN and STANDARD
DEVIATION of each metric across splits, instead of a single number from
one split. If a pattern (e.g. "DIR only beats DIR+AL+EO") holds up
consistently across many splits, it's a real, defensible finding. If it
flips around between splits, that itself is an important, honest result.
"""

import sys, os, io, contextlib, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
os.environ["TF_DETERMINISTIC_OPS"] = "1"

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

N_SEEDS = 15  # number of different random train/test splits to test
SEEDS = list(range(1, N_SEEDS + 1))  # seeds 1 through 15


def silent(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


def get_train_test(split_seed):
    dataset, df = preprocess()
    train, test = dataset.split([0.7], shuffle=True, seed=split_seed)
    return scale_datasets(train, test)


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
    dir_ = DisparateImpactRemover(repair_level=1.0, sensitive_attribute="age")
    train_dir = dir_.fit_transform(train)
    test_dir = dir_.fit_transform(test)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(train_dir.features, train_dir.labels.ravel())
    test_pred = test_dir.copy(deepcopy=True)
    test_pred.labels = lr.predict(test_dir.features).reshape(-1, 1)
    test_pred.scores = lr.predict_proba(test_dir.features)[:, 1].reshape(-1, 1)
    return train_dir, test_dir, test_pred


def run_dir_al_eo(train, test, tf_seed):
    dir_ = DisparateImpactRemover(repair_level=1.0, sensitive_attribute="age")
    train_dir = dir_.fit_transform(train)
    test_dir = dir_.fit_transform(test)

    np.random.seed(tf_seed)
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(tf_seed)
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

    eo = EqOddsPostprocessing(unprivileged_groups=UNPRIVILEGED_GROUPS, privileged_groups=PRIVILEGED_GROUPS, seed=tf_seed)
    eo = eo.fit(train_dir, train_pred)
    test_final = eo.predict(test_pred)
    return test_final, test_dir


def main():
    all_results = {"NONE": [], "DIR only": [], "DIR+AL+EO": []}

    for i, seed in enumerate(SEEDS):
        print(f"\n[{i+1}/{N_SEEDS}] Running split seed={seed} ...")
        train, test = get_train_test(seed)

        _, test_pred_none = run_baseline(train, test)
        m_none = compute_metrics(test, test_pred_none, UNPRIVILEGED_GROUPS, PRIVILEGED_GROUPS)
        all_results["NONE"].append(m_none)
        print(f"  NONE:      Acc={m_none['Acc']:.3f}  SPD={m_none['SPD']:.3f}")

        _, test_dir_true, test_dir_pred = run_dir_only(train, test)
        m_dir = compute_metrics(test_dir_true, test_dir_pred, UNPRIVILEGED_GROUPS, PRIVILEGED_GROUPS)
        all_results["DIR only"].append(m_dir)
        print(f"  DIR only:  Acc={m_dir['Acc']:.3f}  SPD={m_dir['SPD']:.3f}")

        test_final, test_dir_true2 = run_dir_al_eo(train, test, tf_seed=seed)
        m_full = compute_metrics(test_dir_true2, test_final, UNPRIVILEGED_GROUPS, PRIVILEGED_GROUPS)
        all_results["DIR+AL+EO"].append(m_full)
        print(f"  DIR+AL+EO: Acc={m_full['Acc']:.3f}  SPD={m_full['SPD']:.3f}")

    summary_rows = {}
    raw_rows = []
    for model_name, runs in all_results.items():
        df_runs = pd.DataFrame(runs)
        for seed, row in zip(SEEDS, runs):
            raw_rows.append({"model": model_name, "seed": seed, **row})
        summary_rows[model_name] = {}
        for metric in ["SPD", "DI", "EOD", "PED", "Acc"]:
            mean = df_runs[metric].mean()
            std = df_runs[metric].std()
            summary_rows[model_name][f"{metric}_mean"] = round(mean, 4)
            summary_rows[model_name][f"{metric}_std"] = round(std, 4)

    summary_df = pd.DataFrame(summary_rows).T
    raw_df = pd.DataFrame(raw_rows)

    summary_df.to_csv(f"{OUT_DIR}/multi_seed_summary.csv")
    raw_df.to_csv(f"{OUT_DIR}/multi_seed_raw.csv", index=False)

    print("\n" + "=" * 90)
    print(f"MULTI-SEED ROBUSTNESS CHECK ({N_SEEDS} random train/test splits)")
    print("=" * 90)
    pd.set_option("display.width", 200)
    print(summary_df)
    print(f"\nSaved: {OUT_DIR}/multi_seed_summary.csv (aggregated)")
    print(f"Saved: {OUT_DIR}/multi_seed_raw.csv (every individual run)")


if __name__ == "__main__":
    main()