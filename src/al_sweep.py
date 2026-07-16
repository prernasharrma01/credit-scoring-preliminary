"""
Hyperparameter sweep: shows exactly why AdversarialDebiasing's default
setting (adversary_loss_weight=0.1) collapsed on this dataset, and how
lowering it fixed the problem.
"""

import sys, os, io, contextlib, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.inprocessing import AdversarialDebiasing
from data_loader import preprocess, PRIVILEGED_GROUPS, UNPRIVILEGED_GROUPS
from utils import scale_datasets

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def silent(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


def get_train_test():
    dataset, df = preprocess()
    train, test = dataset.split([0.7], shuffle=True, seed=42)
    return scale_datasets(train, test)


def run_al(train, test, adversary_loss_weight):
    np.random.seed(42)
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(42)
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(config=config)
    ad = AdversarialDebiasing(
        privileged_groups=PRIVILEGED_GROUPS, unprivileged_groups=UNPRIVILEGED_GROUPS,
        scope_name="sweep_scope", sess=sess, num_epochs=50, batch_size=128,
        debias=True, adversary_loss_weight=adversary_loss_weight,
    )
    silent(ad.fit, train)
    test_pred = ad.predict(test)
    sess.close()
    return test_pred


def main():
    train, test = get_train_test()
    dir_ = DisparateImpactRemover(repair_level=1.0, sensitive_attribute="age")
    train_dir = dir_.fit_transform(train)
    test_dir = dir_.fit_transform(test)

    weights = [0.1, 0.05, 0.02, 0.01, 0.005]
    rows = []
    for w in weights:
        print(f"Testing adversary_loss_weight={w} ...")
        test_pred = run_al(train_dir, test_dir, w)
        acc = float((test_pred.labels.ravel() == test_dir.labels.ravel()).mean())
        favorable_rate = float((test_pred.labels.ravel() == 1).mean())
        rows.append({"adversary_loss_weight": w, "accuracy": acc, "pct_predicted_favorable": favorable_rate})
        print(f"  -> accuracy={acc:.3f}, % predicted favorable={favorable_rate:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/al_hyperparameter_sweep.csv", index=False)
    print("\n" + "=" * 60)
    print(df.round(3))
    print(f"\nSaved to {OUT_DIR}/al_hyperparameter_sweep.csv")


if __name__ == "__main__":
    main()