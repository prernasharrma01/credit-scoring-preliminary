"""
Data loader for the Statlog German Credit dataset, structured to mirror
the COMPAS setup used in Farayola et al. (2024) so the same fairness
pipeline (Reweighing, DIR, EGR, AL, ROC, EO) can be applied unchanged.

Protected attribute: AGE (privileged = age >= 25, unprivileged = age < 25).
This mirrors AIF360's own built-in GermanDataset convention and is the
standard split used across the credit-scoring fairness literature.

Label convention (AIF360 style):
    favorable_label   = 1.0  -> good credit risk (no default)
    unfavorable_label = 0.0  -> bad credit risk (default)
"""

import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset

RAW_PATH = "/Users/prernasharma/Downloads/preliminary_project/data/german_credit_raw.csv"

COLUMN_NAMES = [
    "checking_account_status", "duration", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment_since", "installment_rate",
    "personal_status_sex", "other_debtors", "present_residence", "property",
    "age", "other_installment_plans", "housing", "existing_credits", "job",
    "num_dependents", "telephone", "foreign_worker", "target"
]

CATEGORICAL_COLS = [
    "checking_account_status", "credit_history", "purpose", "savings_account",
    "employment_since", "personal_status_sex", "other_debtors", "property",
    "other_installment_plans", "housing", "job", "telephone", "foreign_worker"
]

NUMERIC_COLS = [
    "duration", "credit_amount", "installment_rate", "present_residence",
    "existing_credits", "num_dependents"
]

PROTECTED_ATTRIBUTE = "age"
PRIVILEGED_GROUPS = [{"age": 1}]     # age >= 25
UNPRIVILEGED_GROUPS = [{"age": 0}]   # age < 25


def load_raw():
    df = pd.read_csv(RAW_PATH, header=None, names=COLUMN_NAMES)
    return df


def preprocess():
    """
    Returns a BinaryLabelDataset ready for the fairness pipeline, plus
    the underlying pandas dataframe (for inspection/EDA).
    """
    df = load_raw()

    # Label: 1 = good credit (favorable), 2 = bad credit -> recode to 0 (unfavorable)
    df["target"] = df["target"].map({1: 1, 2: 0})

    # Protected attribute: binarize age at 25 (>=25 privileged=1, <25 unprivileged=0)
    df["age"] = df["age"].apply(lambda x: 1 if x >= 25 else 0)

    # One-hot encode categorical columns (excluding the protected attribute 'age')
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    # Ensure all bool dummies become int
    bool_cols = df_encoded.select_dtypes(include="bool").columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    dataset = BinaryLabelDataset(
        df=df_encoded,
        label_names=["target"],
        protected_attribute_names=["age"],
        favorable_label=1.0,
        unfavorable_label=0.0,
    )
    return dataset, df


if __name__ == "__main__":
    dataset, df = preprocess()
    print("Shape:", df.shape)
    print("Target distribution:\n", df["target"].value_counts())
    print("Age group distribution (1=privileged/older>=25, 0=unprivileged/younger<25):")
    print(df["age"].value_counts())
    print("Encoded feature matrix shape:", dataset.features.shape)
