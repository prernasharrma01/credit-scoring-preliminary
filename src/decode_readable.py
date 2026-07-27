"""
Decodes the raw German Credit dataset (A11, A34, etc. codes) into plain
English, so it can be opened and read directly in Excel/Numbers/VS Code.
This is purely for human readability -- the actual pipeline (data_loader.py)
uses the coded version directly, since that's the standard format.
"""

import pandas as pd

COLUMN_NAMES = [
    "checking_account_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment_since", "installment_rate_pct",
    "personal_status_sex", "other_debtors", "present_residence_years", "property",
    "age", "other_installment_plans", "housing", "existing_credits", "job",
    "num_dependents", "telephone", "foreign_worker", "credit_risk"
]

DECODE_MAPS = {
    "checking_account_status": {
        "A11": "< 0 DM", "A12": "0 to 200 DM", "A13": ">= 200 DM", "A14": "no checking account"
    },
    "credit_history": {
        "A30": "no credits taken / all paid duly", "A31": "all credits at this bank paid duly",
        "A32": "existing credits paid duly till now", "A33": "delay in paying off in the past",
        "A34": "critical account / other credits existing"
    },
    "purpose": {
        "A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment",
        "A43": "radio/television", "A44": "domestic appliances", "A45": "repairs",
        "A46": "education", "A47": "vacation", "A48": "retraining", "A49": "business",
        "A410": "other"
    },
    "savings_account": {
        "A61": "< 100 DM", "A62": "100 to 500 DM", "A63": "500 to 1000 DM",
        "A64": ">= 1000 DM", "A65": "unknown / none"
    },
    "employment_since": {
        "A71": "unemployed", "A72": "< 1 year", "A73": "1 to 4 years",
        "A74": "4 to 7 years", "A75": ">= 7 years"
    },
    "personal_status_sex": {
        "A91": "male: divorced/separated", "A92": "female: divorced/separated/married",
        "A93": "male: single", "A94": "male: married/widowed", "A95": "female: single"
    },
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {
        "A121": "real estate", "A122": "savings agreement / life insurance",
        "A123": "car or other", "A124": "unknown / no property"
    },
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {
        "A171": "unemployed / unskilled non-resident", "A172": "unskilled resident",
        "A173": "skilled employee/official", "A174": "management/self-employed/highly qualified"
    },
    "telephone": {"A191": "none", "A192": "yes, registered"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
    "credit_risk": {1: "good", 2: "bad"},
}

df = pd.read_csv("data/german_credit_raw.csv", header=None, names=COLUMN_NAMES)

df_readable = df.copy()
for col, mapping in DECODE_MAPS.items():
    df_readable[col] = df_readable[col].map(mapping)

df_readable.to_csv("data/german_credit_readable.csv", index=False)
print("Saved human-readable version to data/german_credit_readable.csv")
print(df_readable.head())