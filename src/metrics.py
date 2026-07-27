"""
Fairness metrics matching the definitions used in Farayola et al. (2024):
  - SPD: Statistical/Demographic Parity Difference
  - DI:  Disparate Impact
  - EOD: Equal Opportunity Difference
  - PED: Predictive Equality Difference
  - Acc: Accuracy
"""

from aif360.metrics import ClassificationMetric


def compute_metrics(dataset_true, dataset_pred, unprivileged_groups, privileged_groups):
    cm = ClassificationMetric(
        dataset_true, dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )

    # Note: AIF360's built-in metrics all use the (unprivileged - privileged) /
    # (unprivileged / privileged) convention. Cross-checking Farayola et al.'s
    # reported numbers and narrative interpretations against these definitions
    # confirms they used the AIF360 defaults directly (despite their written
    # Eq. 1-4 listing privileged/gi first), so we match that convention exactly
    # for a like-for-like comparison.
    spd = cm.statistical_parity_difference()
    di = cm.disparate_impact()
    eod = cm.equal_opportunity_difference()
    ped = cm.false_positive_rate_difference()
    acc = cm.accuracy()

    return {
        "SPD": spd,
        "DI": di,
        "EOD": eod,
        "PED": ped,
        "Acc": acc,
    }
