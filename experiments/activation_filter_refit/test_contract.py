"""Fitting and testing must retain every task variant on its frozen side."""

import pytest

from experiments.activation_filter_refit.run import check_groups


def test_fit_test_membership_rejects_variant_leakage_and_duplicate_ids() -> None:
    fit = [{"id": "f1", "metadata": {"task_id": "f"}}]
    test = [{"id": "t1", "metadata": {"task_id": "t"}}]
    partition = {"fit_tasks": ["f"], "test_tasks": ["t"]}
    check_groups(fit, test, partition)
    with pytest.raises(ValueError, match="leakage"):
        check_groups(
            fit, test + [{"id": "f2", "metadata": {"task_id": "f"}}], partition
        )
    with pytest.raises(ValueError, match="duplicate"):
        check_groups(fit + fit, test, partition)
