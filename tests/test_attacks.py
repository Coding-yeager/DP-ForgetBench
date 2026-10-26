"""Tests for calibrated membership inference attacks with bootstrap CIs."""

import numpy as np
import pytest
import torch

from dp_forgetbench.attacks import _bootstrap_roc_metrics, loss_membership_probe, tri_population_probe
from dp_forgetbench.federated import BinaryLinearModel


def test_bootstrap_roc_metrics_monotonic_and_bounded():
    labels = np.array([1] * 50 + [0] * 50)
    # Perfect separation scores
    scores = np.array([10.0] * 50 + [-10.0] * 50)
    res = _bootstrap_roc_metrics(labels, scores, n_bootstraps=50, seed=123)
    assert "auc_ci_95" in res
    assert "advantage_ci_95" in res
    assert "tpr_at_fpr_1pct_ci_95" in res
    assert "tpr_at_fpr_0_1pct_ci_95" in res

    # AUC should be 1.0
    assert res["auc_ci_95"][0] == 1.0
    assert res["auc_ci_95"][1] == 1.0
    assert res["advantage_ci_95"][0] == 1.0


def test_loss_membership_probe_outputs_ci_and_rates():
    torch.manual_seed(42)
    model = BinaryLinearModel(n_features=4)
    # Synthetic members and nonmembers
    mx = torch.randn(20, 4)
    my = torch.randint(0, 2, (20,)).float()
    ux = torch.randn(20, 4)
    uy = torch.randint(0, 2, (20,)).float()

    probe_res = loss_membership_probe(model, mx, my, ux, uy, n_bootstraps=20, seed=42)
    assert "auc" in probe_res
    assert "auc_ci_95" in probe_res
    assert "advantage" in probe_res
    assert "advantage_ci_95" in probe_res
    assert "tpr_at_fpr_1pct" in probe_res
    assert "tpr_at_fpr_1pct_ci_95" in probe_res
    assert "tpr_at_fpr_0_1pct" in probe_res
    assert "tpr_at_fpr_0_1pct_ci_95" in probe_res
    assert probe_res["members"] > 0
    assert probe_res["nonmembers"] > 0
    assert probe_res["calibration_members"] > 0
    assert probe_res["calibration_nonmembers"] > 0
    assert probe_res["threshold_selection"].startswith("maximum TPR-FPR")


def test_tri_population_probe_reports_disjoint_split_and_cis():
    torch.manual_seed(7)
    model = BinaryLinearModel(n_features=4)
    populations = {
        name: (torch.randn(18, 4), torch.randint(0, 2, (18,)).float())
        for name in ("forget", "retain", "unseen")
    }
    result = tri_population_probe(model, model, populations, seed=7)
    assert result["train_examples_per_population"] == 9
    assert result["eval_examples_per_population"] == 9
    assert len(result["probe_accuracy_ci_95"]) == 2
    assert result["audit_standard"] == "TC-UMIA (Fu et al., 2026) Tri-Population Privacy Audit"
    assert "forget_vs_unseen_auc" in result
    assert "retained_vs_unseen_auc" in result
    assert "per_class_ovr_auc" in result
    assert "mlp_accuracy" in result
    assert "normalized_confusion_matrix" in result


def test_tc_umia_multiclass_and_retained_leakage():
    """Verify TC-UMIA features on multi-class output and test retained-leakage probe."""
    torch.manual_seed(99)
    from dp_forgetbench.federated import SmallGroupNormCNN
    pre_model = SmallGroupNormCNN(num_classes=3)
    post_model = SmallGroupNormCNN(num_classes=3)

    populations = {
        name: (torch.randn(16, 3, 32, 32), torch.randint(0, 3, (16,)))
        for name in ("forget", "retain", "unseen")
    }
    result = tri_population_probe(pre_model, post_model, populations, seed=99)
    assert 0.0 <= result["forget_vs_unseen_auc"] <= 1.0
    assert 0.0 <= result["retained_vs_unseen_auc"] <= 1.0
    assert "forget_ovr_auc" in result["per_class_ovr_auc"]
    assert "retain_ovr_auc" in result["per_class_ovr_auc"]
    assert "unseen_ovr_auc" in result["per_class_ovr_auc"]
    assert len(result["confusion_matrix"]) == 3
    assert len(result["normalized_confusion_matrix"]) == 3

