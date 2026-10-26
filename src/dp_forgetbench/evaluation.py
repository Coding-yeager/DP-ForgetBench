"""Phase 0/1/2 evaluation: utility, functional alignment, and transparent cost."""

from __future__ import annotations

import torch
from torch import nn

from .attacks import loss_membership_probe, tri_population_probe
from .federated import TrainCost, evaluate_loss_accuracy, flatten_logits, get_device


def _predictive_distribution(logits: torch.Tensor) -> torch.Tensor:
    """Return an (N, K) categorical distribution for either binary or multi-class logits.

    Binary models emit a single logit per example (K=1 or 1-D); multi-class models
    emit one logit per class. Both are mapped to a shared (N, K>=2) simplex so JS
    divergence and downstream probes work identically for either model family.
    """
    if logits.ndim == 1 or logits.shape[-1] == 1:
        p = torch.sigmoid(logits.squeeze(-1)).clamp(1e-7, 1 - 1e-7)
        return torch.stack([p, 1 - p], dim=1)
    return torch.softmax(logits, dim=-1).clamp(1e-7, 1.0)


def js_divergence_to_target(candidate: nn.Module, target: nn.Module, x: torch.Tensor) -> float:
    """Mean Jensen-Shannon divergence between two models' predictive distributions."""
    p_dist = _predictive_distribution(flatten_logits(candidate, x))
    q_dist = _predictive_distribution(flatten_logits(target, x))
    midpoint = 0.5 * (p_dist + q_dist)
    kl_p = (p_dist * (p_dist.log() - midpoint.log())).sum(dim=1)
    kl_q = (q_dist * (q_dist.log() - midpoint.log())).sum(dim=1)
    return float((0.5 * (kl_p + kl_q)).mean().item())


import numpy as np


def compute_bootstrap_ci(data, stat_func, n_bootstraps=500, ci_level=0.95):
    """Computes bootstrap confidence intervals for a given statistic."""
    data = np.array(data)
    n = len(data)
    if n == 0:
        return 0.0, (0.0, 0.0)
    boot_stats = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(stat_func(sample))
    lower_percentile = (1 - ci_level) / 2 * 100
    upper_percentile = (1 + ci_level) / 2 * 100
    lower_bound = float(np.percentile(boot_stats, lower_percentile))
    upper_bound = float(np.percentile(boot_stats, upper_percentile))
    return float(stat_func(data)), (lower_bound, upper_bound)


def compute_method_mub(
    candidate_metrics: dict,
    dp_metrics: dict,
    all_retrain_metrics: list[dict],
    equivalence_margin: float = 0.02,
) -> dict:
    """Compute Marginal Unlearning Benefit (MUB) and 4-way redundancy classification."""
    dp_acc = dp_metrics["test"]["accuracy"]
    u_acc = candidate_metrics["test"]["accuracy"]
    retrain_accs = [m["test"]["accuracy"] for m in all_retrain_metrics]

    dp_js = dp_metrics["alignment_to_retrain"]["forgotten_js_divergence"]
    u_js = candidate_metrics["alignment_to_retrain"]["forgotten_js_divergence"]
    retrain_js = [m["alignment_to_retrain"]["forgotten_js_divergence"] for m in all_retrain_metrics]

    dp_mia = dp_metrics["attack_diagnostics"]["forgotten_vs_unseen_loss_mia"]["advantage"]
    u_mia = candidate_metrics["attack_diagnostics"]["forgotten_vs_unseen_loss_mia"]["advantage"]
    retrain_mia_advs = [m["attack_diagnostics"]["forgotten_vs_unseen_loss_mia"]["advantage"] for m in all_retrain_metrics]

    retrain_acc_q90 = float(np.percentile(retrain_accs, 90))
    retrain_mia_q90 = float(np.percentile(retrain_mia_advs, 90))
    retrain_mia_median = float(np.median(retrain_mia_advs))

    mub_acc = float(u_acc - dp_acc)
    mub_js = float(dp_js - u_js)
    mub_mia = float(abs(dp_mia - retrain_mia_median) - abs(u_mia - retrain_mia_median))

    def mub_mia_stat(sample_retrain_mia):
        med = np.median(sample_retrain_mia)
        return abs(dp_mia - med) - abs(u_mia - med)

    mub_mia_point, mub_mia_ci = compute_bootstrap_ci(retrain_mia_advs, mub_mia_stat)

    dp_within_acc_var = np.percentile(retrain_accs, 10) <= dp_acc <= np.percentile(retrain_accs, 90)
    dp_within_js_var = dp_js <= np.percentile(retrain_js, 90)
    dp_within_mia_var = np.percentile(retrain_mia_advs, 10) <= dp_mia <= np.percentile(retrain_mia_advs, 90)

    dp_is_redundant_base = bool(dp_within_acc_var and dp_within_js_var and dp_within_mia_var)
    unlearning_adds_benefit = bool(mub_acc > equivalence_margin or mub_mia_point > equivalence_margin)
    unlearning_is_harmful = bool(mub_acc < -equivalence_margin or mub_mia_point < -equivalence_margin)

    if dp_is_redundant_base and not unlearning_adds_benefit and not unlearning_is_harmful:
        classification = "REDUNDANT"
    elif unlearning_adds_benefit:
        classification = "UNLEARNING-BENEFICIAL"
    elif unlearning_is_harmful:
        classification = "UNLEARNING-HARMFUL"
    else:
        classification = "INCONCLUSIVE"

    return {
        "mub_test_accuracy": round(mub_acc, 5),
        "mub_forgotten_js_div": round(mub_js, 5),
        "mub_mia_advantage": round(mub_mia_point, 5),
        "mub_mia_ci_95": [round(mub_mia_ci[0], 5), round(mub_mia_ci[1], 5)],
        "retrain_variability_acc_q90": round(retrain_acc_q90, 5),
        "retrain_variability_mia_q90": round(retrain_mia_q90, 5),
        "classification": classification,
    }


def evaluate_against_target(
    *,
    candidate: nn.Module,
    target: nn.Module,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    forgotten_x: torch.Tensor,
    forgotten_y: torch.Tensor,
    retained_x: torch.Tensor,
    retained_y: torch.Tensor,
    unseen_x: torch.Tensor,
    unseen_y: torch.Tensor,
    pre_deletion_model: nn.Module,
    attack_seed: int,
    cost: TrainCost,
    assumptions: str = "",
    privacy_implications: str = "",
) -> dict:
    test_eval = evaluate_loss_accuracy(candidate, test_x, test_y)
    forgotten_eval = evaluate_loss_accuracy(candidate, forgotten_x, forgotten_y)
    retained_eval = evaluate_loss_accuracy(candidate, retained_x, retained_y)

    test_js = js_divergence_to_target(candidate, target, test_x)
    forgotten_js = js_divergence_to_target(candidate, target, forgotten_x)

    forgotten_mia = loss_membership_probe(candidate, forgotten_x, forgotten_y, unseen_x, unseen_y, seed=attack_seed)
    retained_mia = loss_membership_probe(candidate, retained_x, retained_y, unseen_x, unseen_y, seed=attack_seed + 1)
    tri_probe = tri_population_probe(
        pre_deletion_model,
        candidate,
        {"forget": (forgotten_x, forgotten_y), "retain": (retained_x, retained_y), "unseen": (unseen_x, unseen_y)},
        seed=attack_seed,
    )

    cost_dict = cost.as_dict()

    return {
        # Standard benchmark sections (Phase 8/9 requirements)
        "utility": {
            "test_loss": test_eval["loss"],
            "test_accuracy": test_eval["accuracy"],
            "forgotten_loss": forgotten_eval["loss"],
            "forgotten_accuracy": forgotten_eval["accuracy"],
            "retained_loss": retained_eval["loss"],
            "retained_accuracy": retained_eval["accuracy"],
        },
        "alignment_to_retrain": {
            "test_js_divergence": test_js,
            "forgotten_js_divergence": forgotten_js,
        },
        "forgotten_privacy": forgotten_mia,
        "retained_privacy": retained_mia,
        "unseen_privacy": tri_probe,
        "mia_lira": {
            "loss_mia_advantage": forgotten_mia["advantage"],
            "loss_mia_auc": forgotten_mia["auc"],
            "loss_mia_tpr_1pct_fpr": forgotten_mia["tpr_at_fpr_1pct"],
            "loss_mia_tpr_0_1pct_fpr": forgotten_mia["tpr_at_fpr_0_1pct"],
        },
        "runtime": {
            "wall_clock_seconds": cost_dict.get("runtime_seconds", 0.0),
        },
        "communication": {
            "communicated_bytes": cost_dict.get("communicated_bytes", 0),
        },
        "historical_storage": {
            "persistent_storage_bytes": cost_dict.get("persistent_storage_bytes", 0),
        },
        "assumptions": assumptions,
        "privacy_implications": privacy_implications,

        # Backward compatibility aliases
        "test": test_eval,
        "forgotten_client": forgotten_eval,
        "retained_data": retained_eval,
        "attack_diagnostics": {
            "forgotten_vs_unseen_loss_mia": forgotten_mia,
            "retained_vs_unseen_loss_mia": retained_mia,
            "tri_population_pre_post_probe": tri_probe,
        },
        "cost": cost_dict,
    }
