"""Attack diagnostics with fixed, held-out populations.

These diagnostics are intentionally labelled probes. They are valuable Phase 0
sanity checks, but do not substitute for a full LiRA/A-LiRA reproduction on
public datasets.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from torch import nn

from .federated import flatten_logits, get_device


def per_sample_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    """Per-example loss for either a binary (single-logit) or multi-class model."""
    logits = flatten_logits(model, x)
    if logits.ndim == 1 or logits.shape[-1] == 1:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), y.cpu().float(), reduction="none")
    else:
        loss = torch.nn.functional.cross_entropy(logits, y.cpu().long(), reduction="none")
    return loss.numpy()


def _confidence(logits: torch.Tensor) -> torch.Tensor:
    """Scalar confidence signal for attack features: P(true-ish class), model-family agnostic."""
    if logits.ndim == 1 or logits.shape[-1] == 1:
        return torch.sigmoid(logits.squeeze(-1))
    return torch.softmax(logits, dim=-1).max(dim=-1).values


def _balanced_binary_groups(
    member_x: torch.Tensor,
    member_y: torch.Tensor,
    unseen_x: torch.Tensor,
    unseen_y: torch.Tensor,
    limit_per_class: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match membership populations by label before calculating attack scores.

    Iterates over every label actually present in either population, so this
    works for binary (0/1) tasks and multi-class (e.g. 10-way CIFAR-10) tasks
    alike -- a fixed ``(0.0, 1.0)`` label set would silently drop all non-binary
    data instead of raising, which is why this is derived from the data.
    """
    labels_present = torch.unique(torch.cat([member_y, unseen_y])).tolist()
    member_indices, unseen_indices = [], []
    for label in labels_present:
        m = torch.where(member_y == label)[0]
        u = torch.where(unseen_y == label)[0]
        take = min(len(m), len(u), limit_per_class)
        if take == 0:
            continue
        member_indices.append(m[:take])
        unseen_indices.append(u[:take])
    if not member_indices:
        raise ValueError("Attack populations do not have overlapping label support.")
    member_idx = torch.cat(member_indices)
    unseen_idx = torch.cat(unseen_indices)
    return member_x[member_idx], member_y[member_idx], unseen_x[unseen_idx], unseen_y[unseen_idx]


def _bootstrap_roc_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstraps: int = 100,
    seed: int = 42,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    auc_list, adv_list, tpr1_list, tpr01_list = [], [], [], []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        boot_labels = labels[idx]
        if len(np.unique(boot_labels)) < 2:
            continue
        boot_scores = scores[idx]
        try:
            boot_auc = float(roc_auc_score(boot_labels, boot_scores))
            fpr, tpr, _ = roc_curve(boot_labels, boot_scores)
            adv = float(np.max(tpr - fpr))
            def _tpr_at(target_fpr: float) -> float:
                valid = tpr[fpr <= target_fpr]
                return float(valid.max()) if len(valid) else 0.0
            auc_list.append(boot_auc)
            adv_list.append(adv)
            tpr1_list.append(_tpr_at(0.01))
            tpr01_list.append(_tpr_at(0.001))
        except Exception:
            continue
    def _ci(vals: list[float]) -> list[float]:
        if not vals:
            return [0.0, 0.0]
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    return {
        "auc_ci_95": _ci(auc_list),
        "advantage_ci_95": _ci(adv_list),
        "tpr_at_fpr_1pct_ci_95": _ci(tpr1_list),
        "tpr_at_fpr_0_1pct_ci_95": _ci(tpr01_list),
    }


def loss_membership_probe(
    model: nn.Module,
    member_x: torch.Tensor,
    member_y: torch.Tensor,
    unseen_x: torch.Tensor,
    unseen_y: torch.Tensor,
    n_bootstraps: int = 100,
    seed: int = 42,
) -> dict[str, float | int | list[float]]:
    """Held-out loss MIA with label matching and bootstrap confidence intervals.

    The member and unseen populations are split independently, stratified by
    label. ROC/AUC and the attack advantage are reported only on the held-out
    evaluation split. This prevents selecting the best threshold and evaluating
    it on the same examples.
    """
    mx, my, ux, uy = _balanced_binary_groups(member_x, member_y, unseen_x, unseen_y)
    # Lower loss means stronger membership evidence.
    scores = np.concatenate([-per_sample_loss(model, mx, my), -per_sample_loss(model, ux, uy)])
    labels = np.concatenate([np.ones(len(mx), dtype=int), np.zeros(len(ux), dtype=int)])
    calibration_indices = []
    eval_indices = []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        calibration_part, eval_part = train_test_split(indices, test_size=0.5, random_state=seed, stratify=labels[indices])
        calibration_indices.extend(calibration_part.tolist())
        eval_indices.extend(eval_part.tolist())
    calibration_indices = np.asarray(sorted(calibration_indices), dtype=int)
    eval_indices = np.asarray(sorted(eval_indices), dtype=int)
    calibration_labels = labels[calibration_indices]
    calibration_scores = scores[calibration_indices]
    eval_labels = labels[eval_indices]
    eval_scores = scores[eval_indices]
    auc = float(roc_auc_score(eval_labels, eval_scores))
    calibration_fpr, calibration_tpr, calibration_thresholds = roc_curve(calibration_labels, calibration_scores)
    best_index = int(np.argmax(calibration_tpr - calibration_fpr))
    selected_threshold = float(calibration_thresholds[best_index])
    eval_predictions = eval_scores >= selected_threshold
    eval_tpr = float(np.sum(eval_predictions[eval_labels == 1]) / max(1, np.sum(eval_labels == 1)))
    eval_fpr = float(np.sum(eval_predictions[eval_labels == 0]) / max(1, np.sum(eval_labels == 0)))
    advantage = eval_tpr - eval_fpr
    fpr, tpr, _ = roc_curve(eval_labels, eval_scores)
    def tpr_at(target_fpr: float) -> float:
        valid = tpr[fpr <= target_fpr]
        return float(valid.max()) if len(valid) else 0.0
    tpr1 = tpr_at(0.01)
    tpr01 = tpr_at(0.001)
    ci_dict = _bootstrap_roc_metrics(eval_labels, eval_scores, n_bootstraps=n_bootstraps, seed=seed)
    return {
        "auc": auc,
        "auc_ci_95": ci_dict["auc_ci_95"],
        "advantage": advantage,
        "advantage_ci_95": ci_dict["advantage_ci_95"],
        "tpr_at_fpr_1pct": tpr1,
        "tpr_at_fpr_1pct_ci_95": ci_dict["tpr_at_fpr_1pct_ci_95"],
        "tpr_at_fpr_0_1pct": tpr01,
        "tpr_at_fpr_0_1pct_ci_95": ci_dict["tpr_at_fpr_0_1pct_ci_95"],
        "members": int(np.sum(eval_labels == 1)),
        "nonmembers": int(np.sum(eval_labels == 0)),
        "calibration_members": int(len(mx) - np.sum(eval_labels == 1)),
        "calibration_nonmembers": int(len(ux) - np.sum(eval_labels == 0)),
        "selected_threshold": selected_threshold,
        "calibration_advantage": float(np.max(calibration_tpr - calibration_fpr)),
        "threshold_selection": "maximum TPR-FPR on calibration split, applied once to held-out evaluation split",
        "score_definition": "negative per-example cross-entropy loss; higher means more member-like",
        "population_definition": "label-matched member population versus label-matched unseen population",
    }


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def _extract_tc_umia_features(
    pre_model: nn.Module,
    post_model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> np.ndarray:
    """Extract 14-dimensional TC-UMIA (Fu et al., 2026) features per example.

    Fu et al. show that machine unlearning induces distinctive shifts across
    prediction probabilities, entropy, confidence margins, and loss distributions.
    This feature extractor captures:
      1. pre_loss: Cross-entropy loss under M_pre
      2. post_loss: Cross-entropy loss under M_post
      3. loss_delta: post_loss - pre_loss
      4. loss_ratio: (post_loss + 1e-5) / (pre_loss + 1e-5)
      5. pre_prob_true: Probability assigned to true class under M_pre
      6. post_prob_true: Probability assigned to true class under M_post
      7. prob_delta: post_prob_true - pre_prob_true
      8. pre_entropy: Predictive Shannon entropy under M_pre
      9. post_entropy: Predictive Shannon entropy under M_post
      10. entropy_delta: post_entropy - pre_entropy
      11. pre_margin: Top-1 minus Top-2 probability under M_pre
      12. post_margin: Top-1 minus Top-2 probability under M_post
      13. margin_delta: post_margin - pre_margin
      14. total_variation: Total variation distance between M_pre and M_post output distributions
    """
    pre_loss = per_sample_loss(pre_model, x, y)
    post_loss = per_sample_loss(post_model, x, y)
    loss_delta = post_loss - pre_loss
    loss_ratio = (post_loss + 1e-5) / (pre_loss + 1e-5)

    pre_logits = flatten_logits(pre_model, x)
    post_logits = flatten_logits(post_model, x)

    # Determine probability distributions
    if pre_logits.ndim == 1 or pre_logits.shape[-1] == 1:
        p1_pre = torch.sigmoid(pre_logits.squeeze(-1))
        p0_pre = 1.0 - p1_pre
        probs_pre = torch.stack([p0_pre, p1_pre], dim=-1)

        p1_post = torch.sigmoid(post_logits.squeeze(-1))
        p0_post = 1.0 - p1_post
        probs_post = torch.stack([p0_post, p1_post], dim=-1)

        y_bool = y.cpu().bool()
        prob_true_pre = torch.where(y_bool, p1_pre, p0_pre)
        prob_true_post = torch.where(y_bool, p1_post, p0_post)
    else:
        probs_pre = torch.softmax(pre_logits, dim=-1)
        probs_post = torch.softmax(post_logits, dim=-1)
        y_long = y.cpu().long().view(-1, 1)
        prob_true_pre = probs_pre.gather(1, y_long).squeeze(-1)
        prob_true_post = probs_post.gather(1, y_long).squeeze(-1)

    prob_delta = (prob_true_post - prob_true_pre).numpy()
    pre_prob_true_np = prob_true_pre.numpy()
    post_prob_true_np = prob_true_post.numpy()

    # Entropy
    eps = 1e-12
    pre_entropy = (-probs_pre * torch.log(probs_pre + eps)).sum(dim=-1).numpy()
    post_entropy = (-probs_post * torch.log(probs_post + eps)).sum(dim=-1).numpy()
    entropy_delta = post_entropy - pre_entropy

    # Margin (top-1 - top-2)
    sorted_pre = probs_pre.sort(dim=-1, descending=True).values
    pre_margin = (sorted_pre[:, 0] - sorted_pre[:, 1]).numpy()
    sorted_post = probs_post.sort(dim=-1, descending=True).values
    post_margin = (sorted_post[:, 0] - sorted_post[:, 1]).numpy()
    margin_delta = post_margin - pre_margin

    # Total variation distance: 0.5 * sum_k |p_pre,k - p_post,k|
    tv_dist = (0.5 * torch.abs(probs_pre - probs_post).sum(dim=-1)).numpy()

    return np.stack(
        [
            pre_loss,
            post_loss,
            loss_delta,
            loss_ratio,
            pre_prob_true_np,
            post_prob_true_np,
            prob_delta,
            pre_entropy,
            post_entropy,
            entropy_delta,
            pre_margin,
            post_margin,
            margin_delta,
            tv_dist,
        ],
        axis=1,
    )


# Backward-compatible alias
_features = _extract_tc_umia_features


def tri_population_probe(
    pre_model: nn.Module,
    post_model: nn.Module,
    populations: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    seed: int,
) -> dict:
    """Full TC-UMIA (Fu et al., 2026) tri-population audit distinguishing forget, retain, and unseen populations.

    Extracts a 14-dimensional feature vector across prediction losses, probability
    deltas, predictive entropy, confidence margins, and total variation distance.
    Evaluates linear and non-linear (MLP) probes on held-out data to audit:
      1. Forgotten-set removal efficacy (Class 0: forget)
      2. Retained-set privacy leakage (Class 1: retain)
      3. Calibration against unseen non-members (Class 2: unseen)
    """
    names = ("forget", "retain", "unseen")
    if set(populations) != set(names):
        raise ValueError(f"Expected exactly {names} populations.")
    limit = min(len(populations[name][1]) for name in names)
    if limit < 12:
        raise ValueError("Need at least 12 examples in each tri-class population.")
    features, labels = [], []
    for class_id, name in enumerate(names):
        x, y = populations[name]
        features.append(_extract_tc_umia_features(pre_model, post_model, x[:limit], y[:limit]))
        labels.append(np.full(limit, class_id, dtype=int))
    x_all = np.concatenate(features)
    y_all = np.concatenate(labels)

    x_train, x_eval, y_train, y_eval = train_test_split(
        x_all, y_all, test_size=0.5, random_state=seed, stratify=y_all
    )

    # Scale features for numerical stability
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_eval_scaled = scaler.transform(x_eval)

    # Fit probes with ConvergenceWarning cleanly suppressed for quick diagnostic steps
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        # 1. Primary Linear Logistic Probe (calibrated reference)
        probe = LogisticRegression(max_iter=500, random_state=seed)
        probe.fit(x_train_scaled, y_train)
        predicted = probe.predict(x_eval_scaled)
        probabilities = probe.predict_proba(x_eval_scaled)

        # 2. Non-linear Multi-Layer Perceptron (MLP) Probe
        mlp_probe = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=seed)
        try:
            mlp_probe.fit(x_train_scaled, y_train)
            mlp_predicted = mlp_probe.predict(x_eval_scaled)
            mlp_probs = mlp_probe.predict_proba(x_eval_scaled)
            mlp_acc = float((mlp_predicted == y_eval).mean())
            mlp_auc = float(roc_auc_score(y_eval, mlp_probs, multi_class="ovr", average="macro"))
        except Exception:
            mlp_acc = float((predicted == y_eval).mean())
            mlp_auc = float(roc_auc_score(y_eval, probabilities, multi_class="ovr", average="macro"))

    # Bootstrap 95% Confidence Intervals
    rng = np.random.default_rng(seed)
    bootstrap_accuracy = []
    bootstrap_auc = []
    for _ in range(100):
        indices = rng.choice(len(y_eval), size=len(y_eval), replace=True)
        if len(np.unique(y_eval[indices])) < 3:
            continue
        bootstrap_accuracy.append(float((predicted[indices] == y_eval[indices]).mean()))
        bootstrap_auc.append(
            float(roc_auc_score(y_eval[indices], probabilities[indices], multi_class="ovr", average="macro"))
        )

    def _ci(values: list[float]) -> list[float]:
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values else [0.0, 0.0]

    # Per-class One-vs-Rest AUCs
    per_class_auc = {}
    for cid, cname in enumerate(names):
        binary_y = (y_eval == cid).astype(int)
        if len(np.unique(binary_y)) == 2:
            per_class_auc[f"{cname}_ovr_auc"] = float(roc_auc_score(binary_y, probabilities[:, cid]))
        else:
            per_class_auc[f"{cname}_ovr_auc"] = 0.5

    # Pairwise Leakage Probes (Fu et al. key metrics)
    # Forgotten vs Unseen (Class 0 vs Class 2)
    fu_mask = (y_eval == 0) | (y_eval == 2)
    if len(np.unique(y_eval[fu_mask])) == 2:
        # Score difference between P(forget) and P(unseen)
        fu_scores = probabilities[fu_mask, 0] - probabilities[fu_mask, 2]
        fu_labels = (y_eval[fu_mask] == 0).astype(int)
        forget_vs_unseen_auc = float(roc_auc_score(fu_labels, fu_scores))
    else:
        forget_vs_unseen_auc = 0.5

    # Retained vs Unseen (Class 1 vs Class 2 - Retained Set Leakage!)
    ru_mask = (y_eval == 1) | (y_eval == 2)
    if len(np.unique(y_eval[ru_mask])) == 2:
        ru_scores = probabilities[ru_mask, 1] - probabilities[ru_mask, 2]
        ru_labels = (y_eval[ru_mask] == 1).astype(int)
        retained_vs_unseen_auc = float(roc_auc_score(ru_labels, ru_scores))
    else:
        retained_vs_unseen_auc = 0.5

    cm = confusion_matrix(y_eval, predicted, labels=[0, 1, 2])
    cm_norm = (cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)).tolist()

    return {
        "audit_standard": "TC-UMIA (Fu et al., 2026) Tri-Population Privacy Audit",
        "probe_accuracy": float((predicted == y_eval).mean()),
        "macro_ovr_auc": float(roc_auc_score(y_eval, probabilities, multi_class="ovr", average="macro")),
        "probe_accuracy_ci_95": _ci(bootstrap_accuracy),
        "macro_ovr_auc_ci_95": _ci(bootstrap_auc),
        "mlp_accuracy": mlp_acc,
        "mlp_macro_ovr_auc": mlp_auc,
        "confusion_matrix": cm.tolist(),
        "normalized_confusion_matrix": cm_norm,
        "per_class_ovr_auc": per_class_auc,
        "forget_vs_unseen_auc": forget_vs_unseen_auc,
        "retained_vs_unseen_auc": retained_vs_unseen_auc,
        "examples_per_population": int(limit),
        "train_examples_per_population": int(limit // 2),
        "eval_examples_per_population": int(limit - limit // 2),
        "population_labels": {"0": "forget", "1": "retain", "2": "unseen"},
        "feature_definition": (
            "TC-UMIA 14-dim feature vector: [pre_loss, post_loss, loss_delta, loss_ratio, "
            "pre_prob_true, post_prob_true, prob_delta, pre_entropy, post_entropy, "
            "entropy_delta, pre_margin, post_margin, margin_delta, total_variation]"
        ),
        "split_definition": "stratified 50/50 train/evaluation split, random_state=seed",
    }
