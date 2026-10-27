"""TOST (Two One-Sided Tests) equivalence testing for unlearning redundancy.

Implements the pre-registered-style statistical procedure recommended in the AI
agent's review. Replaces the informal 4-way band rule with bootstrap-based
confidence intervals tested against explicit equivalence margins.

Reference: Lakens (2017) "Equivalence Tests: A Practical Primer for t-Tests,
Correlations, and Meta-Analyses". Adapted for paired seed designs.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


# ---------------------------------------------------------------------------
# Core bootstrap TOST
# ---------------------------------------------------------------------------

def tost_paired(
    d: Sequence[float],
    delta: float,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """TOST equivalence test from paired per-seed differences.

    Args:
        d:      Per-seed paired differences (e.g. Acc(M_DP) - Acc(M_R),
                or MUB values). One value per independent seed.
        delta:  The pre-specified equivalence margin. Equivalence is declared
                iff the 90% bootstrap CI is strictly inside (-delta, +delta).
        n_boot: Bootstrap resampling iterations.
        alpha:  One-sided significance level (90% CI ↔ alpha=0.05).
        rng:    Optional numpy Generator for reproducibility.

    Returns:
        dict with keys:
          - mean, sd                 : point estimates of d
          - ci90_low, ci90_high      : 90% bootstrap CI (for TOST)
          - ci95_low, ci95_high      : 95% bootstrap CI (for reporting)
          - delta                    : the margin used
          - equivalent               : True iff CI90 ⊂ (-delta, delta)
          - n_seeds                  : number of seeds
          - interpretation           : human-readable conclusion
    """
    if rng is None:
        rng = np.random.default_rng(0)
    arr = np.asarray(d, dtype=float)
    n = len(arr)
    if n < 2:
        raise ValueError("At least 2 independent seeds required for TOST.")

    boots = np.array([
        rng.choice(arr, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])

    ci90_lo, ci90_hi = float(np.percentile(boots, alpha * 100)), float(np.percentile(boots, (1 - alpha) * 100))
    ci95_lo, ci95_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    equivalent = (-delta < ci90_lo) and (ci90_hi < delta)

    if equivalent:
        interp = f"EQUIVALENT: 90% CI [{ci90_lo:.4f}, {ci90_hi:.4f}] inside (+/-{delta:.4f}). Effect is negligible."
    elif ci90_lo > 0:
        interp = f"BENEFICIAL: 90% CI [{ci90_lo:.4f}, {ci90_hi:.4f}] entirely above 0. Unlearning improves over DP-only."
    elif ci90_hi < 0:
        interp = f"HARMFUL: 90% CI [{ci90_lo:.4f}, {ci90_hi:.4f}] entirely below 0. Unlearning degrades relative to DP-only."
    else:
        interp = f"INCONCLUSIVE: 90% CI [{ci90_lo:.4f}, {ci90_hi:.4f}] overlaps 0 and/or margin +/-{delta:.4f}. More seeds needed."

    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "ci90_low": ci90_lo,
        "ci90_high": ci90_hi,
        "ci95_low": ci95_lo,
        "ci95_high": ci95_hi,
        "delta": delta,
        "equivalent": equivalent,
        "n_seeds": n,
        "interpretation": interp,
    }


def tost_nested(
    seed_example_diffs: list[list[float]],
    delta: float,
    n_outer: int = 5_000,
    n_inner: int = 5_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """Nested two-level bootstrap TOST for E3 (AUC per example within seeds).

    E3 (TC-UMIA AUC) requires resampling seeds first, then examples within
    each seed, because examples within one model evaluation are not i.i.d.

    Args:
        seed_example_diffs: List of per-seed lists of per-example AUC scores.
        delta: Equivalence margin (typically 0.05 AUC).
        n_outer: Outer seed resamples.
        n_inner: Inner example resamples per seed.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_seeds = len(seed_example_diffs)
    if n_seeds < 2:
        raise ValueError("At least 2 seeds required for nested bootstrap.")

    outer_means = []
    for _ in range(n_outer):
        # Resample seeds
        sampled_seed_indices = rng.choice(n_seeds, size=n_seeds, replace=True)
        inner_seed_means = []
        for si in sampled_seed_indices:
            examples = np.asarray(seed_example_diffs[si])
            # Resample examples within that seed
            inner_mean = rng.choice(examples, size=len(examples), replace=True).mean()
            inner_seed_means.append(inner_mean)
        outer_means.append(float(np.mean(inner_seed_means)))

    outer_means_arr = np.asarray(outer_means)
    ci90_lo = float(np.percentile(outer_means_arr, alpha * 100))
    ci90_hi = float(np.percentile(outer_means_arr, (1 - alpha) * 100))
    ci95_lo = float(np.percentile(outer_means_arr, 2.5))
    ci95_hi = float(np.percentile(outer_means_arr, 97.5))

    equivalent = (-delta < ci90_lo) and (ci90_hi < delta)
    grand_mean = float(outer_means_arr.mean())

    return {
        "grand_mean": grand_mean,
        "ci90_low": ci90_lo,
        "ci90_high": ci90_hi,
        "ci95_low": ci95_lo,
        "ci95_high": ci95_hi,
        "delta": delta,
        "equivalent": equivalent,
        "n_seeds": n_seeds,
        "interpretation": (
            f"EQUIVALENT (nested): 90% CI [{ci90_lo:.4f}, {ci90_hi:.4f}] ⊂ (±{delta:.4f})"
            if equivalent
            else f"NOT EQUIVALENT (nested): 90% CI [{ci90_lo:.4f}, {ci90_hi:.4f}]"
        ),
    }


# ---------------------------------------------------------------------------
# Multi-endpoint intersection-union decision
# ---------------------------------------------------------------------------

def intersection_union_decision(
    e1_result: dict,  # Utility TOST
    e2_result: dict,  # JS alignment TOST
    e3_result: dict,  # AUC privacy TOST
) -> dict:
    """Intersection-union test over E1, E2, E3 for the 4-way decision.

    REDUNDANT requires ALL three endpoints to pass equivalence.
    No multiplicity correction needed across endpoints for IUT (Berger & Hsu).
    Consumers should apply Holm or B-Y correction across cells/methods.

    Args:
        e1_result: Output of tost_paired() for utility (accuracy difference).
        e2_result: Output of tost_paired() for JS alignment (MUB_JS).
        e3_result: Output of tost_paired() or tost_nested() for MIA AUC.

    Returns:
        dict with:
          - verdict: REDUNDANT | UNLEARNING-BENEFICIAL | UNLEARNING-HARMFUL | INCONCLUSIVE
          - endpoint_results: dict of individual endpoint verdicts
          - rationale: human-readable explanation
    """
    e1_eq = e1_result.get("equivalent", False)
    e2_eq = e2_result.get("equivalent", False)
    e3_eq = e3_result.get("equivalent", False)

    # Check for beneficial (90% CI strictly above 0 on JS alignment = getting closer to retrain)
    e2_beneficial = e2_result.get("ci90_low", 0.0) > 0
    e3_beneficial = e3_result.get("ci90_low", 0.0) > 0

    # Check for harmful (90% CI strictly below 0 on utility = getting worse)
    e1_harmful = e1_result.get("ci90_high", 0.0) < 0

    if e1_eq and e2_eq and e3_eq:
        verdict = "REDUNDANT"
        rationale = "All three endpoints (utility, JS alignment, privacy AUC) are statistically equivalent to DP-only. Explicit unlearning provides no measurable benefit."
    elif e2_beneficial and e3_beneficial and not e1_harmful:
        verdict = "UNLEARNING-BENEFICIAL"
        rationale = "Unlearning method significantly improves JS alignment and privacy AUC over DP-only without degrading utility."
    elif e1_harmful:
        verdict = "UNLEARNING-HARMFUL"
        rationale = "Unlearning method significantly degrades utility (E1 CI strictly below 0). Not recommended."
    else:
        verdict = "INCONCLUSIVE"
        failing = [k for k, v in [("E1-utility", e1_eq), ("E2-JS", e2_eq), ("E3-AUC", e3_eq)] if not v]
        rationale = f"Cannot declare equivalence. Failing endpoints: {', '.join(failing)}. Consider more seeds or tighter delta."

    return {
        "verdict": verdict,
        "rationale": rationale,
        "endpoint_results": {
            "e1_utility": {
                "equivalent": e1_eq,
                "delta": e1_result.get("delta"),
                "ci90": [e1_result.get("ci90_low"), e1_result.get("ci90_high")],
                "mean_diff": e1_result.get("mean"),
            },
            "e2_js_alignment": {
                "equivalent": e2_eq,
                "delta": e2_result.get("delta"),
                "ci90": [e2_result.get("ci90_low"), e2_result.get("ci90_high")],
                "mean_mub_js": e2_result.get("mean"),
            },
            "e3_privacy_auc": {
                "equivalent": e3_eq,
                "delta": e3_result.get("delta"),
                "ci90": [e3_result.get("ci90_low"), e3_result.get("ci90_high")],
                "mean_auc_excess": e3_result.get("mean"),
            },
        },
    }


# ---------------------------------------------------------------------------
# Power estimation helper
# ---------------------------------------------------------------------------

def estimate_seeds_needed(
    effect_size: float,
    delta: float,
    sigma: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Approximate seed count for TOST to achieve target power.

    Based on the paired t-test approximation: n ≈ (z_α + z_β)² * σ² / (δ - |μ|)².
    Safe for planning only — use simulation for final power analysis.
    """
    from scipy import stats
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    margin = delta - abs(effect_size)
    if margin <= 0:
        return float("inf")
    n = ((z_alpha + z_beta) ** 2 * sigma ** 2) / (margin ** 2)
    return max(2, int(np.ceil(n)))
