"""Print a concise, formatted summary of the latest experimental run.

Usage:
  python scripts/print_run_summary.py
  python scripts/print_run_summary.py --run-dir results/<run-folder>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def get_latest_run_dir(results_root: Path = Path("results")) -> Path:
    subdirs = [p.parent for p in results_root.glob("**/metrics.json")]
    if not subdirs:
        raise FileNotFoundError(f"No completed run directories found in {results_root}")
    return max(subdirs, key=lambda p: (p / "metrics.json").stat().st_mtime)


def print_summary(run_dir: Path) -> None:
    metrics_path = run_dir / "metrics.json"
    metadata_path = run_dir / "run_metadata.json"
    ledger_path = run_dir / "privacy_ledger.json"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}

    print("=" * 76)
    print(f" EXPERIMENTAL RUN SUMMARY: {run_dir.name}")
    print("=" * 76)

    # 1. Threat Model & Privacy Configuration
    print("\n[1] Threat Model & Privacy Contract:")
    sec_agg = metadata.get("secure_aggregation", False)
    eps = ledger.get("epsilon", "N/A")
    delta = ledger.get("delta", "N/A")
    print(f"  * Differential Privacy:       epsilon = {eps:.2f}, delta = {delta}" if isinstance(eps, (int, float)) else f"  * Differential Privacy:       epsilon = {eps}, delta = {delta}")
    print(f"  * Secure Aggregation:        {'ACTIVE (Pairwise Zero-Sum Masking)' if sec_agg else 'INACTIVE (Server sees raw client deltas)'}")
    print(f"  * Privacy Adjacency:         Add/Remove ONE complete client dataset")
    print(f"  * Audit Standard:            {metadata.get('audit_standard', 'Standard Baseline Probe')}")
    print(f"  * Training Validity:         {metrics.get('validity', 'N/A')}")

    # 2. Model Utility Across Populations
    print("\n[2] Model Utility Comparison across Populations:")
    print(f"  {'Method':<32} {'Test Acc':<12} {'Forgot Acc':<12} {'Retain Acc':<12}")
    print("  " + "-" * 66)

    methods = [
        ("DP-Only (M_DP)", "dp_only_no_action"),
        ("Retained Fine-Tuning (M_FT)", "retained_finetune_baseline"),
        ("Exact Retrain (M_R)", "target_retrain"),
    ]
    if "cached_update_reconstruction_baseline" in metrics:
        methods.append(("Cached Reconstruction (M_CR)", "cached_update_reconstruction_baseline"))
    if "federated_eraser_baseline" in metrics:
        methods.append(("FedEraser (M_FE)", "federated_eraser_baseline"))

    for display_name, key in methods:
        data = metrics.get(key, {})
        if data.get("incompatible"):
            print(f"  {display_name:<32} [BLOCKED: Incompatible with Secure Aggregation]")
            continue
        u = data.get("utility", {})
        test_acc = f"{u.get('test_accuracy', 0.0)*100:.2f}%" if "test_accuracy" in u else "N/A"
        forg_acc = f"{u.get('forgotten_accuracy', 0.0)*100:.2f}%" if "forgotten_accuracy" in u else "N/A"
        ret_acc = f"{u.get('retained_accuracy', 0.0)*100:.2f}%" if "retained_accuracy" in u else "N/A"
        print(f"  {display_name:<32} {test_acc:<12} {forg_acc:<12} {ret_acc:<12}")

    # 3. Full TC-UMIA Tri-Population Privacy Audit
    dp_data = metrics.get("dp_only_no_action", {})
    tri_probe = dp_data.get("attack_diagnostics", {}).get("tri_population_pre_post_probe", {})
    if tri_probe:
        print("\n[3] TC-UMIA Tri-Population Privacy Audit (Fu et al., 2026):")
        print(f"  * Feature Vector:            14-dim (Loss, Prob Deltas, Entropy, Margins, TV Distance)")
        print(f"  * Linear Probe Macro AUC:    {tri_probe.get('macro_ovr_auc', 0.0):.3f} (95% CI: {tri_probe.get('macro_ovr_auc_ci_95')})")
        print(f"  * MLP Neural Probe Macro AUC:{tri_probe.get('mlp_macro_ovr_auc', 0.0):.3f} (Accuracy: {tri_probe.get('mlp_accuracy', 0.0)*100:.1f}%)")
        print(f"  * Forgotten vs Unseen AUC:   {tri_probe.get('forget_vs_unseen_auc', 0.0):.3f} (Lower is closer to unlearned reference)")
        print(f"  * Retained vs Unseen AUC:    {tri_probe.get('retained_vs_unseen_auc', 0.0):.3f} (Retained-set privacy leakage)")

        cm_norm = tri_probe.get("normalized_confusion_matrix")
        if cm_norm:
            print("\n  Normalized 3x3 Confusion Matrix:")
            print("                Pred: Forget   Pred: Retain   Pred: Unseen")
            labels = ["True: Forget ", "True: Retain ", "True: Unseen "]
            for lbl, row in zip(labels, cm_norm):
                print(f"    {lbl}  {row[0]*100:6.1f}%       {row[1]*100:6.1f}%       {row[2]*100:6.1f}%")

    # 4. Unlearning Method Redundancy & Baseline Compatibility
    print("\n[4] Unlearning Redundancy & SecAgg Architectural Verdict:")
    for display_name, key in [
        ("Cached Reconstruction (M_CR)", "cached_update_reconstruction_baseline"),
        ("FedEraser (M_FE)", "federated_eraser_baseline"),
    ]:
        data = metrics.get(key, {})
        if data.get("incompatible"):
            print(f"  * {display_name}:")
            print(f"      Status: INCOMPATIBLE WITH SECURE AGGREGATION")
            print(f"      Reason: {data.get('reason')}")
        else:
            mub = data.get("mub", {}).get("classification", "N/A")
            print(f"  * {display_name}: Redundancy Status = {mub}")

    print("\n" + "=" * 76)


def main():
    parser = argparse.ArgumentParser(description="DP-ForgetBench Run Summary Viewer")
    parser.add_argument("--run-dir", type=Path, default=None, help="Path to specific run directory")
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir else get_latest_run_dir()
    print_summary(run_dir)


if __name__ == "__main__":
    main()
