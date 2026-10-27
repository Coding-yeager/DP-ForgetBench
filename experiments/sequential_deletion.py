"""Sequential client deletion experiment.

Implements the multi-deletion design described in the AI agent review.
Runs both post-hoc (delete from finished model) and online (continue training)
regimes across M ∈ {1, 2, 4, 8} deletion requests per cell.

Design:
- Population: N ≥ 100 clients (so 8 deletions is ≤ 8% of federation)
- Arms:   DP-only, fine-tuning (warm-start), exact retrain per prefix
- Endpoints per j: JS on D_forget, MUB_j, TC-UMIA F-vs-U AUC
- Analysis: Tracks slope in j (how fast redundancy degrades)

Usage:
    python experiments/sequential_deletion.py --n_clients 100 --max_deletions 8 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import numpy as np
import torch

from dp_forgetbench.data import make_cifar10_multiclass_federation, apply_client_deletion_requests
from dp_forgetbench.federated import (
    get_device, set_seed, evaluate_loss_accuracy, train_federated,
    finetune_retained, FederatedHistory,
)
from dp_forgetbench.architectures import SmallGroupNormCNN, FrozenBackboneLinear
from dp_forgetbench.privacy import poisson_gaussian_epsilon
from dp_forgetbench.attacks import tri_population_probe
from dp_forgetbench.tost import tost_paired
from dp_forgetbench.evaluation import js_divergence


def make_model_instance(model_type: str, backbone_name: str = "dinov2_small", num_classes: int = 10) -> torch.nn.Module:
    if model_type == "dinov2":
        return FrozenBackboneLinear(backbone_name=backbone_name, num_classes=num_classes)
    return SmallGroupNormCNN(num_classes=num_classes)


def _precompute_features_dict(model: torch.nn.Module, clients: dict, device: torch.device) -> dict:
    from dp_forgetbench.data import ClientDataset
    encoder = model.to(device)
    new_clients = {}
    for cid, client in clients.items():
        if client.x.ndim == 2:
            new_clients[cid] = client
            continue
        with torch.no_grad():
            feats = []
            for i in range(0, len(client.x), 128):
                batch = client.x[i:i+128].to(device)
                feats.append(encoder._encode(batch).cpu())
            new_clients[cid] = ClientDataset(cid, torch.cat(feats, dim=0), client.y)
    return new_clients


def build_base_configs(n_clients: int, target_epsilon: float, rounds: int = 40):
    """Build standard federated + privacy configs."""
    sample_rate = 0.5
    noise_mult = 0.0

    if target_epsilon != float("inf"):
        # Simple binary search for noise_mult achieving target_epsilon
        from dp_forgetbench.privacy import poisson_gaussian_epsilon as rdp_eps
        lo, hi = 0.01, 200.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            try:
                eps = rdp_eps(noise_multiplier=mid, sample_rate=sample_rate, rounds=rounds, delta=1e-5)
            except Exception:
                eps = float("inf")
            if eps > target_epsilon:
                lo = mid
            else:
                hi = mid
        noise_mult = hi

    fed_cfg = {
        "rounds": rounds,
        "local_epochs": 5,
        "local_batch_size": 64,
        "learning_rate": 0.05,
        "client_sample_rate": sample_rate,
        "secure_aggregation": False,
    }
    priv_cfg = {
        "enabled": noise_mult > 0,
        "population_size": n_clients,
        "delta": 1e-5,
        "clip_norm": 1.0,
        "noise_multiplier": noise_mult,
        "sampling": "poisson",
        "release_count": 1,
        "secure_aggregation": False,
    }
    return fed_cfg, priv_cfg, noise_mult


def run_sequential_deletion(
    n_clients: int = 100,
    max_deletions: int = 8,
    target_epsilon: float = 8.0,
    n_seeds: int = 3,
    rounds: int = 40,
    output_dir: Path = ROOT_DIR / "results" / "sequential_deletion",
    model_type: str = "dinov2",
    backbone_name: str = "dinov2_small",
) -> dict:
    """
    Main sequential deletion experiment.

    For each seed:
    1. Train a full M_DP model on all N clients.
    2. Post-hoc delete clients {k_1, ..., k_j} for j in {1,2,4,8}.
    3. Measure JS(M_DP, M_R^(j)) gap growth.
    4. Fine-tune (warm-start) after each prefix deletion.
    5. Compare MUB_j across j.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    data_cfg = {
        "backend": "cifar10_multiclass",
        "dataset_root": str(ROOT_DIR / "data"),
        "download": True,
        "n_clients": n_clients,
        "samples_per_client": 400,
        "heterogeneity_alpha": 0.5,
        "audit_samples": 500,
        "test_samples": 10000,
    }

    fed_cfg, priv_cfg, noise_mult = build_base_configs(n_clients, target_epsilon, rounds)

    # Deletion order: fixed clients by ID for reproducibility
    # Choose clients in the middle of the range to avoid edge cases
    deletion_sequence = list(range(5, 5 + max_deletions))  # e.g. [5, 6, 7, 8, ..., 12]
    deletion_counts = [m for m in [1, 2, 4, 8] if m <= max_deletions]

    all_results = []

    for seed_idx in range(n_seeds):
        train_seed = 1000 + seed_idx * 77
        print(f"\n{'='*60}")
        print(f"  Seed {seed_idx+1}/{n_seeds} (seed={train_seed}) [model={model_type}]")
        set_seed(train_seed)

        federation = make_cifar10_multiclass_federation(data_cfg, seed=train_seed)

        # ---- Step 1: Train full M_DP model ----
        print("  Training full M_DP (all clients)...")
        model_dp = make_model_instance(model_type, backbone_name, num_classes=10).to(device)
        t0 = time.perf_counter()

        # For foundation backbones, precompute features to accelerate FL rounds
        if model_type == "dinov2":
            active_clients = _precompute_features_dict(model_dp, federation.clients, device)
        else:
            active_clients = federation.clients

        train_federated(
            clients=active_clients,
            n_features=None,
            federated_config=fed_cfg,
            privacy_config=priv_cfg,
            seed=train_seed,
            model=model_dp,
        )
        train_time = time.perf_counter() - t0
        m_dp_test = evaluate_loss_accuracy(model_dp, federation.test_x, federation.test_y)
        print(f"    M_DP test acc: {m_dp_test['accuracy']*100:.2f}% in {train_time:.1f}s")

        if m_dp_test["accuracy"] < 0.12:
            print("    [INVALID] M_DP collapsed to chance. Skipping seed.")
            continue

        seed_row = {
            "seed": train_seed,
            "model_type": model_type,
            "backbone_name": backbone_name,
            "m_dp_test_acc": m_dp_test["accuracy"],
            "deletion_results": [],
        }

        # ---- Step 2: Sequential post-hoc deletion ----
        for j in deletion_counts:
            clients_to_delete = deletion_sequence[:j]
            print(f"\n  Post-hoc deletion: j={j} clients {clients_to_delete}")

            # Build deletion config for all j clients
            del_cfg = {"requests": [{"client_id": cid, "fraction": 1.0} for cid in clients_to_delete]}

            try:
                del_result = apply_client_deletion_requests(federation, del_cfg, seed=train_seed)
            except Exception as e:
                print(f"    [ERROR] Deletion failed: {e}")
                continue

            # Retained clients features
            if model_type == "dinov2":
                retained_active = {cid: active_clients[cid] for cid in del_result.retained}
            else:
                retained_active = del_result.retained

            # ---- Exact retrain M_R^(j) ----
            print(f"    Training M_R^(j={j}) from scratch on retained clients...")
            model_retrain = make_model_instance(model_type, backbone_name, num_classes=10).to(device)
            # Use a modified privacy config for retained clients
            retrain_priv = dict(priv_cfg, population_size=len(del_result.retained))
            train_federated(
                clients=retained_active,
                n_features=None,
                federated_config=fed_cfg,
                privacy_config=retrain_priv,
                seed=train_seed + j,
                model=model_retrain,
            )
            m_r_test = evaluate_loss_accuracy(model_retrain, federation.test_x, federation.test_y)
            print(f"    M_R^({j}) test acc: {m_r_test['accuracy']*100:.2f}%")

            # ---- DP-only JS gap ----
            dp_logits_test = torch.softmax(
                _batched_logits(model_dp, federation.test_x, device), dim=-1
            )
            r_logits_test = torch.softmax(
                _batched_logits(model_retrain, federation.test_x, device), dim=-1
            )
            js_dp = float(js_divergence(dp_logits_test, r_logits_test))

            # ---- DP-only forget JS ----
            dp_logits_forget = torch.softmax(
                _batched_logits(model_dp, del_result.forgotten_x, device), dim=-1
            )
            r_logits_forget = torch.softmax(
                _batched_logits(model_retrain, del_result.forgotten_x, device), dim=-1
            )
            js_dp_forget = float(js_divergence(dp_logits_forget, r_logits_forget))

            # ---- Fine-tune (warm-start from M_DP) post-deletion ----
            import copy
            model_ft = copy.deepcopy(model_dp)
            ft_cfg = dict(fed_cfg, rounds=4)
            finetune_retained(
                model=model_ft,
                retained_clients=retained_active,
                federated_config=ft_cfg,
                privacy_config=priv_cfg,
                seed=train_seed + 1000 + j,
            )
            ft_test = evaluate_loss_accuracy(model_ft, federation.test_x, federation.test_y)
            ft_logits_test = torch.softmax(
                _batched_logits(model_ft, federation.test_x, device), dim=-1
            )
            js_ft = float(js_divergence(ft_logits_test, r_logits_test))

            mub_js = js_dp - js_ft  # Positive = FT moved closer to retrain

            # ---- TC-UMIA F-vs-U AUC on cumulative forgotten set ----
            cumulative_forget_x = del_result.forgotten_x
            cumulative_forget_y = del_result.forgotten_y
            probe_limit = min(50, len(cumulative_forget_x), len(federation.test_x))
            populations = {
                "forget": (cumulative_forget_x[:probe_limit], cumulative_forget_y[:probe_limit]),
                "retain": (
                    next(iter(del_result.retained.values())).x[:probe_limit],
                    next(iter(del_result.retained.values())).y[:probe_limit],
                ),
                "unseen": (federation.test_x[:probe_limit], federation.test_y[:probe_limit]),
            }

            try:
                probe_result = tri_population_probe(
                    pre_model=model_dp.cpu(),
                    post_model=model_ft.cpu(),
                    populations=populations,
                    seed=train_seed,
                )
                fvu_auc = probe_result["forget_vs_unseen_auc"]
            except Exception:
                fvu_auc = float("nan")
            finally:
                model_dp.to(device)
                model_ft.to(device)

            del_row = {
                "j": j,
                "deleted_clients": clients_to_delete[:j],
                "m_dp_test_acc": m_dp_test["accuracy"],
                "m_r_test_acc": m_r_test["accuracy"],
                "m_ft_test_acc": ft_test["accuracy"],
                "js_dp_to_retrain_test": js_dp,
                "js_dp_to_retrain_forget": js_dp_forget,
                "js_ft_to_retrain_test": js_ft,
                "mub_js_test": mub_js,
                "tc_umia_fvu_auc": fvu_auc,
                "redundancy_signal": "dp_already_close" if js_dp < 0.05 else "gap_detected",
            }
            seed_row["deletion_results"].append(del_row)

            print(f"    JS(DP, R): {js_dp:.4f} | JS(FT, R): {js_ft:.4f} | MUB_JS: {mub_js:+.4f} | F-vs-U AUC: {fvu_auc:.3f}")

        all_results.append(seed_row)

    # ---- Analysis across seeds ----
    analysis = _analyze_sequential_results(all_results, deletion_counts)

    output = {
        "experiment": "sequential_deletion",
        "n_clients": n_clients,
        "max_deletions": max_deletions,
        "target_epsilon": target_epsilon,
        "noise_multiplier": noise_mult,
        "deletion_sequence": deletion_sequence[:max_deletions],
        "n_seeds": n_seeds,
        "seed_results": all_results,
        "analysis": analysis,
    }

    safe_model = f"{model_type}_{backbone_name}" if model_type == "dinov2" else "cnn"
    out_file = output_dir / f"sequential_{safe_model}_N{n_clients}_eps{target_epsilon}_M{max_deletions}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved -> {out_file}")

    return output


def _batched_logits(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run model inference in batches."""
    model = model.to(device)
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(x), 128):
            outputs.append(model(x[i:i+128].to(device)).cpu())
    return torch.cat(outputs, dim=0) if outputs else torch.empty(0)


def _analyze_sequential_results(all_results: list[dict], deletion_counts: list[int]) -> dict:
    """Compute per-j statistics across seeds."""
    analysis = {}
    for j in deletion_counts:
        js_dps, js_fts, mub_jss, fvu_aucs = [], [], [], []
        for seed_row in all_results:
            for del_row in seed_row["deletion_results"]:
                if del_row["j"] == j:
                    js_dps.append(del_row["js_dp_to_retrain_test"])
                    js_fts.append(del_row["js_ft_to_retrain_test"])
                    mub_jss.append(del_row["mub_js_test"])
                    fvu_aucs.append(del_row["tc_umia_fvu_auc"])

        if not js_dps:
            continue

        analysis[f"j_{j}"] = {
            "n_seeds": len(js_dps),
            "mean_js_dp": float(np.mean(js_dps)),
            "mean_js_ft": float(np.mean(js_fts)),
            "mean_mub_js": float(np.mean(mub_jss)),
            "mean_fvu_auc": float(np.nanmean(fvu_aucs)),
            "dp_gap_growing": len(js_dps) > 1 and js_dps[-1] > js_dps[0],
        }

        # TOST for MUB at this j (are FT gains significant?)
        if len(mub_jss) >= 2:
            tost_result = tost_paired(mub_jss, delta=0.02)
            analysis[f"j_{j}"]["tost_mub_delta002"] = tost_result

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Sequential client deletion experiment")
    parser.add_argument("--model_type", type=str, default="dinov2", choices=["dinov2", "cnn"],
                        help="Model family to run: 'dinov2' (Tier 2 foundation linear probe) or 'cnn' (Tier 0 baseline)")
    parser.add_argument("--backbone", type=str, default="dinov2_small",
                        choices=["dinov2_small", "mobilenet_v3_small", "resnet50", "bit"],
                        help="Backbone model if model_type == 'dinov2'")
    parser.add_argument("--n_clients", type=int, default=100)
    parser.add_argument("--max_deletions", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--output", type=str, default="results/sequential_deletion")
    args = parser.parse_args()

    run_sequential_deletion(
        n_clients=args.n_clients,
        max_deletions=args.max_deletions,
        target_epsilon=args.epsilon,
        n_seeds=args.seeds,
        rounds=args.rounds,
        output_dir=ROOT_DIR / args.output,
        model_type=args.model_type,
        backbone_name=args.backbone,
    )


if __name__ == "__main__":
    main()

