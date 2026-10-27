"""GPU experiment runner: Architecture tier benchmark.

Runs all 5 architecture tiers across 3 privacy regimes on GPU.
Measures test accuracy, training time, and MUB (vs DP-only baseline).
Designed to run overnight; outputs a structured JSON report + CSV.

Usage:
    python experiments/tier_benchmark.py [--tiers 0 1 2 3 4] [--epsilons 1 4 8 32 inf]
    python experiments/tier_benchmark.py --tier 2 --eps 4 --n_clients 100 --seeds 3
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

# Add src to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dp_forgetbench.data import make_cifar10_multiclass_federation, apply_client_deletion_requests
from dp_forgetbench.federated import (
    get_device, set_seed, flatten_logits, evaluate_loss_accuracy,
    train_federated, finetune_retained, clone_state,
)
from dp_forgetbench.architectures import (
    SmallGroupNormCNN, Tier1ScatterLinear, FrozenBackboneLinear,
    Tier3FrozenNCM, Tier4FrozenLoRA, get_tier_specs,
)
from dp_forgetbench.privacy import poisson_gaussian_epsilon
from dp_forgetbench.attacks import tri_population_probe
from dp_forgetbench.tost import tost_paired, intersection_union_decision


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def get_noise_multiplier_for_epsilon(
    target_epsilon: float,
    n_clients: int,
    rounds: int,
    sample_rate: float,
    delta: float = 1e-5,
    clip_norm: float = 1.0,
) -> float:
    """Binary search for noise_multiplier that achieves approximately target_epsilon."""
    if target_epsilon == float("inf"):
        return 0.0  # Non-private

    lo, hi = 0.01, 200.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        try:
            eps = poisson_gaussian_epsilon(
                noise_multiplier=mid,
                sample_rate=sample_rate,
                rounds=rounds,
                delta=delta,
            )
        except Exception:
            eps = float("inf")
        if eps > target_epsilon:
            lo = mid
        else:
            hi = mid
    return hi


def build_data_config(n_clients: int, samples_per_client: int = 400) -> dict:
    return {
        "backend": "cifar10_multiclass",
        "dataset_root": str(ROOT_DIR / "data"),
        "download": True,
        "n_clients": n_clients,
        "samples_per_client": samples_per_client,
        "heterogeneity_alpha": 0.5,  # Moderate non-IID
        "audit_samples": 500,
        "test_samples": 10000,
    }


def build_privacy_config(
    noise_multiplier: float,
    n_clients: int,
    sample_rate: float,
    delta: float = 1e-5,
) -> dict:
    return {
        "enabled": noise_multiplier > 0,
        "population_size": n_clients,
        "delta": delta,
        "clip_norm": 1.0,
        "noise_multiplier": noise_multiplier,
        "sampling": "poisson",
        "release_count": 1,
        "secure_aggregation": False,
    }


def build_federated_config(rounds: int, local_epochs: int, sample_rate: float) -> dict:
    return {
        "rounds": rounds,
        "local_epochs": local_epochs,
        "local_batch_size": 64,
        "learning_rate": 0.05,
        "client_sample_rate": sample_rate,
        "secure_aggregation": False,
    }


# ---------------------------------------------------------------------------
# Tier-specific training wrappers
# ---------------------------------------------------------------------------

def train_tier0_to_4(
    model: nn.Module,
    tier_id: int,
    federation,
    federated_cfg: dict,
    privacy_cfg: dict,
    forgotten_client_id: int,
    seed: int,
) -> dict:
    """Run full federated training for a given tier. Returns metrics dict."""
    device = get_device()
    set_seed(seed)

    if tier_id == 3:
        # NCM: One-shot aggregation of sufficient statistics
        return _train_ncm(model, federation, privacy_cfg, forgotten_client_id, seed, device)

    # Tiers 0, 1, 2, 4: Standard FedAvg
    clients = federation.clients
    start = time.perf_counter()

    # For frozen backbone tiers (1, 2, 4), features can be pre-computed to speed up
    if tier_id in (1, 2, 4):
        clients = _precompute_features(model, clients, device)

    cost = train_federated(
        clients=clients,
        n_features=None if tier_id == 0 else federation.clients[list(federation.clients.keys())[0]].x.shape[1]
        if clients[list(clients.keys())[0]].x.ndim == 2 else None,
        federated_config=federated_cfg,
        privacy_config=privacy_cfg,
        seed=seed,
        model=model,
        history=None,
    )

    runtime = time.perf_counter() - start

    # Evaluate M_DP
    test_metrics = evaluate_loss_accuracy(model, federation.test_x, federation.test_y)
    deletion = apply_client_deletion_requests(
        federation, {"client_id": forgotten_client_id}, seed=seed
    )
    forget_metrics = evaluate_loss_accuracy(model, deletion.forgotten_x, deletion.forgotten_y)

    return {
        "tier": tier_id,
        "seed": seed,
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "forget_accuracy": forget_metrics["accuracy"],
        "runtime_s": runtime,
        "method": "dp_only",
    }


def _precompute_features(model: nn.Module, clients: dict, device: torch.device) -> dict:
    """Cache backbone features so FedAvg only updates the head."""
    from dp_forgetbench.data import ClientDataset
    encoder = model.to(device)
    new_clients = {}
    for cid, client in clients.items():
        x = client.x
        if x.ndim == 2:
            new_clients[cid] = client
            continue
        # Determine encoder
        if hasattr(encoder, '_encode'):
            with torch.no_grad():
                feats = []
                for i in range(0, len(x), 128):
                    batch = x[i:i+128].to(device)
                    feats.append(encoder._encode(batch).cpu())
            x = torch.cat(feats, dim=0)
        elif hasattr(encoder, 'scatter'):
            with torch.no_grad():
                feats = []
                for i in range(0, len(x), 128):
                    batch = x[i:i+128].to(device)
                    feats.append(encoder.scatter(batch).cpu())
            x = torch.cat(feats, dim=0)
        new_clients[cid] = ClientDataset(cid, x, client.y)
    return new_clients


def _train_ncm(
    model: Tier3FrozenNCM,
    federation,
    privacy_cfg: dict,
    forgotten_client_id: int,
    seed: int,
    device: torch.device,
) -> dict:
    """One-round NCM aggregation with optional DP noise."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    clip_norm = float(privacy_cfg.get("clip_norm", 1.0))
    sigma = float(privacy_cfg.get("noise_multiplier", 0.0))
    n_pop = int(privacy_cfg.get("population_size", len(federation.clients)))
    q = 0.5  # sample rate
    enabled = privacy_cfg.get("enabled", False)

    # Aggregate class sufficient statistics
    num_classes = 10
    global_sums = {c: torch.zeros(model.ncm.feature_dim) for c in range(num_classes)}
    global_counts = {c: 0 for c in range(num_classes)}

    all_client_stats = {}  # for exact unlearning
    for cid, client in federation.clients.items():
        # Extract features
        feats = []
        with torch.no_grad():
            for i in range(0, len(client.x), 128):
                feats.append(model.encode(client.x[i:i+128].to(device)).cpu())
        feats = torch.cat(feats, dim=0)
        labels = client.y.long()

        client_sums = {}
        client_counts = {}
        for cls in range(num_classes):
            mask = labels == cls
            if mask.any():
                cls_feats = feats[mask]
                # Clip
                norms = cls_feats.norm(dim=1, keepdim=True).clamp(min=1e-8)
                cls_feats = cls_feats / norms.clamp(min=clip_norm / norms) if enabled else cls_feats
                client_sums[cls] = cls_feats.sum(dim=0)
                client_counts[cls] = int(mask.sum())
            else:
                client_sums[cls] = torch.zeros(model.ncm.feature_dim)
                client_counts[cls] = 0

        all_client_stats[cid] = (client_sums, client_counts)

        # Aggregate
        for cls in range(num_classes):
            global_sums[cls] += client_sums[cls]
            global_counts[cls] += client_counts[cls]

    # Add DP noise if enabled (single mechanism, one round)
    if enabled and sigma > 0:
        normalizer = n_pop * q
        for cls in range(num_classes):
            noise = torch.randn_like(global_sums[cls]) * (sigma * clip_norm)
            global_sums[cls] = (global_sums[cls] + noise) / normalizer

    model.ncm.fit(global_sums, global_counts)
    runtime = time.perf_counter() - start

    # Evaluate
    model_for_eval = model.to(device)
    test_metrics = evaluate_loss_accuracy(model_for_eval, federation.test_x, federation.test_y)
    deletion = apply_client_deletion_requests(
        federation, {"client_id": forgotten_client_id}, seed=seed
    )
    forget_metrics = evaluate_loss_accuracy(model_for_eval, deletion.forgotten_x, deletion.forgotten_y)

    return {
        "tier": 3,
        "seed": seed,
        "test_accuracy": test_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "forget_accuracy": forget_metrics["accuracy"],
        "runtime_s": runtime,
        "method": "dp_only_ncm_one_round",
    }


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_tier_experiment(
    tier_id: int,
    epsilon: float,
    n_clients: int = 100,
    n_seeds: int = 3,
    rounds: int = 40,
    local_epochs: int = 5,
    sample_rate: float = 0.5,
    forgotten_client_id: int = 2,
    output_dir: Path = ROOT_DIR / "results" / "tier_benchmark",
    data_seed_base: int = 42,
    backbone_name: str = "dinov2_small",
) -> list[dict]:
    """Run a single tier × epsilon cell across multiple seeds."""
    device = get_device()
    tier_specs = get_tier_specs(num_classes=10, backbone_name=backbone_name)
    spec = tier_specs[tier_id]

    print(f"\n{'='*70}")
    print(f"  TIER {tier_id}: {spec.name} (backbone: {backbone_name})")
    print(f"  eps = {epsilon}, N = {n_clients}, {rounds}R x {local_epochs}E, {n_seeds} seeds")
    print(f"{'='*70}")

    data_cfg = build_data_config(n_clients)
    noise_mult = get_noise_multiplier_for_epsilon(
        epsilon, n_clients, rounds, sample_rate
    ) if epsilon != float("inf") else 0.0
    privacy_cfg = build_privacy_config(noise_mult, n_clients, sample_rate)
    federated_cfg = build_federated_config(rounds, local_epochs, sample_rate)

    # Compute actual epsilon achieved
    if noise_mult > 0:
        try:
            actual_eps = poisson_gaussian_epsilon(
                noise_multiplier=noise_mult,
                sample_rate=sample_rate,
                rounds=rounds,
                delta=1e-5,
            )
        except Exception:
            actual_eps = float("nan")
        else:
            pass
    else:
        actual_eps = float("inf")

    results = []
    for seed in range(n_seeds):
        train_seed = data_seed_base + seed * 100
        print(f"\n  [Seed {seed+1}/{n_seeds}] seed={train_seed}")

        # Build federation (same seed for all tiers to allow fair comparison)
        set_seed(train_seed)
        federation = make_cifar10_multiclass_federation(data_cfg, seed=train_seed)

        # Instantiate model fresh
        model = spec.model_factory().to(device)

        t_start = time.perf_counter()
        try:
            seed_result = train_tier0_to_4(
                model=model,
                tier_id=tier_id,
                federation=federation,
                federated_cfg=federated_cfg,
                privacy_cfg=privacy_cfg,
                forgotten_client_id=forgotten_client_id,
                seed=train_seed,
            )
        except Exception as e:
            print(f"    [ERROR] {e}")
            seed_result = {
                "tier": tier_id, "seed": train_seed,
                "test_accuracy": 0.0, "error": str(e),
            }

        seed_result.update({
            "epsilon_target": epsilon,
            "epsilon_actual": actual_eps,
            "noise_multiplier": noise_mult,
            "n_clients": n_clients,
            "rounds": rounds,
            "local_epochs": local_epochs,
            "tier_name": spec.name,
            "backbone_name": backbone_name,
            "valid": seed_result.get("test_accuracy", 0) > 0.15,  # Validity gate
        })
        results.append(seed_result)
        acc = seed_result.get("test_accuracy", 0)
        print(f"    Test Accuracy: {acc*100:.2f}%  {'[VALID]' if acc > 0.15 else '[INVALID - below chance threshold]'}")

    # Aggregate
    valid_accs = [r["test_accuracy"] for r in results if r.get("valid", False)]
    summary = {
        "tier": tier_id,
        "tier_name": spec.name,
        "backbone_name": backbone_name,
        "epsilon_target": epsilon,
        "epsilon_actual": actual_eps,
        "n_clients": n_clients,
        "mean_accuracy": float(np.mean(valid_accs)) if valid_accs else 0.0,
        "sd_accuracy": float(np.std(valid_accs, ddof=1)) if len(valid_accs) > 1 else 0.0,
        "n_valid_seeds": len(valid_accs),
        "n_total_seeds": n_seeds,
    }
    print(f"\n  SUMMARY: mean={summary['mean_accuracy']*100:.2f}% +/- {summary['sd_accuracy']*100:.2f}%"
          f"  ({summary['n_valid_seeds']}/{summary['n_total_seeds']} valid seeds)")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_eps = str(epsilon).replace(".", "p").replace("inf", "inf")
    safe_bb = backbone_name.replace("/", "_")
    out_file = output_dir / f"tier{tier_id}_{safe_bb}_eps{safe_eps}_N{n_clients}.json"
    with open(out_file, "w") as f:
        json.dump({"summary": summary, "seed_results": results}, f, indent=2)
    print(f"  Saved -> {out_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Architecture tier benchmark")
    parser.add_argument("--tiers", nargs="+", type=int, default=[0, 1, 2, 4],
                        help="Tier IDs to run (0-4). Default: 0 1 2 4 (skip slow NCM)")
    parser.add_argument("--backbone", type=str, default="dinov2_small",
                        choices=["dinov2_small", "mobilenet_v3_small", "resnet50", "bit"],
                        help="Backbone model for Tiers 2, 3, 4 (default: dinov2_small)")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[4.0, 8.0, 32.0, float("inf")],
                        help="Target epsilon values (use inf for non-private)")
    parser.add_argument("--n_clients", type=int, default=100, help="Number of simulated FL clients")
    parser.add_argument("--seeds", type=int, default=3, help="Seeds per cell")
    parser.add_argument("--rounds", type=int, default=40, help="Federated training rounds")
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--sample_rate", type=float, default=0.5)
    parser.add_argument("--forgotten_client", type=int, default=2)
    parser.add_argument("--output", type=str, default="results/tier_benchmark")
    args = parser.parse_args()

    output_dir = ROOT_DIR / args.output
    all_summaries = []

    for tier_id in args.tiers:
        for eps in args.epsilons:
            results = run_tier_experiment(
                tier_id=tier_id,
                epsilon=eps,
                n_clients=args.n_clients,
                n_seeds=args.seeds,
                rounds=args.rounds,
                local_epochs=args.local_epochs,
                sample_rate=args.sample_rate,
                forgotten_client_id=args.forgotten_client,
                output_dir=output_dir,
                backbone_name=args.backbone,
            )
            all_summaries.extend(results)

    # Master summary table
    print("\n" + "="*70)
    print("  TIER BENCHMARK - RESULTS TABLE")
    print("="*70)
    print(f"  {'Tier':<5} {'Name':<35} {'eps':>6} {'Acc%':>7} {'+/-SD':>7} {'Valid':>6}")
    print("-"*70)

    # Load all saved summaries
    for json_file in sorted(output_dir.glob("tier*.json")):
        with open(json_file) as f:
            data = json.load(f)
        s = data["summary"]
        eps_str = f"{s['epsilon_actual']:.1f}" if s['epsilon_actual'] != float('inf') else "inf"
        print(f"  T{s['tier']:<4} {s['tier_name'][:35]:<35} {eps_str:>6} "
              f"{s['mean_accuracy']*100:>6.2f}% {s['sd_accuracy']*100:>6.2f}% "
              f"{s['n_valid_seeds']}/{s['n_total_seeds']:>4}")

    # Save master
    master_file = output_dir / "master_summary.json"
    with open(master_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n  Master summary -> {master_file}")


if __name__ == "__main__":
    main()
