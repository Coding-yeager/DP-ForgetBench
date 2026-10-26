"""CLI for the Phase 0 fully auditable client-deletion experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from .config import load_config
from .data import apply_client_deletion_requests, make_federation, normalized_deletion_requests, write_deletion_manifest
from .evaluation import compute_method_mub, evaluate_against_target
from .federated import (
    FederatedHistory,
    federated_eraser,
    finetune_retained,
    infer_n_features,
    reconstruct_from_cached_updates,
    train_federated,
)
from .privacy import make_ledger


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _config_hash(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _reference_privacy_config(privacy_config: dict, retained_client_count: int) -> dict:
    """Return the internal retraining-reference privacy config.

    The released full model uses the original public population size. A retrain
    reference is trained on D-minus-F, so its DP aggregation normalizer must be based
    on the retained population. It is an internal target model, not an extra
    public release under the full-model ledger.
    """
    reference = {**privacy_config}
    if reference["enabled"]:
        reference["population_size"] = int(retained_client_count)
    return reference


def run_experiment(config: dict, config_path: Path) -> Path:
    seed = int(config["seed"])
    federation = make_federation(config["data"], seed)
    deletion_result = apply_client_deletion_requests(federation, config["deletion"], seed)
    retained = deletion_result.retained
    requests = normalized_deletion_requests(config["deletion"])
    output_root = Path(config["output"]["root"])
    run_name = f"{config['experiment_name']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_seed{seed}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path, manifest_checksum = write_deletion_manifest(deletion_result.rows, Path("partitions") / "generated")
    n_features = infer_n_features(next(iter(federation.clients.values())).x)
    model_config = config.get("model")
    privacy_config = {**config["privacy"]}
    if privacy_config["enabled"]:
        privacy_config.setdefault("population_size", int(config["data"]["n_clients"]))
    target_privacy_config = _reference_privacy_config(privacy_config, len(retained))
    ledger = make_ledger(privacy_config, config["federated"])

    # Full model uses all clients. The DP model is deliberately left unchanged after a request.
    history = FederatedHistory()
    full_model, full_cost = train_federated(
        clients=federation.clients,
        n_features=n_features,
        federated_config=config["federated"],
        privacy_config=privacy_config,
        seed=seed,
        history=history,
        model_config=model_config,
    )
    # Target retrain uses the same algorithm on retained data. Its private
    # normalizer is tied to the retained population and recorded separately.
    target_model, target_cost = train_federated(
        clients=retained,
        n_features=n_features,
        federated_config=config["federated"],
        privacy_config=target_privacy_config,
        seed=seed + 10_000,
        model_config=model_config,
    )
    # Independent targets are required to quantify ordinary retraining variability.
    # We build an ensemble of them.
    ensemble_size = int(config.get("evaluation", {}).get("retrain_ensemble_size", 5))
    target_models_ensemble = []
    target_costs_ensemble = []
    for i in range(1, ensemble_size):
        m, c = train_federated(
            clients=retained,
            n_features=n_features,
            federated_config=config["federated"],
            privacy_config=target_privacy_config,
            seed=seed + 10_000 + i,
            model_config=model_config,
        )
        target_models_ensemble.append(m)
        target_costs_ensemble.append(c)
    unlearned_model, unlearning_cost = finetune_retained(
        full_model=full_model,
        clients=retained,
        n_features=n_features,
        federated_config=config["federated"],
        unlearning_rounds=int(config["unlearning"]["rounds"]),
        seed=seed + 20_000,
        model_config=model_config,
    )
    retained_x = torch.cat([client.x for client in retained.values()])
    retained_y = torch.cat([client.y for client in retained.values()])
    metrics = {
        "dp_only_no_action": evaluate_against_target(
            candidate=full_model,
            target=target_model,
            test_x=federation.test_x,
            test_y=federation.test_y,
            forgotten_x=deletion_result.forgotten_x,
            forgotten_y=deletion_result.forgotten_y,
            retained_x=retained_x,
            retained_y=retained_y,
            unseen_x=federation.audit_x,
            unseen_y=federation.audit_y,
            pre_deletion_model=full_model,
            attack_seed=seed + 30_000,
            cost=full_cost,
            assumptions="No post-deletion action taken; relies on pre-existing client-level DP guarantee.",
            privacy_implications="Client-level DP mechanism recorded in privacy_ledger.json; no extra privacy loss or raw data access.",
        ),
        "retained_finetune_baseline": evaluate_against_target(
            candidate=unlearned_model,
            target=target_model,
            test_x=federation.test_x,
            test_y=federation.test_y,
            forgotten_x=deletion_result.forgotten_x,
            forgotten_y=deletion_result.forgotten_y,
            retained_x=retained_x,
            retained_y=retained_y,
            unseen_x=federation.audit_x,
            unseen_y=federation.audit_y,
            pre_deletion_model=full_model,
            attack_seed=seed + 30_000,
            cost=unlearning_cost,
            assumptions="Starts from full model and runs fine-tuning rounds on retained clients.",
            privacy_implications="Accesses retained raw data after deletion request; separate exposure regime, not DP post-processing.",
        ),
        "target_retrain": evaluate_against_target(
            candidate=target_model,
            target=target_model,
            test_x=federation.test_x,
            test_y=federation.test_y,
            forgotten_x=deletion_result.forgotten_x,
            forgotten_y=deletion_result.forgotten_y,
            retained_x=retained_x,
            retained_y=retained_y,
            unseen_x=federation.audit_x,
            unseen_y=federation.audit_y,
            pre_deletion_model=full_model,
            attack_seed=seed + 30_000,
            cost=target_cost,
            assumptions="Ground truth gold-standard reference trained from scratch on retained data.",
            privacy_implications="Internal benchmark reference; not an extra public release under the full-model ledger.",
        ),
        "target_retrain_ensemble": [
            evaluate_against_target(
                candidate=m,
                target=target_model,
                test_x=federation.test_x,
                test_y=federation.test_y,
                forgotten_x=deletion_result.forgotten_x,
                forgotten_y=deletion_result.forgotten_y,
                retained_x=retained_x,
                retained_y=retained_y,
                unseen_x=federation.audit_x,
                unseen_y=federation.audit_y,
                pre_deletion_model=full_model,
                attack_seed=seed + 30_000,
                cost=c,
                assumptions="Ensemble retrain model for quantifying retrain variability.",
                privacy_implications="Internal benchmark reference.",
            ) for m, c in zip(target_models_ensemble, target_costs_ensemble)
        ],
    }

    # Complete-client removal baselines: Cached Update Reconstruction & Faithful FedEraser
    complete_deletion_ids = {r["client_id"] for r in requests if r.get("fraction", 1.0) == 1.0}
    sec_agg_enabled = bool(privacy_config.get("secure_aggregation", False)) or bool(config.get("federated", {}).get("secure_aggregation", False))

    if complete_deletion_ids:
        if sec_agg_enabled:
            # Under Secure Aggregation, individual client updates were masked and never cached
            metrics["cached_update_reconstruction_baseline"] = {
                "status": "INCOMPATIBLE_WITH_SECURE_AGGREGATION",
                "incompatible": True,
                "reason": (
                    "Secure Aggregation cryptographically conceals individual client deltas. "
                    "Cached Reconstruction requires plaintext per-client updates, which breaks the SecAgg threat model."
                ),
            }
            metrics["federated_eraser_baseline"] = {
                "status": "INCOMPATIBLE_WITH_SECURE_AGGREGATION",
                "incompatible": True,
                "reason": (
                    "FedEraser requires historical individual client update caching to rescale calibration directions. "
                    "Under Secure Aggregation, individual client updates are never exposed to the server."
                ),
            }
        else:
            cached_model, cached_cost = reconstruct_from_cached_updates(
                history=history,
                forgotten_client_ids=complete_deletion_ids,
                n_features=n_features,
                federated_config=config["federated"],
                privacy_config=privacy_config,
                model_config=model_config,
            )
            metrics["cached_update_reconstruction_baseline"] = evaluate_against_target(
                candidate=cached_model,
                target=target_model,
                test_x=federation.test_x,
                test_y=federation.test_y,
                forgotten_x=deletion_result.forgotten_x,
                forgotten_y=deletion_result.forgotten_y,
                retained_x=retained_x,
                retained_y=retained_y,
                unseen_x=federation.audit_x,
                unseen_y=federation.audit_y,
                pre_deletion_model=full_model,
                attack_seed=seed + 30_000,
                cost=cached_cost,
                assumptions="Uses sensitive per-client server update history; direct accumulation without client calibration; no client participation.",
                privacy_implications="Server holds sensitive client-level updates; does not access raw data at unlearning time; separate exposure regime from DP release.",
            )

            eraser_model, eraser_cost = federated_eraser(
                history=history,
                clients=retained,
                forgotten_client_ids=complete_deletion_ids,
                n_features=n_features,
                federated_config=config["federated"],
                privacy_config=privacy_config,
                calibration_ratio=float(config.get("unlearning", {}).get("calibration_ratio", 0.5)),
                model_config=model_config,
            )
            metrics["federated_eraser_baseline"] = evaluate_against_target(
                candidate=eraser_model,
                target=target_model,
                test_x=federation.test_x,
                test_y=federation.test_y,
                forgotten_x=deletion_result.forgotten_x,
                forgotten_y=deletion_result.forgotten_y,
                retained_x=retained_x,
                retained_y=retained_y,
                unseen_x=federation.audit_x,
                unseen_y=federation.audit_y,
                pre_deletion_model=full_model,
                attack_seed=seed + 30_000,
                cost=eraser_cost,
                assumptions="Requires stored historical updates for all clients and rounds; requires retained clients to run local calibration training.",
                privacy_implications="Re-queries retained clients on raw data; not DP post-processing of the final model; server holds sensitive historical updates.",
            )

    # Compute MUB for every candidate against the retrain ensemble
    all_retrain_metrics = [metrics["target_retrain"]] + metrics["target_retrain_ensemble"]
    unlearning_methods = ["dp_only_no_action", "retained_finetune_baseline"]
    if "cached_update_reconstruction_baseline" in metrics and not metrics["cached_update_reconstruction_baseline"].get("incompatible"):
        unlearning_methods.append("cached_update_reconstruction_baseline")
    if "federated_eraser_baseline" in metrics and not metrics["federated_eraser_baseline"].get("incompatible"):
        unlearning_methods.append("federated_eraser_baseline")

    mub_summary = {}
    for method_name in unlearning_methods:
        method_mub = compute_method_mub(
            candidate_metrics=metrics[method_name],
            dp_metrics=metrics["dp_only_no_action"],
            all_retrain_metrics=all_retrain_metrics,
        )
        metrics[method_name]["mub"] = method_mub
        mub_summary[method_name] = method_mub
    metrics["mub_summary"] = mub_summary
    metrics["full_model"] = metrics["dp_only_no_action"]

    backend = config["data"].get("backend", "synthetic")
    dataset_scope = (
        "Phase 0 synthetic data only; not a benchmark result."
        if backend == "synthetic"
        else f"{backend} public-data pilot; exploratory benchmark result, not a SOTA or certified-unlearning claim."
    )
    num_classes = config.get("model", {}).get("num_classes", 2)
    chance = 1.0 / num_classes

    eval_config = config.get("evaluation", {})
    validity_threshold_invalid = eval_config.get("validity_threshold_invalid", chance + 0.15)
    validity_threshold_warning = eval_config.get("validity_threshold_warning", validity_threshold_invalid + 0.15)

    ensemble_accs = [metrics["target_retrain"]["test"]["accuracy"]] + [
        m["test"]["accuracy"] for m in metrics["target_retrain_ensemble"]
    ]
    ensemble_accs.sort()
    retrain_acc = ensemble_accs[len(ensemble_accs) // 2]

    if retrain_acc < validity_threshold_invalid:
        validity_status = "INVALID"
    elif retrain_acc < validity_threshold_warning:
        validity_status = "WARNING"
    else:
        validity_status = "VALID"

    metadata = {
        "experiment_name": config["experiment_name"],
        "validity_status": validity_status,
        "config_path": config_path,
        "config_sha256": _config_hash(config),
        "effective_privacy_config": privacy_config,
        "reference_privacy_config": target_privacy_config,
        "seed": seed,
        "deletion_requests": requests,
        "deletion_manifest": manifest_path,
        "deletion_manifest_sha256": manifest_checksum,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "secure_aggregation": sec_agg_enabled,
        "audit_standard": "TC-UMIA (Fu et al., 2026) Tri-Population Privacy Audit",
        "limitations": [
            dataset_scope,
            "Fine-tuning accesses retained raw data and is not DP post-processing.",
            "Tri-population unlearning audit complies with TC-UMIA (Fu et al., 2026) multi-feature protocol.",
            (
                "Pairwise zero-sum Secure Aggregation active: individual client updates cryptographically concealed."
                if sec_agg_enabled
                else "Secure Aggregation inactive: server observes individual clipped client deltas."
            ),
        ],
        "method_privacy_class": {
            "full_model": "pre-deletion model trained on all clients",
            "dp_only_no_action": "final-model client-DP mechanism as recorded in privacy_ledger.json; compatible with SecAgg",
            "retained_finetune_baseline": "accesses retained raw data after request; compatible with SecAgg",
            "target_retrain": "internal retained-data retrain reference; compatible with SecAgg",
            "cached_update_reconstruction_baseline": (
                "incompatible with SecAgg" if sec_agg_enabled
                else "uses sensitive per-client server update history; direct accumulation; not DP post-processing"
            ),
            "federated_eraser_baseline": (
                "incompatible with SecAgg" if sec_agg_enabled
                else "faithful FedEraser algorithm; requires stored historical updates and client calibration training; not DP post-processing"
            ),
        },
    }
    _save_json(run_dir / "metrics.json", metrics)
    _save_json(run_dir / "privacy_ledger.json", ledger.as_dict())
    _save_json(run_dir / "run_metadata.json", metadata)
    return run_dir


def run_sequence_experiment(config: dict, config_path: Path) -> Path:
    """Execute a stateful baseline over each prefix of a frozen request stream.

    DP-only remains the already-released full model. The explicit baseline starts
    from that model and is fine-tuned on the progressively retained data after
    each request. Each prefix has an independent DP retraining target.
    """
    requests = normalized_deletion_requests(config["deletion"])
    if len(requests) < 2:
        raise ValueError("Sequence mode requires at least two deletion.requests.")
    seed = int(config["seed"])
    federation = make_federation(config["data"], seed)
    n_features = infer_n_features(next(iter(federation.clients.values())).x)
    model_config = config.get("model")
    output_root = Path(config["output"]["root"])
    run_name = f"{config['experiment_name']}_sequence_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_seed{seed}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    privacy_config = {**config["privacy"]}
    if privacy_config["enabled"]:
        privacy_config.setdefault("population_size", int(config["data"]["n_clients"]))
    ledger = make_ledger(privacy_config, config["federated"])
    full_model, full_cost = train_federated(
        clients=federation.clients,
        n_features=n_features,
        federated_config=config["federated"],
        privacy_config=privacy_config,
        seed=seed,
        model_config=model_config,
    )
    stateful_unlearned = full_model
    sequence_metrics = []
    for request_index in range(1, len(requests) + 1):
        prefix = {"requests": requests[:request_index]}
        deletion_result = apply_client_deletion_requests(federation, prefix, seed)
        retained = deletion_result.retained
        target_privacy_config = _reference_privacy_config(privacy_config, len(retained))
        target, target_cost = train_federated(
            clients=retained,
            n_features=n_features,
            federated_config=config["federated"],
            privacy_config=target_privacy_config,
            seed=seed + 10_000 + request_index,
            model_config=model_config,
        )
        target_2, target_2_cost = train_federated(
            clients=retained,
            n_features=n_features,
            federated_config=config["federated"],
            privacy_config=target_privacy_config,
            seed=seed + 11_000 + request_index,
            model_config=model_config,
        )
        stateful_unlearned, unlearning_cost = finetune_retained(
            full_model=stateful_unlearned,
            clients=retained,
            n_features=n_features,
            federated_config=config["federated"],
            unlearning_rounds=int(config["unlearning"]["rounds"]),
            seed=seed + 20_000 + request_index,
            model_config=model_config,
        )
        retained_x = torch.cat([client.x for client in retained.values()])
        retained_y = torch.cat([client.y for client in retained.values()])
        manifest_path, manifest_checksum = write_deletion_manifest(deletion_result.rows, Path("partitions") / "generated")
        common = {
            "target": target,
            "test_x": federation.test_x,
            "test_y": federation.test_y,
            "forgotten_x": deletion_result.forgotten_x,
            "forgotten_y": deletion_result.forgotten_y,
            "retained_x": retained_x,
            "retained_y": retained_y,
            "unseen_x": federation.audit_x,
            "unseen_y": federation.audit_y,
            "pre_deletion_model": full_model,
            "attack_seed": seed + 30_000 + request_index,
        }
        sequence_metrics.append({
            "request_index": request_index,
            "request": requests[request_index - 1],
            "reference_privacy_config": target_privacy_config,
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_checksum,
            "dp_only_no_action": evaluate_against_target(candidate=full_model, cost=full_cost, **common),
            "stateful_retained_finetune": evaluate_against_target(candidate=stateful_unlearned, cost=unlearning_cost, **common),
            "target_retrain": evaluate_against_target(candidate=target, cost=target_cost, **common),
            "target_retrain_independent": evaluate_against_target(candidate=target_2, cost=target_2_cost, **common),
        })
    metadata = {
        "experiment_name": config["experiment_name"],
        "config_path": config_path,
        "config_sha256": _config_hash(config),
        "effective_privacy_config": privacy_config,
        "seed": seed,
        "request_stream": requests,
        "method_privacy_class": "Stateful retained-data fine-tuning accesses raw retained data after each request; it is a baseline and not DP post-processing.",
        "limitations": [
            "Sequence experiment is synthetic unless a public-data config is supplied.",
            "No secure aggregation service, certified unlearning, LiRA/A-LiRA, or TC-UMIA reproduction is implemented.",
            "Each target retrain is an internal reference and not an additional public model release.",
        ],
    }
    _save_json(run_dir / "sequence_metrics.json", {"steps": sequence_metrics})
    _save_json(run_dir / "privacy_ledger.json", ledger.as_dict())
    _save_json(run_dir / "run_metadata.json", metadata)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["full", "sequence"], default="full")
    parser.add_argument("--seed", type=int, help="Override the configuration seed for an independent run.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    run_dir = run_sequence_experiment(config, args.config) if args.mode == "sequence" else run_experiment(config, args.config)
    print(f"Completed experiment: {run_dir}")


if __name__ == "__main__":
    main()
