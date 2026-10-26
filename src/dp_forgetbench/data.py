"""Deterministic synthetic federation and versioned deletion manifests.

The synthetic source is deliberately used only for Phase 0. It makes every
privacy/deletion invariant testable before adapters for public datasets are added.
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class ClientDataset:
    client_id: int
    x: torch.Tensor
    y: torch.Tensor


@dataclass(frozen=True)
class Federation:
    clients: dict[int, ClientDataset]
    test_x: torch.Tensor
    test_y: torch.Tensor
    audit_x: torch.Tensor
    audit_y: torch.Tensor


@dataclass(frozen=True)
class DeletionResult:
    retained: dict[int, ClientDataset]
    forgotten_x: torch.Tensor
    forgotten_y: torch.Tensor
    rows: list[dict[str, int | float | str]]


def make_synthetic_federation(data_config: dict, seed: int) -> Federation:
    """Build a binary task with client-specific class imbalance and feature shift."""
    rng = np.random.default_rng(seed)
    n_clients = int(data_config["n_clients"])
    samples_per_client = int(data_config["samples_per_client"])
    n_features = int(data_config["n_features"])
    heterogeneity = float(data_config["heterogeneity"])
    if not 0.0 <= heterogeneity <= 1.0:
        raise ValueError("heterogeneity must lie in [0, 1].")

    direction = rng.normal(size=n_features)
    direction = direction / np.linalg.norm(direction)
    clients: dict[int, ClientDataset] = {}
    for client_id in range(n_clients):
        class_probability = 0.5 + heterogeneity * (0.42 if client_id % 2 else -0.42)
        labels = rng.binomial(1, class_probability, size=samples_per_client).astype(np.float32)
        client_shift = rng.normal(0.0, 0.45 * heterogeneity, size=n_features)
        features = rng.normal(0.0, 1.0, size=(samples_per_client, n_features))
        features += ((2.0 * labels - 1.0)[:, None] * direction * 1.25) + client_shift
        clients[client_id] = ClientDataset(
            client_id=client_id,
            x=torch.tensor(features, dtype=torch.float32),
            y=torch.tensor(labels, dtype=torch.float32),
        )

    test_samples = int(data_config["test_samples"])
    test_labels = rng.binomial(1, 0.5, size=test_samples).astype(np.float32)
    test_features = rng.normal(0.0, 1.0, size=(test_samples, n_features))
    test_features += (2.0 * test_labels - 1.0)[:, None] * direction * 1.25
    # This is a held-out population generated from the same global process. It is
    # reserved exclusively for attack calibration/evaluation, never training.
    audit_labels = rng.binomial(1, 0.5, size=test_samples).astype(np.float32)
    audit_features = rng.normal(0.0, 1.0, size=(test_samples, n_features))
    audit_features += (2.0 * audit_labels - 1.0)[:, None] * direction * 1.25
    return Federation(
        clients=clients,
        test_x=torch.tensor(test_features, dtype=torch.float32),
        test_y=torch.tensor(test_labels, dtype=torch.float32),
        audit_x=torch.tensor(audit_features, dtype=torch.float32),
        audit_y=torch.tensor(audit_labels, dtype=torch.float32),
    )


def _load_cifar10_raw(root: str, download: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10 data as (train_images, train_labels, test_images, test_labels).

    Tries torchvision if installed; falls back to loading directly from the
    standard CIFAR-10 python batch files (e.g. root/cifar-10-batches-py).
    """
    try:
        from torchvision.datasets import CIFAR10
        try:
            train = CIFAR10(root=root, train=True, download=False)
            test = CIFAR10(root=root, train=False, download=False)
        except RuntimeError:
            if not download:
                raise
            train = CIFAR10(root=root, train=True, download=True)
            test = CIFAR10(root=root, train=False, download=True)
        return np.asarray(train.data), np.asarray(train.targets), np.asarray(test.data), np.asarray(test.targets)
    except Exception:
        import os
        import pickle
        cifar_dir = Path(root) / "cifar-10-batches-py"
        if not cifar_dir.exists():
            if download:
                import tarfile
                import urllib.request
                Path(root).mkdir(parents=True, exist_ok=True)
                url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
                tar_path = Path(root) / "cifar-10-python.tar.gz"
                urllib.request.urlretrieve(url, tar_path)
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(path=root)
            else:
                raise RuntimeError(
                    f"CIFAR-10 dataset not found in {cifar_dir} and download=False. "
                    "Ensure batches are extracted in data/cifar-10-batches-py or set download: true."
                )
        train_batches = []
        train_targets = []
        for i in range(1, 6):
            batch_path = cifar_dir / f"data_batch_{i}"
            with open(batch_path, "rb") as f:
                entry = pickle.load(f, encoding="bytes")
                train_batches.append(entry[b"data"])
                train_targets.extend(entry[b"labels"])
        train_data = np.vstack(train_batches).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

        test_path = cifar_dir / "test_batch"
        with open(test_path, "rb") as f:
            test_entry = pickle.load(f, encoding="bytes")
            test_data = test_entry[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            test_targets = test_entry[b"labels"]
        return train_data, np.array(train_targets), test_data, np.array(test_targets)


def make_cifar10_binary_federation(data_config: dict, seed: int) -> Federation:
    """Create a small, deterministic real-data Phase 1 federation from CIFAR-10.

    This is a CPU validation configuration using classes 0 and 1 with a linear
    model. It verifies public-data loading, deletion, DP accounting, and attacks;
    it is not the final CIFAR-10/CIFAR-100 multi-class benchmark.
    """
    root = str(data_config.get("dataset_root", "data"))
    download = bool(data_config.get("download", False))
    raw_tr_data, raw_tr_targets, raw_te_data, raw_te_targets = _load_cifar10_raw(root=root, download=download)
    rng = np.random.default_rng(seed)
    train_images = torch.tensor(raw_tr_data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    train_labels = torch.tensor(raw_tr_targets, dtype=torch.long)
    test_images = torch.tensor(raw_te_data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    test_labels = torch.tensor(raw_te_targets, dtype=torch.long)
    train_mask = (train_labels == 0) | (train_labels == 1)
    test_mask = (test_labels == 0) | (test_labels == 1)
    pool_kernel = int(data_config.get("feature_pool_kernel", 1))
    if pool_kernel < 1 or 32 % pool_kernel:
        raise ValueError("feature_pool_kernel must be a positive divisor of 32.")
    if pool_kernel > 1:
        # Fixed public preprocessing: no learned parameters and no private-data
        # fitting. It reduces DP noise exposure for this CPU feasibility pilot.
        train_images = functional.avg_pool2d(train_images, kernel_size=pool_kernel)
        test_images = functional.avg_pool2d(test_images, kernel_size=pool_kernel)
    x_pool, y_pool = train_images[train_mask].flatten(1), train_labels[train_mask].float()
    x_test, y_test = test_images[test_mask].flatten(1), test_labels[test_mask].float()
    x_pool = (x_pool - 0.5) / 0.5
    x_test = (x_test - 0.5) / 0.5
    n_clients = int(data_config["n_clients"])
    samples_per_client = int(data_config["samples_per_client"])
    audit_samples = int(data_config.get("audit_samples", 400))
    test_samples = int(data_config.get("test_samples", len(y_test)))
    required = n_clients * samples_per_client + audit_samples
    if required > len(y_pool):
        raise ValueError(f"Requested {required} train/audit examples but only {len(y_pool)} selected CIFAR-10 examples exist.")
    # Shuffle label-specific pools then assign a skewed, but auditable client sample.
    pools = {label: rng.permutation(torch.where(y_pool == label)[0].numpy()).tolist() for label in (0, 1)}
    clients: dict[int, ClientDataset] = {}
    heterogeneity = float(data_config.get("heterogeneity", 0.75))
    for client_id in range(n_clients):
        p_one = 0.5 + heterogeneity * (0.40 if client_id % 2 else -0.40)
        labels = rng.binomial(1, p_one, size=samples_per_client)
        indices = []
        for label in labels:
            if not pools[int(label)]:
                raise ValueError("Class pool exhausted; reduce samples_per_client or heterogeneity.")
            indices.append(pools[int(label)].pop())
        idx = torch.tensor(indices, dtype=torch.long)
        clients[client_id] = ClientDataset(client_id=client_id, x=x_pool[idx], y=y_pool[idx])
    remaining = [index for label in (0, 1) for index in pools[label]]
    audit_idx = torch.tensor(rng.choice(remaining, size=audit_samples, replace=False), dtype=torch.long)
    test_idx = torch.tensor(rng.choice(len(y_test), size=min(test_samples, len(y_test)), replace=False), dtype=torch.long)
    return Federation(
        clients=clients,
        test_x=x_test[test_idx],
        test_y=y_test[test_idx],
        audit_x=x_pool[audit_idx],
        audit_y=y_pool[audit_idx],
    )


def make_cifar10_multiclass_federation(data_config: dict, seed: int) -> Federation:
    """Create a 10-class, image-tensor CIFAR-10 federation for the Phase 2 CNN model.

    Unlike ``make_cifar10_binary_federation``, this keeps images as ``(N, 3, 32, 32)``
    tensors (no flattening/pooling into fixed public features) so a convolutional
    model can be trained. Client heterogeneity is controlled with a symmetric
    Dirichlet split over the 10 classes: ``alpha -> 0`` yields near-single-class
    clients (highly non-IID) and ``alpha`` large yields near-uniform label mixes,
    matching the standard non-IID-FL convention used by FedAvg-family papers.
    This is a Phase 2 backend: it is CPU-usable for smoke tests but is intended
    to be trained on GPU for the full benchmark grid.
    """
    root = str(data_config.get("dataset_root", "data"))
    download = bool(data_config.get("download", False))
    raw_tr_data, raw_tr_targets, raw_te_data, raw_te_targets = _load_cifar10_raw(root=root, download=download)
    rng = np.random.default_rng(seed)
    train_images = torch.tensor(raw_tr_data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    train_labels = torch.tensor(raw_tr_targets, dtype=torch.long)
    test_images = torch.tensor(raw_te_data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    test_labels = torch.tensor(raw_te_targets, dtype=torch.long)
    # Fixed public per-channel normalization (CIFAR-10 train-set statistics).
    # No learned parameters and no private-data fitting.
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    train_images = (train_images - mean) / std
    test_images = (test_images - mean) / std

    n_clients = int(data_config["n_clients"])
    samples_per_client = int(data_config["samples_per_client"])
    audit_samples = int(data_config.get("audit_samples", 400))
    test_samples = int(data_config.get("test_samples", len(test_labels)))
    alpha = float(data_config.get("heterogeneity_alpha", 100.0))
    if alpha <= 0.0:
        raise ValueError("data.heterogeneity_alpha must be positive (small values are more non-IID).")
    n_classes = 10
    required = n_clients * samples_per_client + audit_samples
    if required > len(train_labels):
        raise ValueError(f"Requested {required} train/audit examples but only {len(train_labels)} CIFAR-10 train examples exist.")

    class_pools = {label: rng.permutation(torch.where(train_labels == label)[0].numpy()).tolist() for label in range(n_classes)}
    # Reserve a held-out audit pool first, drawn evenly across classes, so it never
    # overlaps client training data regardless of the Dirichlet draw below.
    audit_indices: list[int] = []
    per_class_audit = max(1, audit_samples // n_classes)
    for label in range(n_classes):
        take = min(per_class_audit, len(class_pools[label]))
        audit_indices.extend(class_pools[label][:take])
        class_pools[label] = class_pools[label][take:]
    audit_indices = audit_indices[:audit_samples]

    clients: dict[int, ClientDataset] = {}
    class_proportions = rng.dirichlet(alpha=[alpha] * n_classes, size=n_clients)
    for client_id in range(n_clients):
        counts = np.floor(class_proportions[client_id] * samples_per_client).astype(int)
        # Distribute any remainder from flooring back onto the classes with the
        # largest fractional part so totals still sum to samples_per_client.
        remainder = samples_per_client - counts.sum()
        if remainder > 0:
            fractional = class_proportions[client_id] * samples_per_client - counts
            top_up = np.argsort(-fractional)[:remainder]
            counts[top_up] += 1
        indices: list[int] = []
        for label in range(n_classes):
            take = min(int(counts[label]), len(class_pools[label]))
            if take < counts[label]:
                raise ValueError(
                    "Class pool exhausted while building a Dirichlet client split; "
                    "reduce n_clients, samples_per_client, or audit_samples, or raise heterogeneity_alpha."
                )
            indices.extend(class_pools[label][:take])
            class_pools[label] = class_pools[label][take:]
        idx = torch.tensor(indices, dtype=torch.long)
        clients[client_id] = ClientDataset(client_id=client_id, x=train_images[idx], y=train_labels[idx])

    audit_idx = torch.tensor(audit_indices, dtype=torch.long)
    test_idx = torch.tensor(rng.choice(len(test_labels), size=min(test_samples, len(test_labels)), replace=False), dtype=torch.long)
    return Federation(
        clients=clients,
        test_x=test_images[test_idx],
        test_y=test_labels[test_idx],
        audit_x=train_images[audit_idx],
        audit_y=train_labels[audit_idx],
    )


def make_federation(data_config: dict, seed: int) -> Federation:
    backend = data_config.get("backend", "synthetic")
    if backend == "synthetic":
        return make_synthetic_federation(data_config, seed)
    if backend == "cifar10_binary":
        return make_cifar10_binary_federation(data_config, seed)
    if backend == "cifar10_multiclass":
        return make_cifar10_multiclass_federation(data_config, seed)
    raise ValueError(f"Unsupported dataset backend: {backend}")


def write_deletion_manifest(rows: list[dict[str, int | float | str]], directory: str | Path) -> tuple[Path, str]:
    """Write an auditable request manifest and return its content checksum."""
    if not rows:
        raise ValueError("Deletion manifest cannot be empty.")
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    with buffer:
        handle = buffer
        writer = csv.DictWriter(handle, fieldnames=["request_index", "client_id", "sample_index", "fraction", "action"])
        writer.writeheader()
        writer.writerows(rows)
        content = handle.getvalue().encode("utf-8")
    checksum = hashlib.sha256(content).hexdigest()
    path = output_dir / f"deletion_{checksum[:16]}.csv"
    path.write_bytes(content)
    return path, checksum


def normalized_deletion_requests(deletion_config: dict) -> list[dict]:
    """Normalize one or multiple client requests into a frozen cumulative request list."""
    requests = deletion_config.get("requests")
    if requests is None:
        requests = [{"client_id": deletion_config["client_id"], "fraction": deletion_config.get("fraction", 1.0)}]
    if not isinstance(requests, list) or not requests:
        raise ValueError("deletion.requests must be a non-empty list.")
    normalized = []
    seen = set()
    for index, request in enumerate(requests):
        client_id = int(request["client_id"])
        fraction = float(request.get("fraction", 1.0))
        if not 0 < fraction <= 1:
            raise ValueError("Deletion fraction must be in (0, 1].")
        if client_id in seen:
            raise ValueError("A client may appear at most once in a cumulative request list.")
        normalized.append({"request_index": index, "client_id": client_id, "fraction": fraction})
        seen.add(client_id)
    return normalized


def apply_client_deletion_requests(federation: Federation, deletion_config: dict, seed: int) -> DeletionResult:
    """Create retained and forgotten populations without mutating the source federation.

    Partial deletion is deterministic stratified index sampling within a client.
    A request list is cumulative: targets are retrained without all requested data.
    """
    requests = normalized_deletion_requests(deletion_config)
    retained = dict(federation.clients)
    forgotten_x, forgotten_y, rows = [], [], []
    rng = np.random.default_rng(seed + 40_000)
    for request in requests:
        client_id, fraction = request["client_id"], request["fraction"]
        if client_id not in retained:
            raise KeyError(f"Unknown or already removed client: {client_id}")
        client = retained[client_id]
        n_remove = len(client.y) if fraction == 1.0 else max(1, int(round(len(client.y) * fraction)))
        indices = np.sort(rng.choice(len(client.y), size=n_remove, replace=False))
        index_tensor = torch.tensor(indices, dtype=torch.long)
        forgotten_x.append(client.x[index_tensor])
        forgotten_y.append(client.y[index_tensor])
        rows.extend(
            {
                "request_index": request["request_index"],
                "client_id": client_id,
                "sample_index": int(sample_index),
                "fraction": fraction,
                "action": "forget",
            }
            for sample_index in indices
        )
        if n_remove == len(client.y):
            del retained[client_id]
        else:
            keep = torch.ones(len(client.y), dtype=torch.bool)
            keep[index_tensor] = False
            retained[client_id] = ClientDataset(client_id=client_id, x=client.x[keep], y=client.y[keep])
    if not retained:
        raise ValueError("Deletion cannot remove every client.")
    return DeletionResult(
        retained=retained,
        forgotten_x=torch.cat(forgotten_x),
        forgotten_y=torch.cat(forgotten_y),
        rows=rows,
    )


def write_client_deletion_manifest(
    federation: Federation, forgotten_client_id: int, directory: str | Path
) -> tuple[Path, str]:
    """Backwards-compatible wrapper for one complete-client deletion."""
    result = apply_client_deletion_requests(federation, {"client_id": forgotten_client_id}, seed=0)
    return write_deletion_manifest(result.rows, directory)


def retained_clients(federation: Federation, forgotten_client_id: int) -> dict[int, ClientDataset]:
    """Compatibility helper for a complete one-client deletion."""
    return apply_client_deletion_requests(federation, {"client_id": forgotten_client_id}, seed=0).retained
