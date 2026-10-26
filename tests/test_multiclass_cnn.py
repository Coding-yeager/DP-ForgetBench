"""Phase 2 smoke tests: cifar10_multiclass data backend + small_groupnorm_cnn model.

These mock torchvision.datasets.CIFAR10 with small random image tensors so the
tests run in seconds without downloading the real dataset. They check wiring
and shapes, not model quality -- accuracy on random labels is expected to be
near chance.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


class _FakeCIFAR10:
    """Minimal stand-in with the two attributes data.py reads: .data, .targets."""

    def __init__(self, root, train, download=False) -> None:
        n = 400 if train else 150
        rng = np.random.default_rng(0 if train else 1)
        self.data = rng.integers(0, 255, size=(n, 32, 32, 3), dtype=np.uint8)
        self.targets = (np.arange(n) % 10).tolist()


import sys
import types

if "torchvision" not in sys.modules:
    tv_mod = types.ModuleType("torchvision")
    tv_datasets = types.ModuleType("torchvision.datasets")
    tv_datasets.CIFAR10 = _FakeCIFAR10
    tv_mod.datasets = tv_datasets
    sys.modules["torchvision"] = tv_mod
    sys.modules["torchvision.datasets"] = tv_datasets


def test_make_cifar10_multiclass_federation_shapes_and_determinism() -> None:
    from dp_forgetbench.data import make_cifar10_multiclass_federation

    config = {
        "n_clients": 5,
        "samples_per_client": 16,
        "heterogeneity_alpha": 0.5,
        "audit_samples": 40,
        "test_samples": 60,
    }
    with patch("torchvision.datasets.CIFAR10", _FakeCIFAR10):
        federation = make_cifar10_multiclass_federation(config, seed=7)
        federation_again = make_cifar10_multiclass_federation(config, seed=7)

    assert len(federation.clients) == 5
    for client in federation.clients.values():
        assert tuple(client.x.shape) == (16, 3, 32, 32)
        assert client.y.shape == (16,)
        assert client.y.dtype == torch.long
    assert federation.test_x.shape[0] == 60
    assert federation.audit_x.shape[0] == 40
    # All 10 classes should appear somewhere in the audit pool given the fixture.
    assert set(federation.audit_y.tolist()) == set(range(10))
    # Same seed -> identical federation (required for the LiRA CLI's re-derivation).
    assert torch.equal(federation.clients[0].x, federation_again.clients[0].x)
    assert torch.equal(federation.clients[0].y, federation_again.clients[0].y)


def test_make_federation_dispatches_to_multiclass_backend() -> None:
    from dp_forgetbench.data import make_federation

    config = {
        "backend": "cifar10_multiclass",
        "n_clients": 3,
        "samples_per_client": 10,
        "heterogeneity_alpha": 1.0,
        "audit_samples": 20,
        "test_samples": 20,
    }
    with patch("torchvision.datasets.CIFAR10", _FakeCIFAR10):
        federation = make_federation(config, seed=3)
    assert len(federation.clients) == 3


def test_small_groupnorm_cnn_end_to_end_run(tmp_path: Path, monkeypatch) -> None:
    """Full run_experiment pipeline with the CNN model on mocked multiclass data."""
    from dp_forgetbench.config import load_config
    from dp_forgetbench.run import run_experiment

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "phase2_cifar10_multiclass_cnn.yaml")
    config["data"]["n_clients"] = 6
    config["data"]["samples_per_client"] = 12
    config["data"]["audit_samples"] = 30
    config["data"]["test_samples"] = 30
    config["privacy"]["population_size"] = 6
    config["federated"]["rounds"] = 2
    config["federated"]["local_batch_size"] = 6
    config["federated"]["client_sample_rate"] = 0.8
    config["unlearning"]["rounds"] = 1
    config["output"]["root"] = str(tmp_path / "results")
    monkeypatch.chdir(tmp_path)

    with patch("torchvision.datasets.CIFAR10", _FakeCIFAR10):
        output = run_experiment(config, root / "configs" / "phase2_cifar10_multiclass_cnn.yaml")

    assert (output / "metrics.json").exists()
    metrics_text = (output / "metrics.json").read_text(encoding="utf-8")
    assert "dp_only_no_action" in metrics_text
    assert "cached_update_reconstruction_baseline" in metrics_text  # full-client deletion path
    assert "forgotten_vs_unseen_loss_mia" in metrics_text
    assert "tri_population_pre_post_probe" in metrics_text


def test_loss_membership_probe_handles_ten_classes() -> None:
    """Regression test: the old (0.0, 1.0)-only label loop silently dropped 10-class data."""
    from dp_forgetbench.attacks import loss_membership_probe
    from dp_forgetbench.federated import SmallGroupNormCNN

    torch.manual_seed(0)
    model = SmallGroupNormCNN(num_classes=10)
    member_x = torch.randn(40, 3, 32, 32)
    member_y = torch.randint(0, 10, (40,))
    unseen_x = torch.randn(40, 3, 32, 32)
    unseen_y = torch.randint(0, 10, (40,))

    result = loss_membership_probe(model, member_x, member_y, unseen_x, unseen_y)
    # With the old binary-only label loop this would raise
    # "Attack populations do not have overlapping label support."
    assert result["members"] > 0
    assert result["nonmembers"] > 0
    assert 0.0 <= result["auc"] <= 1.0


def test_groupnorm_resnet18_factory_has_cifar_multiclass_shape() -> None:
    from dp_forgetbench.federated import make_model
    from dp_forgetbench.models import GroupNormResNet18

    model = make_model({"name": "groupnorm_resnet18", "num_classes": 10}, None)
    assert isinstance(model, GroupNormResNet18)
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
    logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 10)
