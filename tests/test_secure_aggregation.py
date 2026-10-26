"""Tests for Secure Aggregation (SecAgg) simulator and unlearning compatibility."""

import numpy as np
import pytest
import torch

from dp_forgetbench.federated import (
    ClientDataset,
    FederatedHistory,
    IncompatibleSecurityRegimeError,
    SecureAggregator,
    federated_eraser,
    reconstruct_from_cached_updates,
    train_federated,
)


def test_secure_aggregator_pairwise_zero_sum_cancellation():
    """Verify that simulated pairwise masks cancel out exactly to produce the true sum."""
    rng = np.random.default_rng(42)
    deltas = {
        0: {"w": torch.tensor([1.5, -2.0, 3.0]), "b": torch.tensor([0.5])},
        1: {"w": torch.tensor([-0.5, 4.0, -1.0]), "b": torch.tensor([-1.2])},
        2: {"w": torch.tensor([2.0, 1.0, -0.5]), "b": torch.tensor([0.8])},
        3: {"w": torch.tensor([-1.0, -0.5, 2.5]), "b": torch.tensor([0.1])},
    }

    true_sum_w = sum(d["w"] for d in deltas.values())
    true_sum_b = sum(d["b"] for d in deltas.values())

    masked_aggregate = SecureAggregator.aggregate_masked(deltas, rng)

    assert torch.allclose(masked_aggregate["w"], true_sum_w, atol=1e-6)
    assert torch.allclose(masked_aggregate["b"], true_sum_b, atol=1e-6)


def test_secure_aggregator_single_client():
    rng = np.random.default_rng(7)
    deltas = {0: {"w": torch.tensor([2.5, -1.0])}}
    agg = SecureAggregator.aggregate_masked(deltas, rng)
    assert torch.allclose(agg["w"], torch.tensor([2.5, -1.0]))


def test_secagg_blocks_cached_reconstruction_and_federated_eraser():
    """Verify that unlearning baselines requiring individual client updates fail under SecAgg."""
    torch.manual_seed(123)
    n_features = 4
    clients = {
        0: ClientDataset(0, torch.randn(10, n_features), torch.randint(0, 2, (10,)).float()),
        1: ClientDataset(1, torch.randn(10, n_features), torch.randint(0, 2, (10,)).float()),
        2: ClientDataset(2, torch.randn(10, n_features), torch.randint(0, 2, (10,)).float()),
    }
    fed_cfg = {"rounds": 2, "client_sample_rate": 1.0, "learning_rate": 0.05, "local_epochs": 1, "local_batch_size": 5}
    priv_cfg = {"enabled": True, "clip_norm": 1.0, "noise_multiplier": 0.5, "population_size": 3, "secure_aggregation": True}

    history = FederatedHistory()
    model, cost = train_federated(
        clients=clients,
        n_features=n_features,
        federated_config=fed_cfg,
        privacy_config=priv_cfg,
        seed=123,
        history=history,
    )

    assert history.secure_aggregation is True
    assert all(r.deltas_by_client is None for r in history.rounds)
    assert all(r.aggregate_delta is not None for r in history.rounds)

    # Calling Cached Reconstruction must raise IncompatibleSecurityRegimeError
    with pytest.raises(IncompatibleSecurityRegimeError) as exc_cached:
        reconstruct_from_cached_updates(
            history=history,
            forgotten_client_ids={0},
            n_features=n_features,
            federated_config=fed_cfg,
            privacy_config=priv_cfg,
        )
    assert "Secure Aggregation" in str(exc_cached.value)

    # Calling FedEraser must also raise IncompatibleSecurityRegimeError
    with pytest.raises(IncompatibleSecurityRegimeError) as exc_eraser:
        federated_eraser(
            history=history,
            clients={1: clients[1], 2: clients[2]},
            forgotten_client_ids={0},
            n_features=n_features,
            federated_config=fed_cfg,
            privacy_config=priv_cfg,
        )
    assert "Secure Aggregation" in str(exc_eraser.value)
