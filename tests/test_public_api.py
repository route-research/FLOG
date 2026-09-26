import torch

from flog import FLOGPolicy, compute_reward_metrics
from flog.decoder import CoreImplementationUnavailable, flog_rollout


def test_model_and_metrics():
    policy = FLOGPolicy(hidden_dim=32, encode_layers=1, heads=4, ff_dim=64)
    agent = torch.rand(2, 5, 5)
    city = torch.rand(2, 20, 3)
    agent_static, city_static = policy.encode(agent, city)
    assert agent_static.shape == (2, 5, 32)
    assert city_static.shape == (2, 20, 32)

    rewards = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]]])
    serviceable = torch.tensor([[True, True, True]])
    metrics = compute_reward_metrics(rewards, serviceable)
    assert metrics["total"].shape == (1, 2)


def test_redacted_rollout_is_explicit():
    try:
        flog_rollout()
    except CoreImplementationUnavailable:
        return
    raise AssertionError("redacted rollout must raise CoreImplementationUnavailable")
