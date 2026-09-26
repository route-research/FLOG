from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .metrics import EPS


class CoreImplementationUnavailable(RuntimeError):
    """Raised when a protected algorithmic component is invoked."""


@dataclass
class FLOGResult:
    rewards: torch.Tensor
    lengths: torch.Tensor
    routes: List[List[List[List[int]]]]
    log_probability: torch.Tensor
    entropy: torch.Tensor
    serviceable: torch.Tensor
    policy_steps: torch.Tensor
    pre_recovery_rewards: torch.Tensor
    recovery_rewards: Optional[torch.Tensor] = None
    recovery_lengths: Optional[torch.Tensor] = None


def _repeat_samples(tensor: torch.Tensor, samples: int) -> torch.Tensor:
    shape = tensor.shape
    return (
        tensor.unsqueeze(1)
        .expand(shape[0], samples, *shape[1:])
        .reshape(shape[0] * samples, *shape[1:])
        .contiguous()
    )


def _masked_stats(
    reward: torch.Tensor, serviceable: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    minimum = reward.masked_fill(~serviceable, float("inf")).min(dim=1).values
    maximum = reward.masked_fill(~serviceable, float("-inf")).max(dim=1).values
    has_agent = serviceable.any(dim=1)
    minimum = torch.where(has_agent, minimum, torch.zeros_like(minimum))
    maximum = torch.where(has_agent, maximum, torch.zeros_like(maximum))
    ratio = torch.where(maximum > EPS, minimum / (maximum + EPS), torch.zeros_like(maximum))
    return minimum, maximum, ratio


def _dynamic_features(
    agent_rewards: torch.Tensor,
    route_lengths: torch.Tensor,
    route_sizes: torch.Tensor,
    budgets: torch.Tensor,
    serviceable: torch.Tensor,
    cnum: int,
) -> torch.Tensor:
    """Public dynamic state representation used by the policy backbone."""
    total_scale = agent_rewards.sum(dim=1, keepdim=True).clamp_min(1.0)
    reward_ratio = agent_rewards / total_scale
    budget_ratio = route_lengths / budgets.clamp_min(EPS)
    remaining_ratio = (budgets - route_lengths).clamp_min(0.0) / budgets.clamp_min(EPS)
    unserved = (agent_rewards <= EPS).float()
    route_ratio = (route_sizes.float() - 2.0).clamp_min(0.0) / max(1, cnum)
    return torch.stack(
        [reward_ratio, remaining_ratio, budget_ratio, serviceable.float(), unserved, route_ratio],
        dim=-1,
    )


def _pair_features(
    rewards: torch.Tensor,
    insertion_cost: torch.Tensor,
    route_lengths: torch.Tensor,
    budgets: torch.Tensor,
) -> torch.Tensor:
    """Public pairwise features passed to the policy scoring head."""
    reward_norm = rewards / rewards.max(dim=1, keepdim=True).values.clamp_min(EPS)
    reward_norm = reward_norm.unsqueeze(1).expand_as(insertion_cost)
    budget = budgets.unsqueeze(-1).clamp_min(EPS)
    cost_ratio = insertion_cost / budget
    slack = (budget - route_lengths.unsqueeze(-1) - insertion_cost).clamp_min(0.0) / budget
    return torch.stack([reward_norm, cost_ratio, slack], dim=-1)


def fairness_guided_construction(*args, **kwargs):
    """Protected FGC implementation.

    The review repository exposes the module boundary and public model/data
    interfaces while withholding the implementation of the fairness-guided
    action filtering and tie-breaking logic.
    """
    raise CoreImplementationUnavailable(
        "Fairness-Guided Construction is intentionally redacted in the anonymous review release."
    )


def residual_reward_recovery(*args, **kwargs):
    """Protected residual-reward/fairness-preserving recovery implementation."""
    raise CoreImplementationUnavailable(
        "Residual Reward Recovery is intentionally redacted in the anonymous review release."
    )


def flog_rollout(*args, **kwargs) -> FLOGResult:
    """Public rollout entry point; core decision logic is withheld in this release."""
    raise CoreImplementationUnavailable(
        "The complete FLOG rollout implementation will be released with the full public artifact."
    )
