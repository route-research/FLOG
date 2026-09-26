from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


EPS = 1e-8


def single_node_serviceability(
    merge_coord: torch.Tensor,
    budgets: torch.Tensor,
    anum: int,
    cnum: int,
    city_rewards: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-agent serviceability and per-agent/per-city single-visit costs.

    ``merge_coord`` follows GRIP's layout: cities, starts, ends.  The public review release uses the open-route setting with distinct starts and ends.
    """

    city = merge_coord[:, :cnum]
    start = merge_coord[:, cnum : cnum + anum]
    end = merge_coord[:, cnum + anum : cnum + 2 * anum]
    out = torch.cdist(start, city)
    back = torch.cdist(end, city)
    single_cost = out + back
    feasible = single_cost <= budgets.unsqueeze(-1) + 1e-9
    if city_rewards is not None:
        positive = city_rewards > 0
        feasible = feasible & positive.unsqueeze(1)
    serviceable = feasible.any(dim=-1)
    return serviceable, single_cost


def _masked_min_max(
    reward: torch.Tensor,
    serviceable: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = serviceable
    while mask.dim() < reward.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(reward)
    minimum = reward.masked_fill(~mask, float("inf")).min(dim=-1).values
    maximum = reward.masked_fill(~mask, float("-inf")).max(dim=-1).values
    has_agent = mask.any(dim=-1)
    minimum = torch.where(has_agent, minimum, torch.zeros_like(minimum))
    maximum = torch.where(has_agent, maximum, torch.zeros_like(maximum))
    return minimum, maximum


def compute_reward_metrics(
    per_agent_reward: torch.Tensor,
    serviceable: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute raw and feasibility-aware metrics along the agent dimension."""

    total = per_agent_reward.sum(dim=-1)
    raw_min = per_agent_reward.min(dim=-1).values
    raw_max = per_agent_reward.max(dim=-1).values
    raw_rbr = torch.where(
        raw_max > EPS,
        raw_min / (raw_max + EPS),
        torch.zeros_like(raw_max),
    )

    feasible_min, feasible_max = _masked_min_max(per_agent_reward, serviceable)
    feasible_rbr = torch.where(
        feasible_max > EPS,
        feasible_min / (feasible_max + EPS),
        torch.zeros_like(feasible_max),
    )
    all_serviceable = serviceable.all(dim=-1)
    conditional_mask = all_serviceable
    while conditional_mask.dim() < raw_rbr.dim():
        conditional_mask = conditional_mask.unsqueeze(1)
    conditional_mask = conditional_mask.expand_as(raw_rbr)
    all_serviceable_rbr = torch.where(
        conditional_mask, raw_rbr, torch.zeros_like(raw_rbr)
    )
    all_serviceable_count = conditional_mask.sum()
    all_serviceable_mean_rbr = all_serviceable_rbr.sum() / all_serviceable_count.clamp_min(1)
    return {
        "total": total,
        "min": raw_min,
        "max": raw_max,
        "rbr": raw_rbr,
        "feasible_min": feasible_min,
        "feasible_max": feasible_max,
        "feasible_rbr": feasible_rbr,
        "all_serviceable": all_serviceable,
        "all_serviceable_rbr": all_serviceable_rbr,
        "all_serviceable_count": all_serviceable_count,
        "all_serviceable_mean_rbr": all_serviceable_mean_rbr,
        "structural_infeasible": ~all_serviceable,
        "serviceable_rate": serviceable.float().mean(dim=-1),
    }
