from __future__ import annotations

import glob
import os
import random
import re
from typing import List, Sequence, Tuple

import torch


ALL_SCALES: Tuple[Tuple[int, int], ...] = (
    (5, 50), (5, 100), (5, 200), (5, 300), (5, 500), (5, 1000),
    (7, 70),
    (10, 100), (10, 200), (10, 300), (10, 500), (10, 1000),
    (20, 500), (20, 1000),
)


def safe_torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_scales(text: str) -> List[Tuple[int, int]]:
    if text.strip().lower() == "all":
        return list(ALL_SCALES)
    result: List[Tuple[int, int]] = []
    for token in text.split(","):
        left, right = token.strip().lower().split("x", 1)
        result.append((int(left), int(right)))
    if not result:
        raise ValueError("at least one scale is required")
    return result


def sample_training_scale(mode: str, scales: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
    if mode == "grip":
        anum = int(torch.randint(5, 10, size=(1,)).item())
        cnum = int(torch.randint(anum * 10, 101, size=(1,)).item())
        return anum, cnum
    if mode == "listed":
        return random.choice(list(scales))
    raise ValueError(f"unknown train_sampling={mode}")


class FLOGSyntheticGenerator:
    """Synthetic generator for the equal-budget open-route fTOP setting.

    A distinct start and end are sampled for every agent. A feasible budget is
    first sampled for each agent, then replaced with a common actual budget
    equal to max(mean(original budgets), max direct start-end distance + eps).
    """

    def __init__(
        self,
        const_max_budget: float = 2.0,
        equal_budget_eps: float = 1e-4,
    ):
        self.const_max_budget = float(const_max_budget)
        self.equal_budget_eps = float(equal_budget_eps)

    @staticmethod
    def _distance(coord_a: torch.Tensor, coord_b: torch.Tensor) -> torch.Tensor:
        return torch.cdist(coord_a, coord_b)

    def _coords(self, anum: int, cnum: int):
        while True:
            ends = torch.rand(anum, 2)
            if int((self._distance(ends, ends) < 0.001).sum().item()) == anum:
                break

        while True:
            starts = torch.rand(anum, 2)
            cross = self._distance(starts, ends)
            self_distance = self._distance(starts, starts)
            if not bool((cross < 0.001).any()) and int((self_distance < 0.001).sum()) == anum:
                break

        while True:
            cities = torch.rand(cnum, 2)
            city_distance = self._distance(cities, cities)
            if (
                int((city_distance < 1e-6).sum().item()) == cnum
                and not bool((self._distance(starts, cities) < 1e-6).any())
                and not bool((self._distance(ends, cities) < 1e-6).any())
            ):
                break

        merge = torch.cat([cities, starts, ends], dim=0)
        agent = torch.cat([starts, ends], dim=1)
        return merge, cities, agent

    def _sample_equal_budget(self, agent: torch.Tensor) -> torch.Tensor:
        direct = torch.norm(agent[:, :2] - agent[:, 2:4], dim=1)
        original = []
        for index in range(direct.numel()):
            while True:
                normalized = torch.rand(1)
                if float(normalized.item()) * self.const_max_budget > float(direct[index].item()):
                    original.append(normalized)
                    break
        original_actual = torch.stack(original).view(-1) * self.const_max_budget
        common_actual = torch.maximum(
            original_actual.mean(),
            direct.max() + self.equal_budget_eps,
        )
        if float(common_actual.item()) > self.const_max_budget + 1e-7:
            raise RuntimeError(
                f"equal common budget {common_actual.item():.6f} exceeds "
                f"const_max_budget={self.const_max_budget}"
            )
        common_normalized = common_actual / self.const_max_budget
        return common_normalized.repeat(agent.size(0)).view(-1, 1)

    def getitem(self, anum: int, cnum: int):
        merge, cities, agent = self._coords(anum, cnum)
        rewards = torch.rand(cnum, 1)
        budgets = self._sample_equal_budget(agent)
        city_feature = torch.cat([cities, rewards], dim=1)
        agent_feature = torch.cat([agent, budgets], dim=1)
        dummy = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]).view(1, 5)
        agent_feature = torch.cat([agent_feature, dummy], dim=0)
        return merge, agent_feature, city_feature


def make_training_batch(generator, anum, cnum, batch_size, device):
    items = [generator.getitem(anum, cnum) for _ in range(batch_size)]
    merge = torch.stack([item[0] for item in items]).to(device)
    agent = torch.stack([item[1] for item in items]).to(device)
    city = torch.stack([item[2] for item in items]).to(device)
    return merge, agent, city


def _instance_number(path: str) -> int:
    match = re.search(r"_cnt=(\d+)_", os.path.basename(path))
    return int(match.group(1)) if match else 10**18


def validation_files(dataset_root: str, anum: int, cnum: int, limit: int = 0) -> List[str]:
    model = "mdcfa"
    subdir = "mdmtsp-cfa"
    directory = os.path.join(dataset_root, "valset", subdir, f"{model}_anum={anum}_cnum={cnum}")
    pattern = os.path.join(directory, f"{model}_anum={anum}_cnum={cnum}_cnt=*_with_empty.pt")
    files = sorted(glob.glob(pattern), key=lambda x: (_instance_number(x), x))
    return files[:limit] if limit > 0 else files


def load_validation_instance(path: str):
    item = safe_torch_load(path, "cpu")
    return item["merge_coord"], item["af"], item["cf"]


def load_real_instance(item, anum: int, cnum: int, const_max_budget: float = 2.0):
    merge = item["coords"].float()
    budgets = item["budgets"].float()
    agent = torch.cat(
        [
            merge[cnum : cnum + anum],
            merge[cnum + anum : cnum + 2 * anum],
            (budgets / float(const_max_budget)).unsqueeze(1),
        ],
        dim=1,
    )
    dummy = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0], dtype=agent.dtype).view(1, 5)
    agent = torch.cat([agent, dummy], dim=0)
    city = torch.cat([merge[:cnum], item["rewards"].float().unsqueeze(1)], dim=1)
    return merge, agent, city
