import torch

from flog.data import FLOGSyntheticGenerator


def test_equal_budget_open_route_generator():
    torch.manual_seed(7)
    generator = FLOGSyntheticGenerator(const_max_budget=2.0, equal_budget_eps=1e-4)
    merge, agent, city = generator.getitem(5, 20)
    assert merge.shape == (30, 2)
    assert agent.shape == (6, 5)
    assert city.shape == (20, 3)
    actual_budget = agent[:5, -1] * 2.0
    assert torch.allclose(actual_budget, actual_budget[:1].expand_as(actual_budget))
    direct = torch.norm(agent[:5, :2] - agent[:5, 2:4], dim=1)
    assert bool((actual_budget > direct).all())
