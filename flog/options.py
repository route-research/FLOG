from __future__ import annotations

import argparse
from typing import Optional


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def get_options(args: Optional[list] = None):
    parser = argparse.ArgumentParser(description="FLOG anonymous review release")
    parser.add_argument("--run_mode", choices=["train", "val", "real"], default="val")
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_sampling", choices=["grip", "listed"], default="grip")
    parser.add_argument("--train_scales", type=str, default="5x50,7x70,10x100")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_rollouts", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--encode_layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tanh_clipping", type=float, default=10.0)
    parser.add_argument("--rho_train", type=float, default=0.65)
    parser.add_argument("--rho_recovery", type=float, default=0.65)
    parser.add_argument("--rho_eval", type=float, default=0.50)
    parser.add_argument("--reward_floor", type=float, default=0.90)
    parser.add_argument("--const_max_budget", type=float, default=2.0)
    parser.add_argument("--equal_budget_eps", type=float, default=1e-4)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--dataset_root", type=str, default="data")
    parser.add_argument("--eval_ins", type=int, default=10)
    parser.add_argument("--eval_scales", type=str, default="5x50,5x100,7x70,10x100")
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--real_limit", type=int, default=0)
    parser.add_argument("--cities", type=str, default="Denver,LA,Seattle")

    parsed = parser.parse_args(args)
    for name in ("rho_train", "rho_recovery", "rho_eval", "reward_floor"):
        value = float(getattr(parsed, name))
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name} must be in [0, 1]")
    return parsed
