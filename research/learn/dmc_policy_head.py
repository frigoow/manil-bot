"""Frozen-Q residual policy head.

The existing QGRU remains an immutable, calibrated value estimator. The
small residual head learns only within-state action preferences from paired
all-action labels. At zero initialization its argmax is exactly raw Q's.
"""
import os

import torch
import torch.nn as nn

from .dmc_model import QGRU, load_q_model
from .dmc_tokens import GLOBAL_DIM
from .encoding import ACT_DIM


class ResidualPolicyHead(nn.Module):
    def __init__(self, d_model, hidden=256):
        super().__init__()
        self.d_model = int(d_model)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(self.d_model + GLOBAL_DIM + ACT_DIM, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 1),
        )
        # Function-preserving start: logits are just scaled frozen Q.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, last, glob, act):
        return self.net(torch.cat([last, glob, act], dim=-1)).squeeze(-1)

    def logits(self, q, last, glob, act, tau):
        return 30.0 * q / max(float(tau), 1e-6) + self(last, glob, act)


def save_policy_checkpoint(path, head, ema_shadow, opt, step, config):
    tmp = path + ".tmp"
    torch.save({
        "kind": "dmc_policy_head",
        "model": ema_shadow,
        "raw_model": head.state_dict(),
        "opt": opt.state_dict() if opt is not None else None,
        "step": int(step),
        "config": dict(config),
    }, tmp)
    os.replace(tmp, path)


def _resolve_base_path(head_path, base_path):
    if os.path.exists(base_path) or os.path.isabs(base_path):
        return base_path
    beside = os.path.join(os.path.dirname(os.path.abspath(head_path)),
                          base_path)
    return beside if os.path.exists(beside) else base_path


def load_policy_checkpoint(path, device="cuda", base_model=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("kind") != "dmc_policy_head":
        raise ValueError(f"{path} is not a dmc_policy_head checkpoint")
    cfg = state["config"]
    base_path = base_model or cfg.get("q_model")
    if not base_path:
        raise ValueError("policy checkpoint does not name its frozen Q model")
    q_model = load_q_model(_resolve_base_path(path, base_path), device)
    if not isinstance(q_model, QGRU):
        raise ValueError("dmc_policy_head currently requires a QGRU base")
    head = ResidualPolicyHead(
        q_model.gru.hidden_size, hidden=cfg.get("hidden", 256)).to(device)
    head.load_state_dict(state["model"])
    head.eval()
    return q_model, head, float(cfg["tau"]), cfg
