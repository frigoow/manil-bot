"""Policy-value-belief network for Manille (plan §4, §5.2).

One trunk, three heads:
  - policy: logits over 32 cards (masked to legal at loss/inference time)
  - value:  tanh scalar, expected team points normalized to [-1, 1]
  - belief: per-unseen-card logit offsets over the 3 hidden seats, added to
            the log of the constrained-uniform prior that the encoder emits.
            Zero offsets == uniform-consistent posterior, so the head is
            zero-initialized and all capacity goes into the deviation from
            uniform (the mandatory §5.2 parameterization).
"""
import torch
import torch.nn as nn

from .encoding import OBS_DIM, BELIEF_PRIOR_OFF, BELIEF_MASK_OFF


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)
        return torch.relu(self.norm(x + h))


class ManilleNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=1024, blocks=3):
        super().__init__()
        self.input = nn.Linear(obs_dim, hidden)
        self.blocks = nn.ModuleList(ResidualBlock(hidden) for _ in range(blocks))
        self.policy_head = nn.Linear(hidden, 32)
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh()
        )
        self.belief_head = nn.Linear(hidden, 3 * 32)
        # Zero output must reproduce the constrained-uniform prior exactly.
        nn.init.zeros_(self.belief_head.weight)
        nn.init.zeros_(self.belief_head.bias)

    def forward(self, obs):
        h = torch.relu(self.input(obs))
        for block in self.blocks:
            h = block(h)
        policy_logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        belief_offsets = self.belief_head(h).view(-1, 3, 32)
        return policy_logits, value, belief_offsets


def belief_probs(belief_offsets, prior, mask):
    """Prior-residual softmax with hard constraints (§5.2).

    Args: [B, 3, 32] tensors — offsets from the net, constrained-uniform prior
    and validity mask from the encoder's belief blocks.
    Returns [B, 3, 32] probabilities over hidden seats per card; exactly zero
    on seen cards and hard-masked (seat, card) pairs.
    """
    logits = torch.log(prior.clamp_min(1e-9)) + belief_offsets
    logits = logits.masked_fill(mask < 0.5, -1e9)
    logits = logits - logits.amax(dim=1, keepdim=True)
    return torch.softmax(logits, dim=1) * mask


def split_belief_inputs(obs):
    """Slice the belief prior and mask blocks out of an observation batch."""
    prior = obs[:, BELIEF_PRIOR_OFF : BELIEF_PRIOR_OFF + 96].view(-1, 3, 32)
    mask = obs[:, BELIEF_MASK_OFF : BELIEF_MASK_OFF + 96].view(-1, 3, 32)
    return prior, mask


def load_model(path, device="cuda"):
    model = ManilleNet()
    state = torch.load(path, map_location=device, weights_only=True)
    if "model" in state:  # training checkpoint with optimizer/EMA metadata
        state = state["model"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model
