"""DMC networks (manille_dmc_plan §1–§2, Architecture A).

QNet: action-as-input Q_θ(s, a) -> expected final team points, normalized to
[-1, 1] ((points - 30) / 30). The candidate card's feature vector (ACT_DIM)
is concatenated with the information-state observation (OBS_DIM); one scalar
output per (state, candidate) pair. To act, evaluate all legal candidates and
take argmax (ε-greedy during generation) — no output masking needed.

QOracle: the same MLP scorer with the recorded true-holder labels appended to
the public observation. It is a training-only perfect-information critic.

BeliefNet: obs-only trunk + the mandatory §5.2 belief head (logit offsets on
the constrained-uniform prior, zero-initialized). Trained on the same
self-play stream; the blueprint does not need it to act.
"""
import torch
import torch.nn as nn

from .encoding import ACT_DIM, OBS_DIM
from .dmc_tokens import (BELIEF_CTX_DIM, GLOBAL_DIM, MAX_HAND, MAX_PLAYS,
                         PAD_CARD, RAW_TOKEN_DIM)
from .model import ResidualBlock, belief_probs, split_belief_inputs  # noqa: F401


class QNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=1024, blocks=4):
        super().__init__()
        self.input = nn.Linear(obs_dim + act_dim, hidden)
        self.blocks = nn.ModuleList(ResidualBlock(hidden) for _ in range(blocks))
        self.q_head = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh()
        )

    def forward(self, obs, act):
        h = torch.relu(self.input(torch.cat([obs, act], dim=-1)))
        for block in self.blocks:
            h = block(h)
        return self.q_head(h).squeeze(-1)


class QOracle(QNet):
    """Perfect-information action-value regressor used only during training.

    The replay's ``belief`` target is a one-hot true holder for every unseen
    card, in the acting seat's relative frame. Appending its 3×32 entries to
    the public encoder output gives the critic the hidden hands without
    changing the deployable Q network's input contract.
    """

    BELIEF_DIM = 3 * 32

    def __init__(self, hidden=1024, blocks=4):
        super().__init__(obs_dim=OBS_DIM + self.BELIEF_DIM, hidden=hidden,
                         blocks=blocks)

    def forward(self, obs, belief, act):
        if belief.shape[-2:] != (3, 32):
            raise ValueError("oracle belief input must end in [3, 32]")
        return super().forward(
            torch.cat([obs, belief.reshape(*belief.shape[:-2], -1)], dim=-1),
            act,
        )


class BeliefNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=512, blocks=2):
        super().__init__()
        self.input = nn.Linear(obs_dim, hidden)
        self.blocks = nn.ModuleList(ResidualBlock(hidden) for _ in range(blocks))
        self.belief_head = nn.Linear(hidden, 3 * 32)
        # Zero output must reproduce the constrained-uniform prior exactly.
        nn.init.zeros_(self.belief_head.weight)
        nn.init.zeros_(self.belief_head.bias)

    def forward(self, obs):
        h = torch.relu(self.input(obs))
        for block in self.blocks:
            h = block(h)
        return self.belief_head(h).view(-1, 3, 32)


class QGRU(nn.Module):
    """DouZero-style recurrent history encoder (P0.1 probe candidate):
    GRU over the public play sequence, last hidden state, then the same
    action-as-input Q head pattern as QNet. Tokens are the canonical
    dmc_tokens records (raw history + global + candidate); hand tokens are
    unused (own hand already sits in the global token)."""

    def __init__(self, d_model=256, layers=1, hidden=1024, blocks=2):
        super().__init__()
        self.gru = nn.GRU(RAW_TOKEN_DIM, d_model, layers,
                          batch_first=True)
        self.input = nn.Linear(d_model + GLOBAL_DIM + ACT_DIM, hidden)
        self.blocks = nn.ModuleList(ResidualBlock(hidden)
                                    for _ in range(blocks))
        self.q_head = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh()
        )

    def trunk(self, raw, n_raw):
        """History embedding, candidate-independent: callers evaluating many
        candidates (or many worlds sharing a history row) should run this
        once per unique (raw, n_raw) row and expand the result — the trunk
        does not depend on glob/act."""
        h, _ = self.gru(raw)
        idx = (n_raw - 1).clamp_min(0)  # n_raw==0 -> row 0 is all zeros
        return h[torch.arange(raw.shape[0], device=raw.device), idx]

    def head(self, last, glob, act):
        z = torch.relu(self.input(torch.cat([last, glob, act], dim=-1)))
        for block in self.blocks:
            z = block(z)
        return self.q_head(z).squeeze(-1)

    def forward(self, raw, n_raw, glob, hand, act):
        return self.head(self.trunk(raw, n_raw), glob, act)


class BeliefGRU(nn.Module):
    """Recurrent belief net (P0 probe candidate): GRU over the public play
    sequence, last hidden state + belief-context vector (global token +
    voids + belief prior/mask — same conditioning as the flat BeliefNet),
    MLP head emitting prior-offset logits (same zero-init parameterization
    as BeliefNet — zero output reproduces the constrained-uniform prior
    exactly)."""

    def __init__(self, d_model=256, layers=1, hidden=512, blocks=2,
                 ctx_dim=BELIEF_CTX_DIM):
        super().__init__()
        self.gru = nn.GRU(RAW_TOKEN_DIM, d_model, layers,
                          batch_first=True)
        self.input = nn.Linear(d_model + ctx_dim, hidden)
        self.blocks = nn.ModuleList(ResidualBlock(hidden)
                                    for _ in range(blocks))
        self.belief_head = nn.Linear(hidden, 3 * 32)
        nn.init.zeros_(self.belief_head.weight)
        nn.init.zeros_(self.belief_head.bias)

    def forward(self, raw, n_raw, ctx, hand):
        h, _ = self.gru(raw)
        idx = (n_raw - 1).clamp_min(0)
        last = h[torch.arange(raw.shape[0], device=raw.device), idx]
        z = torch.relu(self.input(torch.cat([last, ctx], dim=-1)))
        for block in self.blocks:
            z = block(z)
        return self.belief_head(z).view(-1, 3, 32)


class QTransformer(nn.Module):
    """Architecture B (plan §4): small transformer history encoder with the
    candidate action as query token.

    Sequence: [global, play tokens (≤31), hand tokens (8), candidate].
    The Q head reads the candidate token's final hidden state. Same DMC
    targets and training loop as QNet — only the encoder changes.
    """
    def __init__(self, d_model=128, layers=4, heads=4):
        super().__init__()
        self.tok_proj = nn.Linear(RAW_TOKEN_DIM, d_model)
        self.glob_proj = nn.Linear(GLOBAL_DIM, d_model)
        self.card_embed = nn.Embedding(33, d_model)  # 32 cards + PAD_CARD
        self.act_proj = nn.Linear(ACT_DIM, d_model)
        # Positions: 1 global + MAX_PLAYS plays + MAX_HAND hand + 1 candidate.
        self.pos = nn.Parameter(
            torch.zeros(1, 1 + MAX_PLAYS + MAX_HAND + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, heads, 4 * d_model, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.q_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh()
        )

    def forward(self, raw, n_raw, glob, hand, act, return_pooled=False):
        """raw [B, MAX_PLAYS, RAW_TOKEN_DIM], n_raw [B] long, glob [B,
        GLOBAL_DIM], hand [B, MAX_HAND] long, act [B, ACT_DIM].
        return_pooled: also return the global token's final hidden state
        (for the P0.1 belief-aux probe variant)."""
        B = raw.shape[0]
        h = torch.cat([
            self.glob_proj(glob).unsqueeze(1),
            self.tok_proj(raw),
            self.card_embed(hand.clamp(0, 32)),
            self.act_proj(act).unsqueeze(1),
        ], dim=1) + self.pos
        # Padding mask: True = ignore (padded play and hand slots).
        device = raw.device
        play_pad = (torch.arange(MAX_PLAYS, device=device)[None, :]
                    >= n_raw[:, None])
        hand_pad = hand >= PAD_CARD
        pad = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=device),
            play_pad, hand_pad,
            torch.zeros(B, 1, dtype=torch.bool, device=device),
        ], dim=1)
        out = self.enc(h, src_key_padding_mask=pad)
        q = self.q_head(out[:, -1]).squeeze(-1)
        if return_pooled:
            return q, out[:, 0]
        return q


def save_checkpoint(path, model, ema_shadow, opt, scaler, step, config):
    """Atomic checkpoint write (tmp + rename): a killed run must never leave
    a half-written file behind."""
    import os

    tmp = path + ".tmp"
    torch.save(
        {
            "model": ema_shadow,
            "raw_model": model.state_dict(),
            "opt": opt.state_dict() if opt is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "config": config,
        },
        tmp,
    )
    os.replace(tmp, path)


def new_feature_column_mask(kind, width):
    """Boolean mask over an input layer's columns: True for the 1.3 tactical
    features. They are appended within each sub-block, so for the GRU they are
    two disjoint ranges (end of the global token, end of the candidate), not a
    single tail."""
    from .encoding import (ACT_DIM, LEGACY_ACT_DIM, LEGACY_OBS_DIM, OBS_DIM)
    from .dmc_tokens import GLOBAL_DIM, LEGACY_GLOBAL_DIM

    mask = torch.zeros(width, dtype=torch.bool)
    if kind == "gru":
        d_model = width - GLOBAL_DIM - ACT_DIM
        glob0 = d_model
        act0 = d_model + GLOBAL_DIM
    elif kind == "mlp":
        glob0, act0 = 0, OBS_DIM
    else:
        raise ValueError(f"no input-column layout for arch {kind!r}")
    legacy_glob = LEGACY_GLOBAL_DIM if kind == "gru" else LEGACY_OBS_DIM
    n_glob = GLOBAL_DIM if kind == "gru" else OBS_DIM
    mask[glob0 + legacy_glob : glob0 + n_glob] = True
    mask[act0 + LEGACY_ACT_DIM : act0 + ACT_DIM] = True
    return mask


def expand_checkpoint_inputs(state, kind, legacy_obs=None, legacy_act=None):
    """Grow a pre-1.3 checkpoint's input layer to the current feature layout.

    The tactical block (plan 1.3) appends columns to the observation, the
    candidate-action vector and the GRU global token. Old columns are copied
    to their new positions and the new ones are ZERO-initialised, so the
    expanded net computes exactly the old function until training moves those
    weights. That is what makes a feature A/B honest: both arms start from
    bit-identical behaviour and differ only in what they are shown.

    `kind` is "mlp", "gru" or "belief". Returns the state dict, modified in
    place; a checkpoint already at the current width is left untouched.
    """
    from .encoding import (ACT_DIM, LEGACY_ACT_DIM, LEGACY_OBS_DIM, OBS_DIM)
    from .dmc_tokens import GLOBAL_DIM, LEGACY_GLOBAL_DIM

    legacy_obs = LEGACY_OBS_DIM if legacy_obs is None else legacy_obs
    legacy_act = LEGACY_ACT_DIM if legacy_act is None else legacy_act

    if kind == "transformer":
        # QTransformer has no single input layer: the global token and the
        # candidate enter through their own projections, so each is widened
        # on its own. Padding at the END is exactly right because the new
        # features are appended, never interleaved.
        for key, width in (("glob_proj.weight", GLOBAL_DIM),
                           ("act_proj.weight", ACT_DIM)):
            w = state.get(key)
            if w is None or w.shape[1] == width:
                continue
            grown = w.new_zeros((w.shape[0], width))
            grown[:, : w.shape[1]] = w
            state[key] = grown
        return state

    w = state.get("input.weight")
    if w is None:
        raise KeyError("checkpoint has no input.weight to expand")
    old_width = w.shape[1]
    if kind == "belief":
        new_width = OBS_DIM
        blocks = [(0, 0, legacy_obs)]                      # (dst, src, n)
    elif kind == "gru":
        d_model = state["gru.weight_ih_l0"].shape[0] // 3
        new_width = d_model + GLOBAL_DIM + ACT_DIM
        blocks = [(0, 0, d_model + LEGACY_GLOBAL_DIM),
                  (d_model + GLOBAL_DIM, d_model + LEGACY_GLOBAL_DIM,
                   legacy_act)]
    else:
        new_width = OBS_DIM + ACT_DIM
        blocks = [(0, 0, legacy_obs), (OBS_DIM, legacy_obs, legacy_act)]
    if old_width == new_width:
        return state
    expected = sum(n for _, _, n in blocks)
    if old_width != expected:
        raise ValueError(
            f"{kind} checkpoint input width {old_width} is neither the "
            f"current {new_width} nor the legacy {expected}")
    grown = w.new_zeros((w.shape[0], new_width))
    for dst, src, n in blocks:
        grown[:, dst : dst + n] = w[:, src : src + n]
    state["input.weight"] = grown
    return state


def make_q_model(config, device="cuda"):
    """Construct a Q model from a checkpoint config dict."""
    if config.get("arch") == "transformer":
        model = QTransformer(
            d_model=config.get("d_model", 128),
            layers=config.get("layers", 4),
            heads=config.get("heads", 4),
        )
    elif config.get("arch") == "gru":
        model = QGRU(
            d_model=config.get("d_model", 256),
            layers=config.get("layers", 1),
            hidden=config.get("hidden", 1024),
            blocks=config.get("blocks", 2),
        )
    else:
        model = QNet(hidden=config.get("hidden", 1024),
                     blocks=config.get("blocks", 4))
    return model.to(device)


_NOT_Q_KINDS = {
    "dmc_iteration_actor":
        "load it with learn.dmc_iteration_policy.load_actor_policy",
    "dmc_iteration_critic":
        "critics are training-only (learn.dmc_iteration_train.load_critic)",
}


def load_q_model(path, device="cuda", hidden=1024, blocks=4):
    """Load EMA weights for inference. Geometry comes from the checkpoint's
    config when present, else the given defaults."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = state.get("config", {"hidden": hidden, "blocks": blocks})
    if config.get("kind") in _NOT_Q_KINDS:
        # Policy-iteration actors emit policy logits and critics emit
        # unbounded paired advantages; both fit the QGRU topology, so
        # without this check either would silently load as a tanh Q model.
        # Every Q consumer (dmc_q, SPARTA, value/likelihood evaluators,
        # the raw-Q policy source) loads through here.
        raise ValueError(f"{path} is a {config['kind']} checkpoint, not a Q "
                         f"model; {_NOT_Q_KINDS[config['kind']]}")
    model = make_q_model(config, device="cpu")
    if "model" in state:  # training checkpoint: EMA weights under "model"
        state = state["model"]
    # Artifacts frozen before the 1.3 tactical block have a narrower input
    # layer. Expanding with zeros keeps them loadable AND bit-identical in
    # behaviour (the new columns contribute nothing), so every existing
    # checkpoint, benchmark and served model keeps working unchanged.
    expand_checkpoint_inputs(state, config.get("arch", "mlp"))
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_oracle_model(path, device="cuda", hidden=1024, blocks=4):
    """Load a training-only QOracle EMA checkpoint.

    Keeping this loader separate from ``load_q_model`` prevents an oracle
    checkpoint from ever being accidentally used by a deployable agent, whose
    observation intentionally contains public information only.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = state.get("config", {})
    if config.get("kind") != "q_oracle":
        raise ValueError(f"{path} is not a QOracle checkpoint")
    model = QOracle(hidden=config.get("hidden", hidden),
                    blocks=config.get("blocks", blocks))
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_belief_model(path, device="cuda", hidden=512, blocks=2):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "config" in state:
        hidden = state["config"].get("hidden", hidden)
        blocks = state["config"].get("blocks", blocks)
    model = BeliefNet(hidden=hidden, blocks=blocks)
    if "model" in state:
        state = state["model"]
    expand_checkpoint_inputs(state, "belief")
    model.load_state_dict(state)
    model.to(device).eval()
    return model
