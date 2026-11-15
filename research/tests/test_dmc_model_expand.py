"""Warm-starting across the 1.3 feature-layout change must be behaviour-
preserving: the expanded net has to compute exactly what the old net did on
the old inputs, or a feature A/B is comparing two different starting points
rather than two feature sets."""
import torch

from learn.dmc_model import BeliefNet, QGRU, QNet, expand_checkpoint_inputs
from learn.dmc_tokens import GLOBAL_DIM, LEGACY_GLOBAL_DIM
from learn.encoding import (ACT_DIM, LEGACY_ACT_DIM, LEGACY_OBS_DIM, OBS_DIM)

torch.manual_seed(0)


def _pad(x, width):
    out = x.new_zeros((x.shape[0], width))
    out[:, : x.shape[1]] = x
    return out


def test_mlp_expansion_is_behaviour_preserving():
    old = QNet(obs_dim=LEGACY_OBS_DIM, act_dim=LEGACY_ACT_DIM,
               hidden=64, blocks=1).eval()
    obs = torch.randn(5, LEGACY_OBS_DIM)
    act = torch.randn(5, LEGACY_ACT_DIM)
    with torch.no_grad():
        before = old(obs, act)
    new = QNet(hidden=64, blocks=1).eval()
    new.load_state_dict(expand_checkpoint_inputs(old.state_dict(), "mlp"))
    with torch.no_grad():
        after = new(_pad(obs, OBS_DIM), _pad(act, ACT_DIM))
    torch.testing.assert_close(before, after)


def test_gru_expansion_is_behaviour_preserving():
    cfg = dict(d_model=16, layers=1, hidden=32, blocks=1)
    old = QGRU(**cfg).eval()
    old.input = torch.nn.Linear(16 + LEGACY_GLOBAL_DIM + LEGACY_ACT_DIM, 32)
    raw = torch.randn(4, 31, 58)
    n_raw = torch.full((4,), 7)
    glob = torch.randn(4, LEGACY_GLOBAL_DIM)
    act = torch.randn(4, LEGACY_ACT_DIM)
    with torch.no_grad():
        before = old(raw, n_raw, glob, None, act)
    new = QGRU(**cfg).eval()
    new.load_state_dict(expand_checkpoint_inputs(old.state_dict(), "gru"))
    with torch.no_grad():
        after = new(raw, n_raw, _pad(glob, GLOBAL_DIM), None,
                    _pad(act, ACT_DIM))
    torch.testing.assert_close(before, after)


def test_belief_expansion_is_behaviour_preserving():
    old = BeliefNet(obs_dim=LEGACY_OBS_DIM, hidden=32, blocks=1).eval()
    obs = torch.randn(3, LEGACY_OBS_DIM)
    with torch.no_grad():
        before = old(obs)
    new = BeliefNet(hidden=32, blocks=1).eval()
    new.load_state_dict(expand_checkpoint_inputs(old.state_dict(), "belief"))
    with torch.no_grad():
        after = new(_pad(obs, OBS_DIM))
    torch.testing.assert_close(before, after)


def test_already_current_checkpoint_is_untouched():
    net = QNet(hidden=32, blocks=1)
    width = net.state_dict()["input.weight"].shape[1]
    out = expand_checkpoint_inputs(net.state_dict(), "mlp")
    assert out["input.weight"].shape[1] == width
