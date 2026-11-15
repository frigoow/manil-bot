"""The native rule agent (cpp/rule_agent.cpp) against the Python reference
(learn/rule_agent.py): same card and same play-out values on the same
sampled worlds, belief probabilities and Q values, at trump and no trump."""
import numpy as np
import pytest

import manille_ismcts as mi
from learn.evaluate import make_deals
from learn.rule_agent import RuleAgent
from manille_env.card import Suit
from manille_env.game_logic import GamePhase, ManilleGame
from tournament import game_to_cpp_arrays

FAST = dict(root_q_device="cpu", trump_rule_known=False, n_worlds=120, playout_worlds=96,
            sim_worlds=20)


def _positions(trump, seed, n_deals, params, meegaan=False):
    """Play deals with the Python agent; yield (agent, game, pc, pp, worlds,
    python card, python values) at every decision with a choice."""
    orig = mi.sample_determinizations
    rec = {}

    def patched(*a, **k):
        out = orig(*a, **k)
        rec["w"] = np.array(out)
        return out

    mi.sample_determinizations = patched
    try:
        agents = [RuleAgent(**params) for _ in range(4)]
        for hands, dealer in make_deals(n_deals, seed):
            g = ManilleGame(dealer=dealer)
            g.hands = [list(h) for h in hands]
            g.select_trump(None if trump is None else Suit(trump))
            g.meegaan_seat, g.terug_seat = None, None
            if meegaan:
                g.meegaan_seat = (dealer + 1) % 4
                assert g.announce_double(g.meegaan_seat)
            g.start_playing()
            for p, a in enumerate(agents):
                a.new_game(g, dealer, p)
            pc, pp = [], []
            while g.phase == GamePhase.PLAYING:
                me = g.current_player
                rec.clear()
                card = agents[me].choose_card(g, me, pc, pp)
                if "w" in rec:
                    yield agents[me], g, list(pc), list(pp), rec["w"], card.to_index(), \
                        agents[me].last_values
                pre = list(g.current_trick)
                g.play_card(me, card)
                if len(pre) == 3:
                    for s, c in pre + [(me, card)]:
                        pc.append(c.to_index())
                        pp.append(s)
    finally:
        mi.sample_determinizations = orig


@pytest.mark.parametrize("trump,seed", [(1, 11), (None, 12)])
def test_native_matches_python(trump, seed):
    n = 0
    for agent, g, pc, pp, worlds, card, values in _positions(trump, seed, 2, FAST):
        arrays = game_to_cpp_arrays(g, pc, pp)
        native, _ = agent.native_choose(g, g.current_player, pc, pp, arrays, worlds)
        assert native == card
        if values:
            assert set(values) == set(agent.last_values)
            for c, v in values.items():
                assert agent.last_values[c] == pytest.approx(v, abs=1e-5)
        n += 1
    assert n > 20


def test_native_matches_python_with_the_called_double_tilt():
    """The meegaan likelihood enters the C++ agent as per-world extra
    log-weights; same cards and values as the Python path."""
    import os
    policy = "runs/policy_state_rates.pt"
    if not os.path.exists(policy):
        pytest.skip("served decision policy missing")
    params = dict(FAST, decl_tilt=policy, decl_tilt_declaration=False, double_tilt="called")
    n = 0
    for agent, g, pc, pp, worlds, card, values in _positions(2, 14, 2, params, meegaan=True):
        assert agent.native_supported()
        arrays = game_to_cpp_arrays(g, pc, pp)
        native, _ = agent.native_choose(g, g.current_player, pc, pp, arrays, worlds)
        assert native == card
        if values:
            for c, v in values.items():
                assert agent.last_values[c] == pytest.approx(v, abs=1e-5)
        n += 1
    assert n > 20


def test_native_fallback_without_worlds():
    """Too few worlds: the Knowledge falls back to the analytic marginals
    and the old lead tiers, in both implementations."""
    for agent, g, pc, pp, worlds, card, values in _positions(0, 13, 1, FAST):
        arrays = game_to_cpp_arrays(g, pc, pp)
        py = RuleAgent(**dict(FAST, n_worlds=0))
        py.new_game(g, agent.dealer, g.current_player)
        expected = py.choose_card(g, g.current_player, pc, pp).to_index()
        native, _ = py.native_choose(g, g.current_player, pc, pp, arrays, None)
        assert native == expected


def test_native_flag_is_used_only_when_supported():
    assert RuleAgent(native=True).native_supported()
    assert RuleAgent(native=True, decl_tilt="x").native_supported()  # passed as extra weights
    assert not RuleAgent(native=True, signal_mode=1).native_supported()


def test_native_matches_python_with_the_ruff_convention():
    """Partnership agreement: singleton opening lead, partner returns it."""
    params = dict(FAST, ruff_convention=True)
    n, tiers = 0, set()
    for agent, g, pc, pp, worlds, card, values in _positions(2, 21, 3, params):
        arrays = game_to_cpp_arrays(g, pc, pp)
        native, why = agent.native_choose(g, g.current_player, pc, pp, arrays, worlds)
        assert native == card
        if "ruff convention" in why:
            tiers.add(why.split(",")[1].split(";")[0].strip())
        n += 1
    assert n > 30


@pytest.mark.parametrize("trump,seed,mix", [(1, 31, 1.0), (None, 32, 1.0), (2, 33, 0.5)])
def test_native_matches_python_with_the_fast_playout_net(trump, seed, mix):
    """The learned continuation (cpp/fast_policy.h) mixed into the play-out."""
    params = dict(FAST, playout_net="artifacts/fast_policy_256.pt", playout_net_mix=mix)
    n = 0
    for agent, g, pc, pp, worlds, card, values in _positions(trump, seed, 2, params):
        arrays = game_to_cpp_arrays(g, pc, pp)
        native, _ = agent.native_choose(g, g.current_player, pc, pp, arrays, worlds)
        assert native == card
        if values:
            for c, v in values.items():
                assert agent.last_values[c] == pytest.approx(v, abs=1e-5)
        n += 1
    assert n > 20


@pytest.mark.parametrize("trump,seed", [(0, 41), (None, 42)])
def test_native_matches_python_with_the_hybrid_continuation(trump, seed):
    """The net plays the first plies after the candidate, the rule policy the rest."""
    params = dict(FAST, playout_net="artifacts/fast_policy_256.pt", playout_net_plies=4)
    n = 0
    for agent, g, pc, pp, worlds, card, values in _positions(trump, seed, 2, params):
        arrays = game_to_cpp_arrays(g, pc, pp)
        native, _ = agent.native_choose(g, g.current_player, pc, pp, arrays, worlds)
        assert native == card
        if values:
            for c, v in values.items():
                assert agent.last_values[c] == pytest.approx(v, abs=1e-5)
        n += 1
    assert n > 20


@pytest.mark.parametrize("trump,seed", [(0, 43), (None, 44)])
def test_native_matches_python_with_the_net_to_trick_end(trump, seed):
    """The hybrid, with the net finishing the trick its plies end in."""
    params = dict(FAST, playout_net="artifacts/fast_policy_256.pt", playout_net_plies=4,
                  playout_net_trick_end=True)
    n = 0
    for agent, g, pc, pp, worlds, card, values in _positions(trump, seed, 2, params):
        arrays = game_to_cpp_arrays(g, pc, pp)
        native, _ = agent.native_choose(g, g.current_player, pc, pp, arrays, worlds)
        assert native == card
        if values:
            for c, v in values.items():
                assert agent.last_values[c] == pytest.approx(v, abs=1e-5)
        n += 1
    assert n > 20


@pytest.mark.parametrize("trump", [1, -1])
def test_net_to_trick_end_is_the_plain_cut_at_the_trick_boundary(trump):
    """With 4 plies and trick_end, the net plays exactly up to the first
    trick boundary at or after ply 4 -- the plain cut at that count."""
    from learn.fast_policy import load_native
    net = load_native("artifacts/fast_policy_256.pt")
    hands, dealer = make_deals(1, 7)[0]
    g = ManilleGame(dealer=dealer)
    g.hands = [list(h) for h in hands]
    g.select_trump(None if trump < 0 else Suit(trump))
    g.start_playing()
    hist = []
    for k in range(4):                # the candidate at every trick position
        me = g.current_player
        legal = sorted(c.to_index() for c in g.get_legal_moves(me))
        worlds = np.array([[sum(1 << c.to_index() for c in g.hands[s]) for s in range(4)]],
                          dtype=np.int64)
        hs = np.array([s for s, _ in hist], dtype=np.int32)
        hc = np.array([c for _, c in hist], dtype=np.int32)
        j = (len(g.current_trick) + 1) % 4   # cards on the trick after the candidate
        rounded = 4 if j == 0 else 8 - j
        args = (trump, dealer, me, legal, worlds, hs, hc, 0, True, 31, 0, net)
        to_end = np.asarray(mi.rule_playout_evaluate(*args, 4, True))
        plain = np.asarray(mi.rule_playout_evaluate(*args, rounded, False))
        assert np.array_equal(to_end, plain)
        card = g.get_legal_moves(me)[0]
        hist.append((me, card.to_index()))
        g.play_card(me, card)


@pytest.mark.parametrize("seed", [51, 52])
def test_native_matches_python_with_the_mull_review_rules(seed):
    """Fable mull review (2026-11-14): play-out bits 128/256/512, low lead
    from a touching run, and the late cash lock -- same cards and values."""
    params = dict(FAST, playout_nt_rules=31 | 896, nt_est_low_run=True,
                  nt_lock_cash_from_trick=3)
    n = 0
    for agent, g, pc, pp, worlds, card, values in _positions(None, seed, 3, params):
        assert agent.native_supported()
        arrays = game_to_cpp_arrays(g, pc, pp)
        native, _ = agent.native_choose(g, g.current_player, pc, pp, arrays, worlds)
        assert native == card
        if values:
            for c, v in values.items():
                assert agent.last_values[c] == pytest.approx(v, abs=1e-5)
        n += 1
    assert n > 20


@pytest.mark.parametrize("trump,seed,extra", [
    (1, 61, dict(def_trump_lock=True)),
    (3, 62, dict(def_trump_lock=True, def_trump_lock_weak=True)),
    (0, 63, dict(def_trump_margin=1.0)),
    (2, 64, dict(def_trump_margin=0.5, def_trump_lock_weak=True)),
])
def test_native_matches_python_with_the_defender_trump_lock(trump, seed, extra):
    """def_trump_lock / _weak / def_trump_margin: the defenders' play-out may
    not (or only by a margin) turn a side-suit lead into a trump lead -- same
    cards and values in both implementations."""
    params = dict(FAST, **extra)
    n = 0
    for agent, g, pc, pp, worlds, card, values in _positions(trump, seed, 3, params):
        assert agent.native_supported()
        arrays = game_to_cpp_arrays(g, pc, pp)
        native, _ = agent.native_choose(g, g.current_player, pc, pp, arrays, worlds)
        assert native == card
        if values:
            for c, v in values.items():
                assert agent.last_values[c] == pytest.approx(v, abs=1e-5)
        n += 1
    assert n > 20
