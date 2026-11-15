"""Golden parity: the C++ rule play-out (cpp/rule_playout.cpp) must give
exactly the same results as the Python reference (learn/rule_playout.py),
for every world, candidate and policy mode. Same discipline as the encoder
mirror: Python is the readable reference, C++ the runtime."""
import random

import numpy as np
import pytest

import manille_ismcts as mi
from learn.evaluate import make_deals, trump_for
from learn.rule_playout import evaluate_candidates, world_weights
from manille_env.card import Suit
from manille_env.game_logic import ManilleGame
from tournament import game_to_cpp_arrays

MODES = [(0, False), (1, False), (2, False), (0, True), (2, True)]


def _states(seed, mull=False):
    """Decision points along one random game: (trump, dealer, me, legal,
    worlds, history)."""
    rng = random.Random(seed)
    hands, dealer = make_deals(1, seed)[0]
    trump = None if mull else int(trump_for(hands, dealer))
    g = ManilleGame(dealer=dealer)
    g.hands = [list(h) for h in hands]
    g.select_trump(None if trump is None else Suit(trump))
    g.start_playing()
    pc, pp, history = [], [], []
    out = []
    for ply in range(30):
        me = g.current_player
        legal = [c.to_index() for c in g.get_legal_moves(me)]
        if ply % 3 == 0 and len(legal) > 1:
            arr = game_to_cpp_arrays(g, pc, pp)
            worlds = mi.sample_determinizations(*arr, me, 6, dealer, True)
            out.append((trump, dealer, me, legal, worlds, list(history)))
        card = rng.choice(g.get_legal_moves(me))
        pre = list(g.current_trick)
        g.play_card(me, card)
        history.append((me, card.to_index()))
        if len(pre) == 3:
            for q, x in pre + [(me, card)]:
                pc.append(x.to_index())
                pp.append(q)
    return out


def _cpp_args(history):
    seats = np.array([s for s, _ in history], dtype=np.int32)
    cards = np.array([c for _, c in history], dtype=np.int32)
    return seats, cards


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("mull", [False, True])
def test_evaluate_parity(seed, mull):
    for trump, dealer, me, legal, worlds, history in _states(seed, mull):
        seats, cards = _cpp_args(history)
        t = -1 if trump is None else trump
        for honest, agent_like in MODES:
            py = evaluate_candidates(trump, dealer, me, legal, worlds, history,
                                     honest, agent_like)
            cpp = mi.rule_playout_evaluate(t, dealer, me, legal, worlds, seats, cards,
                                           honest, agent_like)
            for i, c in enumerate(legal):
                np.testing.assert_array_equal(
                    py[c], cpp[i],
                    err_msg=f"seed {seed} mull {mull} mode {(honest, agent_like)} card {c}")


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("mull", [False, True])
def test_weights_parity(seed, mull):
    for trump, dealer, me, legal, worlds, history in _states(seed, mull):
        seats, cards = _cpp_args(history)
        t = -1 if trump is None else trump
        for honest, agent_like in MODES:
            py = world_weights(trump, dealer, me, worlds, history, 0.5, honest, agent_like)
            cpp = mi.rule_playout_weights(t, dealer, me, worlds, seats, cards, 0.5,
                                          honest, agent_like)
            np.testing.assert_array_equal(py, cpp)

NT_MODES = [(0, True, 15), (2, True, 15), (1, True, 15), (0, True, 1), (0, True, 2), (0, True, 4),
            (0, True, 8), (0, False, 15), (0, True, 31), (2, True, 31), (0, True, 16), (0, False, 31),
            (0, True, 32), (0, True, 64), (0, True, 127), (2, True, 127), (1, True, 127),
            (0, True, 128), (0, True, 256), (0, True, 512), (0, True, 31 | 896),
            (2, True, 31 | 896), (1, True, 1023), (0, False, 31 | 896)]


@pytest.mark.parametrize("seed", range(20))
def test_no_trump_rules_parity(seed):
    """The no-trump rule bits (unblocking, winning with the doomed honour,
    cashing a suit the opponents are out of) in both implementations."""
    for trump, dealer, me, legal, worlds, history in _states(seed, mull=True):
        seats, cards = _cpp_args(history)
        for honest, agent_like, nt in NT_MODES:
            py = evaluate_candidates(None, dealer, me, legal, worlds, history,
                                     honest, agent_like, nt)
            cpp = mi.rule_playout_evaluate(-1, dealer, me, legal, worlds, seats, cards,
                                           honest, agent_like, nt)
            for i, c in enumerate(legal):
                np.testing.assert_array_equal(
                    py[c], cpp[i], err_msg=f"seed {seed} mode {(honest, agent_like, nt)} card {c}")
            pw = world_weights(None, dealer, me, worlds, history, 0.5, honest, agent_like, nt)
            cw = mi.rule_playout_weights(-1, dealer, me, worlds, seats, cards, 0.5,
                                         honest, agent_like, nt)
            np.testing.assert_array_equal(pw, cw)


def test_review_bits_change_no_trump_play_outs():
    """Bits 128/256/512 (cash team bosses, cross to partner, low from a
    run) are live: somewhere in these states they change a play-out."""
    changed = 0
    for seed in range(6):
        for trump, dealer, me, legal, worlds, history in _states(seed, mull=True):
            seats, cards = _cpp_args(history)
            a = mi.rule_playout_evaluate(-1, dealer, me, legal, worlds, seats, cards, 0, True, 31)
            b = mi.rule_playout_evaluate(-1, dealer, me, legal, worlds, seats, cards, 0, True,
                                         31 | 896)
            changed += int(not np.array_equal(a, b))
    assert changed > 0


def test_no_trump_rules_are_inert_with_a_trump_suit():
    for trump, dealer, me, legal, worlds, history in _states(3):
        seats, cards = _cpp_args(history)
        a = mi.rule_playout_evaluate(trump, dealer, me, legal, worlds, seats, cards, 0, True, 0)
        b = mi.rule_playout_evaluate(trump, dealer, me, legal, worlds, seats, cards, 0, True, 31)
        np.testing.assert_array_equal(a, b)


TR_MODES = [(0, True, 1), (0, True, 2), (0, True, 4), (0, True, 8), (0, True, 16), (0, True, 32), (2, True, 32),
            (0, True, 31), (2, True, 31), (0, False, 31)]


@pytest.mark.parametrize("seed", range(20))
def test_trump_rules_parity(seed):
    """The trump rule bits (lone trump through, vet a master on partner's
    trick, singleton lead, discard to partner, doomed honour on a ruff) in
    both implementations."""
    for trump, dealer, me, legal, worlds, history in _states(seed):
        seats, cards = _cpp_args(history)
        for honest, agent_like, tr in TR_MODES:
            py = evaluate_candidates(trump, dealer, me, legal, worlds, history,
                                     honest, agent_like, 31, tr)
            cpp = mi.rule_playout_evaluate(trump, dealer, me, legal, worlds, seats, cards,
                                           honest, agent_like, 31, tr)
            for i, c in enumerate(legal):
                np.testing.assert_array_equal(
                    py[c], cpp[i], err_msg=f"seed {seed} mode {(honest, agent_like, tr)} card {c}")
            pw = world_weights(trump, dealer, me, worlds, history, 0.5, honest, agent_like, 31, tr)
            cw = mi.rule_playout_weights(trump, dealer, me, worlds, seats, cards, 0.5,
                                         honest, agent_like, 31, tr)
            np.testing.assert_array_equal(pw, cw)


def test_trump_rules_are_inert_at_no_trump():
    for trump, dealer, me, legal, worlds, history in _states(4, mull=True):
        seats, cards = _cpp_args(history)
        a = mi.rule_playout_evaluate(-1, dealer, me, legal, worlds, seats, cards, 0, True, 31, 0)
        b = mi.rule_playout_evaluate(-1, dealer, me, legal, worlds, seats, cards, 0, True, 31, 31)
        np.testing.assert_array_equal(a, b)
