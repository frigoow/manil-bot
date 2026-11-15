"""Golden parity tests: the C++ observation encoder (authoritative) and the
Python mirror must produce tensor-identical output on identical states.

These tests also implicitly check that the two engines agree on legality,
because the observer's legal mask is part of the encoded vector.
"""
import random

import numpy as np
import pytest

import manille_ismcts as mi
from learn.encoding import OBS_DIM, encode_mirror
from manille_env.card import Suit
from manille_env.game_logic import GamePhase, ManilleGame
from tournament import game_to_cpp_arrays


def _hands_to_masks(game):
    hands = np.zeros(4, dtype=np.int64)
    for p in range(4):
        for card in game.hands[p]:
            hands[p] |= np.int64(1) << card.to_index()
    return hands


def _play_random_game(seed):
    """Yield (cpp_args, mirror_kwargs) at every ply of a random game."""
    rng = random.Random(seed)
    game = ManilleGame(dealer=seed % 4)
    hands = _hands_to_masks(game)
    trump = int(mi.apply_trump_rule(hands[game.dealer]))
    game.select_trump(Suit(trump))
    game.start_playing()

    played_cards, played_players = [], []
    while game.phase == GamePhase.PLAYING:
        (h, tc, tp, pc, pp, n_played, n_trick, cp, trump_v,
         pts0, pts1, tw0, tw1) = game_to_cpp_arrays(game, played_cards, played_players)
        legal = [c.to_index() for c in game.get_legal_moves(game.current_player)]
        cpp_args = (h, tc, tp, pc, pp, n_played, n_trick, cp, trump_v,
                    pts0, pts1, tw0, tw1)
        mirror_kwargs = dict(
            hands=h, trick_cards=tc, trick_players=tp,
            played_cards=pc, played_players=pp,
            n_played=n_played, n_trick=n_trick, current_player=cp,
            trump=trump_v, pts0=pts0, pts1=pts1,
            declarer=game.dealer, legal_moves=legal,
        )
        yield cpp_args, mirror_kwargs

        pid = game.current_player
        card = rng.choice(game.get_legal_moves(pid))
        pre_trick = list(game.current_trick)
        assert game.play_card(pid, card)
        if len(pre_trick) == 3:
            for trick_pid, trick_card in pre_trick + [(pid, card)]:
                played_cards.append(trick_card.to_index())
                played_players.append(trick_pid)


@pytest.mark.parametrize("seed", range(30))
def test_encoding_parity(seed):
    n_states = 0
    for cpp_args, mk in _play_random_game(seed):
        observer = mk["current_player"]
        cpp_obs = mi.encode_observation(*cpp_args, mk["declarer"], observer)
        py_obs = encode_mirror(observer=observer, **mk)
        assert cpp_obs.shape == (OBS_DIM,)
        np.testing.assert_array_equal(
            cpp_obs, py_obs,
            err_msg=f"seed={seed} ply state {n_states}",
        )
        n_states += 1
    assert n_states == 32


def test_encoding_parity_all_observers():
    """Belief/prior blocks must be consistent from every seat, not just the
    player to move (legal mask is self-only, everything else well-defined)."""
    for cpp_args, mk in list(_play_random_game(7))[::5]:
        for observer in range(4):
            cpp_obs = mi.encode_observation(*cpp_args, mk["declarer"], observer)
            py_obs = encode_mirror(observer=observer, **mk)
            np.testing.assert_array_equal(cpp_obs, py_obs)
