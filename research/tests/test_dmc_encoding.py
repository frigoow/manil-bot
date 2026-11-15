"""Golden parity tests for the DMC candidate-action encoder
(encoding::encode_action / encode_q_observation / encode_q_trajectory).

Same discipline as test_encoding_parity.py: C++ is authoritative, the Python
mirror must produce tensor-identical output, and step-wise queries must agree
with the batch trajectory encoder.
"""
import random

import numpy as np
import pytest

import manille_ismcts as mi
from learn.encoding import (ACT_DIM, ACT_SURE_WIN_OFF, MAX_LEGAL,
                            encode_action_mirror)
from manille_env.card import Suit
from manille_env.game_logic import GamePhase, ManilleGame
from tournament import game_to_cpp_arrays

from tests.test_encoding_parity import _hands_to_masks, _play_random_game


@pytest.mark.parametrize("seed", range(20))
def test_action_feature_parity(seed):
    """Mirror == C++ for every legal candidate at every ply."""
    for cpp_args, mk in _play_random_game(seed):
        observer = mk["current_player"]
        out = mi.encode_q_observation(*cpp_args, mk["declarer"], observer)
        assert out["act"].shape == (MAX_LEGAL, ACT_DIM)
        assert out["obs"].shape == (mi.OBS_DIM,)

        py_legal = sorted(mk["legal_moves"])
        cpp_legal = sorted(int(c) for c in out["legal"][: out["n_legal"]])
        assert cpp_legal == py_legal

        for j in range(out["n_legal"]):
            card = int(out["legal"][j])
            py_act = encode_action_mirror(
                hands=mk["hands"], trick_cards=mk["trick_cards"],
                trick_players=mk["trick_players"],
                played_cards=mk["played_cards"],
                played_players=mk["played_players"],
                n_played=mk["n_played"], n_trick=mk["n_trick"],
                current_player=mk["current_player"], trump=mk["trump"],
                declarer=mk["declarer"], card=card)
            np.testing.assert_array_equal(
                out["act"][j], py_act,
                err_msg=f"seed={seed} card={card}",
            )
        # Padding is zero / -1.
        for j in range(out["n_legal"], MAX_LEGAL):
            assert int(out["legal"][j]) == -1
            assert not out["act"][j].any()


def test_q_trajectory_matches_stepwise():
    """The batch trajectory encoder must agree with per-ply step queries and
    every recorded action must be legal in its ply."""
    rng = random.Random(123)
    game = ManilleGame(dealer=3)
    hands = _hands_to_masks(game)
    trump = int(mi.apply_trump_rule(hands[game.dealer]))
    game.select_trump(Suit(trump))
    game.start_playing()

    played_cards, played_players, actions, step_rows = [], [], [], []
    players = []
    while game.phase == GamePhase.PLAYING:
        cpp_args = game_to_cpp_arrays(game, played_cards, played_players)
        pid = game.current_player
        players.append(pid)
        step_rows.append(mi.encode_q_observation(*cpp_args, game.dealer, pid))
        card = rng.choice(game.get_legal_moves(pid))
        actions.append(card.to_index())
        pre = list(game.current_trick)
        assert game.play_card(pid, card)
        if len(pre) == 3:
            for tp_, tc_ in pre + [(pid, card)]:
                played_cards.append(tc_.to_index())
                played_players.append(tp_)

    traj = mi.encode_q_trajectory(
        hands, trump, game.dealer, np.array(actions, dtype=np.int32))
    assert traj["pts0"] == game.hand_points[0]
    assert traj["pts1"] == game.hand_points[1]

    for t in range(32):
        step = step_rows[t]
        assert int(traj["players"][t]) == players[t]
        np.testing.assert_array_equal(traj["obs"][t], step["obs"])
        np.testing.assert_array_equal(traj["act"][t], step["act"])
        np.testing.assert_array_equal(traj["legal"][t], step["legal"])
        assert int(traj["n_legal"][t]) == int(step["n_legal"])
        assert actions[t] in traj["legal"][t][: traj["n_legal"][t]]


def test_sure_win_is_certain_on_the_real_hands():
    """ACT_SURE_WIN claims certainty, so check it against ground truth.

    The feature is computed from an UPPER BOUND on each remaining seat's hand
    (every unseen card in the suits it is not publicly void in). The real hand
    is a subset, so "no possible hand can legally beat this" must imply "the
    real hand cannot either" — on every seat, on every ply.

    Legality is the point: a seat can HOLD a card that outranks the candidate
    and still be forbidden to play it, because following suit comes first.
    An earlier version of this test compared raw card ranks and failed on
    exactly that case.
    """
    from manille_env.card import Card
    from learn.encoding import _beats, _can_beat, _trick_winner

    checked = 0
    for seed in range(12):
        for cpp_args, mk in _play_random_game(seed):
            n_trick, me, trump = mk["n_trick"], mk["current_player"], mk["trump"]
            if n_trick + 1 >= 4:
                continue
            enc = mi.encode_q_observation(*cpp_args, mk["declarer"], me)
            table = [int(mk["trick_cards"][i]) for i in range(n_trick)]
            seats = [int(mk["trick_players"][i]) for i in range(n_trick)]
            for j in range(int(enc["n_legal"])):
                if enc["act"][j][ACT_SURE_WIN_OFF] != 1.0:
                    continue
                card = int(enc["legal"][j])
                after, after_seats = table + [card], seats + [me]
                lead_idx = _trick_winner(after, len(after), trump)
                assert after_seats[lead_idx] == me, "sure win that is not winning"
                leading = after[lead_idx]
                pairs = [(s, Card.from_index(c))
                         for s, c in zip(after_seats, after)]
                for k in range(n_trick + 1, 4):
                    seat = (me + 1 + (k - n_trick - 1)) % 4
                    assert not _can_beat(
                        seat, int(mk["hands"][seat]), pairs, trump, leading,
                        mk["declarer"]), (
                        f"seed={seed}: sure-win card {card} can be beaten by "
                        f"seat {seat} with its real hand")
                checked += 1
    assert checked > 0, "no sure-win candidates encountered"
