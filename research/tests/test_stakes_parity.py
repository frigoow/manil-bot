"""M0 (manille_match_plan.md): the C++ engine carries the stake multiplier
and both announcements (meegaan / terug meegaan).

The Python engine (manille_env/game_logic.py) is the reference
implementation. These tests drive identical deals, identical trump choices
(including mull) and identical announcement sequences on both engines and
require:

- identical accept/reject outcomes for every announcement call,
- identical multiplier / doubled / redoubled state at every checkpoint,
- identical legality at every ply of the play-out,
- identical terminal hand scores: (winner_points - 30) * multiplier for the
  winner only, 0 for the loser and for a 30-30 tie.

The gating rules (match plan §3, game_logic.py): announcements only during
the doubling window (after trump selection, before play starts); meegaan
only by the team that did NOT make trump, once; terug meegaan only by the
trump-making team, only in answer to an actual meegaan, once — gated on the
doubled/redoubled FLAGS, never on the multiplier, because mull alone
already makes the multiplier 2 (the c63fdbd bug).
"""
import numpy as np
import pytest

import manille_ismcts as mi
from manille_env.card import Card, Suit
from manille_env.game_logic import GamePhase, ManilleGame

TRUMPS = [0, 1, 2, 3, -1]  # four suits and mull


def _deal(seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(32)
    return [[Card.from_index(int(c)) for c in perm[p * 8:(p + 1) * 8]]
            for p in range(4)]


def _masks(hands):
    masks = np.zeros(4, dtype=np.int64)
    for p in range(4):
        for c in hands[p]:
            masks[p] |= np.int64(1) << c.to_index()
    return masks


def _py_game(hands, dealer, trump):
    """Python engine after trump selection (mull = None sets x2)."""
    g = ManilleGame(dealer=dealer)
    g.hands = [list(h) for h in hands]
    assert g.select_trump(None if trump == -1 else Suit(trump)) is True
    return g


def _cpp_game(hands, dealer, trump):
    """C++ engine: construction IS the trump selection (trump=-1 is mull)."""
    return mi.ManilleGame(_masks(hands), trump, dealer)


def _assert_same_stake_state(py_g, cpp_g):
    assert py_g.multiplier == cpp_g.multiplier()
    assert py_g.doubled == cpp_g.doubled()
    assert py_g.redoubled == cpp_g.redoubled()
    assert (py_g.phase == GamePhase.DOUBLING) == cpp_g.doubling_phase()


def _run_script(py_g, cpp_g, script):
    """Apply an announcement script to both engines, pinning accept/reject.

    script: list of (action, player, expected) with action in
    {"double", "redouble", "start"}; expected is the required boolean for
    announce calls (None for "start").
    """
    for action, pid, expected in script:
        if action == "start":
            py_g.start_playing()
            cpp_g.start_playing()
            continue
        if action == "double":
            py_res = py_g.announce_double(pid)
            cpp_res = cpp_g.announce_double(pid)
        else:
            py_res = py_g.announce_redouble(pid)
            cpp_res = cpp_g.announce_redouble(pid)
        assert py_res == cpp_res == expected, (
            f"{action} by {pid}: python {py_res}, cpp {cpp_res}, "
            f"expected {expected}")
        _assert_same_stake_state(py_g, cpp_g)


def _play_out(py_g, cpp_g):
    """Play the deal out on both engines with identical (deterministic)
    cards, requiring identical legality at every ply."""
    if py_g.phase == GamePhase.DOUBLING:
        py_g.start_playing()
        cpp_g.start_playing()
    n_plies = 0
    while py_g.phase == GamePhase.PLAYING:
        pid = py_g.current_player
        py_legal = {c.to_index() for c in py_g.get_legal_moves(pid)}
        cpp_legal = set(cpp_g.valid_moves())
        assert py_legal == cpp_legal, (
            f"ply {n_plies}: python {sorted(py_legal)} != "
            f"cpp {sorted(cpp_legal)}")
        assert pid == cpp_g.current_player()
        card = min(py_legal)
        assert py_g.play_card(pid, Card.from_index(card))
        cpp_g.do_move(card)
        n_plies += 1
    assert n_plies == 32
    assert cpp_g.finished()


def _assert_same_score(py_g, cpp_g):
    """Terminal agreement: hand points, tricks, and the scored result."""
    assert py_g.hand_points == [cpp_g.team_points(0), cpp_g.team_points(1)]
    assert sum(py_g.hand_points) == 60
    assert py_g.tricks_won == [cpp_g.tricks_won(0), cpp_g.tricks_won(1)]
    for team in range(2):
        assert py_g.team_scores[team].score == cpp_g.hand_score(team)
        # get_result is the signed-margin MCTS utility (/30) — deliberately
        # distinct from hand_score (match accounting): denying the winner's
        # margin is real value in a race to 101 (see manille_game.cpp).
        for player in (team, team + 2):
            assert cpp_g.get_result(player) == pytest.approx(
                (cpp_g.team_points(team) - 30.0) / 30.0)
    _assert_same_stake_state(py_g, cpp_g)


# dealer 1 -> trump-making team {1,3}, defending team {0,2}
GATING_SCRIPTS = [
    ("no announcement", []),
    ("meegaan by a defender", [("double", 0, True)]),
    ("meegaan + terug",
     [("double", 0, True), ("redouble", 1, True)]),
    ("meegaan by the dealer rejected", [("double", 1, False)]),
    ("meegaan by the dealer's partner rejected",
     [("double", 3, False)]),
    ("a second opponent cannot compound",
     [("double", 0, True), ("double", 2, False)]),
    ("terug without a meegaan rejected", [("redouble", 1, False)]),
    ("terug by a defender rejected",
     [("double", 0, True), ("redouble", 2, False)]),
    ("a second terug rejected",
     [("double", 0, True), ("redouble", 1, True),
      ("redouble", 3, False)]),
    ("announcement after start_playing rejected",
     [("start", None, None), ("double", 0, False)]),
]


@pytest.mark.parametrize("trump", TRUMPS)
@pytest.mark.parametrize("script_name", range(len(GATING_SCRIPTS)))
def test_announcement_gating_matches_python(trump, script_name):
    name, script = GATING_SCRIPTS[script_name]
    hands = _deal(seed=7)
    py_g = _py_game(hands, dealer=1, trump=trump)
    cpp_g = _cpp_game(hands, dealer=1, trump=trump)
    _assert_same_stake_state(py_g, cpp_g)
    _run_script(py_g, cpp_g, script)


def test_mull_sets_x2_at_trump_selection():
    hands = _deal(seed=3)
    for trump, base in [(0, 1), (1, 1), (2, 1), (3, 1), (-1, 2)]:
        py_g = _py_game(hands, dealer=0, trump=trump)
        cpp_g = _cpp_game(hands, dealer=0, trump=trump)
        assert py_g.multiplier == cpp_g.multiplier() == base


def test_multiplier_reaches_x8_on_mull_meegaan_terug():
    hands = _deal(seed=5)
    py_g = _py_game(hands, dealer=0, trump=-1)
    cpp_g = _cpp_game(hands, dealer=0, trump=-1)
    _run_script(py_g, cpp_g,
                [("double", 1, True), ("redouble", 0, True)])
    assert py_g.multiplier == cpp_g.multiplier() == 8


def test_terug_without_meegaan_rejected_at_mull():
    """The c63fdbd regression: mull alone makes the multiplier 2, so a
    multiplier-based gate would accept a terug that never had a meegaan.
    The flags must forbid it."""
    hands = _deal(seed=5)
    py_g = _py_game(hands, dealer=0, trump=-1)
    cpp_g = _cpp_game(hands, dealer=0, trump=-1)
    assert py_g.multiplier == cpp_g.multiplier() == 2
    _run_script(py_g, cpp_g, [("redouble", 0, False)])
    assert py_g.multiplier == cpp_g.multiplier() == 2


ANNOUNCE_SCRIPTS = [
    [],
    [("double", 1, True)],
    [("double", 1, True), ("redouble", 0, True)],
]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("trump", TRUMPS)
@pytest.mark.parametrize("script", ANNOUNCE_SCRIPTS)
def test_scored_hands_match_python(seed, trump, script):
    hands = _deal(seed)
    py_g = _py_game(hands, dealer=0, trump=trump)
    cpp_g = _cpp_game(hands, dealer=0, trump=trump)
    _run_script(py_g, cpp_g, script)
    _play_out(py_g, cpp_g)
    _assert_same_score(py_g, cpp_g)


def test_announcement_after_first_card_rejected():
    """The window closes when play starts — explicitly via start_playing
    (both engines) and implicitly on the first card (C++; Python's engine
    refuses play_card before start_playing, so the implicit path is
    C++-only)."""
    hands = _deal(seed=11)

    # Explicit close, then a card, then an announcement attempt.
    py_g = _py_game(hands, dealer=0, trump=0)
    cpp_g = _cpp_game(hands, dealer=0, trump=0)
    py_g.start_playing()
    cpp_g.start_playing()
    pid = py_g.current_player
    card = py_g.get_legal_moves(pid)[0]
    assert py_g.play_card(pid, card)
    cpp_g.do_move(card.to_index())
    assert py_g.announce_double(1) is False
    assert cpp_g.announce_double(1) is False
    assert py_g.announce_redouble(0) is False
    assert cpp_g.announce_redouble(0) is False

    # C++-only: a first do_move implicitly closes the window.
    cpp_implicit = _cpp_game(hands, dealer=0, trump=0)
    assert cpp_implicit.doubling_phase() is True
    cpp_implicit.do_move(cpp_implicit.valid_moves()[0])
    assert cpp_implicit.doubling_phase() is False
    assert cpp_implicit.announce_double(1) is False


def _play_out_python_only(seed, trump, dealer=0):
    """Deterministic min-index play-out on the reference engine."""
    py_g = _py_game(_deal(seed), dealer=dealer, trump=trump)
    py_g.start_playing()
    while py_g.phase == GamePhase.PLAYING:
        pid = py_g.current_player
        card = min(py_g.get_legal_moves(pid), key=lambda c: c.to_index())
        assert py_g.play_card(pid, card)
    return py_g


# Tie seeds for the deterministic min-index play-out above (found by scan;
# the play-out is deterministic, so these are stable).
MULL_TIE_SEED = 91   # 30-30 at trump = -1
SUIT_TIE_SEED = 19   # 30-30 at trump = 0


def test_tie_scores_zero_even_at_x8():
    """A 30-30 tie scores 0 for both sides, whatever the multiplier —
    here mull + meegaan + terug = x8."""
    hands = _deal(MULL_TIE_SEED)
    py_g = _py_game(hands, dealer=0, trump=-1)
    cpp_g = _cpp_game(hands, dealer=0, trump=-1)
    _run_script(py_g, cpp_g,
                [("double", 1, True), ("redouble", 0, True)])
    assert py_g.multiplier == cpp_g.multiplier() == 8
    _play_out(py_g, cpp_g)
    assert py_g.hand_points == [30, 30]  # precondition: it really is a tie
    _assert_same_score(py_g, cpp_g)
    assert cpp_g.hand_score(0) == cpp_g.hand_score(1) == 0
    assert all(cpp_g.get_result(p) == 0.0 for p in range(4))


def test_tie_scores_zero_at_suit_trump():
    hands = _deal(SUIT_TIE_SEED)
    py_g = _py_game(hands, dealer=0, trump=0)
    cpp_g = _cpp_game(hands, dealer=0, trump=0)
    _run_script(py_g, cpp_g, [("double", 1, True)])
    assert py_g.multiplier == cpp_g.multiplier() == 2
    _play_out(py_g, cpp_g)
    assert py_g.hand_points == [30, 30]
    _assert_same_score(py_g, cpp_g)
    assert cpp_g.hand_score(0) == cpp_g.hand_score(1) == 0
