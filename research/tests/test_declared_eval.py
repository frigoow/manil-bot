"""Declared-deal harness and the rule agent's declaration / doubling tilt."""
import os

import numpy as np
import pytest

from learn.evaluate import make_deals

POLICY = "runs/policy_state_rates.pt"
pytestmark = pytest.mark.skipif(not os.path.exists(POLICY),
                                reason="served decision policy not present")


def test_calls_are_identical_for_both_seatings_and_legal():
    from learn.declared_eval import declare_deals, load_equity_policy, play_declared
    from learn.rule_agent import RuleAgent
    from tournament import RandomAgent
    deals = make_deals(6, 3)
    calls = declare_deals(deals, load_equity_policy())
    assert calls == declare_deals(deals, load_equity_policy())
    p = dict(n_worlds=30, playout_worlds=8, sim_worlds=10, root_q_w=0.0,
             trump_rule_known=False, decl_tilt=POLICY, double_tilt=True)
    for (hands, dealer), call in zip(deals, calls):
        agents = [RuleAgent(**p), RandomAgent(), RuleAgent(**p), RandomAgent()]
        pts, mult = play_declared(agents, hands, dealer, call)
        assert sum(pts) == 60 and mult in (1, 2, 4, 8)


def test_declaration_tilt_prefers_hands_that_declare_it():
    """A dealer who declared a suit is more likely, under the tilt, to hold
    a hand that would declare it than a hand with none of that suit."""
    from manille_env.card import Suit
    from manille_env.game_logic import ManilleGame
    from learn.rule_agent import RuleAgent
    (hands, dealer), = make_deals(1, 5)
    trump = max(range(4), key=lambda s: sum(c.suit.value == s for c in hands[dealer]))
    g = ManilleGame(dealer=dealer)
    g.hands = [list(h) for h in hands]
    g.select_trump(Suit(trump))
    g.start_playing()
    me = (dealer + 1) % 4
    a = RuleAgent(decl_tilt=POLICY)
    a.new_game(g, dealer, me)
    true = np.array([sum(1 << c.to_index() for c in g.hands[s]) for s in range(4)], dtype=np.int64)
    # swap the dealer's trumps with the next seat's non-trumps: same cards
    # outside, a dealer who would not have declared that suit
    other = (dealer + 2) % 4
    sm = np.int64(0xFF) << np.int64(8 * trump)
    give = true[dealer] & sm
    take = true[other] & ~sm
    k = min(bin(int(give)).count("1"), bin(int(take)).count("1"))
    gb = [b for b in range(32) if int(give) >> b & 1][:k]
    tb = [b for b in range(32) if int(take) >> b & 1][:k]
    alt = true.copy()
    for x, y in zip(gb, tb):
        alt[dealer] ^= (1 << x) | (1 << y)
        alt[other] ^= (1 << x) | (1 << y)
    lw = a.decl_log_weights(g, trump, np.stack([true, alt]), [], me)
    assert lw[0] > lw[1]


def test_called_double_tilt_reads_only_actual_meegaans():
    """double_tilt='called': no weight change on undoubled deals; a called
    meegaan tilts the caller's hand."""
    from manille_env.card import Suit
    from manille_env.game_logic import ManilleGame
    from learn.rule_agent import RuleAgent
    import manille_ismcts as mi
    from tournament import game_to_cpp_arrays
    (hands, dealer), = make_deals(1, 6)
    for doubled in (False, True):
        g = ManilleGame(dealer=dealer)
        g.hands = [list(h) for h in hands]
        g.select_trump(Suit(0))
        caller = (dealer + 1) % 4
        if doubled:
            assert g.announce_double(caller)
        g.meegaan_seat, g.terug_seat = (caller if doubled else None), None
        g.start_playing()
        me = (dealer + 3) % 4  # the other defender: the caller's partner
        a = RuleAgent(decl_tilt=POLICY, decl_tilt_declaration=False, double_tilt="called",
                      trump_rule_known=False)
        a.new_game(g, dealer, me)
        W = np.asarray(mi.sample_determinizations(*game_to_cpp_arrays(g, [], []), me, 64, -1,
                                                  True), dtype=np.int64)
        lw = a.decl_log_weights(g, 0, W, [], me)
        if doubled:
            assert np.ptp(lw) > 0
        else:
            assert np.all(lw == 0)


def test_match_equity_play_at_late_scores():
    """match_equity ranks cards by P(match win); the late-match harness
    returns one equity in [0, 1] per deal."""
    import torch
    from learn import declared_eval as de
    deals = make_deals(3, 80)
    states = de.match_states(3, 80)
    calls = de.declare_deals_at(deals, states, de.load_equity_policy())
    V = np.asarray(torch.load(POLICY, map_location="cpu", weights_only=False)["V"])
    fast = dict(n_worlds=40, playout_worlds=16, sim_worlds=10, root_q_w=0.0,
                trump_rule_known=False)
    de._init_states({"type": "rule", "params": dict(fast, match_equity=POLICY)},
                    {"type": "rule", "params": fast}, deals, calls, states, V)
    eq = [de._deal_equity(i)[1] for i in range(3)]
    assert all(0.0 <= e <= 1.0 for e in eq)
