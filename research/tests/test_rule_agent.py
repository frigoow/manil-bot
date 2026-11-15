"""Tests for the rule-based agent (learn/rule_agent.py) and its play-out
(learn/rule_playout.py). Small, fast checks of the inference and policy
pieces; strength is measured with duplicate-deal matches, not here."""
import numpy as np
import pytest

import manille_ismcts as mi
from learn.evaluate import make_deals, trump_for
from learn.rule_agent import Knowledge, RuleAgent
from learn.rule_playout import Playout, evaluate_candidates, world_weights
from manille_env.card import Card, Rank, Suit
from manille_env.game_logic import ManilleGame
from tournament import RandomAgent, game_to_cpp_arrays, run_game

FAST = dict(n_worlds=30, playout_worlds=8, sim_worlds=10)


def C(s, r):
    return Card(Suit(s), Rank(r))


def _game(hands, trump, dealer):
    g = ManilleGame(dealer=dealer)
    g.hands = [list(h) for h in hands]
    g.select_trump(None if trump is None else Suit(trump))
    g.start_playing()
    return g


@pytest.mark.parametrize("seed", range(3))
def test_full_games_are_legal(seed):
    """run_game raises on an illegal move; play whole deals in all modes."""
    for hands, dealer in make_deals(2, seed):
        for params in (FAST, dict(FAST, signal_mode=2), dict(FAST, playout=False)):
            agents = [RuleAgent(**params), RandomAgent(), RuleAgent(**params), RandomAgent()]
            r = run_game(agents, dealer=dealer, hands=hands, trump=trump_for(hands, dealer))
            assert sum(r.pts) == 60


def test_no_trump_games_are_legal():
    for hands, dealer in make_deals(2, 9):
        agents = [RuleAgent(**FAST), RandomAgent(), RuleAgent(**FAST), RandomAgent()]
        r = run_game(agents, dealer=dealer, hands=hands, mull=True)
        assert sum(r.pts) == 60


def _knowledge_after(hands, trump, dealer, plays, observer):
    g = _game(hands, trump, dealer)
    for pid, card in plays:
        assert g.play_card(pid, card)
    arr = game_to_cpp_arrays(g, [], [])
    enc = mi.encode_q_observation(*arr, dealer, observer)
    return Knowledge(enc, None, observer)


# Hearts led; seat 1 plays K, seat 2 (an opponent of seat 1) follows with the
# 8 without beating -- must-beat means it holds nothing above the K.
HANDS = [[C(1, 0), C(0, 0), C(0, 1), C(0, 2), C(2, 0), C(2, 1), C(2, 2), C(3, 0)],
         [C(1, 5), C(0, 3), C(0, 4), C(0, 5), C(2, 3), C(2, 4), C(2, 5), C(3, 1)],
         [C(1, 1), C(1, 2), C(0, 6), C(0, 7), C(2, 6), C(2, 7), C(3, 2), C(3, 3)],
         [C(1, 3), C(1, 4), C(1, 6), C(1, 7), C(3, 4), C(3, 5), C(3, 6), C(3, 7)]]


def test_followed_without_beating_caps_the_suit():
    k = _knowledge_after(HANDS, 3, 3, [(0, C(1, 0)), (1, C(1, 5)), (2, C(1, 1))], 3)
    # observer 3: seat 2 is relative seat 3 (previous player)
    assert k.cap[3][1] == 5
    # from seat 0's view the A and 10 of hearts are unseen, and can no
    # longer be with seat 2 (relative seat 2 = partner)
    k0 = _knowledge_after(HANDS, 3, 3, [(0, C(1, 0)), (1, C(1, 5)), (2, C(1, 1))], 0)
    assert 2 not in k0.poss[1 * 8 + 6] and 2 not in k0.poss[1 * 8 + 7]


def test_playout_records_public_voids():
    po = Playout(trump=3, declarer=0)
    po._note(2, C(0, 0).to_index(), [(1, C(1, 0).to_index())])
    assert po._void(2, 1) and not po._void(2, 0)


def test_respond_secures_when_an_opponent_behind_can_beat():
    """Third seat with partner winning and the 4th seat able to overtake:
    secure the trick instead of ducking (the game-4 bug)."""
    agent = RuleAgent()
    t = Suit(1)
    trick = [(1, Card.from_index(1 * 8 + 2)), (2, Card.from_index(0 * 8 + 6))]  # 9H, AD (void)
    hands = {3: [C(1, 7), C(1, 0)], 0: [C(1, 3), C(2, 0)]}  # 10H 7H | JH 7C
    play = agent._respond(trick, 3, hands, [0], t)
    assert play == C(1, 7)


def _decision_state(seed=4, plies=9):
    hands, dealer = make_deals(1, seed)[0]
    trump = int(trump_for(hands, dealer))
    g = _game(hands, trump, dealer)
    pc, pp, history = [], [], []
    for _ in range(plies):
        pid = g.current_player
        c = g.get_legal_moves(pid)[0]
        pre = list(g.current_trick)
        g.play_card(pid, c)
        history.append((pid, c.to_index()))
        if len(pre) == 3:
            for q, x in pre + [(pid, c)]:
                pc.append(x.to_index())
                pp.append(q)
    me = g.current_player
    arr = game_to_cpp_arrays(g, pc, pp)
    worlds = mi.sample_determinizations(*arr, me, 12, dealer, True)
    legal = [c.to_index() for c in g.get_legal_moves(me)]
    return trump, dealer, me, legal, worlds, history


def test_evaluate_candidates_is_paired_and_bounded():
    trump, dealer, me, legal, worlds, history = _decision_state()
    out = evaluate_candidates(trump, dealer, me, legal, worlds, history)
    assert set(out) == set(legal)
    for v in out.values():
        assert v.shape == (len(worlds),)
        assert ((v >= 0) & (v <= 60)).all()


def test_world_weights_in_unit_interval():
    trump, dealer, me, legal, worlds, history = _decision_state()
    w = world_weights(trump, dealer, me, worlds, history, 0.5)
    assert w.shape == (len(worlds),)
    assert ((w > 0) & (w <= 1)).all()


def test_signal_target_conventions():
    assert RuleAgent(signal_mode=1)._signal_target(1) == 1
    opp = RuleAgent(signal_mode=2)
    assert opp._signal_target(1) == 0 and opp._signal_target(0) == 1   # hearts <-> diamonds
    assert opp._signal_target(3) == 2 and opp._signal_target(2) == 3   # spades <-> clubs


@pytest.mark.parametrize("seed", range(4))
def test_enumeration_is_the_sampler_support(seed):
    """enumerate_determinizations lists each consistent deal once, and every
    sampled deal is one of them (late states, where the support is small)."""
    import random
    rng = random.Random(seed)
    hands, dealer = make_deals(1, seed)[0]
    trump = int(trump_for(hands, dealer))
    g = _game(hands, trump, dealer)
    pc, pp = [], []
    checked = 0
    for ply in range(31):
        me = g.current_player
        if ply >= 20 and ply % 3 == 0:
            arr = game_to_cpp_arrays(g, pc, pp)
            en = mi.enumerate_determinizations(*arr, me, 5000, dealer)
            if len(en):
                support = {tuple(r) for r in en.tolist()}
                assert len(support) == len(en)
                sampled = mi.sample_determinizations(*arr, me, 500, dealer, True)
                assert {tuple(r) for r in sampled.tolist()} <= support
                checked += 1
        c = rng.choice(g.get_legal_moves(me))
        pre = list(g.current_trick)
        g.play_card(me, c)
        if len(pre) == 3:
            for q, x in pre + [(me, c)]:
                pc.append(x.to_index())
                pp.append(q)
    assert checked > 0


def test_mismatch_counts_reproduce_world_weights():
    """rule_playout_mismatches is world_weights split by seat: eps to the
    total count must give the same weight, and `me` is never counted."""
    trump, dealer, me, legal, worlds, history = _decision_state(plies=13)
    worlds = np.ascontiguousarray(worlds, dtype=np.int64)
    hs = np.array([s for s, _ in history], dtype=np.int32)
    hc = np.array([c for _, c in history], dtype=np.int32)
    for al in (False, True):
        mm = mi.rule_playout_mismatches(trump, dealer, me, worlds, hs, hc, 0, al)
        w = mi.rule_playout_weights(trump, dealer, me, worlds, hs, hc, 0.5, 0, al)
        assert mm.shape == (len(worlds), 4)
        assert (mm[:, me] == 0).all()
        np.testing.assert_allclose(0.5 ** mm.sum(axis=1), w)


def test_belief_likelihood_plays_legal_games():
    for hands, dealer in make_deals(1, 3):
        p = dict(FAST, playout_lik='belief')
        agents = [RuleAgent(**p), RandomAgent(), RuleAgent(**p), RandomAgent()]
        r = run_game(agents, dealer=dealer, hands=hands, trump=trump_for(hands, dealer))
        assert sum(r.pts) == 60


def test_belief_weighted_knowledge_plays_legal_games():
    for hands, dealer in make_deals(1, 4):
        p = dict(FAST, playout_lik='belief', lik_knowledge=True)
        agents = [RuleAgent(**p), RandomAgent(), RuleAgent(**p), RandomAgent()]
        r = run_game(agents, dealer=dealer, hands=hands, trump=trump_for(hands, dealer))
        assert sum(r.pts) == 60


def test_root_q_blend_plays_legal_games_and_reports_values():
    for hands, dealer in make_deals(1, 6):
        p = dict(FAST, root_q_w=0.4, root_q_device='cpu')
        agents = [RuleAgent(**p), RandomAgent(), RuleAgent(**p), RandomAgent()]
        r = run_game(agents, dealer=dealer, hands=hands, trump=trump_for(hands, dealer))
        assert sum(r.pts) == 60


def test_symmetrized_root_q_blend_plays_legal_games():
    for hands, dealer in make_deals(1, 10):
        p = dict(FAST, root_q_w=0.4, root_q_sym=True, root_q_device='cpu')
        agents = [RuleAgent(**p), RandomAgent(), RuleAgent(**p), RandomAgent()]
        r = run_game(agents, dealer=dealer, hands=hands, trump=trump_for(hands, dealer))
        assert sum(r.pts) == 60


def test_no_trump_deals_switch_to_the_mull_nets():
    """One agent across a trump deal and a mull deal: the blend and the
    belief weights load the trump nets for the first and the no-trump
    lineage for the second."""
    (hands, dealer), = make_deals(1, 11)
    p = dict(FAST, root_q_w=0.4, root_q_device='cpu')
    agents = [RuleAgent(**p), RandomAgent(), RuleAgent(**p), RandomAgent()]
    assert sum(run_game(agents, dealer=dealer, hands=hands,
                        trump=trump_for(hands, dealer)).pts) == 60
    assert sum(run_game(agents, dealer=dealer, hands=hands, mull=True).pts) == 60
    a = agents[0]
    assert set(a._root_q) == {a.p['root_q_model'], a.p['mull_root_q_model']}
    assert set(a._belief_model) == {a.p['lik_belief_model'], a.p['mull_lik_belief_model']}



def test_no_trump_blend_weight_applies_only_at_no_trump():
    """mull_root_q_w with root_q_w=0: the net is consulted on the mull deal
    only (the trump net is never loaded)."""
    (hands, dealer), = make_deals(1, 13)
    p = dict(FAST, root_q_w=0.0, mull_root_q_w=0.6, root_q_device='cpu')
    agents = [RuleAgent(**p), RandomAgent(), RuleAgent(**p), RandomAgent()]
    assert sum(run_game(agents, dealer=dealer, hands=hands,
                        trump=trump_for(hands, dealer)).pts) == 60
    assert not getattr(agents[0], '_root_q', None)
    assert sum(run_game(agents, dealer=dealer, hands=hands, mull=True).pts) == 60
    assert set(agents[0]._root_q) == {agents[0].p['mull_root_q_model']}


def test_no_trump_rules_play_legal_games():
    for hands, dealer in make_deals(2, 14):
        p = dict(FAST, nt_tiers=True, playout_nt_rules=15, root_q_w=0.0)
        agents = [RuleAgent(**p), RandomAgent(), RuleAgent(**p), RandomAgent()]
        assert sum(run_game(agents, dealer=dealer, hands=hands, mull=True).pts) == 60


def test_boss_counts_cards_the_opponents_cannot_beat():
    """Hearts led by seat 1 (K), seat 2 follows low without beating: seat 2
    holds nothing above the K. Seen from seat 3 (partner of seat 1), with
    seat 0 also out of hearts the remaining hearts are opponent-proof."""
    k = _knowledge_after(HANDS, None, 0, [(1, C(1, 5)), (2, C(1, 1))], 3)
    assert k.cap[(2 - 3) % 4][1] <= 5          # seat 2, relative to observer 3
    assert k.is_master(C(1, 7).to_index()) == k.is_boss(C(1, 7).to_index())
