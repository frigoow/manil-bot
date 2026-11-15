"""QNetAgent: legal play and argmax semantics."""
import numpy as np
import pytest
import torch

import manille_ismcts as mi
from learn.dmc_agent import QNetAgent
from learn.dmc_model import QNet
from learn.evaluate import make_deals, trump_for
from manille_env.card import Card
from tournament import RandomAgent, game_to_cpp_arrays, run_game


def _agent(seed=0):
    torch.manual_seed(seed)
    return QNetAgent(QNet(), device="cpu")


def test_qnet_agent_plays_legal_full_game():
    deals = make_deals(3, 0)
    for hands, declarer in deals:
        trump = trump_for(hands, declarer)
        agents = [_agent(0), RandomAgent(), _agent(1), RandomAgent()]
        result = run_game(agents, dealer=declarer, hands=hands, trump=trump)
        assert sum(result.pts) == 60


def test_qnet_agent_picks_argmax():
    deals = make_deals(1, 0)
    hands, declarer = deals[0]
    trump = trump_for(hands, declarer)
    model = QNet()
    agent = QNetAgent(model, device="cpu")
    agents = [agent, RandomAgent(), _agent(1), RandomAgent()]

    # Walk a game; at every QNetAgent decision, recompute the expected argmax
    # from a fresh encode and compare.
    from manille_env.game_logic import GamePhase, ManilleGame
    from manille_env.card import Suit
    game = ManilleGame(dealer=declarer)
    game.hands = [list(h) for h in hands]
    game.select_trump(Suit(trump))
    game.start_playing()
    played_cards, played_players = [], []
    for i, a in enumerate(agents):
        a.new_game(game, declarer, i)
    pid = game.current_player
    while game.phase == GamePhase.PLAYING:
        pid = game.current_player
        if pid % 2 == 0:
            arrays = game_to_cpp_arrays(game, played_cards, played_players)
            enc = mi.encode_q_observation(*arrays, declarer, pid)
            n = int(enc["n_legal"])
            obs = np.repeat(enc["obs"][None, :], n, axis=0)
            with torch.no_grad():
                q = model(torch.from_numpy(obs),
                          torch.from_numpy(enc["act"][:n])).numpy()
            expected = int(enc["legal"][int(np.argmax(q))])
            card = agent.choose_card(game, pid, played_cards, played_players)
            assert card.to_index() == expected
        else:
            card = agents[pid].choose_card(game, pid, played_cards,
                                           played_players)
        pre = list(game.current_trick)
        assert game.play_card(pid, card)
        if len(pre) == 3:
            for tp_, tc_ in pre + [(pid, card)]:
                played_cards.append(tc_.to_index())
                played_players.append(tp_)


def test_qnet_agent_epsilon_stays_legal():
    deals = make_deals(2, 5)
    for hands, declarer in deals:
        trump = trump_for(hands, declarer)
        agent = QNetAgent(QNet(), device="cpu", epsilon=1.0, seed=3)
        agents = [agent, RandomAgent(), agent, RandomAgent()]
        result = run_game(agents, dealer=declarer, hands=hands, trump=trump)
        assert sum(result.pts) == 60


def test_qtransformer_agent_plays_legal_full_game():
    from learn.dmc_model import QTransformer
    deals = make_deals(2, 11)
    for hands, declarer in deals:
        trump = trump_for(hands, declarer)
        agents = [QNetAgent(QTransformer(), device="cpu"), RandomAgent(),
                  QNetAgent(QTransformer(), device="cpu"), RandomAgent()]
        result = run_game(agents, dealer=declarer, hands=hands, trump=trump)
        assert sum(result.pts) == 60


def _permuted_game(hands, declarer, trump, perm):
    """The same deal with its suits relabelled: an exact automorphism."""
    from manille_env.card import Suit, permute_card_index
    from manille_env.game_logic import ManilleGame

    game = ManilleGame(dealer=declarer)
    game.hands = [[Card.from_index(permute_card_index(c.to_index(), perm))
                   for c in h] for h in hands]
    game.select_trump(Suit(perm[trump]) if trump >= 0 else None)
    game.start_playing()
    return game


def test_symmetrized_agent_is_suit_equivariant():
    """The point of the wrapper: relabelling the suits must not move Q.

    The raw agent fails this on ~21% of non-forced plies (the whole reason
    the wrapper exists), so the raw arm is asserted too -- otherwise a
    degenerate model would let the symmetrized assertion pass for free."""
    from learn.dmc_agent import SymmetrizedQNetAgent
    from manille_env.card import permute_card_index

    torch.manual_seed(4)
    model = QNet()
    hands, declarer = make_deals(1, 11)[0]
    trump = trump_for(hands, declarer)
    # A permutation that fixes trump and 3-cycles the other three suits.
    others = [s for s in range(4) if s != trump]
    perm = [0] * 4
    perm[trump] = trump
    for i, s in enumerate(others):
        perm[s] = others[(i + 1) % 3]
    perm = tuple(perm)

    raw_gap, sym_gap = 0.0, 0.0
    for agent_cls, gap_name in ((QNetAgent, "raw"), (SymmetrizedQNetAgent,
                                                     "sym")):
        a = agent_cls(model, device="cpu")
        b = agent_cls(model, device="cpu")
        g0 = _permuted_game(hands, declarer, trump, (0, 1, 2, 3))
        g1 = _permuted_game(hands, declarer, trump, perm)
        a.new_game(g0, declarer, g0.current_player)
        b.new_game(g1, declarer, g1.current_player)
        pid = g0.current_player
        l0, q0 = a.q_values(g0, pid, [], [])
        l1, q1 = b.q_values(g1, pid, [], [])
        aligned = [q1[list(l1).index(permute_card_index(int(c), perm))]
                   for c in l0]
        gap = float(np.abs(np.asarray(aligned) - np.asarray(q0)).max())
        if gap_name == "raw":
            raw_gap = gap
        else:
            sym_gap = gap

    assert raw_gap > 1e-5, "untrained QNet was already suit-invariant"
    assert sym_gap < 1e-5, f"symmetrized agent not equivariant: {sym_gap}"


def test_symmetrized_agent_plays_legal_full_game():
    from learn.dmc_agent import SymmetrizedQNetAgent

    torch.manual_seed(5)
    model = QNet()
    for hands, declarer in make_deals(2, 12):
        trump = trump_for(hands, declarer)
        agents = [SymmetrizedQNetAgent(model, device="cpu"), RandomAgent(),
                  SymmetrizedQNetAgent(model, device="cpu"), RandomAgent()]
        result = run_game(agents, dealer=declarer, hands=hands, trump=trump)
        assert sum(result.pts) == 60


def test_symmetrized_agent_batches_all_orbit_views_in_one_forward():
    from learn.dmc_agent import SymmetrizedQNetAgent

    class CountingQNet(QNet):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, obs, act):
            self.calls += 1
            return super().forward(obs, act)

    model = CountingQNet()
    hands, declarer = make_deals(1, 121)[0]
    trump = trump_for(hands, declarer)
    game = _permuted_game(hands, declarer, trump, (0, 1, 2, 3))
    agent = SymmetrizedQNetAgent(model, device="cpu")
    agent.new_game(game, declarer, game.current_player)
    agent.q_values(game, game.current_player, [], [])
    assert model.calls == 1


def test_orbit_q_aggregations():
    from learn.dmc_agent import aggregate_orbit_q

    q = np.array([[0.0, 2.0, 1.0],
                  [100.0, 0.0, 1.0],
                  [0.0, 3.0, 2.0]])
    assert np.allclose(aggregate_orbit_q(q), q.mean(axis=0))
    assert np.allclose(aggregate_orbit_q(q, "median"), [0.0, 2.0, 1.0])
    # One extreme calibration row dominates the mean, but not equal-scale
    # or rank voting: both preserve the 2 > 1 preference of two views.
    assert np.argmax(aggregate_orbit_q(q, "mean")) == 0
    assert np.argmax(aggregate_orbit_q(q, "zscore")) == 1
    assert np.argmax(aggregate_orbit_q(q, "rank")) == 1
    with pytest.raises(ValueError, match="unknown orbit aggregation"):
        aggregate_orbit_q(q, "bogus")


def test_safe_orbit_never_breaks_the_trump_rule():
    """The orbit the agent averages over must stay on-distribution.

    apply_trump_rule breaks ties on the LOWEST suit index, so a trump-fixing
    permutation can still carry a tied suit below trump and change the pick.
    The safe set is the permutations preserving {s < trump} and {s > trump};
    this asserts both that it is safe and that the naive trump-fixing set is
    NOT -- otherwise the restriction would be pointless.
    """
    import manille_ismcts as mi
    from learn.dmc_agent import SymmetrizedQNetAgent
    from manille_env.card import permute_hand_mask

    torch.manual_seed(6)
    safe = SymmetrizedQNetAgent(QNet(), device="cpu")
    naive = SymmetrizedQNetAgent(QNet(), device="cpu", full_orbit=True)
    assert [len(safe._perms(t)) for t in range(4)] == [6, 2, 2, 6]
    assert [len(naive._perms(t)) for t in range(4)] == [6] * 4
    assert len(safe._perms(-1)) == 24          # mull: no rule, whole group

    rng = np.random.default_rng(0)
    naive_broke = False
    for _ in range(3000):
        mask = 0
        for c in rng.choice(32, size=8, replace=False):
            mask |= 1 << int(c)
        trump = mi.apply_trump_rule(mask)
        for p in safe._perms(trump):
            assert mi.apply_trump_rule(permute_hand_mask(mask, p)) == trump
        for p in naive._perms(trump):
            if mi.apply_trump_rule(permute_hand_mask(mask, p)) != trump:
                naive_broke = True
    assert naive_broke, "naive orbit never broke the rule — sample too small?"


def test_declarer_aware_orbit_expands_only_from_own_known_hand():
    import manille_ismcts as mi
    from learn.dmc_agent import SymmetrizedQNetAgent
    from manille_env.card import permute_hand_mask

    model = QNet()
    agent = SymmetrizedQNetAgent(
        model, device="cpu", declarer_aware=True)
    # Find a middle-trump deal where the actual hand permits more than the
    # globally conservative two views.
    for hands, declarer in make_deals(200, 122):
        trump = trump_for(hands, declarer)
        if trump not in (1, 2):
            continue
        game = _permuted_game(hands, declarer, trump, (0, 1, 2, 3))
        agent.new_game(game, declarer, declarer)
        expanded = agent._decision_perms(trump, declarer)
        if len(expanded) > len(agent._perms(trump)):
            break
    else:
        raise AssertionError("sample found no expandable declarer hand")

    assert len(expanded) > len(agent._perms(trump))
    for perm in expanded:
        assert mi.apply_trump_rule(
            permute_hand_mask(agent.initial_hand_mask, perm)) == trump
    # A different seat cannot use the declarer's private hand condition.
    assert agent._decision_perms(trump, (declarer + 1) % 4) == \
        agent._perms(trump)
