"""SPARTA-DMC: legality, deviation-threshold behavior, weight sanity."""
import numpy as np
import pytest
import torch

import manille_ismcts as mi
from learn.dmc_model import QNet
from learn.dmc_sparta import SpartaDMCAgent, plpw_q_log_weights
from learn.evaluate import make_deals, trump_for
from tournament import RandomAgent, game_to_cpp_arrays, run_game


def _agent(seed=0, **kw):
    torch.manual_seed(seed)
    return SpartaDMCAgent(QNet(), device="cpu", **kw)


def test_sparta_dmc_plays_legal_full_games():
    deals = make_deals(2, 13)
    for hands, declarer in deals:
        trump = trump_for(hands, declarer)
        agents = [_agent(0, n_particles=8), RandomAgent(),
                  _agent(1, n_particles=8), RandomAgent()]
        result = run_game(agents, dealer=declarer, hands=hands, trump=trump)
        assert sum(result.pts) == 60


def test_sparta_dmc_huge_delta_matches_blueprint():
    """With an enormous deviation threshold, SPARTA must always play the
    blueprint action."""
    agent = _agent(0, n_particles=8, delta=1e9)
    bp = agent.blueprint
    deals = make_deals(1, 31)
    hands, declarer = deals[0]
    trump = trump_for(hands, declarer)
    from manille_env.game_logic import GamePhase, ManilleGame
    from manille_env.card import Suit
    game = ManilleGame(dealer=declarer)
    game.hands = [list(h) for h in hands]
    game.select_trump(Suit(trump))
    game.start_playing()
    pc, pp = [], []
    agent.new_game(game, declarer, 0)
    bp.new_game(game, declarer, 0)
    while game.phase == GamePhase.PLAYING and len(pc) < 12:
        pid = game.current_player
        c1 = agent.choose_card(game, pid, pc, pp)
        c2 = bp.choose_card(game, pid, pc, pp)
        assert c1.to_index() == c2.to_index()
        pre = list(game.current_trick)
        assert game.play_card(pid, c1)
        if len(pre) == 3:
            for tp_, tc_ in pre + [(pid, c1)]:
                pc.append(tc_.to_index())
                pp.append(tp_)


def test_plpw_q_weights_sanity():
    """Q-likelihood weights: finite where expected, and the true world gets
    a finite likelihood."""
    torch.manual_seed(0)
    model = QNet()
    deals = make_deals(1, 41)
    hands, declarer = deals[0]
    trump = trump_for(hands, declarer)
    from manille_env.game_logic import GamePhase, ManilleGame
    from manille_env.card import Suit
    rng = np.random.default_rng(0)
    game = ManilleGame(dealer=declarer)
    game.hands = [list(h) for h in hands]
    game.select_trump(Suit(trump))
    game.start_playing()
    pc, pp = [], []
    while game.phase == GamePhase.PLAYING and len(pc) < 8:
        pid = game.current_player
        legal = game.get_legal_moves(pid)
        card = legal[int(rng.integers(len(legal)))]
        pre = list(game.current_trick)
        assert game.play_card(pid, card)
        if len(pre) == 3:
            for tp_, tc_ in pre + [(pid, card)]:
                pc.append(tc_.to_index())
                pp.append(tp_)
    arrays = game_to_cpp_arrays(game, pc, pp)
    pid = game.current_player
    obs_cards = np.array(pc, dtype=np.int32)
    obs_players = np.array(pp, dtype=np.int32)
    particles = mi.sample_determinizations(*arrays, pid, 16)
    logw = plpw_q_log_weights(model, "cpu", particles, obs_players,
                              obs_cards, int(arrays[8]), declarer, pid,
                              tau=0.2)
    # The true world (current holdings) must be among the particles' set of
    # consistent worlds conceptually; its likelihood should not be the worst.
    world = np.array([int(arrays[0][p]) for p in range(4)], dtype=np.int64)
    logw_true = plpw_q_log_weights(model, "cpu", world[None, :],
                                   obs_players, obs_cards, int(arrays[8]),
                                   declarer, pid, tau=0.2)
    assert np.isfinite(logw).sum() >= 1
    assert np.isfinite(logw_true[0])


def test_exact_leaves_match_exact_solve():
    """A world already at trick >= 6 must terminate with the exact minimax
    value from mi.exact_solve (banked points included), no Q calls needed."""
    import numpy as np
    import manille_ismcts as mi
    from learn.dmc_sparta import q_blueprint_rollouts
    from learn.evaluate import make_deals, trump_for
    from manille_env.card import Card, Suit
    from manille_env.game_logic import GamePhase, ManilleGame
    from tournament import game_to_cpp_arrays
    from learn.sparta import inject_world

    deals = make_deals(1, 51)
    hands, declarer = deals[0]
    trump = trump_for(hands, declarer)
    rng = np.random.default_rng(0)
    game = ManilleGame(dealer=declarer)
    game.hands = [list(h) for h in hands]
    game.select_trump(Suit(trump))
    game.start_playing()
    pc, pp = [], []
    while (len(pc) + len(game.current_trick)) // 4 < 6:
        pid = game.current_player
        legal = game.get_legal_moves(pid)
        card = legal[int(rng.integers(len(legal)))]
        pre = list(game.current_trick)
        assert game.play_card(pid, card)
        if len(pre) == 3:
            for tp_, tc_ in pre + [(pid, card)]:
                pc.append(tc_.to_index())
                pp.append(tp_)
    assert (len(pc) + len(game.current_trick)) // 4 >= 6

    masks = np.zeros(4, dtype=np.int64)
    for p in range(4):
        for c in game.hands[p]:
            masks[p] |= np.int64(1) << c.to_index()
    world_game = inject_world(declarer, int(game.trump_suit), masks,
                              game.current_trick, game.hand_points,
                              game.tricks_won, game.current_player)
    worlds = [(world_game, list(pc), list(pp))]
    pts = q_blueprint_rollouts(worlds, declarer, None, "cpu", None, 0.0,
                               True)

    arrays = game_to_cpp_arrays(game, pc, pp)
    res = mi.exact_solve(*arrays, declarer, masks)
    vals = res["pts0"]
    v = max(vals) if game.current_player % 2 == 0 else min(vals)
    assert pts[0, 0] == v
    assert pts[0, 1] == 60.0 - v


def test_sparta_dmc_ucb_plays_legal_full_games():
    deals = make_deals(2, 17)
    for hands, declarer in deals:
        trump = trump_for(hands, declarer)
        agents = [_agent(0, n_particles=8, alloc="ucb", ucb_chunk=4),
                  RandomAgent(),
                  _agent(1, n_particles=8, alloc="ucb", ucb_chunk=4),
                  RandomAgent()]
        result = run_game(agents, dealer=declarer, hands=hands, trump=trump)
        assert sum(result.pts) == 60


def test_sparta_dmc_belief_weighted_worlds_smoke():
    """With B_phi mixed into the world weighting (lam_belief > 0), analyze
    still returns finite per-arm values covering every legal card."""
    from learn.dmc_model import BeliefNet
    agent = _agent(0, n_particles=8, belief_model=BeliefNet(), lam_belief=1.0)
    deals = make_deals(1, 41)
    hands, declarer = deals[0]
    trump = trump_for(hands, declarer)
    from manille_env.game_logic import GamePhase, ManilleGame
    from manille_env.card import Suit
    rng = np.random.default_rng(0)
    game = ManilleGame(dealer=declarer)
    game.hands = [list(h) for h in hands]
    game.select_trump(Suit(trump))
    game.start_playing()
    pc, pp = [], []
    while game.phase == GamePhase.PLAYING and len(pc) < 8:
        pid = game.current_player
        legal = game.get_legal_moves(pid)
        card = legal[int(rng.integers(len(legal)))]
        pre = list(game.current_trick)
        assert game.play_card(pid, card)
        if len(pre) == 3:
            for tp_, tc_ in pre + [(pid, card)]:
                pc.append(tc_.to_index())
                pp.append(tp_)
    agent.new_game(game, declarer, game.current_player)
    pid = game.current_player
    legal = game.get_legal_moves(pid)
    a_bp, a_star, q, counts = agent.analyze(game, pid, pc, pp)
    assert set(q) == set(legal)
    assert all(np.isfinite(v) for v in q.values())
    assert a_bp in legal and a_star in legal


def test_sparta_dmc_belief_pure_no_nan_at_trick0():
    """Regression (W2 bug, 2026-07-22): with lam=0 (pure B_phi worlds), -inf
    Q log-weights — a particle that replayed zero observed decisions, e.g.
    the trick-0 leader — must not hit 0 * -inf = NaN in the composed world
    weights (which silently zeroed/uniformed early-game worlds)."""
    import warnings
    from learn.dmc_model import BeliefNet
    agent = _agent(0, n_particles=8, belief_model=BeliefNet(), lam=0.0,
                   lam_belief=1.0)
    deals = make_deals(1, 43)
    hands, declarer = deals[0]
    trump = trump_for(hands, declarer)
    from manille_env.game_logic import ManilleGame
    from manille_env.card import Suit
    game = ManilleGame(dealer=declarer)
    game.hands = [list(h) for h in hands]
    game.select_trump(Suit(trump))
    game.start_playing()
    agent.new_game(game, declarer, game.current_player)
    pid = game.current_player  # trick-0 leader: zero observed plays
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        a_bp, a_star, q, counts = agent.analyze(game, pid, [], [])
    legal = game.get_legal_moves(pid)
    assert set(q) == set(legal)
    assert all(np.isfinite(v) for v in q.values())
    assert a_bp in legal and a_star in legal


def _midgame(seed=41, plies=8):
    """A deal driven `plies` random plays in: a non-trivial decision state."""
    from manille_env.card import Suit
    from manille_env.game_logic import GamePhase, ManilleGame

    deals = make_deals(1, seed)
    hands, declarer = deals[0]
    trump = trump_for(hands, declarer)
    rng = np.random.default_rng(0)
    game = ManilleGame(dealer=declarer)
    game.hands = [list(h) for h in hands]
    game.select_trump(Suit(trump))
    game.start_playing()
    pc, pp = [], []
    while game.phase == GamePhase.PLAYING and len(pc) < plies:
        pid = game.current_player
        legal = game.get_legal_moves(pid)
        card = legal[int(rng.integers(len(legal)))]
        pre = list(game.current_trick)
        assert game.play_card(pid, card)
        if len(pre) == 3:
            for tp_, tc_ in pre + [(pid, card)]:
                pc.append(tc_.to_index())
                pp.append(tp_)
    return game, declarer, pc, pp


def test_sparta_dmc_tier3_trigger_and_legal_games():
    """Tier-3: negative eps never triggers; huge eps always triggers on
    multi-legal decisions (with finite refined values); full games legal."""
    game, declarer, pc, pp = _midgame()
    pid = game.current_player
    legal = game.get_legal_moves(pid)
    assert len(legal) > 1

    off = _agent(0, n_particles=8, alloc="ucb", ucb_chunk=4, tier3=True,
                 t3_eps=-1.0, t3_worlds=8)
    off.new_game(game, declarer, pid)
    off.analyze(game, pid, pc, pp)
    assert off.t3_stats["triggered"] == 0

    on = _agent(0, n_particles=8, alloc="ucb", ucb_chunk=4, tier3=True,
                t3_eps=1e9, t3_worlds=8)
    on.new_game(game, declarer, pid)
    a_bp, a_star, q, counts = on.analyze(game, pid, pc, pp)
    assert on.t3_stats["triggered"] == 1
    assert set(q) == set(legal)
    assert all(np.isfinite(v) for v in q.values())
    assert a_bp in legal and a_star in legal

    deals = make_deals(1, 19)
    for hands, declarer in deals:
        trump = trump_for(hands, declarer)
        agents = [_agent(0, n_particles=8, alloc="ucb", ucb_chunk=4,
                         tier3=True, t3_eps=1e9, t3_worlds=8),
                  RandomAgent(),
                  _agent(1, n_particles=8, alloc="ucb", ucb_chunk=4,
                         tier3=True, t3_eps=1e9, t3_worlds=8),
                  RandomAgent()]
        result = run_game(agents, dealer=declarer, hands=hands, trump=trump)
        assert sum(result.pts) == 60


def test_sparta_dmc_gru_model_paths():
    """SPARTA's Q-eval paths accept a token model (QGRU): analyze, rollout
    and plpw weight computation must not hit the MLP-only (obs, act) call
    (the SPARTA-on-GRU deployment bug)."""
    from learn.dmc_model import QGRU
    torch.manual_seed(0)
    game, declarer, pc, pp = _midgame()
    pid = game.current_player
    agent = SpartaDMCAgent(QGRU(d_model=64, layers=1, hidden=128, blocks=1),
                           device="cpu", n_particles=8, alloc="ucb",
                           ucb_chunk=4)
    agent.new_game(game, declarer, pid)
    a_bp, a_star, q, counts = agent.analyze(game, pid, pc, pp)
    legal = game.get_legal_moves(pid)
    assert set(q) == set(legal)
    assert all(np.isfinite(v) for v in q.values())
    assert a_bp in legal and a_star in legal


def test_tokens_batch_matches_encode_live():
    """Vectorized tokens_batch must equal per-row encode_live on real
    observations (including the empty-history row), and return hand=None
    when own_masks is omitted (the QGRU path)."""
    from learn.dmc_tokens import encode_live, tokens_batch

    rows_obs, rows_n, rows_mask = [], [], []
    for plies in (0, 5, 9, 13):
        game, declarer, pc, pp = _midgame(plies=plies)
        pid = game.current_player
        arrays = game_to_cpp_arrays(game, pc, pp)
        enc = mi.encode_q_observation(*arrays, declarer, pid)
        own = 0
        for c in game.hands[pid]:
            own |= 1 << c.to_index()
        rows_obs.append(enc["obs"])
        rows_n.append(len(pc) + len(game.current_trick))
        rows_mask.append(own)
    raw, n_raw, glob, hand = tokens_batch(np.stack(rows_obs), rows_n,
                                          rows_mask)
    for i in range(len(rows_obs)):
        t = encode_live(rows_obs[i], rows_n[i], rows_mask[i])
        assert np.array_equal(raw[i], t["raw"])
        assert int(n_raw[i]) == t["n_raw"]
        assert np.array_equal(glob[i], t["glob"])
        assert np.array_equal(hand[i], t["hand"])
    assert tokens_batch(np.stack(rows_obs), rows_n)[3] is None


def test_q_eval_batch_gru_dedup_matches_naive():
    """QGRU trunk dedup (trunk once per unique row, pooled embedding
    expanded per candidate) must be numerically identical to the naive
    per-candidate full forward."""
    from learn.dmc_model import QGRU
    from learn.dmc_sparta import q_eval_batch
    from learn.dmc_tokens import MAX_HAND, PAD_CARD, tokens_batch

    torch.manual_seed(1)
    model = QGRU(d_model=32, layers=1, hidden=64, blocks=1).eval()
    rows_obs, rows_meta, acts, counts = [], [], [], []
    for plies in (0, 5, 9):
        game, declarer, pc, pp = _midgame(plies=plies)
        pid = game.current_player
        arrays = game_to_cpp_arrays(game, pc, pp)
        enc = mi.encode_q_observation(*arrays, declarer, pid)
        n = int(enc["n_legal"])
        rows_obs.append(enc["obs"])
        rows_meta.append((len(pc) + len(game.current_trick), 0))
        acts.append(enc["act"][:n])
        counts.append(n)
    act = np.concatenate(acts, axis=0)
    q = q_eval_batch(model, "cpu", np.stack(rows_obs), act,
                     token_meta=rows_meta, counts=counts).numpy()

    raw, n_raw, glob, _ = tokens_batch(np.stack(rows_obs),
                                       [m[0] for m in rows_meta])
    pad_hand = np.full((sum(counts), MAX_HAND), PAD_CARD, dtype=np.int64)
    with torch.no_grad():
        q_naive = model(
            torch.from_numpy(np.repeat(raw, counts, axis=0)),
            torch.from_numpy(np.repeat(n_raw, counts)),
            torch.from_numpy(np.repeat(glob, counts, axis=0)),
            torch.from_numpy(pad_hand),
            torch.from_numpy(act)).numpy()
    assert q.shape == q_naive.shape == (sum(counts),)
    assert np.allclose(q, q_naive, atol=1e-6)


def test_q_eval_batch_applies_student_only_to_selected_rows():
    """DAgger continuation swaps policy on one seat without changing Q rows
    belonging to the three frozen-blueprint seats."""
    from learn.dmc_model import QGRU
    from learn.dmc_sparta import q_eval_batch

    class HighestCardHead(torch.nn.Module):
        def logits(self, q, last, glob, act, tau):
            return act[:, :32].argmax(dim=1).float()

    torch.manual_seed(2)
    model = QGRU(d_model=16, layers=1, hidden=32, blocks=1).eval()
    rows_obs, rows_meta, acts, counts = [], [], [], []
    for plies in (0, 5):
        game, declarer, pc, pp = _midgame(plies=plies)
        arrays = game_to_cpp_arrays(game, pc, pp)
        enc = mi.encode_q_observation(*arrays, declarer,
                                      game.current_player)
        n = int(enc["n_legal"])
        rows_obs.append(enc["obs"])
        rows_meta.append((len(pc) + len(game.current_trick), 0))
        acts.append(enc["act"][:n])
        counts.append(n)
    act = np.concatenate(acts)
    baseline = q_eval_batch(
        model, "cpu", np.stack(rows_obs), act, token_meta=rows_meta,
        counts=counts).numpy()
    routed = q_eval_batch(
        model, "cpu", np.stack(rows_obs), act, token_meta=rows_meta,
        counts=counts, policy_head=HighestCardHead(), policy_tau=2.0,
        policy_rows=[True, False]).numpy()
    first = counts[0]
    np.testing.assert_array_equal(
        routed[:first], acts[0][:, :32].argmax(axis=1))
    np.testing.assert_allclose(routed[first:], baseline[first:], atol=1e-6)


def test_arm_values_cross_arm_batching_exact():
    """Concatenating arms into one rollout batch must not change any
    per-world value vs evaluating each arm alone: worlds evolve
    independently, so the batched call must be bitwise identical.
    (Note: equivalence must be checked in-process on shared particles —
    mi.sample_determinizations is not seeded per call.)"""
    from learn.dmc_model import QNet

    torch.manual_seed(0)
    agent = SpartaDMCAgent(QNet().eval(), device="cpu", n_particles=8,
                           seed=7)
    game, declarer, pc, pp = _midgame()
    pid = game.current_player
    agent.new_game(game, declarer, pid)
    legal = game.get_legal_moves(pid)
    assert len(legal) > 1
    arrays = game_to_cpp_arrays(game, pc, pp)
    particles = mi.sample_determinizations(*arrays, pid, 8,
                                           declarer=declarer)
    vals = agent._arm_values(game, pid, pc, pp, legal, particles, pid % 2,
                             int(arrays[8]), 0, np.arange(8))
    per = np.stack([
        agent._arm_values(game, pid, pc, pp, [c], particles, pid % 2,
                          int(arrays[8]), 0, np.arange(8))[0]
        for c in legal])
    assert vals.shape == (len(legal), 8)
    assert np.array_equal(vals, per)


def test_rollout_memo_and_dedup_equivalent():
    """Row dedup + q/leaf memos must not change rollout outcomes: a batch
    with duplicated worlds must give each world the same value as rolling
    it out alone, and the duplicates must hit the memo."""
    from learn.dmc_model import QNet
    from learn.dmc_sparta import q_blueprint_rollouts
    from learn.sparta import inject_world

    torch.manual_seed(0)
    model = QNet().eval()
    game, declarer, pc, pp = _midgame()
    pid = game.current_player
    arrays = game_to_cpp_arrays(game, pc, pp)
    particles = mi.sample_determinizations(*arrays, pid, 4,
                                           declarer=declarer)

    def build(ks):
        worlds = []
        for k in ks:
            wg = inject_world(declarer, int(arrays[8]), particles[k],
                              game.current_trick, game.hand_points,
                              game.tricks_won, game.current_player)
            worlds.append((wg, list(pc), list(pp)))
        return worlds

    ks = [0, 1, 2, 3, 0, 1, 2, 3]  # duplicates on purpose
    stats = {"q_hit": 0, "q_miss": 0, "leaf_hit": 0, "leaf_miss": 0}
    pts = q_blueprint_rollouts(build(ks), declarer, model, "cpu", None,
                               0.0, True, 6, q_memo={}, leaf_memo={},
                               memo_stats=stats)
    singles = np.concatenate([
        q_blueprint_rollouts(build([k]), declarer, model, "cpu", None,
                             0.0, True, 6)
        for k in ks])
    assert np.allclose(pts, singles, atol=1e-5)
    assert np.array_equal(pts[:4], pts[4:])  # duplicates identical
    assert stats["q_hit"] > 0
    assert stats["leaf_hit"] > 0


def test_sparta_dmc_belief_proposal_smoke():
    """Belief-as-proposal worlds (proposal_oversample > 0): analyze returns
    finite values over exactly the legal set; the searched world count may
    collapse below n_particles (duplicates fold into multiplicities)."""
    from learn.dmc_model import BeliefNet

    agent = _agent(0, n_particles=8, alloc="ucb", ucb_chunk=4,
                   belief_model=BeliefNet(), lam_belief=0.0,
                   proposal_oversample=4)
    game, declarer, pc, pp = _midgame()
    pid = game.current_player
    agent.new_game(game, declarer, pid)
    a_bp, a_star, q, counts = agent.analyze(game, pid, pc, pp)
    legal = game.get_legal_moves(pid)
    assert set(q) == set(legal)
    assert all(np.isfinite(v) for v in q.values())
    assert a_bp in legal and a_star in legal
    assert all(0 < c <= 8 for c in counts.values())


def test_sparta_dmc_soft_choice_world_weights():
    """Soft choice-likelihood worlds (Phase 6 free trump): with a choice
    model attached, determinizations must be sampled UNCONSTRAINED
    (declarer=-1 — the hard rule constraint is wrong under net-chosen
    trump) and the world weights must tilt toward worlds where the
    declarer's reconstructed initial hand makes the observed trump likely
    under the chooser."""
    import learn.dmc_sparta as sparta_mod
    from learn.inference import initial_hands

    class CountSuits(torch.nn.Module):
        """Stub chooser: V(hand, suit) = #cards of that suit (the rule's
        'most cards' heuristic, so a low-temp softmax is near-hard)."""
        def forward(self, hands, suits):
            oh = torch.nn.functional.one_hot(
                torch.arange(32) // 8, 4).float()
            return ((hands @ oh) * suits).sum(-1)

    game, declarer, pc, pp = _midgame()
    pid = game.current_player
    arrays = game_to_cpp_arrays(game, pc, pp)
    trump = int(arrays[8])

    seen = {}
    orig_sample = mi.sample_determinizations
    orig_norm = sparta_mod.normalize_weights

    def sample_spy(*a, **kw):
        seen["declarer"] = kw.get("declarer", -1)
        seen["particles"] = orig_sample(*a, **kw)
        return seen["particles"]

    def norm_spy(logw, n, eps=0.0):
        seen["logw"] = np.asarray(logw, dtype=float).copy()
        return orig_norm(logw, n, eps)

    agent = _agent(0, n_particles=16, alloc="ucb", ucb_chunk=4,
                   choice_model=CountSuits(), choice_temp=0.01)
    agent.new_game(game, declarer, pid)
    sparta_mod.mi.sample_determinizations = sample_spy
    sparta_mod.normalize_weights = norm_spy
    try:
        a_bp, a_star, q, counts = agent.analyze(game, pid, pc, pp)
    finally:
        sparta_mod.mi.sample_determinizations = orig_sample
        sparta_mod.normalize_weights = orig_norm

    legal = game.get_legal_moves(pid)
    assert set(q) == set(legal)
    assert a_bp in legal and a_star in legal
    assert seen["declarer"] == -1  # unconstrained sampling

    obs_cards = np.array(pc, dtype=np.int32)
    obs_players = np.array(pp, dtype=np.int32)
    trump_mask = sum(1 << c for c in range(trump * 8, trump * 8 + 8))
    trump_counts = np.array([
        bin(initial_hands(p, obs_players, obs_cards)[declarer]
            & trump_mask).count("1")
        for p in seen["particles"]])
    logw = seen["logw"]
    finite = np.isfinite(logw)
    assert finite.sum() >= 8
    # choice_temp 0.01 => ~100 nats per trump card, dwarfing the Q term:
    # the extreme weights must align with the extremes of trump count.
    top = int(np.argmax(np.where(finite, logw, -np.inf)))
    bot = int(np.argmin(np.where(finite, logw, np.inf)))
    assert trump_counts[top] == trump_counts[finite].max()
    assert trump_counts[bot] == trump_counts[finite].min()

    # The deal-time hook: the harness asks this agent for trump.
    hand = (1 << 0) | (1 << 1) | (1 << 8) | (1 << 9) | (1 << 10) | (1 << 16)
    assert agent.choose_trump_suit(hand) == 1

    # Control: no chooser attached -> the hard-constraint path (declarer).
    agent2 = _agent(1, n_particles=8)
    agent2.new_game(game, declarer, pid)
    sparta_mod.mi.sample_determinizations = sample_spy
    try:
        agent2.analyze(game, pid, pc, pp)
    finally:
        sparta_mod.mi.sample_determinizations = orig_sample
    assert seen["declarer"] == declarer


def test_sparta_dmc_soft_choice_tilt_survives_trick0_lead():
    """Trick-0 leader with a choice model: every particle's PLPW logw is
    -inf (zero observed plays), which must NOT drown the finite choice
    tilt into the all-degenerate uniform fallback — the first decision is
    where the unconstrained sampler needs the trump information most."""
    import learn.dmc_sparta as sparta_mod

    class CountSuits(torch.nn.Module):
        def forward(self, hands, suits):
            oh = torch.nn.functional.one_hot(
                torch.arange(32) // 8, 4).float()
            return ((hands @ oh) * suits).sum(-1)

    game, declarer, pc, pp = _midgame(plies=0)  # trick-0 lead, nothing seen
    pid = game.current_player
    assert not pc and not game.current_trick

    seen = {}
    orig_norm = sparta_mod.normalize_weights

    def norm_spy(logw, n, eps=0.0):
        seen["logw"] = np.asarray(logw, dtype=float).copy()
        return orig_norm(logw, n, eps)

    agent = _agent(0, n_particles=16, alloc="ucb", ucb_chunk=4,
                   choice_model=CountSuits(), choice_temp=0.01)
    agent.new_game(game, declarer, pid)
    sparta_mod.normalize_weights = norm_spy
    try:
        agent.analyze(game, pid, pc, pp)
    finally:
        sparta_mod.normalize_weights = orig_norm

    logw = seen["logw"]
    assert np.isfinite(logw).all()  # tilt survived, no uniform fallback
    assert logw.max() - logw.min() > 1.0  # genuinely tilted, not flat


def test_sparta_dmc_decl_tilt_unconstrained_and_rule_declarations():
    """M6.3 equity tilt (decl_tilt_model): worlds must be sampled
    UNCONSTRAINED (declarer=-1) and tilted toward worlds whose declarer
    hand justifies the observed declaration -- while declarations stay
    with the rule (choice_model is None, so trump_for_seat never asks
    this agent)."""
    import learn.dmc_sparta as sparta_mod
    from learn.evaluate import trump_for_seat
    from learn.inference import initial_hands

    class TrumpCountTilt:
        """Stub equity tilt: log-likelihood ~ declarer's trump-card count,
        plus a mull slot at the count of suit 0. Records the declaration
        context it was queried with — it was previously never given one."""
        seen = []

        def log_likelihood(self, hand_masks, observed_decl, us=0, them=0):
            self.seen.append((us, them))
            vals = []
            for m in hand_masks:
                vals.append([bin(int(m) & sum(1 << c
                             for c in range(s * 8, s * 8 + 8))).count("1")
                             for s in range(4)] + [0.0])
            z = np.array(vals, dtype=float) * 100.0
            z = z - z.max(axis=1, keepdims=True)
            lse = np.log(np.exp(z).sum(axis=1))
            obs = 4 if int(observed_decl) in (-1, 4) else int(observed_decl)
            return z[:, obs] - lse

    game, declarer, pc, pp = _midgame()
    pid = game.current_player
    arrays = game_to_cpp_arrays(game, pc, pp)
    trump = int(arrays[8])

    seen = {}
    orig_sample = mi.sample_determinizations
    orig_norm = sparta_mod.normalize_weights

    def sample_spy(*a, **kw):
        seen["declarer"] = kw.get("declarer", -1)
        seen["particles"] = orig_sample(*a, **kw)
        return seen["particles"]

    def norm_spy(logw, n, eps=0.0):
        seen["logw"] = np.asarray(logw, dtype=float).copy()
        return orig_norm(logw, n, eps)

    agent = _agent(0, n_particles=16, alloc="ucb", ucb_chunk=4,
                   decl_tilt_model=TrumpCountTilt())
    agent.new_game(game, declarer, pid)
    sparta_mod.mi.sample_determinizations = sample_spy
    sparta_mod.normalize_weights = norm_spy
    try:
        a_bp, a_star, q, counts = agent.analyze(game, pid, pc, pp)
    finally:
        sparta_mod.mi.sample_determinizations = orig_sample
        sparta_mod.normalize_weights = orig_norm

    legal = game.get_legal_moves(pid)
    assert set(q) == set(legal)
    assert a_bp in legal and a_star in legal
    assert seen["declarer"] == -1  # unconstrained sampling

    obs_cards = np.array(pc, dtype=np.int32)
    obs_players = np.array(pp, dtype=np.int32)
    trump_mask = sum(1 << c for c in range(trump * 8, trump * 8 + 8))
    trump_counts = np.array([
        bin(initial_hands(p, obs_players, obs_cards)[declarer]
            & trump_mask).count("1")
        for p in seen["particles"]])
    logw = seen["logw"]
    finite = np.isfinite(logw)
    assert finite.sum() >= 8
    top = int(np.argmax(np.where(finite, logw, -np.inf)))
    bot = int(np.argmin(np.where(finite, logw, np.inf)))
    assert trump_counts[top] == trump_counts[finite].max()
    assert trump_counts[bot] == trump_counts[finite].min()

    # Tilt only: the harness must still get the RULE's trump from this
    # agent's seat -- decl_tilt never takes over declarations.
    hands = game.hands
    assert trump_for_seat([agent, agent, agent, agent], hands,
                          declarer) == trump_for(hands, declarer)


def test_sparta_ismcts_handoff_agent_paths():
    """Arm C (endgame_leak_plan.md): pre-handoff plies must go through
    SPARTA's choose_card (no solver call); plies at/after handoff_trick
    must call the C++ solver with SPARTA's weighted particle pool injected
    (particles [n,4] int64, weights finite, summing to 1)."""
    from learn.dmc_endgame_handoff import SpartaIsmctsHandoffAgent

    agent = SpartaIsmctsHandoffAgent(QNet(), device="cpu", n_particles=8,
                                     ismcts_iterations=50)

    class SpySolver:  # pybind attrs are read-only; wrap at our level
        def __init__(self, real):
            self.real, self.calls = real, []

        def solve(self, *a, **kw):
            self.calls.append(kw)
            return self.real.solve(*a, **kw)

    agent.solver = SpySolver(agent.solver)
    calls = agent.solver.calls

    game, declarer, pc, pp = _midgame(plies=4)  # trick 1: SPARTA path
    pid = game.current_player
    agent.new_game(game, declarer, pid)
    agent.choose_card(game, pid, pc, pp)
    assert calls == []  # pre-handoff: no ISMCTS

    game, declarer, pc, pp = _midgame(plies=24)  # trick 6: handoff
    pid = game.current_player
    legal = game.get_legal_moves(pid)
    assert len(legal) > 1
    agent.new_game(game, declarer, pid)
    card = agent.choose_card(game, pid, pc, pp)
    assert card in legal
    assert len(calls) == 1
    parts, w = calls[0]["particles"], calls[0]["particle_weights"]
    assert parts.ndim == 2 and parts.shape[1] == 4
    assert np.isfinite(w).all() and abs(w.sum() - 1.0) < 1e-6


def test_paired_evidence_and_conf_rule():
    """paired_evidence must compute the weighted mean/SE of the per-world
    difference on the common world prefix, and the 'conf' deviation rule
    must use it (deviating only on significant paired evidence) while
    falling back to the delta rule when there is no paired data."""
    agent = _agent(0, n_particles=8, alloc="ucb", ucb_chunk=4)
    game, declarer, pc, pp = _midgame()
    pid = game.current_player
    agent.new_game(game, declarer, pid)
    a_bp, a_star, q, counts = agent.analyze(game, pid, pc, pp)
    legal = game.get_legal_moves(pid)
    assert set(agent._last_samples) == set(legal)

    # Reference computation on the common prefix.
    v_star, w_star = agent._last_samples[a_star]
    v_bp, _ = agent._last_samples[a_bp]
    k = min(len(v_star), len(v_bp))
    m, se, n = agent.paired_evidence(a_star, a_bp)
    assert n == k
    if k >= 2:
        d = np.asarray(v_star[:k]) - np.asarray(v_bp[:k])
        w = np.asarray(w_star[:k], dtype=np.float64)
        ref_m = float((w * d).sum() / w.sum())
        assert abs(m - ref_m) < 1e-9
        assert se >= 0.0
        # The paired mean must equal the difference of the weighted means
        # when both arms saw every world (no UCB pruning).
        if len(v_star) == len(v_bp):
            assert abs(m - (q[a_star] - q[a_bp])) < 1e-6

    # conf rule: c=0 deviates whenever the paired mean is positive;
    # a huge c never deviates (evidence can't clear the bar).
    conf_hi = _agent(0, n_particles=8, alloc="ucb", ucb_chunk=4,
                     deviate_rule="conf", conf_c=1e9)
    conf_hi.new_game(game, declarer, pid)
    assert conf_hi.choose_card(game, pid, pc, pp) in legal
    # unknown arms -> no paired data -> None (delta-rule fallback path)
    assert agent.paired_evidence(a_star, "not-a-card")[1] is None


def test_conf_rule_falls_back_when_no_samples():
    """A forced move and a fresh agent must not crash the conf rule."""
    agent = _agent(0, n_particles=8, deviate_rule="conf", conf_c=1.0)
    game, declarer, pc, pp = _midgame()
    pid = game.current_player
    agent.new_game(game, declarer, pid)
    m, se, n = agent.paired_evidence(None, None)
    assert m is None and se is None and n == 0
    assert agent.choose_card(game, pid, pc, pp) in game.get_legal_moves(pid)


def test_paired_stats_beat_unpaired_when_worlds_dominate():
    """Why the racing rule is paired at all: across-world variance swamps
    the effect, and it cancels in the difference."""
    from learn.dmc_sparta import weighted_paired_stats

    rng = np.random.default_rng(0)
    world = rng.normal(0, 10.0, 200)      # deal difficulty, huge
    a = world + 0.5                        # arm A is 0.5 better, every world
    b = world
    w = np.ones(200)

    m, se, n = weighted_paired_stats(a, b, w)
    assert n == 200
    assert abs(m - 0.5) < 1e-9
    assert se < 1e-9                       # paired: the world term cancels
    # Unpaired, the same data cannot resolve 0.5 at all.
    unpaired_se = np.sqrt(a.var(ddof=1) / 200 + b.var(ddof=1) / 200)
    assert unpaired_se > 0.5


def test_delta_and_ess_standard_errors_differ_when_weights_correlate():
    """The ESS form assumes weights are uncorrelated with the squared
    deviations. PLPW's are not: the heaviest worlds are the ones the
    blueprint finds likely, not the ones where the arms disagree."""
    from learn.dmc_sparta import weighted_paired_stats

    d = np.array([0.0, 0.0, 0.0, 0.0, 10.0])
    w = np.array([1.0, 1.0, 1.0, 1.0, 0.01])   # the outlier is nearly ignored
    _, se_delta, _ = weighted_paired_stats(d, np.zeros(5), w)
    _, se_ess, _ = weighted_paired_stats(d, np.zeros(5), w, ess_variance=True)
    assert se_delta != pytest.approx(se_ess, rel=1e-3)
    # A world carrying 1% of the weight cannot contribute a full-size
    # uncertainty; the ESS form lets it.
    assert se_delta < se_ess


def test_uniform_weights_make_the_two_estimators_agree():
    """Sanity: they only diverge because of the weights."""
    from learn.dmc_sparta import weighted_paired_stats

    rng = np.random.default_rng(1)
    d = rng.normal(0, 1, 64)
    w = np.ones(64)
    _, a, _ = weighted_paired_stats(d, np.zeros(64), w)
    _, b, _ = weighted_paired_stats(d, np.zeros(64), w, ess_variance=True)
    assert a == pytest.approx(b, rel=1e-9)


def test_paired_stats_refuse_to_judge_a_single_world():
    """Pruning on one world is how a winning arm gets thrown away."""
    from learn.dmc_sparta import weighted_paired_stats

    m, se, n = weighted_paired_stats([1.0], [0.0], [1.0])
    assert se is None and n == 1
