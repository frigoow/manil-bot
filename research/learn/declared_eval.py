"""Duplicate-deal matches with REAL declarations: the equity decision layer
(learn/decision_layer.py, the policy that declares for bot seats in the GUI)
picks trump or mull for the dealer and makes the meegaan / terug calls, and
the agents play what was declared. This is the deployment setting: trump is
no longer the C++ rule's, a mull is the dealer's choice (so it says something
about the hand), and doubles are on the table.

The calls depend only on the hands and the match score, and both teams use
the same policy, so a deal gets the same declaration and the same doubles in
both seatings: the pair stays a duplicate pair.

Scores are reported as raw hand points per deal (play strength, comparable
with every other match in this repo) and as multiplied points (what the
match score sees: mull x2, meegaan x2, terug x2).

    from learn.declared_eval import run_declared_match
    raw, mult, info = run_declared_match(spec0, spec1, 2000, seed=70)
"""
import multiprocessing as mp
import os

import numpy as np

from manille_env.card import Suit
from manille_env.game_logic import GamePhase, ManilleGame

from .evaluate import build_agent, make_deals
from .match_harness import MULL, hand_mask

DEFAULT_POLICY = "runs/policy_state_rates.pt"


def load_equity_policy(path=DEFAULT_POLICY, device="cpu"):
    """The served decision policy, built exactly as gui/models_registry.py
    builds it (symmetrized margin net, score-conditional response rates)."""
    import torch

    from .decision_layer import EquityPolicy, MarginOracle, ResponseRates
    from .dmc_margin_train import load_margin_net

    spec = torch.load(path, map_location="cpu", weights_only=False)
    cond = bool(torch.load(spec["net"], map_location="cpu", weights_only=False)
                .get("config", {}).get("cond", False))
    oracle = MarginOracle(load_margin_net(spec["net"], device), device,
                          cond=cond, symmetrize=True)
    rates = None
    if "rates_p_meegaan" in spec:
        rates = ResponseRates(spec["rates_step"], spec["rates_p_meegaan"],
                              spec["rates_p_terug"])
    return EquityPolicy(oracle, spec["V"], spec["p_meegaan"], spec["p_terug"],
                        spec["enable_meegaan"], spec["enable_terug"],
                        spec.get("price_terug", spec["enable_terug"]),
                        rates=rates, symmetrize_rates=True)


def declare_deals(deals, policy, score=(0, 0)):
    """[(declaration, meegaan_seat, terug_seat)] per deal at a fixed match
    score (declaring team first). Defenders are asked from the dealer's
    left and the first yes takes it; the trump-makers answer dealer first,
    as in learn/match_harness.run_match."""
    us, them = score
    out = []
    for hands, dealer in deals:
        decl = policy.declare(hand_mask(hands[dealer]), us, them)
        mee = ter = None
        for off in (1, 3):
            seat = (dealer + off) % 4
            if policy.meegaan(hand_mask(hands[seat]), decl, off, them, us):
                mee = seat
                break
        if mee is not None:
            for off in (0, 2):
                seat = (dealer + off) % 4
                if policy.terug(hand_mask(hands[seat]), decl, off, us, them):
                    ter = seat
                    break
        out.append((int(decl), mee, ter))
    return out


class MullAware:
    """Two play agents, dispatched per deal on the declaration (the
    blueprint lineages are separate nets for suit and no-trump play)."""

    def __init__(self, trump_agent, mull_agent):
        self.trump, self.mull, self.active = trump_agent, mull_agent, None

    def new_game(self, game, dealer, player_id):
        self.active = self.mull if game.no_trump else self.trump
        self.active.new_game(game, dealer, player_id)

    def choose_card(self, game, player_id, played_cards, played_players):
        return self.active.choose_card(game, player_id, played_cards, played_players)

    def on_card_played(self, *a):
        if hasattr(self.active, "on_card_played"):
            self.active.on_card_played(*a)


def build_declared_agent(spec):
    """build_agent, plus {"mull": spec} for a separate no-trump agent."""
    if "mull" in spec:
        base = {k: v for k, v in spec.items() if k != "mull"}
        return MullAware(build_agent(base), build_agent(spec["mull"]))
    return build_agent(spec)


def play_declared(agents, hands, dealer, call, score=(0, 0)):
    """One deal: (raw team points, multiplier)."""
    decl, mee, ter = call
    game = ManilleGame(dealer=dealer)
    game.hands = [list(h) for h in hands]
    game.select_trump(Suit(decl) if decl < MULL else None)
    if mee is not None:
        assert game.announce_double(mee), "engine refused meegaan"
        if ter is not None:
            assert game.announce_redouble(ter), "engine refused terug"
    # who made the calls, and the score the calls were made at: the engine
    # keeps only the team-level flags, the agents' likelihoods want seats
    game.meegaan_seat, game.terug_seat = mee, ter
    game.decl_score = score
    game.start_playing()
    for p, a in enumerate(agents):
        a.new_game(game, dealer, p)
    pc, pp = [], []
    while game.phase == GamePhase.PLAYING:
        pid = game.current_player
        card = agents[pid].choose_card(game, pid, pc, pp)
        pre = list(game.current_trick)
        if not game.play_card(pid, card):
            raise RuntimeError(f"illegal card {card} for player {pid}")
        if len(pre) == 3:
            for tp, tc in pre + [(pid, card)]:
                pc.append(tc.to_index())
                pp.append(tp)
    return list(game.hand_points), game.multiplier


_W = {}


def share_loaded_models():
    """Load every checkpoint once per process. Each of the four seats
    builds its own agents, and a rule agent or mull-aware blueprint holds
    two or three ~150 MB GRU nets; unshared, 14 workers ran the machine out
    of memory (2026-11-12). The nets are inference-only, so sharing one
    module between agents is safe."""
    from . import dmc_model
    if getattr(dmc_model, "_shared_loads", False):
        return
    cache = {}

    def wrap(fn):
        def load(path, device="cpu", *a, **k):
            key = (fn.__name__, str(path), str(device), a, tuple(sorted(k.items())))
            if key not in cache:
                cache[key] = fn(path, device, *a, **k)
            return cache[key]
        return load
    dmc_model.load_q_model = wrap(dmc_model.load_q_model)
    dmc_model.load_belief_model = wrap(dmc_model.load_belief_model)
    dmc_model._shared_loads = True


def _init(spec0, spec1, deals, calls):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    share_loaded_models()
    _W.update(t0=[build_declared_agent(spec0), build_declared_agent(spec0)],
              t1=[build_declared_agent(spec1), build_declared_agent(spec1)],
              deals=deals, calls=calls)


def _deal(i):
    hands, dealer = _W["deals"][i]
    call = _W["calls"][i]
    t0, t1 = _W["t0"], _W["t1"]
    raw, mult = [], []
    for swapped in (False, True):
        agents = [t1[0], t0[0], t1[1], t0[1]] if swapped else [t0[0], t1[0], t0[1], t1[1]]
        pts, m = play_declared(agents, hands, dealer, call)
        diff = pts[1] - pts[0] if swapped else pts[0] - pts[1]
        raw.append(diff)
        # scored like the match: only the winning side's margin over 30 counts
        mult.append(np.sign(diff) * (abs(diff) / 2) * m)
    return i, float(np.mean(raw)), float(np.mean(mult))


def run_declared_match(spec0, spec1, n_deals, seed=0, processes=None,
                       policy_path=DEFAULT_POLICY, deals=None, calls=None):
    """-> (raw diffs [n], multiplied diffs [n], info). diffs are team-0
    (spec0) minus team-1 points per deal, mean of the two seatings."""
    deals = deals if deals is not None else make_deals(n_deals, seed)
    if calls is None:
        calls = declare_deals(deals, load_equity_policy(policy_path))
    processes = processes or max(1, (mp.cpu_count() or 2) - 1)
    raw = np.zeros(len(deals))
    mult = np.zeros(len(deals))
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes, initializer=_init,
                  initargs=(spec0, spec1, deals, calls)) as pool:
        for i, r, m in pool.imap_unordered(_deal, range(len(deals))):
            raw[i], mult[i] = r, m
    decl = np.array([c[0] for c in calls])
    info = dict(mull_rate=float(np.mean(decl == MULL)),
                meegaan_rate=float(np.mean([c[1] is not None for c in calls])),
                terug_rate=float(np.mean([c[2] is not None for c in calls])),
                calls=calls)
    return raw, mult, info


# --- late-match states: play for the match, not the margin ------------------

def match_states(n, seed, lo=60, hi=100):
    """Per-deal match scores (team 0, team 1) for seating A, uniform in
    [lo, hi] -- the end of a match, where points and match equity part."""
    rng = np.random.default_rng((seed, 0x5C))
    return [tuple(int(x) for x in rng.integers(lo, hi + 1, size=2)) for _ in range(n)]


def declare_deals_at(deals, states, policy):
    """Calls per deal at its own match score (scores belong to the seats'
    teams, so both seatings see the same calls)."""
    out = []
    for (hands, dealer), st in zip(deals, states):
        d_team = dealer % 2
        out.append(declare_deals([(hands, dealer)], policy,
                                 (st[d_team], st[1 - d_team]))[0])
    return out


def _deal_equity(i):
    """Both seatings of deal i at its match score. Returns spec0's mean
    P(match win) after the deal (it sits on team 0 in A, team 1 in B)."""
    hands, dealer = _W["deals"][i]
    call, st = _W["calls"][i], _W["states"][i]
    V = _W["V"]
    t0, t1 = _W["t0"], _W["t1"]
    eqs = []
    for swapped in (False, True):
        agents = [t1[0], t0[0], t1[1], t0[1]] if swapped else [t0[0], t1[0], t0[1], t1[1]]
        game = ManilleGame(dealer=dealer)
        game.team_scores[0].score, game.team_scores[1].score = st
        mine = 1 if swapped else 0
        pts, mult = _play_at(agents, hands, dealer, call, game)
        from .decision_layer import equity_vector
        us, them = st[mine], st[1 - mine]
        next_bit = 1 if (dealer + 1) % 2 == mine else 0
        eq = equity_vector(V, us, them, next_bit, mult)
        eqs.append(float(eq[pts[mine] - 30 + 30]))
    return i, float(np.mean(eqs))


def _play_at(agents, hands, dealer, call, game):
    decl, mee, ter = call
    game.hands = [list(h) for h in hands]
    game.select_trump(Suit(decl) if decl < MULL else None)
    if mee is not None:
        assert game.announce_double(mee)
        if ter is not None:
            assert game.announce_redouble(ter)
    game.meegaan_seat, game.terug_seat = mee, ter
    d_team = dealer % 2
    game.decl_score = (game.team_scores[d_team].score, game.team_scores[1 - d_team].score)
    game.start_playing()
    for p, a in enumerate(agents):
        a.new_game(game, dealer, p)
    pc, pp = [], []
    while game.phase == GamePhase.PLAYING:
        pid = game.current_player
        card = agents[pid].choose_card(game, pid, pc, pp)
        pre = list(game.current_trick)
        if not game.play_card(pid, card):
            raise RuntimeError(f"illegal card {card} for player {pid}")
        if len(pre) == 3:
            for tp, tc in pre + [(pid, card)]:
                pc.append(tc.to_index())
                pp.append(tp)
    return list(game.hand_points), game.multiplier


def _init_states(spec0, spec1, deals, calls, states, V):
    _init(spec0, spec1, deals, calls)
    _W.update(states=states, V=V)


def run_state_match(spec0, spec1, n_deals, seed=0, processes=None,
                    policy_path=DEFAULT_POLICY, deals=None, states=None, calls=None):
    """-> per-deal P(match win) of spec0 after the deal, averaged over the
    two seatings, at late-match scores. Pair two arms on the same deals and
    states and the difference is the match equity the play is worth."""
    import torch
    deals = deals if deals is not None else make_deals(n_deals, seed)
    states = states if states is not None else match_states(len(deals), seed)
    if calls is None:
        calls = declare_deals_at(deals, states, load_equity_policy(policy_path))
    V = np.asarray(torch.load(policy_path, map_location="cpu", weights_only=False)["V"])
    processes = processes or max(1, (mp.cpu_count() or 2) - 1)
    out = np.zeros(len(deals))
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes, initializer=_init_states,
                  initargs=(spec0, spec1, deals, calls, states, V)) as pool:
        for i, e in pool.imap_unordered(_deal_equity, range(len(deals))):
            out[i] = e
    return out

