"""Duplicate-deal evaluation harness with bootstrap CIs (plan §7).

Every matchup plays each deal twice with the teams swapped across seat pairs,
so deal luck cancels. Agents are constructed per worker process from
picklable specs; deals are generated from a fixed seed so matchups are
reproducible and comparable.

Usage:
    python -m learn.evaluate ordering --deals 200
    python -m learn.evaluate nn-vs-random --model ckpt.pt --deals 500
    python -m learn.evaluate nn-vs-ismcts --model ckpt.pt --iters 100 --deals 200
    python -m learn.evaluate belief-vs-uniform --model ckpt.pt --iters 150 --deals 1000
"""
import argparse
import multiprocessing as mp
import os
import time

import numpy as np

import manille_ismcts as mi
from manille_env.card import Card, Suit
from tournament import Agent, ISMCTSAgent, RandomAgent, run_game


# ---------------------------------------------------------------------------
# Deals
# ---------------------------------------------------------------------------

def make_deals(n, seed):
    rng = np.random.default_rng(seed)
    deals = []
    for i in range(n):
        perm = rng.permutation(32)
        hands = [
            [Card.from_index(int(c)) for c in perm[p * 8 : (p + 1) * 8]]
            for p in range(4)
        ]
        deals.append((hands, i % 4))
    return deals


def trump_for(hands, declarer):
    mask = 0
    for c in hands[declarer]:
        mask |= 1 << c.to_index()
    return Suit(int(mi.apply_trump_rule(mask)))


# ---------------------------------------------------------------------------
# Agent specs (picklable; built once per worker)
# ---------------------------------------------------------------------------

class _UniformRolloutBatchedAgent(Agent):
    """BatchedISMCTS with a uniform rollout policy and no belief particles —
    the solver-equivalence baseline for scripts/check_wave.py (review §1.3)."""

    def __init__(self, sims=150, wave=8):
        import manille_ismcts as mi
        from learn.encoding import LEGAL_MASK_OFF
        self._legal_off = LEGAL_MASK_OFF
        self.solver = mi.BatchedISMCTS(sims=sims, wave=wave, mode="rollout")
        self.dealer = 0

    def new_game(self, game, dealer, player_id):
        self.dealer = dealer

    def _uniform_fn(self, obs):
        import numpy as np
        legal = obs[:, self._legal_off : self._legal_off + 32]
        probs = (legal / legal.sum(axis=1, keepdims=True)).astype(np.float32)
        return probs, np.zeros(obs.shape[0], dtype=np.float32)

    def choose_card(self, game, player_id, played_cards, played_players):
        from tournament import game_to_cpp_arrays
        arrays = game_to_cpp_arrays(game, played_cards, played_players)
        r = self.solver.solve(*arrays, self.dealer,
                              policy_value_fn=self._uniform_fn)
        best = int(r["best_move"])
        for c in game.get_legal_moves(player_id):
            if c.to_index() == best:
                return c
        return game.get_legal_moves(player_id)[0]


def build_agent(spec):
    if spec["type"] == "random":
        return RandomAgent()
    if spec["type"] == "rule":
        from .rule_agent import RuleAgent
        return RuleAgent(**spec.get("params", {}))
    if spec["type"] == "uniform_batched":
        return _UniformRolloutBatchedAgent(spec.get("sims", 150),
                                           spec.get("wave", 8))
    if spec["type"] == "ismcts":
        return ISMCTSAgent("SO", spec["iters"])
    if spec["type"] == "ismcts_pi":
        # Perfect-information (cheating) bound for 0b — eval-only.
        return ISMCTSAgent("SO", spec["iters"], perfect_info=True)
    if spec["type"] == "ismcts_pt":
        # Partner-transparent bound for 0b.2: partner's hand known,
        # opponents randomized — ceiling of any convention system.
        return ISMCTSAgent("SO", spec["iters"], partner_transparent=True)
    if spec["type"] == "ismcts_time":
        # Root-parallel plain ISMCTS, wall-clock budget per move (Phase B
        # champion). B1 benchmark definition: hard per-move budget on the
        # full machine, single game, forced moves effectively exempt.
        from tournament import TimeBoxedISMCTSAgent
        return TimeBoxedISMCTSAgent(
            budget_ms=spec.get("budget_ms", 500),
            threads=spec.get("threads", 15),
        )
    if spec["type"] == "garnish":
        # C5 root-garnish hybrid: root-parallel plain + one jit policy call
        # blended into the root choice (visits * prior^lam).
        from tournament import GarnishedRootAgent
        return GarnishedRootAgent(
            spec["jit"],
            budget_ms=spec.get("budget_ms", 500),
            threads=spec.get("threads", 15),
            lam=spec.get("lam", 0.5),
            device="cuda" if spec.get("cuda", True) else "cpu",
        )
    if spec["type"] == "fastnn":
        from .agents import FastNNAgent
        from .model import load_model
        device = "cuda" if spec.get("cuda", False) else "cpu"
        return FastNNAgent(load_model(spec["model"], device), device=device,
                           temperature=spec.get("temperature", 0.0))
    if spec["type"] == "dmc_q":
        # DMC blueprint: action-as-input Q, argmax over legal candidates.
        # Optional "choice_model": free-trump convention (Phase 6) — when
        # this agent holds the dealer seat, run_match asks it for trump
        # instead of applying the rule.
        from .dmc_agent import QNetAgent
        from .dmc_model import load_q_model
        device = "cuda" if spec.get("cuda", True) else "cpu"
        choice_model = None
        if spec.get("choice_model"):
            from .dmc_trump_choice import load_choice_model
            choice_model = load_choice_model(spec["choice_model"], device)
        return QNetAgent(
            load_q_model(spec["model"], device),
            device=device,
            epsilon=spec.get("epsilon", 0.0),
            choice_model=choice_model,
        )
    if spec["type"] == "dmc_q_sym":
        # Same blueprint, with the suit symmetry averaged in at test time
        # over the relabellings the trump RULE keeps consistent: 6 for
        # diamonds/spades, 2 for hearts/clubs. No retraining.
        from .dmc_agent import SymmetrizedQNetAgent
        from .dmc_model import load_q_model
        device = "cuda" if spec.get("cuda", True) else "cpu"
        return SymmetrizedQNetAgent(
            load_q_model(spec["model"], device),
            device=device,
            epsilon=spec.get("epsilon", 0.0),
            full_orbit=spec.get("full_orbit", False),
            declarer_aware=spec.get("declarer_aware", False),
            aggregation=spec.get("aggregation", "mean"),
        )
    if spec["type"] == "dmc_iter":
        # Policy-iteration policy (plan §5.1): "policy" is "actor" (a
        # dmc_iteration_actor checkpoint) or "raw_q" (the round-0 adapter
        # over a Q checkpoint). CPU unless cuda is requested explicitly —
        # the deployment contract is one actor on CPU.
        from .dmc_iteration_policy import IterationPolicyAgent, load_policy
        device = "cuda" if spec.get("cuda", False) else "cpu"
        return IterationPolicyAgent(load_policy(
            {"kind": spec.get("policy", "actor"), "path": spec["model"]},
            device))
    if spec["type"] == "dmc_pi":
        from .dmc_agent import PolicyHeadAgent
        from .dmc_policy_head import load_policy_checkpoint
        device = "cuda" if spec.get("cuda", True) else "cpu"
        q_model, head, tau, _ = load_policy_checkpoint(
            spec["model"], device=device, base_model=spec.get("base_model"))
        return PolicyHeadAgent(
            q_model, head, tau, device=device,
            epsilon=spec.get("epsilon", 0.0))
    if spec["type"] == "dmc_hybrid":
        # Endgame hybrid: blueprint until trick `threshold`, then
        # belief-weighted exact minimax (plan §5).
        from .dmc_hybrid import EndgameHybridAgent
        from .dmc_model import load_belief_model, load_q_model
        device = "cuda" if spec.get("cuda", True) else "cpu"
        return EndgameHybridAgent(
            load_q_model(spec["model"], device),
            load_belief_model(spec["belief"], device),
            device=device,
            threshold=spec.get("threshold", 6),
            n_particles=spec.get("particles", 128),
        )
    if spec["type"] == "dmc_sparta":
        # Tier-2 SPARTA on the DMC blueprint (plan §5): 1-ply search over
        # Q-likelihood-weighted worlds, deviation threshold. Optional
        # "belief": mix B_φ marginal log-weights into the world weighting
        # (lam_belief; pure B_φ worlds at lam=0). Optional "choice_model"
        # (Phase 6 free trump): this agent chooses trump on the dealer seat,
        # and worlds are sampled unconstrained + tilted by the chooser's
        # trump likelihood (choice_temp) instead of the hard rule constraint.
        from .dmc_model import load_belief_model, load_q_model
        from .dmc_sparta import SpartaDMCAgent
        device = "cuda" if spec.get("cuda", True) else "cpu"
        belief_model = (load_belief_model(spec["belief"], device)
                        if spec.get("belief") else None)
        choice_model = None
        if spec.get("choice_model"):
            from .dmc_trump_choice import load_choice_model
            choice_model = load_choice_model(spec["choice_model"], device)
        # M6.3: "decl_tilt" = equity-chooser world tilt (policy file from
        # m3_fixed_point --policy-out), "decl_temp" its softmax temperature.
        # Tilt only -- declarations stay with the rule.
        decl_tilt_model = None
        if spec.get("decl_tilt"):
            from .decision_layer import EquityChoiceModel
            decl_tilt_model = EquityChoiceModel(
                spec["decl_tilt"], device,
                temp=spec.get("decl_temp", 0.01))
        return SpartaDMCAgent(
            load_q_model(spec["model"], device),
            device=device,
            n_particles=spec.get("particles", 64),
            delta=spec.get("delta", 0.6),
            tau_likelihood=spec.get("tau_likelihood", 0.2),
            temperature=spec.get("temperature", 0.0),
            lam=spec.get("lam", 1.0),
            eps=spec.get("eps", 0.0),
            exact_leaves=spec.get("exact_leaves", True),
            exact_trick=spec.get("exact_trick", 6),
            ismcts_trick=spec.get("ismcts_trick"),
            ismcts_iters=spec.get("ismcts_iters", 1000),
            alloc=spec.get("alloc", "fixed"),
            ucb_chunk=spec.get("ucb_chunk", 8),
            paired_pruning=spec.get("paired_pruning", True),
            belief_model=belief_model,
            lam_belief=spec.get("lam_belief", 1.0 if belief_model else 0.0),
            proposal_oversample=spec.get("proposal_oversample", 0),
            deviate_rule=spec.get("deviate_rule", "delta"),
            conf_c=spec.get("conf_c", 1.0),
            tier3=spec.get("tier3", False),
            t3_eps=spec.get("t3_eps", 1.0),
            t3_arms=spec.get("t3_arms", 2),
            t3_worlds=spec.get("t3_worlds", 48),
            choice_model=choice_model,
            choice_temp=spec.get("choice_temp", 0.1),
            decl_tilt_model=decl_tilt_model,
        )
    if spec["type"] == "dmc_sparta_qleaf":
        # Depth-limited SPARTA probe (learn/dmc_sparta_qleaf.py): Q-net leaf
        # after the current trick instead of a full blueprint rollout.
        from .dmc_model import load_q_model
        from .dmc_sparta_qleaf import SpartaQLeafAgent
        device = "cuda" if spec.get("cuda", True) else "cpu"
        return SpartaQLeafAgent(
            load_q_model(spec["model"], device), device=device,
            n_particles=spec.get("particles", 128),
            delta=spec.get("delta", 1.0),
            tau_likelihood=spec.get("tau_likelihood", 0.1),
            exact_leaves=spec.get("exact_leaves", True),
            exact_trick=spec.get("exact_trick", 6),
            alloc=spec.get("alloc", "ucb"),
            deviate_rule=spec.get("deviate_rule", "conf"),
            conf_c=spec.get("conf_c", 1.0),
            leaf_depth=spec.get("leaf_depth", 0),
        )
    if spec["type"] == "dmc_sparta_ismcts_handoff":
        # SPARTA until trick handoff_trick, then C++ ISMCTS with SPARTA's
        # belief-weighted particle pool injected as determinizations
        # (endgame_leak_plan.md arm C — the info-set-correct endgame).
        from .dmc_endgame_handoff import SpartaIsmctsHandoffAgent
        from .dmc_model import load_q_model
        device = "cuda" if spec.get("cuda", True) else "cpu"
        return SpartaIsmctsHandoffAgent(
            load_q_model(spec["model"], device),
            device=device,
            n_particles=spec.get("particles", 128),
            delta=spec.get("delta", 1.0),
            tau_likelihood=spec.get("tau_likelihood", 0.1),
            temperature=spec.get("temperature", 0.0),
            lam=spec.get("lam", 1.0),
            eps=spec.get("eps", 0.0),
            exact_leaves=spec.get("exact_leaves", True),
            exact_trick=spec.get("exact_trick", 6),
            ismcts_trick=spec.get("ismcts_trick"),
            ismcts_iters=spec.get("ismcts_iters", 1000),
            alloc=spec.get("alloc", "ucb"),
            ucb_chunk=spec.get("ucb_chunk", 8),
            handoff_trick=spec.get("handoff_trick", 6),
            ismcts_iterations=spec.get("iters", 5000),
        )
    if spec["type"] == "belief_ismcts":
        from .agents import BeliefGuidedISMCTSAgent
        from .model import load_model
        device = "cuda" if spec.get("cuda", False) else "cpu"
        return BeliefGuidedISMCTSAgent(
            load_model(spec["model"], device),
            iterations=spec["iters"],
            n_particles=spec.get("particles", 128),
            device=device,
        )
    if spec["type"] == "batched_ismcts":
        from .agents import BatchedNNAgent
        from .model import load_model
        device = "cuda" if spec.get("cuda", False) else "cpu"
        return BatchedNNAgent(
            load_model(spec["model"], device),
            sims=spec.get("sims", 150),
            wave=spec.get("wave", 8),
            mode=spec.get("mode", "rollout"),
            n_particles=spec.get("particles", 128),
            device=device,
            temperature=spec.get("temperature", 1.0),
            puct=spec.get("puct", False),
            c_puct=spec.get("c_puct", 2.0),
            dirichlet_epsilon=spec.get("dirichlet_epsilon", 0.0),
            jit_path=spec.get("jit"),
            partner_belief=spec.get("partner_belief", False),
        )
    if spec["type"] == "plpw_ismcts":
        from .agents import PLPWGuidedISMCTSAgent
        from .model import load_model
        device = "cuda" if spec.get("cuda", False) else "cpu"
        return PLPWGuidedISMCTSAgent(
            load_model(spec["model"], device),
            iterations=spec["iters"],
            n_particles=spec.get("particles", 64),
            device=device,
            lam=spec.get("lam", 1.0),
            eps=spec.get("eps", 0.0),
        )
    if spec["type"] == "sparta":
        from .sparta import SpartaAgent
        from .model import load_model
        device = "cuda" if spec.get("cuda", False) else "cpu"
        return SpartaAgent(
            load_model(spec["model"], device),
            n_particles=spec.get("particles", 64),
            device=device,
            lam=spec.get("lam", 1.0),
            eps=spec.get("eps", 0.0),
            delta=spec.get("delta", 0.6),
            temperature=spec.get("temperature", 1.0),
        )
    raise ValueError(f"unknown agent spec {spec}")


# ---------------------------------------------------------------------------
# Match runner (multiprocess over deals)
# ---------------------------------------------------------------------------

_DEALS = None
_TEAM0 = None
_TEAM1 = None
_MULL = False


def _build_team(spec):
    """A team is two agents. A single spec runs the same policy in both
    seats; a (spec_a, spec_b) pair puts a different policy in each, which
    is what cross-play needs. Member 0 and member 1 stay partners across
    the duplicate swap (see _play_deal), so a mixed pairing is stable."""
    if isinstance(spec, (list, tuple)):
        a, b = spec
        return [build_agent(a), build_agent(b)]
    return [build_agent(spec), build_agent(spec)]


def _worker_init(deals, spec0, spec1, mull=False):
    global _DEALS, _TEAM0, _TEAM1, _MULL
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    _DEALS = deals
    _TEAM0 = _build_team(spec0)
    _TEAM1 = _build_team(spec1)
    _MULL = mull


def trump_for_seat(agents, hands, declarer):
    """The trump for one game: the declarer-seat agent's own convention.
    Agents carrying a choice model (Phase 6 free trump) choose; everyone
    else gets the deterministic rule. In duplicate play the dealer seat
    changes teams between the two replays, so each side plays the deal
    under its own convention — that is the G-R2 comparison."""
    ag = agents[declarer]
    if getattr(ag, "choice_model", None) is not None:
        mask = 0
        for c in hands[declarer]:
            mask |= 1 << c.to_index()
        return Suit(ag.choose_trump_suit(mask))
    return trump_for(hands, declarer)


# The determinization samplers are the ONLY component whose failure is a
# property of the deal rather than of the run: the trump-constrained sampler
# exhausts its rejection budget stochastically, and the uniform sampler can
# walk into an infeasible branch. Those deals are legitimately retried and,
# if they persist, dropped.
#
# Nothing else is. Catching bare RuntimeError meant CUDA OOM, illegal moves,
# model-loading failures and ordinary bugs were all silently retried and then
# dropped as "sampler skips" -- which is exactly how G-R3 2026-07-24 produced
# a match with 293/400 deals dropped that still reported a number. Worse, the
# dropping is state-dependent: whatever makes a deal fail may correlate with
# how it would have scored, so a biased subset survives. Everything outside
# this list now aborts the match.
_SAMPLER_ERRORS = (
    "exhausted rejection budget",
    "no feasible hidden-card assignment",
    "no eligible hidden hand",
    "infeasible branch while sampling",
    "failed to select a sampling branch",
    "determinization failed",
    "Determinization did not return ManilleGame",
)


def is_sampler_error(msg):
    """True when this RuntimeError is a determinization failure on this deal
    (retryable, then droppable) rather than a failure of the run."""
    return any(m in msg for m in _SAMPLER_ERRORS)


def _play_deal_both(idx):
    """Both swapped plays of deal idx: ("ok", idx, [diff, diff], [pts,
    pts]) with team-0 minus team-1 points per play, or ("skip", idx, msg)
    after persistent determinization failures."""
    hands, declarer = _DEALS[idx]
    diffs = []
    pts_t0 = []
    for swapped in (False, True):
        if swapped:
            agents = [_TEAM1[0], _TEAM0[0], _TEAM1[1], _TEAM0[1]]
        else:
            agents = [_TEAM0[0], _TEAM1[0], _TEAM0[1], _TEAM1[1]]
        trump = None if _MULL else trump_for_seat(agents, hands, declarer)
        # Retry only determinization failures (see _SAMPLER_ERRORS); every
        # other RuntimeError propagates and kills the match, because a
        # silently dropped deal is indistinguishable from a scored one in
        # the final number and the dropping is not deal-independent.
        result = None
        last_err = ""
        for _ in range(3):
            try:
                result = run_game(agents, dealer=declarer, hands=hands,
                                  trump=trump, mull=_MULL)
                break
            except RuntimeError as e:
                last_err = str(e).split("\n")[0][:120]
                if not is_sampler_error(str(e)):
                    raise
                continue
        if result is None:
            return ("skip", idx, last_err)
        t0_pts = result.pts[1] if swapped else result.pts[0]
        t1_pts = result.pts[0] if swapped else result.pts[1]
        diffs.append(t0_pts - t1_pts)
        pts_t0.append(t0_pts)
    return ("ok", idx, diffs, pts_t0)


def _play_deal(idx):
    r = _play_deal_both(idx)
    if r[0] == "skip":
        return ("skip", r[2])
    return float(np.mean(r[2])), float(np.mean(r[3]))


def deal_fingerprint(hands, declarer):
    """Identity of an evaluation deal's starting position (hands in seat
    order + declarer), for pairing matches by content, not by position."""
    import hashlib

    h = hashlib.sha256()
    for hand in hands:
        h.update(bytes(sorted(c.to_index() for c in hand)))
        h.update(b"|")
    h.update(bytes([declarer]))
    return h.hexdigest()


def run_match_detailed(spec0, spec1, n_deals, seed=0, processes=None,
                       progress_every=0, mull=False, log=print):
    """run_match with per-deal provenance, for gates that must pair by
    deal identity and fail closed on skips.

    Returns a dict: deal_ids (index within the seed's deal stream),
    deal_fingerprints, plays [n, 2] (team-0 minus team-1 points for the
    unswapped and swapped play), diffs (their mean — exactly run_match's
    convention), t0_points, skipped [(id, msg)], complete, seed, elapsed.
    Skipped deals are NOT silently dropped from the accounting: `complete`
    is False and the caller decides (gates must refuse)."""
    if processes is None:
        processes = max(1, (mp.cpu_count() or 2) - 1)
    deals = make_deals(n_deals, seed)
    ctx = mp.get_context("spawn")
    t_start = time.time()
    rows = {}
    skipped = []
    with ctx.Pool(processes, initializer=_worker_init,
                  initargs=(deals, spec0, spec1, mull)) as pool:
        for i, r in enumerate(pool.imap(_play_deal_both, range(n_deals),
                                        chunksize=4)):
            if r[0] == "skip":
                skipped.append((int(r[1]), r[2]))
            else:
                rows[int(r[1])] = (r[2], r[3])
            if progress_every and (i + 1) % progress_every == 0 and rows:
                done = np.array([np.mean(v[0]) for v in rows.values()])
                m, lo, hi = bootstrap_ci(done)
                el = time.time() - t_start
                eta = el / (i + 1) * (n_deals - i - 1) / 60.0
                log(f"    [{i + 1}/{n_deals}] running {m:+.2f} "
                    f"[{lo:+.2f}, {hi:+.2f}]  {el:.0f}s, eta {eta:.0f} min")
    ids = sorted(rows)
    plays = np.array([rows[i][0] for i in ids], dtype=np.float64).reshape(
        -1, 2)
    pts = np.array([rows[i][1] for i in ids], dtype=np.float64).reshape(-1, 2)
    return {
        "seed": int(seed),
        "n_requested": int(n_deals),
        "deal_ids": np.array(ids, dtype=np.int64),
        "deal_fingerprints": [deal_fingerprint(*deals[i]) for i in ids],
        "plays": plays,
        "diffs": plays.mean(axis=1),
        "t0_points": pts.mean(axis=1),
        "skipped": skipped,
        "complete": not skipped and len(ids) == n_deals,
        "elapsed": time.time() - t_start,
        "mull": bool(mull),
    }


def run_match(spec0, spec1, n_deals, seed=0, processes=None,
              progress_every=0, mull=False):
    """Play `n_deals` duplicate deals; returns (diffs, t0_points) arrays.

    diffs[i] = mean team0 minus team1 points over the two swapped plays of
    deal i; t0_points[i] = mean team0 points on deal i. Results come back
    in deal order (ordered imap), so per-deal pairing across matches run
    with the same seed is valid when no deals were skipped (skips are
    dropped — verify lengths before pairing). progress_every > 0 prints a
    running mean diff every N completed deals. mull=True plays every deal
    at no trump (the unconditional mull lineage, Phase B).
    """
    if processes is None:
        processes = max(1, (mp.cpu_count() or 2) - 1)
    deals = make_deals(n_deals, seed)
    ctx = mp.get_context("spawn")
    t_start = time.time()
    results = []
    with ctx.Pool(processes, initializer=_worker_init,
                  initargs=(deals, spec0, spec1, mull)) as pool:
        for i, r in enumerate(pool.imap(_play_deal, range(n_deals),
                                        chunksize=4)):
            results.append(r)
            if progress_every and (i + 1) % progress_every == 0:
                done = [x[0] for x in results
                        if not (isinstance(x, tuple) and x[0] == "skip")]
                if done:
                    # Running CI, not just the mean: a long gate is worth
                    # watching for a trend, and the mean alone cannot say
                    # whether it has separated from zero yet.
                    m, lo, hi = bootstrap_ci(np.asarray(done, dtype=float))
                    el = time.time() - t_start
                    eta = el / (i + 1) * (n_deals - i - 1) / 60.0
                    print(f"    [{i + 1}/{n_deals}] running {m:+.2f} "
                          f"[{lo:+.2f}, {hi:+.2f}]  {el:.0f}s, "
                          f"eta {eta:.0f} min", flush=True)
    elapsed = time.time() - t_start
    skips = [r[1] for r in results if isinstance(r, tuple) and r[0] == "skip"]
    if skips:
        from collections import Counter
        print(f"  ({len(skips)} deals skipped; error breakdown:")
        for msg, n in Counter(skips).most_common(5):
            print(f"    {n}x {msg or '<no message>'}")
        print("  )")
    results = [r for r in results
               if r is not None and not (isinstance(r, tuple)
                                         and r[0] == "skip")]
    diffs = np.array([r[0] for r in results])
    t0_points = np.array([r[1] for r in results])
    return diffs, t0_points, elapsed


def bootstrap_ci(values, n_boot=10000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def report(name, diffs, t0_points, elapsed):
    mean, lo, hi = bootstrap_ci(diffs)
    print(f"{name}")
    print(f"  diff (team0 - team1): {mean:+.2f} pts/deal  95% CI [{lo:+.2f}, {hi:+.2f}]"
          f"  ({len(diffs)} deals, {elapsed:.0f}s)")
    print(f"  team0 points: {t0_points.mean():.2f} pts/deal")
    return mean, lo, hi


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["ordering", "nn-vs-random", "nn-vs-ismcts",
                                         "fastnn-vs-fastnn", "pi-vs-ismcts",
                                         "pt-vs-ismcts",
                                         "belief-vs-uniform", "plpw-vs-uniform",
                                         "plpw-vs-belief",
                                         "sparta-vs-ismcts", "sparta-vs-random",
                                         "actor-vs-ismcts",
                                         "actor-vs-actor"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model2", type=str, default=None)
    parser.add_argument("--deals", type=int, default=200)
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument("--particles", type=int, default=128)
    parser.add_argument("--sims", type=int, default=150)
    parser.add_argument("--wave", type=int, default=8)
    parser.add_argument("--leaf", choices=["rollout", "value"], default="rollout")
    parser.add_argument("--puct", action="store_true")
    parser.add_argument("--lam", type=float, default=1.0,
                        help="PLPW likelihood temperature (plpw-vs-uniform)")
    parser.add_argument("--eps", type=float, default=0.0,
                        help="PLPW uniform weight floor (plpw-vs-uniform)")
    parser.add_argument("--delta", type=float, default=0.6,
                        help="SPARTA deviation threshold in points "
                             "(sparta-vs-ismcts)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="blueprint rollout temperature; 0 = argmax "
                             "(sparta-vs-ismcts)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--processes", type=int, default=None)
    args = parser.parse_args()

    kw = dict(n_deals=args.deals, seed=args.seed, processes=args.processes)

    if args.mode == "ordering":
        d, p, e = run_match({"type": "ismcts", "iters": 1000},
                            {"type": "ismcts", "iters": 100}, **kw)
        report("ISMCTS-1000 vs ISMCTS-100", d, p, e)
        d, p, e = run_match({"type": "ismcts", "iters": 100},
                            {"type": "random"}, **kw)
        report("ISMCTS-100 vs random", d, p, e)
    elif args.mode == "nn-vs-random":
        d, p, e = run_match({"type": "fastnn", "model": args.model},
                            {"type": "random"}, **kw)
        report(f"FastNN({args.model}) vs random", d, p, e)
    elif args.mode == "nn-vs-ismcts":
        d, p, e = run_match({"type": "fastnn", "model": args.model},
                            {"type": "ismcts", "iters": args.iters}, **kw)
        report(f"FastNN({args.model}) vs ISMCTS-{args.iters}", d, p, e)
    elif args.mode == "fastnn-vs-fastnn":
        d, p, e = run_match({"type": "fastnn", "model": args.model},
                            {"type": "fastnn", "model": args.model2}, **kw)
        report(f"FastNN({args.model}) vs FastNN({args.model2})", d, p, e)
    elif args.mode == "pi-vs-ismcts":
        # 0b.1: cheating solver vs normal — bounds all hidden-info value.
        d, p, e = run_match({"type": "ismcts_pi", "iters": args.iters},
                            {"type": "ismcts", "iters": args.iters}, **kw)
        report(f"PerfectInfo-ISMCTS-{args.iters} vs ISMCTS-{args.iters}", d, p, e)
    elif args.mode == "pt-vs-ismcts":
        # 0b.2: partner-transparent pair vs normal — signaling ceiling.
        d, p, e = run_match({"type": "ismcts_pt", "iters": args.iters},
                            {"type": "ismcts", "iters": args.iters}, **kw)
        report(f"PartnerTransparent-ISMCTS-{args.iters} vs ISMCTS-{args.iters}",
               d, p, e)
    elif args.mode == "belief-vs-uniform":
        d, p, e = run_match(
            {"type": "belief_ismcts", "model": args.model,
             "iters": args.iters, "particles": args.particles},
            {"type": "ismcts", "iters": args.iters}, **kw)
        report(f"BeliefISMCTS-{args.iters} vs uniform ISMCTS-{args.iters}", d, p, e)
    elif args.mode == "plpw-vs-uniform":
        d, p, e = run_match(
            {"type": "plpw_ismcts", "model": args.model,
             "iters": args.iters, "particles": args.particles,
             "lam": args.lam, "eps": args.eps},
            {"type": "ismcts", "iters": args.iters}, **kw)
        report(f"PLPW-ISMCTS-{args.iters} (lam={args.lam}, eps={args.eps}) "
               f"vs uniform ISMCTS-{args.iters}", d, p, e)
    elif args.mode == "plpw-vs-belief":
        # E3 head-to-head: computed inverse vs learned B_phi, matched budget.
        # --model = PLPW blueprint (best-policy ckpt), --model2 = B_phi ckpt.
        d, p, e = run_match(
            {"type": "plpw_ismcts", "model": args.model,
             "iters": args.iters, "particles": args.particles,
             "lam": args.lam, "eps": args.eps},
            {"type": "belief_ismcts", "model": args.model2,
             "iters": args.iters, "particles": args.particles}, **kw)
        report(f"PLPW({args.model}) vs B_phi({args.model2}), "
               f"ISMCTS-{args.iters} matched", d, p, e)
    elif args.mode == "sparta-vs-ismcts":
        d, p, e = run_match(
            {"type": "sparta", "model": args.model,
             "particles": args.particles, "lam": args.lam, "eps": args.eps,
             "delta": args.delta, "temperature": args.temperature},
            {"type": "ismcts", "iters": args.iters}, **kw)
        report(f"SPARTA(particles={args.particles}, delta={args.delta}, "
               f"T={args.temperature}) vs ISMCTS-{args.iters}", d, p, e)
    elif args.mode == "sparta-vs-random":
        d, p, e = run_match(
            {"type": "sparta", "model": args.model,
             "particles": args.particles, "lam": args.lam, "eps": args.eps,
             "delta": args.delta, "temperature": args.temperature},
            {"type": "random"}, **kw)
        report(f"SPARTA(particles={args.particles}, delta={args.delta}, "
               f"T={args.temperature}) vs random", d, p, e)
    elif args.mode == "actor-vs-ismcts":
        d, p, e = run_match(
            {"type": "batched_ismcts", "model": args.model, "sims": args.sims,
             "wave": args.wave, "mode": args.leaf, "particles": args.particles,
             "puct": args.puct},
            {"type": "ismcts", "iters": args.iters}, **kw)
        report(f"Actor({args.leaf}{' puct' if args.puct else ''}, sims={args.sims}) vs ISMCTS-{args.iters}", d, p, e)
    elif args.mode == "actor-vs-actor":
        d, p, e = run_match(
            {"type": "batched_ismcts", "model": args.model, "sims": args.sims,
             "wave": args.wave, "mode": args.leaf, "particles": args.particles},
            {"type": "batched_ismcts", "model": args.model2, "sims": args.sims,
             "wave": args.wave, "mode": args.leaf, "particles": args.particles},
            **kw)
        report(f"Actor({args.model}) vs Actor({args.model2})", d, p, e)


if __name__ == "__main__":
    main()
