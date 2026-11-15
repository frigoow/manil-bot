"""Tier-2 SPARTA on the DMC blueprint (manille_dmc_plan §5).

Same design as learn/sparta.py — 1-ply search over belief-filtered worlds
with all seats following the blueprint, a deviation threshold making the
noisy Q-estimate safe to act on — but the blueprint is the DMC Q-net, and
the world weights are the blueprint's own action likelihoods: "would the
blueprint have played what I observed" (softmax over Q of the legal
candidates, the Q-analogue of PLPW).

Tier-3 (optional, tier3=True): selective paranoid 2-ply. When the top-2
arms are within t3_eps points, the close arms are re-evaluated with an
adversarial response layer: after the candidate, each top-weight world
advances (forced moves auto-played) to the next non-forced decision; an
opponent to move branches over ALL their legal responses and the world
takes the MIN (opponents like ISMCTS-5000 do not play the blueprint).
Own-team response nodes are NOT branched — they stay info-state-consistent
blueprint rollouts (per-world max over partner responses would
overestimate coordination: PFM bias). End/start-of-trick split stats are
kept on the agent (t3_stats) for the mandatory gate diagnostic.

    spec: {"type": "dmc_sparta", "model": q_ckpt, "particles": 64,
           "delta": 0.6, "tau_likelihood": 0.2, "temperature": 0.0}

Optional: pass belief_model (the DMC BeliefNet) and lam_belief > 0 to mix
B_φ marginal log-weights into the world weighting alongside (or instead
of, lam=0) the Q-likelihood — the Phase-3 B_φ-worlds variant.
"""
import copy
from collections import defaultdict

import numpy as np
import torch
import zlib

import manille_ismcts as mi
from manille_env.card import Card
from manille_env.game_logic import GamePhase
from tournament import Agent, game_to_cpp_arrays

from .dmc_agent import QNetAgent
from .dmc_hybrid import dmc_belief_log_weights
from .dmc_model import QGRU, QTransformer
from .dmc_tokens import tokens_batch
from .dmc_trump_choice import choice_log_likelihood, suit_values
from .inference import initial_hands, legal_completion, normalize_weights
from .sparta import inject_world


def weighted_paired_stats(v_a, v_b, w, ess_variance=False):
    """(mean, se, n) of the PER-WORLD difference v_a - v_b under weights w.

    Both arms are evaluated on the same worlds, so the difference is paired
    and the across-world variance -- which dwarfs the effect -- cancels.
    Comparing two independent weighted means instead throws that away.

    `se` is the delta-method standard error of a self-normalized weighted
    mean:

        se^2 = sum(w_i^2 (d_i - mean)^2) / (sum w_i)^2

    NOT weighted-variance / ESS. Those agree only when the weights are
    uncorrelated with the squared deviations, and here they are correlated
    by construction: PLPW gives its largest weights to the worlds the
    blueprint finds most likely, which are not the worlds where the arms
    disagree most. `ess_variance=True` reproduces the older ESS form, which
    the adopted deviation rule (A1, conf c=1.0) was gated with -- changing
    that rule silently would move the champion without a match behind it.
    """
    k = min(len(v_a), len(v_b))
    if k < 2:
        return None, None, k
    d = np.asarray(v_a[:k], dtype=np.float64) - np.asarray(v_b[:k],
                                                           dtype=np.float64)
    w = np.asarray(w[:k], dtype=np.float64)
    sw = w.sum()
    if sw <= 1e-12:
        return None, None, k
    m = float((w * d).sum() / sw)
    if ess_variance:
        var = float((w * (d - m) ** 2).sum() / sw)
        n_eff = float(sw * sw / max((w * w).sum(), 1e-12))
        return m, float(np.sqrt(max(var, 0.0) / max(n_eff, 1.0))), k
    se2 = float((w * w * (d - m) ** 2).sum() / (sw * sw))
    return m, float(np.sqrt(max(se2, 0.0))), k


def q_eval_batch(q_model, device, obs_rows, act_rows, token_meta=None,
                 counts=None, chunk_rows=512, policy_head=None,
                 policy_tau=1.0, policy_rows=None):
    """Batched Q forward, arch-aware. MLP: q(obs, act) with obs/act rows
    already expanded per candidate. Token models (QTransformer/QGRU):
    token_meta = list of (n_plays, own_mask) per UNIQUE obs row, and
    counts = candidates per unique row. QGRU's trunk is candidate-
    independent, so it runs once per unique row and only the pooled
    embedding is expanded; QTransformer keeps the candidate token inside
    the attention and needs the full per-candidate token expansion.

    chunk_rows bounds the unique rows per GRU forward: the trunk's
    activations are ~rows × 31 × d_model floats, and unbounded PLPW
    batches (~5k rows ≈ 350 MB) OOM'd the shared 12 GB card at 10 match
    workers (G-R3 2026-07-24). Chunking caps the transient at ~35 MB."""
    if policy_head is not None and not isinstance(q_model, QGRU):
        raise ValueError("residual rollout policy requires QGRU")
    with torch.no_grad():
        if isinstance(q_model, QGRU):
            assert token_meta is not None and len(token_meta) == len(obs_rows)
            n_plays = [m[0] for m in token_meta]
            offs = (np.concatenate([[0], np.cumsum(counts)])
                    if counts is not None else None)
            qs = []
            for lo in range(0, len(obs_rows), chunk_rows):
                hi = min(lo + chunk_rows, len(obs_rows))
                raw, n_raw, glob, _ = tokens_batch(obs_rows[lo:hi],
                                                   n_plays[lo:hi])
                last = q_model.trunk(torch.from_numpy(raw).to(device),
                                     torch.from_numpy(n_raw).to(device))
                glob_t = torch.from_numpy(glob).to(device)
                if counts is not None:
                    reps = torch.as_tensor(counts[lo:hi], device=device)
                    last = torch.repeat_interleave(last, reps, dim=0)
                    glob_t = torch.repeat_interleave(glob_t, reps, dim=0)
                    act_t = torch.from_numpy(
                        act_rows[offs[lo]:offs[hi]]).to(device)
                else:
                    act_t = torch.from_numpy(act_rows[lo:hi]).to(device)
                q_chunk = q_model.head(last, glob_t, act_t)
                if policy_head is not None:
                    if policy_rows is None:
                        use_policy = torch.ones(
                            len(q_chunk), dtype=torch.bool, device=device)
                    else:
                        row_mask = torch.as_tensor(
                            policy_rows[lo:hi], dtype=torch.bool,
                            device=device)
                        use_policy = (torch.repeat_interleave(row_mask, reps)
                                      if counts is not None else row_mask)
                    logits = policy_head.logits(
                        q_chunk, last, glob_t, act_t, policy_tau)
                    q_chunk = torch.where(use_policy, logits, q_chunk)
                qs.append(q_chunk)
            q = torch.cat(qs) if len(qs) > 1 else qs[0]
        elif isinstance(q_model, QTransformer):
            assert token_meta is not None and len(token_meta) == len(obs_rows)
            raw, n_raw, glob, hand = tokens_batch(
                obs_rows, [m[0] for m in token_meta],
                [m[1] for m in token_meta])
            if counts is not None:
                raw = np.repeat(raw, counts, axis=0)
                n_raw = np.repeat(n_raw, counts)
                glob = np.repeat(glob, counts, axis=0)
                hand = np.repeat(hand, counts, axis=0)
            q = q_model(
                torch.from_numpy(raw).to(device),
                torch.from_numpy(n_raw).to(device),
                torch.from_numpy(glob).to(device),
                torch.from_numpy(hand).to(device),
                torch.from_numpy(act_rows).to(device),
            )
        else:
            qs = []
            for lo in range(0, len(obs_rows), 8192):
                hi = min(lo + 8192, len(obs_rows))
                qs.append(q_model(
                    torch.from_numpy(obs_rows[lo:hi]).to(device),
                    torch.from_numpy(act_rows[lo:hi]).to(device)))
            q = torch.cat(qs) if len(qs) > 1 else qs[0]
    return q


def plpw_q_log_weights(q_model, device, particles, obs_players, obs_cards,
                       trump, declarer, me, tau=0.2):
    """log w(p) per particle: sum over observed non-`me` plays of the Q
    pseudo-policy log-prob softmax(Q/tau) of the card actually played.
    Replay machinery shared with learn/inference.py (PLPW)."""
    n = particles.shape[0]
    need_hand = isinstance(q_model, QTransformer)  # QGRU ignores hand tokens
    rows_obs, rows_act, rows_legal, rows_card, rows_particle, rows_meta = \
        [], [], [], [], [], []
    for k in range(n):
        hands = initial_hands(particles[k], obs_players, obs_cards)
        actions = legal_completion(hands, trump, declarer, obs_cards)
        out = mi.encode_q_trajectory(
            np.ascontiguousarray(np.array(hands, dtype=np.int64)),
            int(trump), int(declarer), actions)
        for t in range(len(obs_cards)):
            if int(out["players"][t]) != me:
                seat = int(out["players"][t])
                n_legal = int(out["n_legal"][t])
                rows_obs.append(out["obs"][t])
                rows_act.append(out["act"][t][:n_legal])
                rows_legal.append(out["legal"][t][:n_legal])
                rows_card.append(int(obs_cards[t]))
                rows_particle.append(k)
                own_mask = 0
                if need_hand:
                    own_mask = int(hands[seat])
                    for j in range(t):
                        if int(obs_players[j]) == seat:
                            own_mask &= ~(1 << int(obs_cards[j]))
                rows_meta.append((t, own_mask))

    logw = np.full(n, -np.inf)
    if rows_obs:
        counts = [r.shape[0] for r in rows_act]
        act = np.concatenate(rows_act, axis=0)
        if isinstance(q_model, (QTransformer, QGRU)):
            q = q_eval_batch(q_model, device, np.stack(rows_obs), act,
                             token_meta=rows_meta, counts=counts)
        else:
            obs = np.repeat(np.stack(rows_obs), counts, axis=0)
            q = q_eval_batch(q_model, device, obs, act)
        q = q.float().cpu().numpy() / max(tau, 1e-6)
        picked = np.zeros(len(rows_card))
        pos = 0
        for j in range(len(rows_card)):
            n_legal = counts[j]
            q_j = q[pos : pos + n_legal]
            q_j = q_j - q_j.max()
            log_z = np.log(np.exp(q_j).sum())
            idx = list(rows_legal[j]).index(rows_card[j])
            picked[j] = q_j[idx] - log_z
            pos += n_legal
        contrib = np.zeros(n)
        np.add.at(contrib, np.asarray(rows_particle), picked)
        has = np.zeros(n, dtype=bool)
        has[np.unique(rows_particle)] = True
        logw[has] = contrib[has]
    return logw


def q_blueprint_rollouts(worlds, declarer, q_model, device, rngs,
                         temperature=0.0, exact_leaves=False, exact_trick=6,
                         q_memo=None, leaf_memo=None, memo_stats=None,
                         override=None, rollout_student=None):
    """Play each world out with all seats following the Q blueprint (argmax,
    or softmax(Q/temperature) sampling). worlds mutated in place; returns
    final hand_points [K, 2]. With exact_leaves, worlds stop at
    exact_trick and finish by full-information exact minimax instead of
    blueprint rollout — removes the blueprint's endgame leak from the
    Q estimates and is cheaper than rolling out.

    q_memo / leaf_memo: optional dicts reusing Q rows (keyed by obs bytes;
    identical obs implies identical candidate set) and exact-leaf values
    across waves/arms — pass per-decision dicts to dedupe transposed
    states. Within one call duplicate rows are always forwarded once.

    override=(seat, spec, n_decisions): for that seat's first n_decisions
    decisions in each world, `spec` picks the card instead of the
    blueprint. This is the M9 joint-deviation hook: SPARTA evaluates
    changes to the *searching* seat's action only, so a joint local
    optimum — where no unilateral change helps but a simultaneous one does
    — is invisible to it by construction.

    spec is either
      ("qrank", k)  the seat's k-th best action under its OWN Q, k=0 being
                    the blueprint. Free: the Q row is already computed, so
                    this only changes argmax to argsort. This is the basis
                    to prefer — it perturbs *around* the blueprint, and it
                    is defined at every information set by construction.
      callable      rule(game, pid, legal) -> Card, reading only the acting
                    seat's own legal set and public state. Coarse hand
                    rules measured far worse than the blueprint (M9 smoke:
                    blueprint won 57-100% of decisions), which drags the
                    gap negative rather than merely understating it.

    rollout_student=(seat, residual_head, tau) replaces greedy Q only when
    that rotating seat acts. The same batched information-set encodings are
    used, so DAgger can change trajectory and branch continuation together
    without introducing a clairvoyant or per-world policy path.

    Either way nothing consults hidden cards: both forms are policies on
    the acting seat's own information set."""
    student_seat, student_head, student_tau = (
        rollout_student if rollout_student else (None, None, 1.0))
    ov_seat, ov_spec, ov_budget = override if override else (None, None, 0)
    # A q-rank override still needs the network row, so it stays on the Q
    # path and only changes how that row is reduced to a card.
    ov_qrank = (ov_spec[1] if isinstance(ov_spec, tuple)
                and ov_spec[0] == "qrank" else None)
    ov_rule = ov_spec if callable(ov_spec) else None
    ov_used = defaultdict(int)
    final = {k: np.array(worlds[k][0].hand_points, dtype=np.float64)
             for k in range(len(worlds))
             if worlds[k][0].phase != GamePhase.PLAYING}
    active = [k for k in range(len(worlds))
              if worlds[k][0].phase == GamePhase.PLAYING]
    while active:
        if exact_leaves:
            solved = {}
            for k in active:
                game, pc, pp = worlds[k]
                trick = (len(pc) + len(game.current_trick)) // 4
                if trick < exact_trick:
                    continue
                masks = np.zeros(4, dtype=np.int64)
                for p in range(4):
                    for c in game.hands[p]:
                        masks[p] |= np.int64(1) << c.to_index()
                key = None
                if leaf_memo is not None:
                    key = (masks.tobytes(), game.current_player,
                           tuple((p, c.to_index())
                                 for p, c in game.current_trick),
                           tuple(game.hand_points))
                    if key in leaf_memo:
                        if memo_stats is not None:
                            memo_stats["leaf_hit"] += 1
                        solved[k] = leaf_memo[key].copy()
                        continue
                arrays = game_to_cpp_arrays(game, pc, pp)
                res = mi.exact_solve(*arrays, declarer, masks)
                vals = res["pts0"]
                v = (max(vals) if game.current_player % 2 == 0
                     else min(vals))
                solved[k] = np.array([v, 60.0 - v])
                if leaf_memo is not None:
                    leaf_memo[key] = solved[k].copy()
                    if memo_stats is not None:
                        memo_stats["leaf_miss"] += 1
            final.update(solved)
            active = [k for k in active if k not in solved]
            if not active:
                break
        # Split the wave: worlds where the overridden seat is on turn (and
        # still has budget) never reach the network at all.
        rule_ks, q_ks, rank_ks = [], [], set()
        for k in active:
            on_turn = (ov_seat is not None
                       and worlds[k][0].current_player == ov_seat
                       and ov_used[k] < ov_budget)
            if on_turn and ov_rule is not None:
                rule_ks.append(k)
            else:
                if on_turn:
                    rank_ks.add(k)
                q_ks.append(k)
        need_hand = isinstance(q_model, QTransformer)
        rows = []
        metas = []
        for k in q_ks:
            game, pc, pp = worlds[k]
            arrays = game_to_cpp_arrays(game, pc, pp)
            enc = mi.encode_q_observation(*arrays, declarer,
                                          game.current_player)
            n = int(enc["n_legal"])
            rows.append((enc["obs"], enc["act"][:n], enc["legal"][:n]))
            own_mask = 0
            if need_hand:
                for c in game.hands[game.current_player]:
                    own_mask |= 1 << c.to_index()
            metas.append((len(pc) + len(game.current_trick), own_mask))
        # Dedup by obs before the forward: duplicates (transposed worlds,
        # repeated states across arms) reuse one Q evaluation.
        memo = q_memo if q_memo is not None else {}
        policy_rows = [worlds[k][0].current_player == student_seat
                       for k in q_ks]
        # The same observation can legitimately be evaluated under Q or the
        # student during tests with transposed worlds, so policy identity is
        # part of the memo key.
        keys = [(r[0].tobytes(), policy_rows[j])
                for j, r in enumerate(rows)]
        need = []
        for j, key in enumerate(keys):
            if key not in memo:
                memo[key] = None  # claimed by row j of this wave
                need.append(j)
        if memo_stats is not None:
            memo_stats["q_hit"] += len(rows) - len(need)
            memo_stats["q_miss"] += len(need)
        if need:
            counts = [rows[j][1].shape[0] for j in need]
            act = np.concatenate([rows[j][1] for j in need], axis=0)
            if isinstance(q_model, (QTransformer, QGRU)):
                q = q_eval_batch(q_model, device,
                                 np.stack([rows[j][0] for j in need]), act,
                                 token_meta=[metas[j] for j in need],
                                 counts=counts, policy_head=student_head,
                                 policy_tau=student_tau,
                                 policy_rows=[policy_rows[j] for j in need])
            else:
                obs = np.repeat(np.stack([rows[j][0] for j in need]),
                                counts, axis=0)
                q = q_eval_batch(q_model, device, obs, act)
            q = q.float().cpu().numpy()
            pos = 0
            for j, n in zip(need, counts):
                memo[keys[j]] = q[pos : pos + n]
                pos += n
        picks = []
        for j, k in enumerate(q_ks):
            q_j = memo[keys[j]]
            legal_j = rows[j][2]
            if k in rank_ks:
                # k-th best by the seat's own Q; clamp so a short legal set
                # degrades to "worst legal" rather than wrapping round to
                # the blueprint's own choice and silently measuring nothing.
                order = np.argsort(-np.asarray(q_j))
                r = min(ov_qrank, len(order) - 1)
                picks.append((k, Card.from_index(int(legal_j[order[r]]))))
                ov_used[k] += 1
                continue
            if temperature > 0:
                z = q_j / temperature
                z -= z.max()
                p = np.exp(z)
                p /= p.sum()
                picks.append((k, Card.from_index(int(rngs[k].choice(legal_j,
                                                                    p=p)))))
            else:
                picks.append((k, Card.from_index(
                    int(legal_j[int(np.argmax(q_j))]))))
        for k in rule_ks:
            game = worlds[k][0]
            pid = game.current_player
            legal = game.get_legal_moves(pid)
            picks.append((k, ov_rule(game, pid, legal)))
            ov_used[k] += 1
        for k, card in picks:
            game, pc, pp = worlds[k]
            pid = game.current_player
            pre = list(game.current_trick)
            if not game.play_card(pid, card):
                raise ValueError("Q-blueprint rollout played an illegal card")
            if len(pre) == 3:
                for tp_, tc_ in pre + [(pid, card)]:
                    pc.append(tc_.to_index())
                    pp.append(tp_)
        active = [k for k in active
                  if worlds[k][0].phase == GamePhase.PLAYING]
    for k in range(len(worlds)):
        if k not in final:
            final[k] = np.array(worlds[k][0].hand_points, dtype=np.float64)
    return np.array([final[k] for k in range(len(worlds))],
                    dtype=np.float64)


class SpartaDMCAgent(Agent):
    """1-ply SPARTA over Q-likelihood-weighted worlds, deviation threshold.

    alloc="fixed": one rollout per world per arm (baseline). alloc="ucb":
    SPARTA-style allocation — rounds of `ucb_chunk` worlds per alive arm,
    prune arms whose UCB (mean + 2 se) falls below the leader's LCB; pruned
    arms stop consuming rollouts, concentrating the budget on the arms that
    matter. Same total per-arm budget cap as fixed."""

    def __init__(self, q_model, device="cuda", n_particles=64, delta=0.6,
                 tau_likelihood=0.2, temperature=0.0, lam=1.0, eps=0.0,
                 seed=0, exact_leaves=True, exact_trick=6, alloc="fixed",
                 ucb_chunk=8, ucb_c=2.0, paired_ess_variance=True,
                 paired_pruning=True,
                 belief_model=None, lam_belief=0.0,
                 proposal_oversample=0,
                 deviate_rule="delta", conf_c=1.0, margin_gate=0.0,
                 tier3=False, t3_eps=1.0, t3_arms=2, t3_worlds=48,
                 choice_model=None, choice_temp=0.1, decl_tilt_model=None,
                 ismcts_trick=None, ismcts_iters=1000):
        self.q_model = q_model
        self.device = device
        self.n_particles = n_particles
        self.delta = delta
        self.tau_likelihood = tau_likelihood
        self.temperature = temperature
        self.lam = lam
        self.eps = eps
        self.seed = seed
        self.exact_leaves = exact_leaves
        self.exact_trick = exact_trick
        # ISMCTS endgame handoff (endgame_leak_plan.md arm C). From this
        # 0-based trick index onward the decision goes to the C++ ISMCTS
        # solver instead of SPARTA, fed the belief-weighted particle pool
        # this agent already computes.
        #
        # Why it might beat exact leaves against a human: exact leaves are
        # PIMC confined to the endgame -- full-information minimax per world
        # models PARTNER as clairvoyant, so dumping a Manille scores as
        # "partner takes it". ISMCTS shares one action per information set
        # across determinizations, so it has no strategy fusion and models
        # everyone as uninformed. PIMC already lost a full-game head-to-head
        # to ISMCTS in this project; this asks whether the endgame remnant
        # loses too. The standing note "ISMCTS at trick 6+ is strictly
        # weaker than exact-minimax leaves" was reasoning, never measured,
        # and it only holds WITHIN a determinized world.
        #
        # Indexing matches exact_trick: trick = (n_played + n_in_trick) // 4
        # over 8 tricks, so 6 is the last TWO tricks and 5 the last three.
        self.ismcts_trick = ismcts_trick
        self.ismcts_iters = ismcts_iters
        self._solver = None
        self.alloc = alloc
        self.ucb_chunk = ucb_chunk
        # Pruning threshold on the paired difference against the leader, in
        # standard errors. 2.0 keeps the old two-sigma bar.
        self.ucb_c = ucb_c
        # Keep the ESS-based paired SE for the ADOPTED deviation rule (A1,
        # conf c=1.0, gated at 400 deals). The delta-method form is the
        # correct one, but swapping it changes the champion's deviation
        # decisions, so it stays behind this flag until a match says so.
        self.paired_ess_variance = paired_ess_variance
        # The paired racing rule saves ~25% wall time but changes which arms
        # receive the full rollout budget. Keep the pre-change unweighted,
        # unpaired rule reachable for a direct strength gate; default True is
        # the currently served implementation.
        self.paired_pruning = paired_pruning
        # Declaration context for the M6.3 world tilt. Defaults reproduce
        # the previous behaviour (representative 0-0 state, declaration
        # assumed to come from this policy); the GUI supplies the real
        # values per hand via set_declaration_context.
        self.decl_us, self.decl_them = 0, 0
        self.declarer_is_bot = True
        self.belief_model = belief_model
        self.lam_belief = lam_belief
        self.proposal_oversample = proposal_oversample
        # "delta": deviate when the mean gap exceeds self.delta (default,
        # the gated champion rule). "conf": deviate when the PAIRED
        # per-world evidence is significant (see paired_evidence) — same
        # compute, uses the per-world spread the delta rule throws away.
        self.deviate_rule = deviate_rule
        self.conf_c = conf_c
        # Margin gate (points): skip the search entirely when the blueprint's
        # own top-two Q gap already exceeds this, and just play its argmax.
        # 0.0 disables it.
        #
        # MEASURED AND REJECTED (2026-07-26, scripts/sparta_margin_gate_probe,
        # runs/margin_gate_probe_mlp40.log, 797 plies): the blueprint's margin
        # barely predicts where the search will disagree with it — median
        # margin on deviating plies 0.21 vs 0.35 overall — so the gate is
        # only marginally better than skipping plies at random. Skipping 58%
        # of plies loses 46% of all delta-rule deviations for 2.27x; the safe
        # end buys 1.29x for 15% of them. And the deviations it drops are
        # worth MORE than average (mean gap 2.1-2.7 pts vs 2.03 overall):
        # a ply where the net is confident and the search still overrules it
        # is the net being confidently wrong, which is exactly the correction
        # worth paying for. Left in place (default off) so the negative
        # result is reproducible, not because it should be used.
        self.margin_gate = margin_gate
        self.gate_stats = {"gated": 0, "searched": 0}
        self._last_samples = {}
        self.tier3 = tier3
        self.t3_eps = t3_eps
        self.t3_arms = t3_arms
        self.t3_worlds = t3_worlds
        # Phase 6 free trump: when a chooser is attached, worlds are sampled
        # UNCONSTRAINED (the hard rule constraint is wrong under net-chosen
        # trump) and tilted by P_chooser(observed trump | declarer's hand).
        self.choice_model = choice_model
        self.choice_temp = choice_temp
        # M6.3: same unconstrained + soft-tilt mechanics, but with the
        # equity chooser's five-declaration likelihood (mull included) and
        # WITHOUT taking over declarations -- trump_for_seat only asks
        # choice_model, so a tilt-only agent still declares by the rule.
        self.decl_tilt_model = decl_tilt_model
        self.t3_stats = {"triggered": 0, "start": 0, "end": 0, "flips": 0}
        self.last_analysis = None
        self.memo_stats = {"q_hit": 0, "q_miss": 0,
                           "leaf_hit": 0, "leaf_miss": 0}
        self._q_memo = {}
        self._leaf_memo = {}
        self.dealer = 0
        self.blueprint = QNetAgent(q_model, device)

    def set_declaration_context(self, us, them, declarer_is_bot=True):
        """Match score (from the DECLARING team's view) and whether this
        policy actually modelled the declaration.

        Both matter to the M6.3 world tilt and neither was being supplied.
        The score because D1 argmaxes match-win probability, so the same
        hand declares differently at 0-0 and at 90-50 -- tilting worlds by
        a 0-0 likelihood during a real match weights the wrong hands.
        `declarer_is_bot` because the tilt is a model of THIS chooser: when
        a human declared, it is a model of the wrong agent, and a confident
        wrong likelihood is worse than none. In that case the tilt is
        dropped and the worlds stay uniform over declarations, which is the
        honest prior until a calibrated human model exists."""
        self.decl_us, self.decl_them = int(us), int(them)
        self.declarer_is_bot = bool(declarer_is_bot)

    def new_game(self, game, dealer, player_id):
        self.dealer = dealer
        self.blueprint.new_game(game, dealer, player_id)

    def set_rollout_student(self, seat=None, head=None, tau=1.0):
        """Use a residual policy on exactly one seat in branch rollouts."""
        self.rollout_student = ((int(seat), head, float(tau))
                                if seat is not None else None)

    def choose_trump_suit(self, hand_mask):
        """Free-trump convention (Phase 6): the harness calls this on the
        declarer-seat agent when a choice model is attached; deterministic
        argmax at eval time (same contract as QNetAgent)."""
        return int(np.argmax(
            suit_values(self.choice_model, hand_mask, self.device)))

    def _arm_values(self, game, player_id, played_cards, played_players,
                    cards, particles, my_team, trump, base_seed,
                    seed_offsets, override=None):
        """Terminal points for my team from playing each of `cards` in each
        world — one batched rollout across arms × worlds (worlds evolve
        independently, so cross-arm concatenation is exact). seed_offsets
        are the global world indices: RNG streams stay paired across arms
        and identical to the per-arm implementation."""
        worlds = []
        for card in cards:
            for k in range(len(particles)):
                w_game = inject_world(
                    self.dealer, trump, particles[k], game.current_trick,
                    game.hand_points, game.tricks_won, game.current_player)
                pc_w, pp_w = list(played_cards), list(played_players)
                pre = list(w_game.current_trick)
                if not w_game.play_card(player_id, card):
                    raise ValueError("candidate illegal in a world whose "
                                     "hand matches mine — injector bug")
                if len(pre) == 3:
                    for tp_, tc_ in pre + [(player_id, card)]:
                        pc_w.append(tc_.to_index())
                        pp_w.append(tp_)
                worlds.append((w_game, pc_w, pp_w))
        rngs = None
        if self.temperature > 0:
            rngs = [np.random.default_rng(base_seed + int(o))
                    for _ in cards for o in seed_offsets]
        pts = q_blueprint_rollouts(worlds, self.dealer, self.q_model,
                                   self.device, rngs, self.temperature,
                                   self.exact_leaves, self.exact_trick,
                                   q_memo=self._q_memo,
                                   leaf_memo=self._leaf_memo,
                                   memo_stats=self.memo_stats,
                                   override=override,
                                   rollout_student=getattr(
                                       self, "rollout_student", None))
        return pts[:, my_team].reshape(len(cards), len(particles))

    def _search_fixed(self, game, player_id, played_cards, played_players,
                      legal, particles, weights, my_team, trump, base_seed):
        """Returns (best card, per-card weighted values, rollout counts)."""
        vals = self._arm_values(game, player_id, played_cards,
                                played_players, legal, particles, my_team,
                                trump, base_seed,
                                np.arange(len(particles)))
        q, counts = {}, {}
        self._last_samples = {}
        for i, card in enumerate(legal):
            q[card] = float((vals[i] * weights).sum() / weights.sum())
            counts[card] = vals.shape[1]
            self._last_samples[card] = (np.asarray(vals[i]),
                                        np.asarray(weights))
        return max(q, key=q.get), q, counts

    def _search_ucb(self, game, player_id, played_cards, played_players,
                    legal, particles, weights, my_team, trump, base_seed):
        """SPARTA allocation: equal chunks per alive arm per round (paired
        across arms), prune arms with UCB < leader LCB."""
        samples = {c: ([], []) for c in legal}  # card -> (pts, weights)
        alive = list(legal)
        n_worlds = len(particles)  # < n_particles when proposal collapses
        chunk = min(self.ucb_chunk, n_worlds)
        n_rounds = (n_worlds + chunk - 1) // chunk
        for r in range(n_rounds):
            lo = r * chunk
            hi = min(lo + chunk, n_worlds)
            sub = np.arange(lo, hi)
            vals = self._arm_values(
                game, player_id, played_cards, played_players, alive,
                particles[sub], my_team, trump, base_seed, sub)
            for i, card in enumerate(alive):
                samples[card][0].extend(vals[i])
                samples[card][1].extend(weights[sub])
            # Prune on PAIRED differences against the current leader.
            #
            # This used to take a likelihood-weighted mean and bracket it
            # with an UNWEIGHTED std/sqrt(n) -- two different estimators
            # stapled together, and the interval is not valid when the PLPW
            # weights have low ESS. It also compared arms as if they were
            # independent samples, when every arm is evaluated on the same
            # worlds: the paired difference cancels the across-world variance
            # that otherwise swamps the effect being measured.
            means = {}
            for card in alive:
                v = np.asarray(samples[card][0])
                w = np.asarray(samples[card][1])
                means[card] = float((v * w).sum() / max(w.sum(), 1e-12))
            leader = max(alive, key=means.get)
            if self.paired_pruning:
                lv, lw = samples[leader]
                survivors = [leader]
                for card in alive:
                    if card is leader:
                        continue
                    m, se, n = weighted_paired_stats(
                        samples[card][0], lv, lw,
                        ess_variance=self.paired_ess_variance)
                    # No usable paired evidence yet -> the arm survives;
                    # pruning on one world is how a winner gets thrown away.
                    if se is None or m + self.ucb_c * se >= 0.0:
                        survivors.append(card)
            else:
                # Exact pre-443fef8 racing rule: weighted means but
                # UNWEIGHTED, independent-arm standard errors. This estimator
                # mismatch is retained only as the control arm for the match
                # gate; it must not silently become a second production mode.
                ses = {}
                for card in alive:
                    v = np.asarray(samples[card][0])
                    ses[card] = (float(v.std(ddof=1)) / np.sqrt(len(v))
                                 if len(v) > 1 else 1e9)
                lcb = means[leader] - self.ucb_c * ses[leader]
                survivors = [
                    card for card in alive
                    if means[card] + self.ucb_c * ses[card] >= lcb
                ]
            alive = survivors
            if len(alive) == 1:
                break
        q, counts = {}, {}
        self._last_samples = {}
        for card in legal:
            v = np.asarray(samples[card][0])
            w = np.asarray(samples[card][1])
            q[card] = float((v * w).sum() / max(w.sum(), 1e-12))
            counts[card] = len(v)
            # Worlds are appended in the same order for every alive arm, so
            # index k is the same world across arms — the common prefix is
            # what the paired deviation rule compares on.
            self._last_samples[card] = (v, w)
        return max(q, key=q.get), q, counts

    def _paranoid_refine(self, game, player_id, played_cards, played_players,
                         arms, particles, weights, my_team, trump, base_seed):
        """Tier-3 re-evaluation of the close arms with an adversarial
        response layer. In each top-weight world: play the candidate,
        auto-play forced moves to the next non-forced decision; an opponent
        to move with >1 legal branches over ALL their responses and the
        world takes the MIN (they do not play the blueprint). Own-team
        response nodes and terminals roll out as in 1-ply (info-state-
        consistent; per-world max over partner responses would overestimate
        coordination — PFM bias). Returns {card: refined weighted value}.
        """
        M = min(self.t3_worlds, len(particles))
        idx = np.argsort(weights)[-M:]
        sub_w = weights[idx]
        refined = {}
        for ai, card in enumerate(arms):
            # Play the candidate in each world; advance forced moves.
            states = []
            for k in idx:
                wg = inject_world(self.dealer, trump, particles[k],
                                  game.current_trick, game.hand_points,
                                  game.tricks_won, game.current_player)
                pc_w = list(played_cards)
                pp_w = list(played_players)
                pre = list(wg.current_trick)
                if not wg.play_card(player_id, card):
                    raise ValueError("candidate illegal in a world whose "
                                     "hand matches mine — injector bug")
                if len(pre) == 3:
                    for tp_, tc_ in pre + [(player_id, card)]:
                        pc_w.append(tc_.to_index())
                        pp_w.append(tp_)
                while wg.phase == GamePhase.PLAYING:
                    pid2 = wg.current_player
                    legal2 = wg.get_legal_moves(pid2)
                    if len(legal2) != 1:
                        break
                    pre2 = list(wg.current_trick)
                    wg.play_card(pid2, legal2[0])
                    if len(pre2) == 3:
                        for tp_, tc_ in pre2 + [(pid2, legal2[0])]:
                            pc_w.append(tc_.to_index())
                            pp_w.append(tp_)
                states.append((wg, pc_w, pp_w))
            # Branch at opponent response nodes; everything else rolls out.
            branch = []  # (mk, response_card, wg, pc_w, pp_w)
            direct = []  # (mk, None, wg, pc_w, pp_w)
            for mk, (wg, pc_w, pp_w) in enumerate(states):
                if wg.phase == GamePhase.PLAYING:
                    pid2 = wg.current_player
                    legal2 = wg.get_legal_moves(pid2)
                    if len(legal2) > 1 and pid2 % 2 != my_team:
                        for r in legal2:
                            wg2 = copy.deepcopy(wg)
                            pc2 = list(pc_w)
                            pp2 = list(pp_w)
                            pre2 = list(wg2.current_trick)
                            wg2.play_card(pid2, r)
                            if len(pre2) == 3:
                                for tp_, tc_ in pre2 + [(pid2, r)]:
                                    pc2.append(tc_.to_index())
                                    pp2.append(tp_)
                            branch.append((mk, r, wg2, pc2, pp2))
                        continue
                direct.append((mk, None, wg, pc_w, pp_w))
            # One batched rollout over direct + branch worlds.
            all_worlds = [(wg, pc, pp) for (_, _, wg, pc, pp)
                          in direct + branch]
            rngs = [np.random.default_rng(base_seed + 7919 * (ai + 1) + j)
                    for j in range(len(all_worlds))]
            pts = q_blueprint_rollouts(
                all_worlds, self.dealer, self.q_model, self.device, rngs,
                self.temperature, self.exact_leaves, self.exact_trick,
                q_memo=self._q_memo, leaf_memo=self._leaf_memo,
                memo_stats=self.memo_stats)
            vals = pts[:, my_team]
            per_k = {}
            for j, (mk, _, _, _, _) in enumerate(direct):
                per_k.setdefault(mk, []).append(vals[j])
            off = len(direct)
            for j, (mk, r, _, _, _) in enumerate(branch):
                per_k.setdefault(mk, []).append(vals[off + j])
            v = np.empty(len(states))
            for mk in range(len(states)):
                vs = per_k[mk]
                v[mk] = min(vs) if len(vs) > 1 else vs[0]
            refined[card] = float((v * sub_w).sum() / sub_w.sum())
        return refined

    def _build_worlds(self, game, player_id, played_cards, played_players):
        """Sample determinizations and compute their search weights:
        hard-constraint (rule trump) or unconstrained + soft choice tilt
        (Phase 6), Q-likelihood (PLPW), optional B_φ proposal/composition.
        Returns (particles, weights, arrays, obs_cards, obs_players,
        base_seed). Shared by analyze() and the ISMCTS endgame handoff
        (endgame_leak_plan.md arm C), which injects this exact pool into
        the C++ solver as its determinization set."""
        arrays = game_to_cpp_arrays(game, played_cards, played_players)
        trump = int(arrays[8])
        obs_cards = np.array(
            list(played_cards) + [c.to_index() for _, c in game.current_trick],
            dtype=np.int32)
        obs_players = np.array(
            list(played_players) + [p for p, _ in game.current_trick],
            dtype=np.int32)
        # Paired rollouts: same RNG streams across action arms (also seeds
        # the belief-proposal resampler below).
        ctx = zlib.crc32(obs_cards.tobytes())
        base_seed = (self.seed * 1000003 + ctx) % (2**63 - 1)
        log_mult = 0.0
        proposal = (self.proposal_oversample > 0
                    and self.belief_model is not None)
        # Under net-chosen trump (Phase 6) or the M6.3 equity tilt the hard
        # rule constraint on the declarer's hand is wrong, so worlds are
        # sampled unconstrained and re-weighted softly below.
        decl = (-1 if (self.choice_model is not None
                       or self.decl_tilt_model is not None)
                else self.dealer)
        if proposal:
            # Belief-as-proposal (Step-3 probe): oversample K× from the
            # hard-constraint sampler, SIR-resample down by B_φ weights,
            # collapse duplicates into log-multiplicities (exact at
            # temperature 0 — duplicate worlds roll out identically).
            # B_φ steers WHICH worlds get searched; the search weights
            # below stay Q-likelihood-only (no ESS shrinkage from
            # multiplying belief into the weights — the W2 failure mode).
            cand = mi.sample_determinizations(
                *arrays, player_id,
                self.n_particles * self.proposal_oversample,
                declarer=decl, allow_short=True)
            obs = mi.encode_observation(*arrays, self.dealer, player_id)
            bw = normalize_weights(
                dmc_belief_log_weights(self.belief_model, self.device, obs,
                                       cand, arrays, player_id),
                len(cand), 0.0)
            rng = np.random.default_rng(base_seed ^ 0x9E3779B9)
            pos = (rng.random()
                   + np.arange(self.n_particles)) / self.n_particles
            idx = np.minimum(np.searchsorted(np.cumsum(bw), pos),
                             len(cand) - 1)
            particles, mult = np.unique(cand[idx], axis=0,
                                        return_counts=True)
            log_mult = np.log(mult.astype(np.float64))
        else:
            # allow_short: in the ~0.01% of states where the trump rule
            # pins the declarer's hand hard, individual draws exhaust the
            # rejection budget. Search on the worlds that landed rather than
            # failing the decision — the alternative killed a 20k-deal
            # labelling run four times over.
            particles = mi.sample_determinizations(
                *arrays, player_id, self.n_particles, declarer=decl,
                allow_short=True)
        logw = plpw_q_log_weights(
            self.q_model, self.device, particles, obs_players, obs_cards,
            trump, self.dealer, player_id, self.tau_likelihood)
        if (self.belief_model is not None and self.lam_belief > 0.0
                and not proposal):
            # Composed world weights. -inf Q entries mean "this particle
            # replayed zero observed decisions" — no likelihood information,
            # NOT evidence against; map them to 0 before mixing so
            # lam=0 (pure B_phi) never computes 0 * -inf = NaN (the W2 bug).
            logw_q = np.where(np.isfinite(logw), logw, 0.0)
            total_logw = self.lam * logw_q
            obs = mi.encode_observation(*arrays, self.dealer, player_id)
            total_logw = total_logw + self.lam_belief * \
                dmc_belief_log_weights(
                    self.belief_model, self.device, obs, particles, arrays,
                    player_id)
        else:
            total_logw = self.lam * logw
        if proposal:
            # -inf = "replayed zero observed decisions": no Q information,
            # not evidence against — keep the multiplicity weight (same
            # rationale as the W2 0*-inf fix above).
            total_logw = np.where(np.isfinite(total_logw), total_logw,
                                  0.0) + log_mult
        if self.decl_tilt_model is not None and self.declarer_is_bot:
            # M6.3 equity tilt (handoff): tilt worlds by the equity
            # chooser's five-declaration likelihood P(observed decl |
            # declarer's reconstructed initial hand) -- mull included --
            # replacing the hard rule constraint. Same -inf mapping
            # rationale as the choice_model branch below.
            masks = [initial_hands(p, obs_players, obs_cards)[self.dealer]
                     for p in particles]
            total_logw = np.where(np.isfinite(total_logw), total_logw, 0.0)
            total_logw = total_logw + self.decl_tilt_model.log_likelihood(
                masks, trump, us=self.decl_us, them=self.decl_them)
        elif self.choice_model is not None:
            # Soft trump-choice likelihood (Phase 6 free trump): tilt worlds
            # by P_chooser(observed trump | declarer's reconstructed initial
            # hand) instead of the hard rule constraint. -inf Q entries mean
            # "replayed zero observed decisions" (the trick-0 leader): no Q
            # information, NOT evidence against — map them to 0 first, or
            # -inf + finite drowns the tilt and the normalize fallback goes
            # uniform exactly where the unconstrained sampler needs the
            # choice information most (W2 rationale; proposal path is
            # already mapped above).
            masks = [initial_hands(p, obs_players, obs_cards)[self.dealer]
                     for p in particles]
            total_logw = np.where(np.isfinite(total_logw), total_logw, 0.0)
            total_logw = total_logw + choice_log_likelihood(
                self.choice_model, masks, trump, self.device,
                self.choice_temp)
        weights = normalize_weights(total_logw, len(particles), self.eps)
        return particles, weights, arrays, obs_cards, obs_players, base_seed

    def analyze(self, game, player_id, played_cards, played_players):
        """One SPARTA evaluation of the current decision, without the
        deviation threshold. Returns (a_bp, a_star, q, counts): the blueprint
        action, the search action, per-legal-card weighted search values and
        per-card rollout sample counts (UCB prunes arms, so counts differ).
        Forced moves (<=1 legal) return trivial entries. Does not mutate
        `game`. The distillation teacher's labeling entry point."""
        legal = game.get_legal_moves(player_id)
        if len(legal) <= 1:
            self.last_analysis = (legal[0], legal[0], {legal[0]: 0.0},
                                  {legal[0]: 0})
            return self.last_analysis

        # Fresh transposition memos per decision: entries are keyed on the
        # full info-state obs / leaf state, so they stay valid across arms,
        # rounds and Tier-3 refines within this decision.
        self._q_memo = {}
        self._leaf_memo = {}
        bp_legal, bp_q = self.blueprint.q_values(game, player_id, played_cards,
                                                 played_players)
        order = np.argsort(bp_q)[::-1]
        a_bp = Card.from_index(int(bp_legal[order[0]]))
        if self.margin_gate > 0.0 and len(order) >= 2:
            # Q is a normalized team-points return in [-1, 1]; the gate is
            # specified in points, so compare on the points scale.
            margin = float(bp_q[order[0]] - bp_q[order[1]]) * 30.0
            if margin > self.margin_gate:
                self.gate_stats["gated"] += 1
                self._last_samples = {}
                self.last_analysis = (a_bp, a_bp, {a_bp: margin}, {a_bp: 0})
                return self.last_analysis
        self.gate_stats["searched"] += 1
        (particles, weights, arrays, obs_cards, obs_players,
         base_seed) = self._build_worlds(game, player_id, played_cards,
                                         played_players)
        trump = int(arrays[8])

        my_team = player_id % 2
        if self.alloc == "ucb":
            a_star, q, counts = self._search_ucb(
                game, player_id, played_cards, played_players, legal,
                particles, weights, my_team, trump, base_seed)
        else:
            a_star, q, counts = self._search_fixed(
                game, player_id, played_cards, played_players, legal,
                particles, weights, my_team, trump, base_seed)
        if self.tier3:
            ranked = sorted(q, key=q.get, reverse=True)
            top = ranked[: min(self.t3_arms, len(ranked))]
            if len(top) >= 2 and q[top[0]] - q[top[1]] < self.t3_eps:
                end_of_trick = len(game.current_trick) >= 2
                self.t3_stats["triggered"] += 1
                self.t3_stats["end" if end_of_trick else "start"] += 1
                refined = self._paranoid_refine(
                    game, player_id, played_cards, played_players, top,
                    particles, weights, my_team, trump, base_seed)
                q.update(refined)
                new_star = max(q, key=q.get)
                if new_star != a_star:
                    self.t3_stats["flips"] += 1
                a_star = new_star
        # Kept for callers that want the search's own per-card values after
        # choose_card (the GUI hint reads these instead of re-running the
        # search or falling back to the blueprint's flat Q spread).
        self.last_analysis = (a_bp, a_star, q, counts)
        return self.last_analysis

    def paired_evidence(self, a_star, a_bp):
        """Weighted mean and standard error of the PER-WORLD difference
        v(a_star) - v(a_bp), over the worlds both arms were evaluated on.

        The delta rule thresholds the difference of two weighted means and
        ignores how much the worlds agreed. But both arms are evaluated on
        the same worlds, so the per-world difference is paired: the huge
        across-world variance cancels and what is left is the actual noise
        of THIS decision. Some decisions have every world agreeing on a
        0.3-point edge (real signal a fixed delta discards); others have a
        1.2-point mean with worlds disagreeing wildly (noise a fixed delta
        acts on). Returns (mean, se, n_paired); se is None when there is
        not enough paired data to judge.

        The estimator itself lives in `weighted_paired_stats`, shared with
        the arm-pruning path so the search cannot race on one rule and
        deviate on another."""
        s = getattr(self, "_last_samples", None)
        if not s or a_star not in s or a_bp not in s:
            return None, None, 0
        v_star, w_star = s[a_star]
        v_bp, _ = s[a_bp]
        return weighted_paired_stats(v_star, v_bp, w_star,
                                     ess_variance=self.paired_ess_variance)

    @property
    def solver(self):
        if self._solver is None:
            from manille_ismcts import ManilleSolverSO
            self._solver = ManilleSolverSO(iterations=self.ismcts_iters)
        return self._solver

    @solver.setter
    def solver(self, value):
        # SpartaIsmctsHandoffAgent (learn/dmc_endgame_handoff.py) predates
        # this lazy property and installs its own eagerly-built solver with
        # its own iteration count. Without a setter that plain assignment
        # raises, which is how it broke when the property landed.
        self._solver = value

    def _ismcts_endgame(self, game, player_id, played_cards, played_players):
        """Hand the decision to ISMCTS, on OUR belief-weighted worlds.

        The pool is the same one SPARTA would have searched, injected as the
        solver's determinization set (CLAUDE.md's documented hook), so this
        swaps the endgame search without giving up the learned belief."""
        particles, weights, arrays, _, _, _ = self._build_worlds(
            game, player_id, played_cards, played_players)
        result = self.solver.solve(*arrays, particles=particles,
                                   particle_weights=weights)
        best = int(result["best_move"])
        legal = game.get_legal_moves(player_id)
        for card in legal:
            if card.to_index() == best:
                return card
        raise RuntimeError(
            f"ISMCTS endgame returned illegal card {best} for seat "
            f"{player_id}; legal were {[c.to_index() for c in legal]}")

    def choose_card(self, game, player_id, played_cards, played_players) -> Card:
        legal = game.get_legal_moves(player_id)
        if len(legal) <= 1:
            self.last_analysis = None  # never leave a stale analysis behind
            return legal[0]
        if self.ismcts_trick is not None:
            trick = (len(played_cards) + len(game.current_trick)) // 4
            if trick >= self.ismcts_trick:
                # No SPARTA analysis on this path; the GUI hint falls back to
                # blueprint Q and still badges the card actually played.
                self.last_analysis = None
                return self._ismcts_endgame(game, player_id, played_cards,
                                            played_players)
        a_bp, a_star, q, _ = self.analyze(game, player_id, played_cards,
                                          played_players)
        if a_star == a_bp:
            return a_bp
        if self.deviate_rule == "conf":
            m, se, _ = self.paired_evidence(a_star, a_bp)
            if se is not None:
                return a_star if m - self.conf_c * se > 0.0 else a_bp
            # not enough paired data — fall back to the delta rule
        if q[a_star] - q.get(a_bp, -1e9) > self.delta:
            return a_star
        return a_bp
