"""Policy-Likelihood Particle Weigher (PLPW) — SPARTA's computed Bayesian
inverse for Manille (sparta_manille_plan.md §3, Tier 1).

Instead of learned B_phi marginals, candidate worlds from the existing
hard-constraint sampler are weighted by the blueprint policy's likelihood of
the observed actions in each world:

    log w(p) = lam * sum over observed plies t whose actor != me of
               log pi_blueprint(a_t | obs_actor,t(p))

`obs_actor,t(p)` is the actor's counterfactual view at ply t in world p,
produced by replaying the deal in that world (`mi.encode_trajectory`).
Forced plays (one legal move) contribute log 1 = 0 automatically — the
"forced plays carry no information" property falls out for free.

Sharpness knobs (the Skat cheating-agent lesson, plan §3.4):
  lam in (0, 1]  likelihood temperature (0 -> uniform weights, 1 -> full
                 inverse)
  eps in [0, 1)  uniform floor mixed into the normalized weights so no
                 consistent world's weight hits 0

CLI (E1 validator, plan §4.3 / §7): per-trick log-loss of the true hands
under (a) constrained-uniform particle marginals, (b) B_phi marginals,
(c) PLPW-weighted particle marginals:

    python -m learn.inference validate --model runs/m1a.pt \
        --data data/validation_150 --deals 100 --particles 64
"""
import argparse

import numpy as np
import torch

import manille_ismcts as mi
from manille_env.card import Card, Suit
from manille_env.game_logic import GamePhase, ManilleGame
from tournament import game_to_cpp_arrays

from .encoding import LEGAL_MASK_OFF
from .model import belief_probs, load_model, split_belief_inputs
from .replay import encode_deal, load_deals


def initial_hands(particle, obs_players, obs_cards):
    """Reconstruct the 4 initial 8-card hand masks of a world: the particle's
    current holdings plus every observed card given back to its player."""
    hands = [int(m) for m in particle]
    for p, c in zip(obs_players, obs_cards):
        hands[int(p)] |= 1 << int(c)
    return hands


def legal_completion(hands, trump, declarer, prefix_cards):
    """Replay the observed prefix in the particle world and complete it with
    first-legal moves, returning a full 32-ply action sequence that
    `mi.encode_trajectory` will accept. A prefix card that is illegal in the
    particle world means the constraint sampler produced a
    history-inconsistent world — that is a sampler bug, so surface it."""
    game = ManilleGame(dealer=declarer)
    game.hands = [
        [Card.from_index(c) for c in range(32) if mask >> c & 1]
        for mask in hands
    ]
    # trump == -1 is mull (no trump): the engine models it as trump_suit
    # None, and Suit(-1) would throw.
    game.select_trump(Suit(trump) if trump >= 0 else None)
    game.start_playing()
    actions = []
    for c in prefix_cards:
        pid = game.current_player
        if not game.play_card(pid, Card.from_index(int(c))):
            raise ValueError(
                f"particle replay: observed card {c} illegal at ply "
                f"{len(actions)} — constraint-sampler bug")
        actions.append(int(c))
    while game.phase == GamePhase.PLAYING and len(actions) < 32:
        pid = game.current_player
        move = game.get_legal_moves(pid)[0]
        game.play_card(pid, move)
        actions.append(move.to_index())
    return np.array(actions, dtype=np.int32)


def plpw_log_weights(model, device, particles, obs_players, obs_cards,
                     trump, declarer, me):
    """log w(p) for each particle (pre-temperature, unnormalized).

    model: callable obs[N, OBS_DIM] -> (logits, values, offsets); the policy
    is legal-masked log-softmaxed per row. Only observed plies whose actor
    is not `me` contribute.
    """
    n = particles.shape[0]
    rows_obs, rows_action, rows_particle = [], [], []
    for k in range(n):
        hands = initial_hands(particles[k], obs_players, obs_cards)
        actions = legal_completion(hands, trump, declarer, obs_cards)
        out = mi.encode_trajectory(
            np.ascontiguousarray(np.array(hands, dtype=np.int64)),
            int(trump), int(declarer), actions)
        obs, players = out["obs"], out["players"]
        for t in range(len(obs_cards)):
            if int(players[t]) != me:
                rows_obs.append(obs[t])
                rows_action.append(int(obs_cards[t]))
                rows_particle.append(k)

    logw = np.zeros(n)
    if rows_obs:
        batch = np.stack(rows_obs)
        with torch.no_grad():
            t = torch.from_numpy(batch).to(device)
            logits, _, _ = model(t)
            legal = t[:, LEGAL_MASK_OFF : LEGAL_MASK_OFF + 32]
            logp = torch.log_softmax(
                logits.float().masked_fill(legal < 0.5, -1e9), dim=-1)
            picked = logp[np.arange(len(rows_action)),
                          np.array(rows_action)].cpu().numpy()
        np.add.at(logw, np.asarray(rows_particle), picked)
    return logw


def normalize_weights(logw, n, eps=0.0):
    """softmax-normalize log-weights with degenerate fallback and optional
    uniform floor (eps-mix, plan §3.4)."""
    if not np.isfinite(logw).any():
        w = np.ones(n)  # degenerate: fall back to hard-constraint-only
    else:
        logw = np.where(np.isfinite(logw), logw, -1e30)
        w = np.exp(logw - logw.max())
        if not np.isfinite(w).all() or w.sum() <= 1e-12:
            w = np.ones(n)  # degenerate: fall back to hard-constraint-only
    w = w / w.sum()
    if eps > 0.0:
        w = (1.0 - eps) * w + eps / n
    return w


def plpw_weights(model, device, arrays, game, me, played_cards, played_players,
                 declarer, n_particles, lam=1.0, eps=0.0):
    """Agent-facing entry (drop-in for belief_weighted_particles): sample
    candidate worlds and weight them by the computed policy-likelihood
    inverse. Returns (particles [n, 4] int64, weights [n] summing to 1)."""
    particles = mi.sample_determinizations(*arrays, me, n_particles,
                                           declarer=declarer)
    obs_cards = list(played_cards) + [c.to_index() for _, c in game.current_trick]
    obs_players = list(played_players) + [p for p, _ in game.current_trick]
    trump = int(arrays[8])
    logw = plpw_log_weights(model, device, particles, obs_players, obs_cards,
                            trump, declarer, me)
    return particles, normalize_weights(lam * logw, n_particles, eps)


# ---------------------------------------------------------------------------
# E1 validator: how much inverse-accuracy does B_phi's amortization leave on
# the table? Per-trick log-loss of the true hands under three marginals.
# ---------------------------------------------------------------------------

def _particle_marginals(particles, me, weights=None):
    """P(card c held by relative seat r) as [3, 32] from particle masks."""
    cards = np.arange(32)
    marg = np.zeros((3, 32))
    for r in range(3):
        seat = (me + r + 1) % 4
        bits = (particles[:, seat][:, None] >> cards[None, :]) & 1  # [n, 32]
        if weights is None:
            marg[r] = bits.mean(axis=0)
        else:
            marg[r] = (bits * weights[:, None]).sum(axis=0) / weights.sum()
    return marg


def validate_marginals(model, device, deals, n_particles, lam, eps, max_deals):
    """Accumulate per-trick -log P(true holder), Brier score, and top-1
    accuracy for (uniform, B_phi, PLPW). R11: NLL with a 1e-9 floor charges
    ~20.7 when no surviving particle holds the true assignment — a
    discretization tax B_phi never pays — so read Brier/top-1 alongside
    before calling close results (and run E1 with --eps ~0.05, >=256
    particles)."""
    nll = np.zeros((3, 8))
    brier = np.zeros((3, 8))
    top1 = np.zeros((3, 8))
    cnt = np.zeros(8)
    for deal in deals[:max_deals]:
        trump, declarer = int(deal["trump"]), int(deal["declarer"])
        true_hands = [int(deal["hands"][p]) for p in range(4)]
        holder = np.full(32, -1, dtype=np.int64)
        for p in range(4):
            for c in range(32):
                if true_hands[p] >> c & 1:
                    holder[c] = p

        out = encode_deal(deal)  # (obs, players) per ply, C++ parity-checked
        obs_rows, players = out

        with torch.no_grad():
            t = torch.from_numpy(obs_rows).to(device)
            _, _, offsets = model(t)
            prior, mask = split_belief_inputs(t)
            bphi = belief_probs(offsets, prior, mask).cpu().numpy()  # [32,3,32]

        # Replay the true deal in the Python engine for per-ply public state.
        game = ManilleGame(dealer=declarer)
        game.hands = [
            [Card.from_index(c) for c in range(32) if m >> c & 1]
            for m in true_hands
        ]
        game.select_trump(Suit(trump) if trump >= 0 else None)
        game.start_playing()
        played_cards, played_players = [], []
        actions = deal["plies"]["card"].astype(np.int64)

        for ply in range(32):
            me = game.current_player
            assert me == int(players[ply]), "replay/encoder player mismatch"
            arrays = game_to_cpp_arrays(game, played_cards, played_players)

            seen = np.zeros(32, dtype=bool)
            for c in range(32):
                if int(arrays[0][me]) >> c & 1:
                    seen[c] = True
            for c in arrays[3][: arrays[5]]:
                seen[int(c)] = True
            for c in arrays[1][: arrays[6]]:
                seen[int(c)] = True
            unseen = np.where(~seen)[0]

            if len(unseen):
                obs_cards = list(played_cards) + [
                    c.to_index() for _, c in game.current_trick]
                obs_players = list(played_players) + [
                    p for p, _ in game.current_trick]
                particles = mi.sample_determinizations(
                    *arrays, me, n_particles, declarer=declarer)
                logw = plpw_log_weights(model, device, particles, obs_players,
                                        obs_cards, trump, declarer, me)
                w = normalize_weights(lam * logw, n_particles, eps)
                marginals = (
                    _particle_marginals(particles, me),
                    bphi[ply],
                    _particle_marginals(particles, me, w),
                )
                trick = ply // 4
                for c in unseen:
                    slot = (int(holder[c]) - me) % 4
                    assert slot != 0, "unseen card held by observer"
                    onehot = np.eye(3)[slot - 1]
                    for m_i, marg in enumerate(marginals):
                        p = marg[:, c]
                        nll[m_i, trick] += -np.log(p[slot - 1] + 1e-9)
                        brier[m_i, trick] += ((p - onehot) ** 2).sum()
                        top1[m_i, trick] += float(np.argmax(p) == slot - 1)
                    cnt[trick] += 1

            # advance the true deal
            card = Card.from_index(int(actions[ply]))
            pre = list(game.current_trick)
            assert game.play_card(me, card)
            if len(pre) == 3:
                for tp_, tc_ in pre + [(me, card)]:
                    played_cards.append(tc_.to_index())
                    played_players.append(tp_)

    return nll, brier, top1, cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["validate"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="shard directory")
    ap.add_argument("--deals", type=int, default=100)
    ap.add_argument("--particles", type=int, default=64)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = load_model(args.model, args.device)
    deals = load_deals(args.data)
    print(f"E1: marginal log-loss of true hands, {min(args.deals, len(deals))} "
          f"deals, {args.particles} particles, lam={args.lam}, eps={args.eps}")
    nll, brier, top1, cnt = validate_marginals(
        model, args.device, deals, args.particles, args.lam, args.eps,
        args.deals)
    names = ("uniform-particles", "B_phi", "PLPW")
    print("per-trick NLL of true holder (lower is better):")
    for trick in range(8):
        n = max(cnt[trick], 1.0)
        row = "  ".join(f"{names[i]} {nll[i, trick] / n:.4f}" for i in range(3))
        print(f"  trick {trick}: {row}   (n={int(cnt[trick])})")
    total = max(cnt.sum(), 1.0)
    print("overall NLL:   " + "  ".join(
        f"{names[i]} {nll[i].sum() / total:.4f}" for i in range(3)))
    print("overall Brier: " + "  ".join(
        f"{names[i]} {brier[i].sum() / total:.4f}" for i in range(3)))
    print("overall top-1: " + "  ".join(
        f"{names[i]} {top1[i].sum() / total:.4f}" for i in range(3)))


if __name__ == "__main__":
    main()
