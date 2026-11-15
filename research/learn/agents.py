"""NN-backed agents for the evaluation harness.

All agents follow the tournament.py Agent protocol and play through the
Python engine; the C++ encoder produces their observations and the C++
trump rule fixes the trump distribution (set by the harness, not here).
"""
import numpy as np
import torch

import manille_ismcts as mi
from manille_env.card import Card
from tournament import Agent, game_to_cpp_arrays

from .model import belief_probs, split_belief_inputs


def encode_current(game, player_id, played_cards, played_players, declarer):
    """Encode the current state from `player_id`'s perspective."""
    (hands, tc, tp, pc, pp, n_played, n_trick, cp, trump,
     pts0, pts1, tw0, tw1) = game_to_cpp_arrays(game, played_cards, played_players)
    return mi.encode_observation(
        hands, tc, tp, pc, pp, n_played, n_trick, cp, trump,
        pts0, pts1, tw0, tw1, declarer, player_id,
    )


def make_policy_value_fn(model, device, temperature=1.0):
    """Batched callback for mcts::BatchedISMCTS: obs [N, OBS_DIM] ->
    (legal-masked policy probs [N, 32], values [N])."""
    from .encoding import LEGAL_MASK_OFF

    def fn(obs):
        with torch.no_grad():
            t = torch.from_numpy(obs).to(device)
            logits, values, _ = model(t)
            legal = t[:, LEGAL_MASK_OFF : LEGAL_MASK_OFF + 32]
            logits = logits.float().masked_fill(legal < 0.5, -1e9)
            if temperature != 1.0:
                logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy(), values.float().cpu().numpy()

    return fn


def belief_weighted_particles(model, device, obs, arrays, player_id,
                              n_particles, declarer):
    """Sample candidate worlds from the hard-constraint sampler and weight
    them by B_phi marginals (the M1a-validated determinization scheme).
    `declarer` activates the trump-rule constraint in the sampler."""
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        _, _, offsets = model(obs_t)
        prior, mask = split_belief_inputs(obs_t)
        probs = belief_probs(offsets, prior, mask)
    log_probs = np.log(probs.squeeze(0).cpu().numpy() + 1e-9)

    particles = mi.sample_determinizations(*arrays, player_id, n_particles,
                                           declarer=declarer)

    seen = np.zeros(32, dtype=bool)
    hand_mask = int(arrays[0][player_id])
    for c in range(32):
        if hand_mask & (1 << c):
            seen[c] = True
    for c in arrays[3][: arrays[5]]:
        seen[int(c)] = True
    for c in arrays[1][: arrays[6]]:
        seen[int(c)] = True
    unseen = ~seen

    cards = np.arange(32)
    logw = np.zeros(n_particles)
    for r in range(3):
        seat = (player_id + r + 1) % 4
        bits = (particles[:, seat][:, None] >> cards[None, :]) & 1  # [n, 32]
        logw += (bits * log_probs[r][None, :] * unseen[None, :]).sum(axis=1)
    logw -= logw.max()
    weights = np.exp(logw)
    if not np.isfinite(weights).all() or weights.sum() <= 1e-12:
        weights = np.ones(n_particles)  # degenerate: fall back to uniform
    return particles, weights


class FastNNAgent(Agent):
    """pi_theta forward pass only, no search. The 'fast agent' deployment mode."""

    def __init__(self, model, device="cuda", temperature=0.0):
        self.model = model
        self.device = device
        self.temperature = temperature
        self.dealer = 0

    def new_game(self, game, dealer, player_id):
        self.dealer = dealer

    def choose_card(self, game, player_id, played_cards, played_players) -> Card:
        obs = encode_current(game, player_id, played_cards, played_players, self.dealer)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            logits, _, _ = self.model(obs_t)
            logits = logits.squeeze(0).cpu().numpy()

        legal = game.get_legal_moves(player_id)
        legal_idx = np.array([c.to_index() for c in legal])
        masked = logits[legal_idx]
        if self.temperature > 0:
            z = masked / self.temperature
            probs = np.exp(z - z.max())
            probs /= probs.sum()
            chosen = int(np.random.choice(legal_idx, p=probs))
        else:
            chosen = int(legal_idx[np.argmax(masked)])
        return Card.from_index(chosen)


class BeliefGuidedISMCTSAgent(Agent):
    """ISMCTS with determinizations importance-weighted by B_phi marginals.

    Candidate worlds come from the C++ hard-constraint sampler; each candidate
    is weighted by the belief model's per-card probability of the assigned
    holder, then passed to the solver through the particle-pool hook.
    """

    def __init__(self, model, solver_type="SO", iterations=150,
                 n_particles=128, device="cuda"):
        if solver_type != "SO":
            raise ValueError("BeliefGuided agent supports SO only")
        self.model = model
        self.device = device
        self.iterations = iterations
        self.n_particles = n_particles
        self.dealer = 0
        self.solver = mi.ManilleSolverSO(iterations=iterations, perfect_info=False)

    def new_game(self, game, dealer, player_id):
        self.dealer = dealer

    def _belief(self, obs):
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            _, _, offsets = self.model(obs_t)
            prior, mask = split_belief_inputs(obs_t)
            probs = belief_probs(offsets, prior, mask)
        return probs.squeeze(0).cpu().numpy()  # [3, 32]

    def _particle_weights(self, particles, seen_mask, log_probs):
        """log-weights: sum over unseen cards of log P(assigned holder)."""
        n = particles.shape[0]
        cards = np.arange(32)
        unseen = ~seen_mask  # [32] bool
        logw = np.zeros(n)
        for k in range(n):
            logw_k = 0.0
            for r in range(3):
                seat_mask = (int(particles[k][(self._obs + r + 1) % 4]) >> cards) & 1
                logw_k += (seat_mask * log_probs[r] * unseen).sum()
            logw[k] = logw_k
        logw -= logw.max()
        weights = np.exp(logw)
        if not np.isfinite(weights).all() or weights.sum() <= 1e-12:
            return np.ones(n)  # degenerate: fall back to uniform
        return weights

    def choose_card(self, game, player_id, played_cards, played_players) -> Card:
        arrays = game_to_cpp_arrays(game, played_cards, played_players)
        obs = encode_current(game, player_id, played_cards, played_players, self.dealer)
        log_probs = np.log(self._belief(obs) + 1e-9)
        self._obs = player_id

        particles = np.stack([
            mi.sample_determinization(*arrays, player_id,
                                      declarer=self.dealer)
            for _ in range(self.n_particles)
        ])

        seen = np.zeros(32, dtype=bool)
        hand_mask = int(arrays[0][player_id])
        for c in range(32):
            if hand_mask & (1 << c):
                seen[c] = True
        for c in arrays[3][: arrays[5]]:
            seen[int(c)] = True
        for c in arrays[1][: arrays[6]]:
            seen[int(c)] = True

        weights = self._particle_weights(particles, seen, log_probs)

        result = self.solver.solve(
            *arrays, particles=particles, particle_weights=weights
        )
        best_idx = int(result["best_move"])
        for card in game.get_legal_moves(player_id):
            if card.to_index() == best_idx:
                return card
        return game.get_legal_moves(player_id)[0]


class PLPWGuidedISMCTSAgent(Agent):
    """ISMCTS with determinizations weighted by the computed policy-likelihood
    inverse (PLPW — SPARTA/Skat-PI, see learn/inference.py) instead of the
    learned B_phi marginals. Same particle-pool hook, same reference solver;
    only the weight computation differs.

    lam: likelihood temperature (1 = full inverse, ->0 = uniform).
    eps: uniform floor mixed into the weights (over-sharpness guard).
    """

    def __init__(self, model, iterations=150, n_particles=64, device="cpu",
                 lam=1.0, eps=0.0):
        self.model = model
        self.device = device
        self.iterations = iterations
        self.n_particles = n_particles
        self.lam = lam
        self.eps = eps
        self.dealer = 0
        self.solver = mi.ManilleSolverSO(iterations=iterations, perfect_info=False)

    def new_game(self, game, dealer, player_id):
        self.dealer = dealer

    def choose_card(self, game, player_id, played_cards, played_players) -> Card:
        from .inference import plpw_weights

        arrays = game_to_cpp_arrays(game, played_cards, played_players)
        particles, weights = plpw_weights(
            self.model, self.device, arrays, game, player_id,
            played_cards, played_players, self.dealer,
            self.n_particles, self.lam, self.eps)
        result = self.solver.solve(
            *arrays, particles=particles, particle_weights=weights
        )
        best_idx = int(result["best_move"])
        for card in game.get_legal_moves(player_id):
            if card.to_index() == best_idx:
                return card
        return game.get_legal_moves(player_id)[0]


class BatchedNNAgent(Agent):
    """The Phase-2 ExIt actor: ISMCTS with B_phi-weighted determinizations
    and network leaf evaluation (pi rollouts or v leaf eval), run through the
    wave-batched solver.

    With jit_path set (strength_program.md C1), inference runs in-process via
    libtorch (NNPolicyValue + solve_nn, no Python callback): ~16x more sims
    per second, which is what makes the 0.5 s wall-clock regime meaningful.
    """

    def __init__(self, model, sims=150, wave=8, mode="rollout",
                 n_particles=128, device="cuda", temperature=1.0,
                 exploration=0.7, puct=False, c_puct=2.0,
                 dirichlet_epsilon=0.0, jit_path=None, partner_belief=False):
        self.model = model
        self.device = device
        self.sims = sims
        self.n_particles = n_particles
        self.dealer = 0
        self.solver = mi.BatchedISMCTS(sims=sims, wave=wave, mode=mode,
                                       puct=puct, exploration=exploration,
                                       c_puct=c_puct,
                                       dirichlet_epsilon=dirichlet_epsilon,
                                       partner_belief=partner_belief)
        self.nn = mi.NNPolicyValue(jit_path, device) if jit_path else None
        self.pv_fn = None if self.nn else make_policy_value_fn(
            model, device, temperature)

    def new_game(self, game, dealer, player_id):
        self.dealer = dealer

    def choose_card(self, game, player_id, played_cards, played_players) -> Card:
        arrays = game_to_cpp_arrays(game, played_cards, played_players)
        particles, weights = None, None
        if self.n_particles > 0:
            obs = encode_current(game, player_id, played_cards, played_players,
                                 self.dealer)
            particles, weights = belief_weighted_particles(
                self.model, self.device, obs, arrays, player_id,
                self.n_particles, self.dealer)
        if self.nn is not None:
            result = self.solver.solve_nn(
                *arrays, self.dealer, particles=particles,
                particle_weights=weights, nn=self.nn,
            )
        else:
            result = self.solver.solve(
                *arrays, self.dealer, particles=particles,
                particle_weights=weights, policy_value_fn=self.pv_fn,
            )
        best_idx = int(result["best_move"])
        for card in game.get_legal_moves(player_id):
            if card.to_index() == best_idx:
                return card
        return game.get_legal_moves(player_id)[0]
