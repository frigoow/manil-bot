"""Endgame hybrid agent (manille_dmc_plan §5).

Blueprint play until trick `threshold` (default 6), then exact search: sample
consistent worlds from the hard-constraint sampler, weight them by the DMC
belief net's per-card log-probs, and for each legal candidate take the
weighted mean of the world's full-information exact minimax value
(manille_ismcts.exact_solve). This patches the NN's structural endgame leak
while keeping the pure-NN blueprint as the benchmarked artifact — the hybrid
is an eval-time agent only, never used in training data.
"""
import numpy as np
import torch

import manille_ismcts as mi
from manille_env.card import Card
from tournament import Agent, game_to_cpp_arrays

from .dmc_agent import QNetAgent
from .model import belief_probs, split_belief_inputs


def dmc_belief_log_weights(belief_model, device, obs, particles, arrays,
                           player_id):
    """Per-particle log-weights from the DMC BeliefNet's marginals:
    sum over unseen cards of log P(holder = the seat the particle assigns).
    Shared by the hybrid (exact endgame) and SPARTA (world weighting)."""
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        offsets = belief_model(obs_t)
        prior, mask = split_belief_inputs(obs_t)
        probs = belief_probs(offsets, prior, mask)
    log_probs = np.log(probs.squeeze(0).cpu().numpy() + 1e-9)

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
    logw = np.zeros(particles.shape[0])
    for r in range(3):
        seat = (player_id + r + 1) % 4
        bits = (particles[:, seat][:, None] >> cards[None, :]) & 1
        logw += (bits * log_probs[r][None, :] * unseen[None, :]).sum(axis=1)
    return logw


def dmc_belief_weighted_particles(belief_model, device, obs, arrays,
                                  player_id, n_particles, declarer):
    """sample_determinizations weighted by the DMC BeliefNet's marginals
    (same scheme as agents.belief_weighted_particles, but the DMC belief
    net's forward returns offsets only). `declarer` activates the
    trump-rule constraint in the sampler."""
    particles = mi.sample_determinizations(*arrays, player_id, n_particles,
                                           declarer=declarer)
    logw = dmc_belief_log_weights(belief_model, device, obs, particles,
                                  arrays, player_id)
    logw -= logw.max()
    weights = np.exp(logw)
    if not np.isfinite(weights).all() or weights.sum() <= 1e-12:
        weights = np.ones(n_particles)  # degenerate: uniform fallback
    return particles, weights


class EndgameHybridAgent(Agent):
    """Blueprint outside the endgame; belief-weighted exact search at
    trick >= threshold."""

    def __init__(self, q_model, belief_model, device="cuda", threshold=6,
                 n_particles=128):
        self.blueprint = QNetAgent(q_model, device)
        self.belief_model = belief_model
        self.device = device
        self.threshold = threshold
        self.n_particles = n_particles
        self.dealer = 0

    def new_game(self, game, dealer, player_id):
        self.dealer = dealer
        self.blueprint.new_game(game, dealer, player_id)

    def choose_card(self, game, player_id, played_cards, played_players) -> Card:
        trick = (len(played_cards) + len(game.current_trick)) // 4
        if trick < self.threshold:
            return self.blueprint.choose_card(game, player_id, played_cards,
                                              played_players)
        legal = game.get_legal_moves(player_id)
        if len(legal) <= 1:
            return legal[0]

        arrays = game_to_cpp_arrays(game, played_cards, played_players)
        obs = mi.encode_observation(*arrays, self.dealer, player_id)
        particles, weights = dmc_belief_weighted_particles(
            self.belief_model, self.device, obs, arrays, player_id,
            self.n_particles, self.dealer)

        my_team = player_id % 2
        acc = {}
        for k in range(len(particles)):
            res = mi.exact_solve(*arrays, self.dealer, particles[k])
            for mv, v in zip(res["moves"], res["pts0"]):
                pts_mine = v if my_team == 0 else 60.0 - v
                acc[int(mv)] = acc.get(int(mv), 0.0) + weights[k] * pts_mine
        best = max(acc, key=acc.get)
        return Card.from_index(best)
