"""Tier-2 SPARTA agent for Manille (sparta_manille_plan.md §5).

One-ply search in SPARTA's own image: at each decision, sample PLPW worlds
(the computed policy-likelihood inverse, learn/inference.py); for each legal
action, roll every world out with all four seats following the blueprint
policy; Q(a) = the weighted mean terminal team points. Play the blueprint
action unless the best search action beats it by more than `delta` points —
SPARTA's deviation threshold, which makes a noisy Q-estimate safe to act on.

Variance reduction from the paper: the same worlds are used across actions
(paired comparisons — the trick duplicate-deal eval uses), and each world's
rollout reuses the same RNG stream across actions.

Deliberately simple first cut: fixed allocation (one rollout per world per
action) instead of SPARTA's UCB allocation with 2-sigma pruning — that is an
efficiency refinement, not a correctness one, and belongs behind a profile.
Honest framing (plan §2): against adversarial opponents this is a best
response to an opponent model, not a guaranteed improvement.
"""
import numpy as np
import torch
import zlib

import manille_ismcts as mi
from manille_env.card import Card, Suit
from manille_env.game_logic import GamePhase, ManilleGame
from tournament import Agent, game_to_cpp_arrays

from .agents import encode_current
from .encoding import LEGAL_MASK_OFF
from .inference import plpw_weights


def inject_world(dealer, trump, particle, current_trick, hand_points,
                 tricks_won, current_player):
    """A fresh Python game set to the mid-deal state of one particle world
    (the particle's current holdings; public trick/score state)."""
    game = ManilleGame(dealer=dealer)
    game.hands = [
        [Card.from_index(c) for c in range(32) if int(mask) >> c & 1]
        for mask in particle
    ]
    # trump == -1 is mull (no trump): the engine models it as trump_suit
    # None, and Suit(-1) would throw.
    game.select_trump(Suit(trump) if trump >= 0 else None)
    game.start_playing()
    game.current_trick = list(current_trick)
    game.hand_points = list(hand_points)
    game.tricks_won = list(tricks_won)
    game.current_player = current_player
    return game


def blueprint_rollouts(worlds, declarer, model, device, rngs,
                       temperature=1.0):
    """Play each world out with all seats sampling from the blueprint.
    worlds: list of (game, played_cards, played_players), mutated in place.
    temperature < 0.5-ish sharpens the rollout policy (R11: T=1 sampling
    adds Q-variance vs SPARTA's deterministic blueprint); 0 = argmax.
    Returns the final hand_points [K, 2]."""
    active = [k for k in range(len(worlds))
              if worlds[k][0].phase == GamePhase.PLAYING]
    while active:
        obs_batch = []
        for k in active:
            game, pc, pp = worlds[k]
            arrays = game_to_cpp_arrays(game, pc, pp)
            obs_batch.append(mi.encode_observation(
                *arrays, declarer, game.current_player))
        batch = np.stack(obs_batch)
        with torch.no_grad():
            t = torch.from_numpy(batch).to(device)
            logits, _, _ = model(t)
            legal = t[:, LEGAL_MASK_OFF : LEGAL_MASK_OFF + 32]
            logits = logits.float().masked_fill(legal < 0.5, -1e9)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1
                                      ).cpu().numpy()
            else:
                probs = None
                argmax = logits.argmax(dim=-1).cpu().numpy()
        for j, k in enumerate(active):
            game, pc, pp = worlds[k]
            pid = game.current_player
            if probs is None:
                card = Card.from_index(int(argmax[j]))
            else:
                p = probs[j]
                p = p / p.sum()
                card = Card.from_index(int(rngs[k].choice(32, p=p)))
            pre = list(game.current_trick)
            if not game.play_card(pid, card):
                raise ValueError("blueprint rollout played an illegal card")
            if len(pre) == 3:
                for tp_, tc_ in pre + [(pid, card)]:
                    pc.append(tc_.to_index())
                    pp.append(tp_)
        active = [k for k in active
                  if worlds[k][0].phase == GamePhase.PLAYING]
    return np.array([w[0].hand_points for w in worlds], dtype=np.float64)


class SpartaAgent(Agent):
    """1-ply SPARTA search over PLPW worlds with a deviation threshold."""

    def __init__(self, model, n_particles=64, device="cpu", lam=1.0, eps=0.0,
                 delta=0.6, temperature=1.0, seed=0):
        self.model = model
        self.device = device
        self.n_particles = n_particles
        self.lam = lam
        self.eps = eps
        self.delta = delta
        self.temperature = temperature
        self.seed = seed
        self.dealer = 0

    def new_game(self, game, dealer, player_id):
        self.dealer = dealer

    def _blueprint_action(self, game, player_id, played_cards, played_players):
        obs = encode_current(game, player_id, played_cards, played_players,
                             self.dealer)
        with torch.no_grad():
            t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            logits, _, _ = self.model(t)
            logits = logits.squeeze(0).cpu().numpy()
        legal = game.get_legal_moves(player_id)
        legal_idx = np.array([c.to_index() for c in legal])
        return Card.from_index(int(legal_idx[np.argmax(logits[legal_idx])]))

    def choose_card(self, game, player_id, played_cards, played_players) -> Card:
        legal = game.get_legal_moves(player_id)
        if len(legal) == 1:
            return legal[0]

        a_bp = self._blueprint_action(game, player_id, played_cards,
                                      played_players)
        arrays = game_to_cpp_arrays(game, played_cards, played_players)
        particles, weights = plpw_weights(
            self.model, self.device, arrays, game, player_id,
            played_cards, played_players, self.dealer,
            self.n_particles, self.lam, self.eps)
        trump = int(arrays[8])
        my_team = player_id % 2

        q = {}
        # R11: pair luck within a decision (same streams across action arms)
        # but decorrelate across decisions — mix the observed-card context
        # into the seed so every decision doesn't reuse the same 64 streams.
        ctx = zlib.crc32(np.array(
            list(played_cards) + [c.to_index() for _, c in game.current_trick],
            dtype=np.int32).tobytes())
        base_seed = (self.seed * 1000003 + ctx) % (2**63 - 1)
        for card in legal:
            worlds = []
            # Paired rollouts: world k reuses the same RNG stream in every
            # action arm, so action comparisons share luck.
            rngs = [np.random.default_rng(base_seed + k)
                    for k in range(self.n_particles)]
            for k in range(self.n_particles):
                w_game = inject_world(
                    self.dealer, trump, particles[k], game.current_trick,
                    game.hand_points, game.tricks_won, game.current_player)
                pc_w, pp_w = list(played_cards), list(played_players)
                pre = list(w_game.current_trick)
                # My hand is identical in every particle, so `card` is legal.
                if not w_game.play_card(player_id, card):
                    raise ValueError("candidate action illegal in a world "
                                     "whose hand matches mine — injector bug")
                if len(pre) == 3:
                    for tp_, tc_ in pre + [(player_id, card)]:
                        pc_w.append(tc_.to_index())
                        pp_w.append(tp_)
                worlds.append((w_game, pc_w, pp_w))
            pts = blueprint_rollouts(worlds, self.dealer, self.model,
                                     self.device, rngs, self.temperature)
            q[card] = float((pts[:, my_team] * weights).sum() / weights.sum())

        a_star = max(q, key=q.get)
        if q[a_star] - q.get(a_bp, -1e9) > self.delta:
            return a_star
        return a_bp
