"""DMC Q-network agent for the evaluation harness.

Pure blueprint play: one encode_q_observation call, all legal candidates
batch-evaluated by Q_θ(s, a), argmax (ε-greedy only if epsilon > 0, which
evaluations never use). No search, no teacher. Architecture is detected
from the model: QTransformer gets raw-history tokens (built from the same
C++ observation — public information only), QNet gets (obs, act).
"""
import itertools

import numpy as np
import torch

import manille_ismcts as mi
from manille_env.card import Card, permute_card_index, permute_hand_mask
from tournament import Agent, game_to_cpp_arrays

from .dmc_model import QGRU, QTransformer
from .dmc_tokens import encode_live, tokens_batch


class QNetAgent(Agent):
    def __init__(self, model, device="cuda", epsilon=0.0, seed=None,
                 choice_model=None):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.dealer = 0
        self.rng = np.random.default_rng(seed)
        self.choice_model = choice_model

    def choose_trump_suit(self, hand_mask):
        """Free-trump convention (Phase 6): the harness calls this on the
        declarer-seat agent when a choice model is attached; deterministic
        argmax at eval time."""
        from .dmc_trump_choice import suit_values
        return int(np.argmax(
            suit_values(self.choice_model, hand_mask, self.device)))

    def new_game(self, game, dealer, player_id):
        self.dealer = dealer

    def q_values(self, game, player_id, played_cards, played_players):
        """(legal card indices, Q values) for the current decision.

        Split out of choose_card because callers that already pay for this
        forward want the whole vector, not just the argmax — SPARTA's margin
        gate reads the blueprint's top-two gap to decide whether searching
        this ply is worth anything."""
        arrays = game_to_cpp_arrays(game, played_cards, played_players)
        enc = mi.encode_q_observation(*arrays, self.dealer, player_id)
        return enc["legal"][:int(enc["n_legal"])], self._q_from_enc(
            enc, game, player_id, played_cards)

    def _q_from_enc(self, enc, game, player_id, played_cards):
        n = int(enc["n_legal"])
        act = enc["act"][:n]
        with torch.no_grad():
            if isinstance(self.model, QGRU):
                # Candidate-independent trunk once, expanded over candidates.
                n_plays = len(played_cards) + len(game.current_trick)
                raw, n_raw, glob, _ = tokens_batch(enc["obs"][None, :],
                                                   [n_plays])
                last = self.model.trunk(
                    torch.from_numpy(raw).to(self.device),
                    torch.from_numpy(n_raw).to(self.device))
                q = self.model.head(
                    last.expand(n, -1),
                    torch.from_numpy(glob).to(self.device).expand(n, -1),
                    torch.from_numpy(act).to(self.device),
                )
            elif isinstance(self.model, QTransformer):
                own_mask = 0
                for c in game.hands[player_id]:
                    own_mask |= 1 << c.to_index()
                n_plays = len(played_cards) + len(game.current_trick)
                tok = encode_live(enc["obs"], n_plays, own_mask)
                rep = lambda key: np.repeat(tok[key][None, :], n, axis=0)
                q = self.model(
                    torch.from_numpy(rep("raw")).to(self.device),
                    torch.from_numpy(
                        np.full(n, tok["n_raw"], dtype=np.int64)
                    ).to(self.device),
                    torch.from_numpy(rep("glob")).to(self.device),
                    torch.from_numpy(rep("hand")).to(self.device),
                    torch.from_numpy(act).to(self.device),
                )
            else:
                obs = np.repeat(enc["obs"][None, :], n, axis=0)
                q = self.model(
                    torch.from_numpy(obs).to(self.device),
                    torch.from_numpy(act).to(self.device),
                )
        return q.float().cpu().numpy().reshape(-1)

    def choose_card(self, game, player_id, played_cards, played_players) -> Card:
        # Encode before the epsilon draw: the exploration branch samples from
        # enc["legal"], so the RNG stream (and the self-play data it drives)
        # stays identical to before q_values was split out.
        arrays = game_to_cpp_arrays(game, played_cards, played_players)
        enc = mi.encode_q_observation(*arrays, self.dealer, player_id)
        n = int(enc["n_legal"])
        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            return Card.from_index(int(enc["legal"][self.rng.integers(n)]))
        q = self._q_from_enc(enc, game, player_id, played_cards)
        return Card.from_index(int(enc["legal"][int(np.argmax(q))]))


class PolicyHeadAgent(QNetAgent):
    """Greedy residual policy on an immutable QGRU evaluator."""

    def __init__(self, model, policy_head, tau, **kwargs):
        super().__init__(model, **kwargs)
        if not isinstance(model, QGRU):
            raise ValueError("PolicyHeadAgent currently requires QGRU")
        self.policy_head = policy_head
        self.tau = float(tau)

    def policy_values(self, game, player_id, played_cards, played_players):
        arrays = game_to_cpp_arrays(game, played_cards, played_players)
        enc = mi.encode_q_observation(*arrays, self.dealer, player_id)
        n = int(enc["n_legal"])
        raw, n_raw, glob, _ = tokens_batch(
            enc["obs"][None, :],
            [len(played_cards) + len(game.current_trick)])
        act = torch.from_numpy(enc["act"][:n]).to(self.device)
        with torch.no_grad():
            last = self.model.trunk(
                torch.from_numpy(raw).to(self.device),
                torch.from_numpy(n_raw).to(self.device))
            last = last.expand(n, -1)
            glob_t = torch.from_numpy(glob).to(self.device).expand(n, -1)
            q = self.model.head(last, glob_t, act)
            logits = self.policy_head.logits(
                q, last, glob_t, act, self.tau)
        return enc["legal"][:n], logits.float().cpu().numpy()

    def choose_card(self, game, player_id, played_cards, played_players):
        legal, logits = self.policy_values(
            game, player_id, played_cards, played_players)
        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            return Card.from_index(int(legal[self.rng.integers(len(legal))]))
        return Card.from_index(int(legal[int(np.argmax(logits))]))


def _permute_arrays(arrays, perm):
    """Relabel every card-bearing field of a game_to_cpp_arrays tuple.

    All of them, together: hands, the in-progress trick, the played history
    and the trump index. A half-permuted state is a legal-looking state that
    describes a different game, so this deliberately rebuilds the whole
    tuple rather than mutating parts of it."""
    (hands, trick_cards, trick_players, pc, pp, n_played, n_trick,
     current_player, trump, *rest) = arrays

    ph = np.array([permute_hand_mask(int(m), perm) for m in hands],
                  dtype=np.int64)
    def pcards(a):
        out = np.array(a, dtype=np.int32, copy=True)
        hit = out >= 0                       # -1 is padding, not a card
        out[hit] = [permute_card_index(int(c), perm) for c in out[hit]]
        return out

    return (ph, pcards(trick_cards), trick_players, pcards(pc), pp,
            n_played, n_trick, current_player,
            trump if trump < 0 else perm[int(trump)], *rest)


def aggregate_orbit_q(q_views, method="mean"):
    """Combine aligned [views, legal-actions] orbit predictions.

    ``mean`` is the shipped projection.  The alternatives are deliberately
    inference-only experimental knobs: median rejects one badly calibrated
    view, zscore gives each view equal within-state scale, and rank keeps
    only pairwise action ordering.  All are permutation-equivariant once the
    candidate columns have been aligned.
    """
    q_views = np.asarray(q_views, dtype=np.float64)
    if q_views.ndim != 2 or q_views.shape[0] == 0:
        raise ValueError("q_views must be a non-empty [views, actions] array")
    if method == "mean":
        return q_views.mean(axis=0)
    if method == "median":
        return np.median(q_views, axis=0)
    if method == "zscore":
        centered = q_views - q_views.mean(axis=1, keepdims=True)
        scale = q_views.std(axis=1, keepdims=True)
        # A constant view contains no preference signal.  Keeping its row at
        # zero avoids turning fp noise into a vote.
        normalized = np.divide(centered, scale, out=np.zeros_like(centered),
                               where=scale > 1e-8)
        return normalized.mean(axis=0)
    if method == "rank":
        # Mean pairwise win score (ties contribute 1/2) is a Borda-style
        # rank aggregate without candidate-index tie breaking.
        wins = (q_views[:, :, None] > q_views[:, None, :]).sum(axis=2)
        ties = (q_views[:, :, None] == q_views[:, None, :]).sum(axis=2)
        return (wins + 0.5 * ties).mean(axis=0)
    raise ValueError(f"unknown orbit aggregation: {method}")


class SymmetrizedQNetAgent(QNetAgent):
    """QNetAgent with the suit symmetry averaged in at test time.

    Suit relabelling is an exact automorphism, so Q(sigma(s), sigma(a)) ==
    Q(s, a) must hold and the blueprint badly violates it: 21.5% of
    non-forced plies change their argmax under a pure relabelling
    (scripts/suit_symmetry_probe.py). This averages Q over the relabellings
    that FIX trump -- 6 of them, since trump is a distinguished suit and
    moving it would change the game -- which restores the invariance exactly
    at 6x the forward cost, with no retraining.

    Fixing trump is NOT sufficient to stay on-distribution, which is the
    trap here. Trump is not a free label: these nets only ever see states
    where trump came from `apply_trump_rule`, and that rule breaks ties on
    the LOWEST suit index. So a permutation can carry a suit that tied with
    trump down past it and make the rule pick something else -- measured at
    1.59% of hands and 0.72% of trump-fixing orbit elements, concentrated
    entirely in hearts and clubs (diamonds and spades are immune, being the
    extreme indices). A permutation is safe exactly when it maps
    {s : s < trump} and {s : s > trump} each to itself, since only a
    lower-index suit can steal the pick. That leaves 6 relabellings for
    diamonds and spades but only 2 for hearts and clubs.

    An agent cannot check the sharper per-deal condition: it would need the
    declarer's hand, which is hidden. Training-time augmentation can and
    does (`dmc_replay.augmentation_perms` asks the engine's own rule), which
    is one reason the augmented net should beat this wrapper.

    `full_orbit=True` restores the naive 6 for A/B measurement. It is what
    the first gate ran, so its +0.55 is an orbit-ensemble result that
    includes some off-distribution states -- a real measurement, but not the
    exact projection it was first described as.
    """

    def __init__(self, *args, full_orbit=False, declarer_aware=False,
                 aggregation="mean",
                 **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.model, QTransformer):
            # Its token path rebuilds own_mask from game.hands, which this
            # wrapper does not permute -- it would silently mix a permuted
            # observation with an unpermuted hand.
            raise NotImplementedError(
                "SymmetrizedQNetAgent does not support QTransformer")
        self.full_orbit = full_orbit
        self.declarer_aware = declarer_aware
        if aggregation not in {"mean", "median", "zscore", "rank"}:
            raise ValueError(f"unknown orbit aggregation: {aggregation}")
        self.aggregation = aggregation
        self.initial_hand_mask = None

    def new_game(self, game, dealer, player_id):
        super().new_game(game, dealer, player_id)
        self.initial_hand_mask = 0
        for card in game.hands[player_id]:
            self.initial_hand_mask |= 1 << card.to_index()

    def _perms(self, trump):
        trump = int(trump)
        if trump < 0:                      # mull: no trump, no rule, all 24
            return list(itertools.permutations(range(4)))
        keep = [p for p in itertools.permutations(range(4))
                if p[trump] == trump]
        if self.full_orbit:
            return keep
        return [p for p in keep
                if all((p[s] < trump) == (s < trump) for s in range(4))]

    def _decision_perms(self, trump, player_id):
        """Safe views available at this information set.

        Everyone can use the public, order-preserving orbit. The declarer
        additionally knows the exact initial hand that selected trump, so it
        can retain every trump-fixing relabelling for which the engine's rule
        still selects that trump. No private information from another seat is
        consulted.
        """
        if (not self.declarer_aware or trump < 0 or player_id != self.dealer
                or self.initial_hand_mask is None):
            return self._perms(trump)
        keep = [p for p in itertools.permutations(range(4))
                if p[trump] == trump]
        return [p for p in keep if mi.apply_trump_rule(
            permute_hand_mask(self.initial_hand_mask, p)) == trump]

    def q_values(self, game, player_id, played_cards, played_players):
        arrays = game_to_cpp_arrays(game, played_cards, played_players)
        base = mi.encode_q_observation(*arrays, self.dealer, player_id)
        n = int(base["n_legal"])
        legal = np.array(base["legal"][:n], dtype=np.int64)
        perms = self._decision_perms(int(arrays[8]), player_id)
        encs = [mi.encode_q_observation(*_permute_arrays(arrays, perm),
                                        self.dealer, player_id)
                for perm in perms]

        # Evaluate every orbit view in one device call. Sequential forwards
        # pay the GRU trunk and accelerator launch overhead 2--6 times even
        # though the whole orbit is available up front. Rows are view-major,
        # with all legal candidates for one view contiguous.
        acts = np.concatenate([enc["act"][:n] for enc in encs], axis=0)
        with torch.no_grad():
            if isinstance(self.model, QGRU):
                n_plays = len(played_cards) + len(game.current_trick)
                obs = np.stack([enc["obs"] for enc in encs], axis=0)
                raw, n_raw, glob, _ = tokens_batch(
                    obs, [n_plays] * len(encs))
                last = self.model.trunk(
                    torch.from_numpy(raw).to(self.device),
                    torch.from_numpy(n_raw).to(self.device))
                q_all = self.model.head(
                    last.repeat_interleave(n, dim=0),
                    torch.from_numpy(glob).to(self.device)
                    .repeat_interleave(n, dim=0),
                    torch.from_numpy(acts).to(self.device))
            else:
                obs = np.repeat(
                    np.stack([enc["obs"] for enc in encs], axis=0),
                    n, axis=0)
                q_all = self.model(
                    torch.from_numpy(obs).to(self.device),
                    torch.from_numpy(acts).to(self.device))
        q_views = q_all.float().cpu().numpy().reshape(len(encs), n)

        aligned = np.empty((len(perms), n), dtype=np.float64)
        for view, (perm, enc, q) in enumerate(zip(perms, encs, q_views)):
            # Same candidates, different labels and order: index by the
            # permuted identity of each base candidate rather than assuming
            # the encoder emits them in a matching order.
            plegal = list(enc["legal"][:int(enc["n_legal"])])
            assert len(plegal) == n, "permutation changed the legal set"
            for j, card in enumerate(legal):
                aligned[view, j] = q[
                    plegal.index(permute_card_index(int(card), perm))]
        return legal, aggregate_orbit_q(aligned, self.aggregation)

    def choose_card(self, game, player_id, played_cards, played_players):
        legal, q = self.q_values(game, player_id, played_cards,
                                 played_players)
        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            return Card.from_index(int(legal[self.rng.integers(len(legal))]))
        return Card.from_index(int(legal[int(np.argmax(q))]))
