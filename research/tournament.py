"""
Tournament runner for Manille agents.

Runs duplicate-deal matchups between agents: every unique deal is played four
times with rotated hand assignments, so each seat receives each hand once and
deal luck cancels between the two teams. This is the variance-reduction
protocol the evaluation harness is built on.

Usage:
    python tournament.py --games 200 --iterations 1000
    python tournament.py --games 100 --iterations 500 --matchups SO_vs_random

Available matchup names: SO_vs_random, MO_vs_random, SO_vs_MO, all
"""
import argparse
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from manille_env.card import Card
from manille_env.game_logic import ManilleGame, GamePhase

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("tournament")


# ---------------------------------------------------------------------------
# C++ state conversion
# ---------------------------------------------------------------------------

def game_to_cpp_arrays(
    game: ManilleGame,
    played_cards: List[int],
    played_players: List[int],
) -> Tuple:
    """
    Convert a Python ManilleGame + completed-trick history to C++ solver inputs.

    ManilleGame doesn't store the full per-card play history, so the tournament
    runner maintains completed trick history in `played_cards` and
    `played_players` separately. The current in-progress trick stays in
    `game.current_trick`.

    Returns:
        (hands, trick_cards, trick_players, pc, pp, n_played,
         n_trick, current_player, trump_suit,
         pts0, pts1, tricks_won0, tricks_won1)
    """
    hands = np.zeros(4, dtype=np.int64)
    for p in range(4):
        for card in game.hands[p]:
            hands[p] |= np.int64(1) << card.to_index()

    trick_cards = np.full(4, -1, dtype=np.int32)
    trick_players = np.full(4, -1, dtype=np.int32)
    for i, (pid, card) in enumerate(game.current_trick):
        trick_cards[i] = card.to_index()
        trick_players[i] = pid

    n_played = len(played_cards)
    pc = np.array(played_cards + [-1] * (32 - n_played), dtype=np.int32)
    pp = np.array(played_players + [-1] * (32 - n_played), dtype=np.int32)

    trump = game.trump_suit.value if game.trump_suit is not None else -1

    return (
        hands, trick_cards, trick_players, pc, pp, n_played,
        len(game.current_trick), game.current_player, trump,
        game.hand_points[0], game.hand_points[1],
        game.tricks_won[0], game.tricks_won[1],
    )


# ---------------------------------------------------------------------------
# Agent base
# ---------------------------------------------------------------------------

class Agent:
    """
    All agents implement:
      new_game(game, dealer, player_id)     — called once per hand after trump is selected
      on_card_played(game, player, card_idx, played_cards, played_players)
          — called after every card play, with the updated public state
      choose_card(game, player_id, played_cards, played_players) -> Card
    """

    def new_game(self, game: ManilleGame, dealer: int, player_id: int) -> None:
        pass

    def on_card_played(
        self,
        game: ManilleGame,
        player: int,
        card_idx: int,
        played_cards: List[int],
        played_players: List[int],
    ) -> None:
        pass

    def choose_card(
        self,
        game: ManilleGame,
        player_id: int,
        played_cards: List[int],
        played_players: List[int],
    ) -> Card:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Random agent
# ---------------------------------------------------------------------------

class RandomAgent(Agent):
    """Picks a uniformly random legal card. The sanity-floor baseline."""

    def choose_card(self, game, player_id, played_cards, played_players) -> Card:
        legal = game.get_legal_moves(player_id)
        return random.choice(legal)


# ---------------------------------------------------------------------------
# ISMCTS agent
# ---------------------------------------------------------------------------

class ISMCTSAgent(Agent):
    """
    ISMCTS agent backed by the C++ solver.

    Determinizations come from the C++ hard-constraint sampler. The solver also
    accepts `particles` / `particle_weights`, which override that sampler with a
    weighted pool of candidate worlds — the hook for feeding learned beliefs
    into determinization.

    Args:
        solver_type: 'SO' (single-observer) or 'MO' (multi-observer)
        iterations:  MCTS simulation budget per move
        perfect_info: if True, the solver sees all hands (cheating bound —
            0b in strength_program.md; eval-only, never for training data)
        partner_transparent: if True, determinizations fix the partner's true
            hand and randomize only opponents (0b.2 signaling-ceiling bound;
            eval-only)
    """

    def __init__(self, solver_type: str = 'SO', iterations: int = 1000,
                 perfect_info: bool = False,
                 partner_transparent: bool = False):
        from manille_ismcts import ManilleSolverSO, ManilleSolverMO

        if solver_type not in {'SO', 'MO'}:
            raise ValueError(f"Unknown solver_type {solver_type!r}. Use 'SO' or 'MO'.")

        self.iterations = iterations
        self.solver_type = solver_type
        self.partner_transparent = partner_transparent
        self.solver = (ManilleSolverSO if solver_type == 'SO' else ManilleSolverMO)(
            iterations=iterations, perfect_info=perfect_info,
        )

    def choose_card(
        self,
        game: ManilleGame,
        player_id: int,
        played_cards: List[int],
        played_players: List[int],
    ) -> Card:
        (hands, tc, tp, pc, pp, n_played, n_trick, cp, trump,
         pts0, pts1, tw0, tw1) = game_to_cpp_arrays(game, played_cards, played_players)

        result = self.solver.solve(
            hands, tc, tp, pc, pp, n_played, n_trick, cp, trump,
            pts0, pts1, tw0, tw1,
            partner_transparent=self.partner_transparent,
        )
        best_idx = int(result['best_move'])

        legal = game.get_legal_moves(player_id)
        for card in legal:
            if card.to_index() == best_idx:
                return card

        # Fallback (should not happen if solver is correct)
        logger.warning(
            "Solver best_move %d not in legal moves for p%d; falling back to random.",
            best_idx, player_id,
        )
        return random.choice(legal)


class TimeBoxedISMCTSAgent(Agent):
    """Root-parallel plain ISMCTS under a wall-clock budget per move — the
    0.5 s champion baseline (strength_program.md Phase B). Single game, all
    cores available; the C++ solver auto-tunes iterations to the budget."""

    def __init__(self, budget_ms: int = 500, threads: int = 15):
        from manille_ismcts import ManilleSolverTime

        self.solver = ManilleSolverTime(threads=threads, budget_ms=budget_ms)

    def choose_card(
        self,
        game: ManilleGame,
        player_id: int,
        played_cards: List[int],
        played_players: List[int],
    ) -> Card:
        (hands, tc, tp, pc, pp, n_played, n_trick, cp, trump,
         pts0, pts1, tw0, tw1) = game_to_cpp_arrays(game, played_cards, played_players)

        result = self.solver.solve(
            hands, tc, tp, pc, pp, n_played, n_trick, cp, trump,
            pts0, pts1, tw0, tw1,
        )
        best_idx = int(result['best_move'])

        legal = game.get_legal_moves(player_id)
        for card in legal:
            if card.to_index() == best_idx:
                return card
        return legal[0]


class GarnishedRootAgent(Agent):
    """Root-garnish hybrid (strength_program.md C5): plain root-parallel
    ISMCTS at a wall-clock budget, plus ONE NN policy call at the root,
    blended into the final move choice as visits * prior^lam. Costs ~1 ms
    of the budget; documents how much of the NN edge is root-deep."""

    def __init__(self, jit_path, budget_ms=500, threads=15, lam=0.5,
                 device="cuda"):
        from manille_ismcts import ManilleSolverTime, NNPolicyValue

        self.solver = ManilleSolverTime(threads=threads, budget_ms=budget_ms)
        self.nn = NNPolicyValue(jit_path, device)
        self.lam = lam
        self.dealer = 0

    def new_game(self, game, dealer, player_id):
        self.dealer = dealer

    def choose_card(
        self,
        game: ManilleGame,
        player_id: int,
        played_cards: List[int],
        played_players: List[int],
    ) -> Card:
        (hands, tc, tp, pc, pp, n_played, n_trick, cp, trump,
         pts0, pts1, tw0, tw1) = game_to_cpp_arrays(game, played_cards, played_players)
        result = self.solver.solve(
            hands, tc, tp, pc, pp, n_played, n_trick, cp, trump,
            pts0, pts1, tw0, tw1,
        )
        visits = result["visit_counts"]

        from manille_ismcts import encode_observation

        obs = encode_observation(hands, tc, tp, pc, pp, n_played, n_trick,
                                 cp, trump, pts0, pts1, tw0, tw1,
                                 self.dealer, player_id)
        prior, _ = self.nn.eval(obs.reshape(1, -1).astype(np.float32))
        prior = prior[0]

        legal = game.get_legal_moves(player_id)
        best_card, best_score = legal[0], -1.0
        for card in legal:
            idx = card.to_index()
            score = visits.get(idx, 0) * prior[idx] ** self.lam
            if score > best_score:
                best_card, best_score = card, score
        return best_card


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

@dataclass
class GameResult:
    pts: List[int]      # [team0_pts, team1_pts]
    tricks: List[int]   # [team0_tricks, team1_tricks]


def _rotate_hands(hands, rotation: int):
    """Rotate hands by `rotation` positions (circular left-shift)."""
    return hands[rotation:] + hands[:rotation]


def run_game(agents: List[Agent], dealer: int = 0, hands=None, trump=None,
             mull: bool = False) -> GameResult:
    """
    Run a single hand of Manille.

    Args:
        agents:  list of 4 Agent instances, indexed by player_id
        dealer:  player id of the dealer for this hand
        hands:   optional pre-dealt hands (list of 4 Card lists); if None,
                 cards are dealt randomly
        trump:   optional explicit trump Suit; if None, the engine's
                 auto_select_trump heuristic is used. The learning stack
                 always passes the C++ rule-based selection explicitly.
        mull:    play at no trump (trump = -1) regardless of hands — the
                 unconditional mull lineage (Phase B).

    Returns:
        GameResult with final points and tricks per team
    """
    game = ManilleGame(dealer=dealer)
    if hands is not None:
        game.hands = [list(h) for h in hands]  # copy so rotations don't alias
    if mull:
        game.select_trump(None)
    else:
        if trump is None:
            trump = game.auto_select_trump()
        game.select_trump(trump)
    game.start_playing()

    for player_id, agent in enumerate(agents):
        agent.new_game(game, dealer, player_id)

    played_cards: List[int] = []
    played_players: List[int] = []

    while game.phase == GamePhase.PLAYING:
        pid = game.current_player
        card = agents[pid].choose_card(game, pid, played_cards, played_players)
        card_idx = card.to_index()

        pre_trick = list(game.current_trick)
        if not game.play_card(pid, card):
            raise RuntimeError(f"Agent selected illegal move {card} for player {pid}.")

        if len(pre_trick) == 3:
            completed_trick = pre_trick + [(pid, card)]
            for trick_pid, trick_card in completed_trick:
                played_cards.append(trick_card.to_index())
                played_players.append(trick_pid)

        # Notify all agents after the public state has been updated.
        for agent in agents:
            agent.on_card_played(game, pid, card_idx, played_cards, played_players)

    return GameResult(
        pts=list(game.hand_points),
        tricks=list(game.tricks_won),
    )


# ---------------------------------------------------------------------------
# Tournament runner
# ---------------------------------------------------------------------------

@dataclass
class MatchupStats:
    name: str
    t0_pts: List[int] = field(default_factory=list)
    t1_pts: List[int] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.t0_pts)

    @property
    def t0_mean(self) -> float:
        return float(np.mean(self.t0_pts)) if self.t0_pts else 0.0

    @property
    def t1_mean(self) -> float:
        return float(np.mean(self.t1_pts)) if self.t1_pts else 0.0

    @property
    def t0_win_pct(self) -> float:
        if not self.t0_pts:
            return 0.0
        wins = sum(1 for a, b in zip(self.t0_pts, self.t1_pts) if a > b)
        return 100.0 * wins / self.n


def run_tournament(
    matchups: List[Tuple[str, List[Agent], List[Agent]]],
    n_games: int = 200,
    dealer_rotation: bool = True,
) -> List[MatchupStats]:
    """
    Run a set of matchups and return statistics.

    Each unique deal is played 4 times with rotated hand assignments so that
    each player position receives each hand once.  This eliminates deal-luck
    variance between team0 and team1.  n_games is the number of unique deals;
    total hands played = n_games * 4.

    Args:
        matchups: list of (name, team0_agents, team1_agents)
                  team0_agents: 2 agents for players 0 and 2
                  team1_agents: 2 agents for players 1 and 3
        n_games:          number of unique deals per matchup
        dealer_rotation:  cycle the dealer across the 4 rotations

    Returns:
        List of MatchupStats (one per matchup)
    """
    all_stats = []

    # Pre-deal all hands once so every matchup uses the same deals
    base_games = [ManilleGame() for _ in range(n_games)]
    deals = [g.hands for g in base_games]

    for name, team0, team1 in matchups:
        stats = MatchupStats(name=name)
        agents = [team0[0], team1[0], team0[1], team1[1]]  # p0,p1,p2,p3

        t0 = time.time()
        for deal_idx, base_hands in enumerate(deals):
            for rotation in range(4):
                dealer = (deal_idx * 4 + rotation) % 4 if dealer_rotation else 0
                rotated = _rotate_hands(base_hands, rotation)
                result = run_game(agents, dealer=dealer, hands=rotated)
                stats.t0_pts.append(result.pts[0])
                stats.t1_pts.append(result.pts[1])

        elapsed = time.time() - t0
        total_hands = n_games * 4
        logger.info(
            "%-35s  %d deals × 4 rotations = %d hands in %.1fs (%.1f hands/s)",
            name, n_games, total_hands, elapsed, total_hands / elapsed,
        )
        all_stats.append(stats)

    return all_stats


def print_results(all_stats: List[MatchupStats]) -> None:
    header = f"{'Matchup':<40} {'T0 pts/hand':>12} {'T1 pts/hand':>12} {'T0 win%':>9} {'N':>6}"
    print(header)
    print("-" * len(header))
    for s in all_stats:
        print(
            f"{s.name:<40} {s.t0_mean:>12.1f} {s.t1_mean:>12.1f} "
            f"{s.t0_win_pct:>8.1f}% {s.n:>6}"
        )


# ---------------------------------------------------------------------------
# Predefined matchups
# ---------------------------------------------------------------------------

def build_matchups(iterations: int) -> dict:
    """Return a dict of named matchup factory functions."""

    def so():
        return ISMCTSAgent('SO', iterations)

    def mo():
        return ISMCTSAgent('MO', iterations)

    def rnd():
        return RandomAgent()

    return {
        'SO_vs_random': lambda: ('SO vs random', [so(), so()], [rnd(), rnd()]),
        'MO_vs_random': lambda: ('MO vs random', [mo(), mo()], [rnd(), rnd()]),
        'SO_vs_MO':     lambda: ('SO vs MO',     [so(), so()], [mo(), mo()]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--games', type=int, default=100,
                        help='unique deals per matchup (each played 4x rotated)')
    parser.add_argument('--iterations', type=int, default=1000,
                        help='ISMCTS iterations per move')
    parser.add_argument('--matchups', nargs='+', default=['SO_vs_random'],
                        help="matchup names, or 'all'")
    parser.add_argument('--verbose', action='store_true',
                        help='enable INFO-level logging')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)

    catalogue = build_matchups(args.iterations)
    names = list(catalogue) if 'all' in args.matchups else args.matchups

    unknown = [n for n in names if n not in catalogue]
    if unknown:
        print(f"Unknown matchup(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"Available: {', '.join(catalogue)}, all", file=sys.stderr)
        sys.exit(1)

    matchups = [catalogue[n]() for n in names]
    print(f"Running {len(matchups)} matchup(s), {args.games} deals each, "
          f"{args.iterations} iterations/move\n")
    print_results(run_tournament(matchups, n_games=args.games))


if __name__ == '__main__':
    main()
