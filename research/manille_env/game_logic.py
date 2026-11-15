"""Core game logic for Manille."""
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .card import Card, Deck, Suit, get_legal_moves


class GamePhase(Enum):
    """Phases of a Manille game."""
    TRUMP_SELECTION = 0  # Dealer selects trump
    DOUBLING = 1  # Opponents can say "Ik ga mee", dealer's team can counter
    PLAYING = 2  # Playing tricks
    SCORING = 3  # Trick complete, scoring the hand


@dataclass
class TeamScore:
    """Track scores for a team."""
    team_id: int
    score: int = 0
    rounds_won: int = 0


class ManilleGame:
    """
    Manages the complete Manille game state and logic.

    Teams: Players 0&2 vs Players 1&3
    Dealer rotates each hand.
    """

    TOTAL_POINTS = 60  # Total points in deck (4×10=20, 4×A=16, 4×K=12, 4×Q=8, 4×J=4)
    POINTS_THRESHOLD = 30  # Halfway point - score is difference from this
    WINNING_SCORE = 101  # First to 101 wins

    def __init__(self, dealer: int = 0, swap_teams: bool = False):
        """
        Initialize a new Manille game.

        Args:
            dealer: Starting dealer (0-3)
            swap_teams: If True, swap team assignments (for mirror matches)
                        Normal: Team 0 = {0,2}, Team 1 = {1,3}
                        Swapped: Team 0 = {1,3}, Team 1 = {0,2}
        """
        self.dealer = dealer
        self.swap_teams = swap_teams
        self.current_player = (dealer + 1) % 4  # Player left of dealer leads

        # Game state
        self.phase = GamePhase.TRUMP_SELECTION
        self.trump_suit: Optional[Suit] = None
        self.no_trump = False
        self.multiplier = 1  # mull x2, meegaan x2, terug meegaan x2 -> max 8
        self.doubled = False    # defending team said "ik ga mee"
        self.redoubled = False  # trump-making team answered "ik ga tegen"

        # Cards
        self.hands: List[List[Card]] = [[] for _ in range(4)]
        self.current_trick: List[Tuple[int, Card]] = []  # (player_id, card)
        self.tricks_won = [0, 0]  # Tricks won by each team

        # Scoring
        self.team_scores = [
            TeamScore(team_id=0),  # Team 0: Players 0 & 2
            TeamScore(team_id=1),  # Team 1: Players 1 & 3
        ]
        self.hand_points = [0, 0]  # Points this hand for each team

        # Deal initial cards
        self._deal_cards()

    def _deal_cards(self):
        """Deal cards to all players."""
        deck = Deck()
        deck.shuffle()
        self.hands = deck.deal(num_players=4)

    def get_player_team(self, player_id: int) -> int:
        """Get team ID for a player (0 or 1)."""
        if hasattr(self, 'swap_teams') and self.swap_teams:
            # Swapped: Team 0 = {1,3}, Team 1 = {0,2}
            return (player_id + 1) % 2
        else:
            # Normal: Team 0 = {0,2}, Team 1 = {1,3}
            return player_id % 2

    def get_partner(self, player_id: int) -> int:
        """Get partner's player ID."""
        return (player_id + 2) % 4

    def auto_select_trump(self) -> Suit:
        """
        Automatically select trump suit based on dealer's hand.
        Chooses the suit with most cards, breaking ties by highest card.

        Returns:
            The selected trump suit
        """
        dealer_hand = self.hands[self.dealer]

        # Count cards per suit
        suit_cards = {suit: [] for suit in Suit}
        for card in dealer_hand:
            suit_cards[card.suit].append(card)

        # Find best suit: most cards, then most points, then highest rank
        best_suit = None
        best_count = 0
        best_points = -1
        best_rank = -1

        for suit, cards in suit_cards.items():
            if not cards:
                continue

            count = len(cards)
            points = sum(card.points for card in cards)
            max_rank = max(card.rank for card in cards)

            if (count > best_count or
                (count == best_count and points > best_points) or
                (count == best_count and points == best_points and max_rank > best_rank)):
                best_suit = suit
                best_count = count
                best_points = points
                best_rank = max_rank

        return best_suit if best_suit is not None else Suit.DIAMONDS

    def select_trump(self, trump_suit: Optional[Suit]) -> bool:
        """
        Dealer selects trump suit.

        Args:
            trump_suit: The trump suit, or None for no trump

        Returns:
            True if successful, False if not dealer's turn
        """
        if self.phase != GamePhase.TRUMP_SELECTION:
            return False

        if trump_suit is None:
            self.no_trump = True
            self.multiplier = 2  # No trump doubles the score
        else:
            self.trump_suit = trump_suit

        self.phase = GamePhase.DOUBLING
        return True

    def announce_double(self, player_id: int) -> bool:
        """
        Defending team announces "Ik ga mee" (double the stakes).

        ONE announcement per side: only the team that did not make trump
        may say it, and only once — it is a team call, not a per-player
        one, so a second opponent cannot compound it.

        Args:
            player_id: Player making the announcement

        Returns:
            True if successful, False if not allowed
        """
        if self.phase != GamePhase.DOUBLING:
            return False

        # Only the team that did not make trump
        if self.get_player_team(player_id) == self.get_player_team(self.dealer):
            return False

        if self.doubled:
            return False

        self.doubled = True
        self.multiplier *= 2
        return True

    def announce_redouble(self, player_id: int) -> bool:
        """
        Trump-making team answers "Ik ga tegen" (double again).

        Only the team that made trump, only in answer to an actual
        meegaan, and only once. Note this must test `self.doubled` rather
        than the multiplier: mull already sets the multiplier to 2 on its
        own, so a multiplier test would let the trump-makers redouble
        against an announcement that never happened.

        Args:
            player_id: Player making the announcement

        Returns:
            True if successful, False if not allowed
        """
        if self.phase != GamePhase.DOUBLING:
            return False

        if not self.doubled or self.redoubled:
            return False

        if self.get_player_team(player_id) != self.get_player_team(self.dealer):
            return False

        self.redoubled = True
        self.multiplier *= 2
        return True

    def start_playing(self):
        """Move from doubling phase to playing phase."""
        if self.phase == GamePhase.DOUBLING:
            self.phase = GamePhase.PLAYING

    def get_legal_moves(self, player_id: int) -> List[Card]:
        """Get legal moves for the current player."""
        if self.phase != GamePhase.PLAYING:
            return []

        if player_id != self.current_player:
            return []

        led_suit = self.current_trick[0][1].suit if self.current_trick else None

        return get_legal_moves(
            hand=self.hands[player_id],
            led_suit=led_suit,
            current_trick=self.current_trick,
            trump_suit=self.trump_suit,
            player_id=player_id
        )

    def play_card(self, player_id: int, card: Card) -> bool:
        """
        Play a card from player's hand.

        Args:
            player_id: Player playing the card
            card: Card to play

        Returns:
            True if successful, False if illegal move
        """
        if self.phase != GamePhase.PLAYING:
            return False

        if player_id != self.current_player:
            return False

        # Check if card is legal
        legal_moves = self.get_legal_moves(player_id)
        if card not in legal_moves:
            return False

        # Remove from hand and add to trick
        self.hands[player_id].remove(card)
        self.current_trick.append((player_id, card))

        # Check if trick is complete
        if len(self.current_trick) == 4:
            self._resolve_trick()
        else:
            # Next player's turn
            self.current_player = (self.current_player + 1) % 4

        return True

    def _resolve_trick(self):
        """Resolve the current trick and determine winner."""
        # Find winning card
        winning_player, winning_card = self.current_trick[0]

        for player_id, card in self.current_trick[1:]:
            if card.beats(winning_card, self.trump_suit):
                winning_card = card
                winning_player = player_id

        # Award trick to winning team
        winning_team = self.get_player_team(winning_player)
        self.tricks_won[winning_team] += 1

        # Count points in trick
        trick_points = sum(card.points for _, card in self.current_trick)
        self.hand_points[winning_team] += trick_points

        # Clear trick and set winner as next leader
        self.current_trick = []
        self.current_player = winning_player

        # Check if hand is complete (8 tricks played)
        if sum(self.tricks_won) == 8:
            self._score_hand()

    def _score_hand(self):
        """Score the completed hand and prepare for next hand."""
        self.phase = GamePhase.SCORING

        # A 30-30 tie simply scores nothing for either side: the score is
        # the winner's margin over the 30-point threshold, and at 30-30
        # there is no winner and no margin.
        if self.hand_points[0] != self.hand_points[1]:
            # Normal scoring: difference from threshold (30)
            # Team with more points scores (their_points - 30)
            if self.hand_points[0] > self.hand_points[1]:
                winning_team = 0
                score_diff = self.hand_points[0] - self.POINTS_THRESHOLD
            else:
                winning_team = 1
                score_diff = self.hand_points[1] - self.POINTS_THRESHOLD

            self.team_scores[winning_team].score += score_diff * self.multiplier

    def start_new_hand(self):
        """Start a new hand with next dealer."""
        # Rotate dealer
        self.dealer = (self.dealer + 1) % 4
        self.current_player = (self.dealer + 1) % 4

        # Reset hand state
        self.phase = GamePhase.TRUMP_SELECTION
        self.trump_suit = None
        self.no_trump = False
        self.multiplier = 1
        self.doubled = False
        self.redoubled = False
        self.tricks_won = [0, 0]
        self.hand_points = [0, 0]
        self.current_trick = []

        # Deal new cards
        self._deal_cards()

    def is_game_over(self) -> bool:
        """Check if game is over (team reached winning score)."""
        return (self.team_scores[0].score >= self.WINNING_SCORE or
                self.team_scores[1].score >= self.WINNING_SCORE)

    def get_winning_team(self) -> Optional[int]:
        """Get winning team if game is over."""
        if not self.is_game_over():
            return None

        if self.team_scores[0].score >= self.WINNING_SCORE:
            return 0
        return 1

    def get_state_dict(self) -> dict:
        """Get current game state as dictionary for observations."""
        return {
            'phase': self.phase,
            'dealer': self.dealer,
            'current_player': self.current_player,
            'trump_suit': self.trump_suit,
            'no_trump': self.no_trump,
            'multiplier': self.multiplier,
            'tricks_won': self.tricks_won.copy(),
            'hand_points': self.hand_points.copy(),
            'team_scores': [
                {'score': self.team_scores[0].score},
                {'score': self.team_scores[1].score}
            ],
            'current_trick_size': len(self.current_trick),
        }
