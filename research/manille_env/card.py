"""Card and Deck classes for Manille game."""
from enum import IntEnum
from typing import List
import random


class Suit(IntEnum):
    """Card suits."""
    DIAMONDS = 0
    HEARTS = 1
    CLUBS = 2
    SPADES = 3

    def __str__(self):
        return ['♦', '♥', '♣', '♠'][self.value]


class Rank(IntEnum):
    """Card ranks in gameplay order (not point order)."""
    SEVEN = 0
    EIGHT = 1
    NINE = 2
    JACK = 3
    QUEEN = 4
    KING = 5
    ACE = 6
    TEN = 7  # Manille - highest rank

    def __str__(self):
        names = ['7', '8', '9', 'J', 'Q', 'K', 'A', '10']
        return names[self.value]


class Card:
    """Represents a single playing card."""

    # Point values for scoring
    POINT_VALUES = {
        Rank.TEN: 5,
        Rank.ACE: 4,
        Rank.KING: 3,
        Rank.QUEEN: 2,
        Rank.JACK: 1,
        Rank.NINE: 0,
        Rank.EIGHT: 0,
        Rank.SEVEN: 0,
    }

    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit
        self.rank = rank
        self.points = self.POINT_VALUES[rank]

    def __str__(self):
        return f"{self.rank}{self.suit}"

    def __repr__(self):
        return f"Card({self.suit.name}, {self.rank.name})"

    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.rank == other.rank

    def __hash__(self):
        return hash((self.suit, self.rank))

    def beats(self, other: 'Card', trump_suit: Suit | None) -> bool:
        """
        Determine if this card beats another card.

        Args:
            other: The card to compare against
            trump_suit: The current trump suit (None if no trump)

        Returns:
            True if this card beats the other card
        """
        # Trump beats non-trump
        if trump_suit is not None:
            if self.suit == trump_suit and other.suit != trump_suit:
                return True
            if self.suit != trump_suit and other.suit == trump_suit:
                return False

        # Same suit: higher rank wins
        if self.suit == other.suit:
            return self.rank > other.rank

        # Different suits (and no trump advantage): original card wins
        return False

    def to_index(self) -> int:
        """Convert card to unique index (0-31) for array representations."""
        return self.suit * 8 + self.rank

    @staticmethod
    def from_index(index: int) -> 'Card':
        """Create card from index (0-31)."""
        suit = Suit(index // 8)
        rank = Rank(index % 8)
        return Card(suit, rank)


# Suit relabelling. Manille ranks suits only through the trump choice, so a
# permutation of the four suit labels is an exact automorphism of the game:
# relabel a deal's hands, trump and played sequence together and every legal
# move, trick and score is unchanged. That makes it valid data augmentation
# and a valid test-time average, and it is why `perm` must be applied to
# EVERY card-bearing field of a state at once -- a half-permuted state is
# silently corrupt, not merely wrong.
# `perm` maps old suit -> new suit, so perm[Suit.HEARTS] is where hearts go.

def permute_card_index(index: int, perm) -> int:
    """Relabel the suit of one 0-31 card index."""
    return perm[int(index) // 8] * 8 + int(index) % 8


def permute_hand_mask(mask: int, perm) -> int:
    """Relabel suits in a 32-bit hand bitmask (bit c = card index c)."""
    out = 0
    for suit in range(4):
        bits = (int(mask) >> (suit * 8)) & 0xFF
        out |= bits << (perm[suit] * 8)
    return out


class Deck:
    """Represents a 32-card Manille deck."""

    def __init__(self):
        """Initialize a full 32-card deck."""
        self.cards: List[Card] = [
            Card(suit, rank)
            for suit in Suit
            for rank in Rank
        ]

    def shuffle(self):
        """Shuffle the deck in place."""
        random.shuffle(self.cards)

    def deal(self, num_players: int = 4) -> List[List[Card]]:
        """
        Deal cards to players following Manille rules.

        Deals in batches of 3-2-3 to each player (8 cards total per player).

        Args:
            num_players: Number of players (default 4)

        Returns:
            List of hands, where each hand is a list of Cards
        """
        if len(self.cards) != 32:
            raise ValueError("Deck must have 32 cards to deal")

        if num_players != 4:
            raise ValueError("Manille requires exactly 4 players")

        hands = [[] for _ in range(num_players)]
        card_idx = 0

        # Deal in batches: 3-2-3 per player
        for batch_size in [3, 2, 3]:
            for player in range(num_players):
                for _ in range(batch_size):
                    hands[player].append(self.cards[card_idx])
                    card_idx += 1

        return hands

    def __len__(self):
        return len(self.cards)

    def __str__(self):
        return f"Deck({len(self.cards)} cards)"


# Utility functions
def get_suit_cards(hand: List[Card], suit: Suit) -> List[Card]:
    """Get all cards of a specific suit from a hand."""
    return [card for card in hand if card.suit == suit]


def get_legal_moves(hand: List[Card], led_suit: Suit | None,
                    current_trick: List[tuple[int, Card]],
                    trump_suit: Suit | None,
                    player_id: int) -> List[Card]:
    """
    Determine legal moves for a player.

    Rules:
    - Must follow suit if possible
    - If opponent is winning and you can beat them (while following suit), you must
    - If your partner is winning, no obligation to beat
    - Cannot undertrump: if plain suit led and trumped, must play higher trump
      (unless only have lower trumps and no other cards)

    Args:
        hand: Player's current hand
        led_suit: The suit that was led this trick (None if first to play)
        current_trick: List of (player_id, card) tuples played so far
        trump_suit: Current trump suit (None if no trump)
        player_id: The current player's ID (0-3)

    Returns:
        List of legal cards to play
    """
    # First to play: any card is legal
    if led_suit is None:
        return hand.copy()

    # Get cards that follow suit
    following_cards = get_suit_cards(hand, led_suit)

    # CAN follow suit: must follow suit with additional rules
    if following_cards:
        # Determine partner and if partner/opponent is winning
        partner_id = (player_id + 2) % 4

        # Find current winning card
        winning_card = current_trick[0][1]
        winning_player = current_trick[0][0]

        for pid, card in current_trick[1:]:
            if card.beats(winning_card, trump_suit):
                winning_card = card
                winning_player = pid

        # If partner is winning, can play any card that follows suit
        if winning_player == partner_id:
            return following_cards

        # Opponent is winning: must beat if possible while following suit
        cards_that_beat = [
            card for card in following_cards
            if card.beats(winning_card, trump_suit)
        ]

        if cards_that_beat:
            return cards_that_beat
        else:
            return following_cards

    # CANNOT follow suit: must trump if possible (unless partner winning)
    if trump_suit is None:
        # No trump in game: can play anything
        return hand.copy()

    # Check if partner is winning
    partner_id = (player_id + 2) % 4
    winning_card = current_trick[0][1]
    winning_player = current_trick[0][0]
    for pid, card in current_trick[1:]:
        if card.beats(winning_card, trump_suit):
            winning_card = card
            winning_player = pid

    if winning_player == partner_id:
        # Partner winning: can play anything
        return hand.copy()

    # Opponent winning: must trump if possible
    trumps_in_hand = get_suit_cards(hand, trump_suit)
    non_trumps_in_hand = [card for card in hand if card.suit != trump_suit]

    if not trumps_in_hand:
        # No trumps: can play anything
        return hand.copy()

    # Check if a trump has been played on this trick
    trump_played = None
    for pid, card in current_trick:
        if card.suit == trump_suit:
            if trump_played is None or card.rank > trump_played.rank:
                trump_played = card

    if trump_played is None:
        # No trump played yet: must trump (any trump is legal)
        return trumps_in_hand

    # Trump was played: must overtrump if possible
    higher_trumps = [
        card for card in trumps_in_hand
        if card.rank > trump_played.rank
    ]

    if higher_trumps:
        return higher_trumps

    # Cannot overtrump: cannot undertrump, must play non-trump
    if non_trumps_in_hand:
        return non_trumps_in_hand
    else:
        # Only have lower trumps: forced to undertrump
        return trumps_in_hand
