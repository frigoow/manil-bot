#ifndef MANILLE_CARD_H
#define MANILLE_CARD_H

#include <cstdint>
#include <ostream>

struct Card {
    int8_t suit;  // 0-3 (Diamonds, Hearts, Clubs, Spades)
    int8_t rank;  // 0-7 (7=Seven .. 7=Ten/Manille)

    Card() : suit(0), rank(0) {}
    Card(int8_t s, int8_t r) : suit(s), rank(r) {}

    int index() const { return suit * 8 + rank; }

    static Card from_index(int idx) {
        return Card(static_cast<int8_t>(idx / 8), static_cast<int8_t>(idx % 8));
    }

    int points() const {
        // 7=TEN=5, 6=ACE=4, 5=KING=3, 4=QUEEN=2, 3=JACK=1, others=0
        static const int pt[] = {0, 0, 0, 1, 2, 3, 4, 5};
        return pt[rank];
    }

    bool beats(Card other, int trump_suit) const {
        if (trump_suit >= 0) {
            if (suit == trump_suit && other.suit != trump_suit) return true;
            if (suit != trump_suit && other.suit == trump_suit) return false;
        }
        if (suit == other.suit) return rank > other.rank;
        return false;
    }

    bool operator==(Card o) const { return suit == o.suit && rank == o.rank; }
    bool operator!=(Card o) const { return !(*this == o); }
    bool operator<(Card o) const { return index() < o.index(); }

    friend std::ostream& operator<<(std::ostream& os, Card c) {
        static const char* suits[] = {"D", "H", "C", "S"};
        static const char* ranks[] = {"7", "8", "9", "J", "Q", "K", "A", "10"};
        return os << ranks[c.rank] << suits[c.suit];
    }
};

#endif // MANILLE_CARD_H
