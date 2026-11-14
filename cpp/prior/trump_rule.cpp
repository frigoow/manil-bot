#include "prior/trump_rule.h"

#include "card.h"
#include "bitops.h"

namespace prior {
namespace {

int popcount(int64_t mask) {
    return manille::popcount64(static_cast<unsigned long long>(mask));
}

} // namespace

int apply_trump_rule(int64_t hand_mask) {
    int best_suit = 0;
    int best_count = -1;
    int best_points = -1;

    for (int suit = 0; suit < 4; ++suit) {
        const int64_t suit_bits = (hand_mask >> (suit * 8)) & int64_t(0xFF);
        const int count = popcount(suit_bits);

        int points = 0;
        for (int rank = 0; rank < 8; ++rank) {
            if ((suit_bits & (int64_t(1) << rank)) == 0)
                continue;
            points += Card(static_cast<int8_t>(suit), static_cast<int8_t>(rank)).points();
        }

        if (count > best_count
            || (count == best_count && points > best_points)
            || (count == best_count && points == best_points && suit < best_suit)) {
            best_suit = suit;
            best_count = count;
            best_points = points;
        }
    }

    return best_suit;
}

double trump_consistency(int64_t hand_mask, int observed_suit) {
    if (observed_suit < 0 || observed_suit > 3)
        return 0.0;
    return apply_trump_rule(hand_mask) == observed_suit ? 1.0 : 0.0;
}

} // namespace prior
