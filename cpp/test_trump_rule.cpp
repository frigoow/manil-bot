#include "prior/trump_rule.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <string>

namespace {

constexpr int DIAMONDS = 0;
constexpr int HEARTS = 1;
constexpr int CLUBS = 2;
constexpr int SPADES = 3;

int idx(int suit, int rank) {
    return suit * 8 + rank;
}

int64_t maskOf(std::initializer_list<int> cards) {
    int64_t mask = 0;
    for (const int card : cards)
        mask |= int64_t(1) << card;
    return mask;
}

[[noreturn]] void fail(const std::string& message) {
    std::fprintf(stderr, "test_trump_rule failure: %s\n", message.c_str());
    std::exit(1);
}

void expectRule(const std::string& label, int64_t hand_mask, int expected_suit) {
    const int actual = prior::apply_trump_rule(hand_mask);
    if (actual != expected_suit) {
        fail(label + ": expected suit " + std::to_string(expected_suit)
             + ", got " + std::to_string(actual));
    }

    if (prior::trump_consistency(hand_mask, expected_suit) != 1.0)
        fail(label + ": trump_consistency should return 1.0 for the chosen suit");

    const int other_suit = (expected_suit + 1) % 4;
    if (prior::trump_consistency(hand_mask, other_suit) != 0.0)
        fail(label + ": trump_consistency should return 0.0 for a wrong suit");
}

} // namespace

int main() {
    expectRule(
        "five hearts beats singletons",
        maskOf({
            idx(HEARTS, 0), idx(HEARTS, 1), idx(HEARTS, 2), idx(HEARTS, 5), idx(HEARTS, 6),
            idx(DIAMONDS, 0), idx(CLUBS, 0), idx(SPADES, 0)
        }),
        HEARTS
    );

    expectRule(
        "count tie broken by suit points",
        maskOf({
            idx(HEARTS, 4), idx(HEARTS, 5), idx(HEARTS, 6),
            idx(SPADES, 1), idx(SPADES, 2), idx(SPADES, 3),
            idx(DIAMONDS, 0), idx(CLUBS, 0)
        }),
        HEARTS
    );

    expectRule(
        "double tie broken by lower suit index",
        maskOf({
            idx(HEARTS, 3), idx(HEARTS, 4), idx(HEARTS, 5),
            idx(SPADES, 3), idx(SPADES, 4), idx(SPADES, 5),
            idx(DIAMONDS, 0), idx(CLUBS, 0)
        }),
        HEARTS
    );

    expectRule(
        "two two two two split uses hcp then suit order",
        maskOf({
            idx(DIAMONDS, 0), idx(DIAMONDS, 7),
            idx(HEARTS, 0), idx(HEARTS, 6),
            idx(CLUBS, 4), idx(CLUBS, 5),
            idx(SPADES, 0), idx(SPADES, 3)
        }),
        DIAMONDS
    );

    expectRule(
        "count dominates even against a manille doubleton",
        maskOf({
            idx(HEARTS, 0), idx(HEARTS, 7),
            idx(SPADES, 0), idx(SPADES, 1), idx(SPADES, 2),
            idx(DIAMONDS, 4), idx(DIAMONDS, 6),
            idx(CLUBS, 3)
        }),
        SPADES
    );

    if (prior::trump_consistency(maskOf({idx(DIAMONDS, 0)}), -1) != 0.0)
        fail("invalid observed suit should return 0.0");

    std::printf("test_trump_rule: all checks passed\n");
    return 0;
}
