#include "manille_game.h"
#include "prior/trump_rule.h"
#include "sampler/trump_constrained.h"
#include "sampler/uniform.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <numeric>
#include <random>
#include <string>
#include "bitops.h"

namespace {

constexpr int HEARTS = 1;

std::mt19937& rng() {
    static std::mt19937 engine(123456789u);
    return engine;
}

[[noreturn]] void fail(const std::string& message) {
    std::fprintf(stderr, "test_trump_constrained_sampler failure: %s\n", message.c_str());
    std::exit(1);
}

int popcount(int64_t mask) {
    return manille::popcount64(static_cast<unsigned long long>(mask));
}

int suitCount(int64_t hand_mask, int suit) {
    const int64_t suit_bits = (hand_mask >> (suit * 8)) & int64_t(0xFF);
    return popcount(suit_bits);
}

struct HiddenDeal {
    std::array<int64_t, 4> hands{};
    int declarer = 0;
    int trump_suit = 0;
};

struct PublicStateFixture {
    HiddenDeal deal;
    ManilleGame game;
    sampler::PublicState public_state;
};

HiddenDeal randomHiddenDeal(int desired_trump = -1) {
    std::array<int, 32> cards{};
    std::iota(cards.begin(), cards.end(), 0);

    for (;;) {
        std::shuffle(cards.begin(), cards.end(), rng());

        HiddenDeal deal;
        deal.declarer = std::uniform_int_distribution<int>(0, 3)(rng());
        for (int player = 0; player < 4; ++player) {
            deal.hands[player] = 0;
            for (int i = player * 8; i < (player + 1) * 8; ++i)
                deal.hands[player] |= int64_t(1) << cards[i];
        }

        deal.trump_suit = prior::apply_trump_rule(deal.hands[deal.declarer]);
        if (desired_trump < 0 || deal.trump_suit == desired_trump)
            return deal;
    }
}

HiddenDeal randomHiddenDealWithTrumpCount(int trump_suit, int min_trump_count) {
    for (;;) {
        HiddenDeal deal = randomHiddenDeal(trump_suit);
        if (suitCount(deal.hands[deal.declarer], trump_suit) >= min_trump_count)
            return deal;
    }
}

ManilleGame startGame(const HiddenDeal& deal, bool perfect_info = false) {
    const std::array<int32_t, 4> trick_cards = {0, 0, 0, 0};
    const std::array<int32_t, 4> trick_players = {0, 0, 0, 0};
    return ManilleGame(
        deal.hands,
        trick_cards,
        trick_players,
        0,
        (deal.declarer + 1) % 4,
        deal.trump_suit,
        0,
        0,
        0,
        0,
        perfect_info,
        deal.declarer
    );
}

PublicStateFixture randomPublicState(
    int desired_trump,
    int min_plies,
    int max_plies
) {
    HiddenDeal deal = randomHiddenDeal(desired_trump);
    ManilleGame game = startGame(deal, false);

    const int target_plies = std::uniform_int_distribution<int>(min_plies, max_plies)(rng());
    for (int ply = 0; ply < target_plies; ++ply) {
        const auto moves = game.validMoves();
        if (moves.empty())
            break;

        const int move_idx = std::uniform_int_distribution<int>(
            0,
            static_cast<int>(moves.size()) - 1
        )(rng());
        game.doMove(moves[move_idx]);
    }

    int observer = std::uniform_int_distribution<int>(0, 2)(rng());
    if (observer >= deal.declarer)
        observer++;

    PublicStateFixture fixture{deal, game, sampler::PublicState::fromGame(game, observer)};
    if (fixture.public_state.declarer != deal.declarer)
        fail("public state lost the declarer seat");
    return fixture;
}

void testMarginalConsistency() {
    sampler::TrumpConstrainedDealSampler sampler(1000, false);
    const PublicStateFixture fixture = randomPublicState(HEARTS, 0, 20);

    for (int i = 0; i < 10000; ++i) {
        const sampler::Deal deal = sampler.sampleDeal(fixture.public_state);
        const int64_t declarer_original =
            sampler::TrumpConstrainedDealSampler::reconstructOriginalDeclarerHand(
                fixture.public_state,
                deal
            );
        if (prior::apply_trump_rule(declarer_original) != HEARTS)
            fail("marginal consistency check found an invalid sampled deal");
    }
}

void testDistributionShift() {
    const HiddenDeal base_deal = randomHiddenDealWithTrumpCount(HEARTS, 5);
    const ManilleGame game = startGame(base_deal, false);

    int observer = -1;
    int fewest_hearts = 9;
    for (int player = 0; player < 4; ++player) {
        if (player == base_deal.declarer)
            continue;

        const int hearts = suitCount(base_deal.hands[player], HEARTS);
        if (hearts < fewest_hearts) {
            fewest_hearts = hearts;
            observer = player;
        }
    }

    if (observer < 0)
        fail("failed to choose a non-declarer observer for the distribution shift test");

    const sampler::PublicState public_state = sampler::PublicState::fromGame(game, observer);

    sampler::UniformDealSampler uniform_sampler;
    sampler::TrumpConstrainedDealSampler trump_sampler(1000, false);

    double uniform_hearts = 0.0;
    double constrained_hearts = 0.0;
    for (int i = 0; i < 1000; ++i) {
        const sampler::Deal uniform_deal = uniform_sampler.sampleDeal(public_state);
        const sampler::Deal constrained_deal = trump_sampler.sampleDeal(public_state);

        uniform_hearts += suitCount(
            sampler::TrumpConstrainedDealSampler::reconstructOriginalDeclarerHand(
                public_state,
                uniform_deal
            ),
            HEARTS
        );
        constrained_hearts += suitCount(
            sampler::TrumpConstrainedDealSampler::reconstructOriginalDeclarerHand(
                public_state,
                constrained_deal
            ),
            HEARTS
        );
    }

    uniform_hearts /= 1000.0;
    constrained_hearts /= 1000.0;

    std::printf(
        "test_trump_constrained_sampler: uniform hearts=%.3f constrained hearts=%.3f\n",
        uniform_hearts,
        constrained_hearts
    );

    if (constrained_hearts < 3.3)
        fail("constrained sampler heart count stayed too low after conditioning on hearts trump");
    if (uniform_hearts > 3.25)
        fail("uniform sampler heart count was unexpectedly high for the baseline");
    if (constrained_hearts <= uniform_hearts + 0.75)
        fail("constrained sampler did not separate enough from the uniform baseline");
}

void testAcceptanceRate() {
    sampler::TrumpConstrainedDealSampler sampler(1000, false);

    for (int state_idx = 0; state_idx < 100; ++state_idx) {
        const PublicStateFixture fixture = randomPublicState(-1, 4, 24);

        const auto before = sampler.rollingStats();
        for (int sample_idx = 0; sample_idx < 50; ++sample_idx)
            sampler.sampleDeal(fixture.public_state);
        const auto after = sampler.rollingStats();

        const uint64_t attempts = after.attempts - before.attempts;
        const uint64_t accepted = after.accepted - before.accepted;
        const double acceptance_rate = attempts > 0
            ? static_cast<double>(accepted) / static_cast<double>(attempts)
            : 1.0;

        if (acceptance_rate < 0.05)
            fail("acceptance rate dropped below 5% on the random public-state suite");
    }
}

void testThroughput() {
    sampler::TrumpConstrainedDealSampler sampler(1000, false);
    const PublicStateFixture fixture = randomPublicState(-1, 12, 16);

    const int samples = 2000;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < samples; ++i)
        sampler.sampleDeal(fixture.public_state);
    const auto end = std::chrono::steady_clock::now();

    const double seconds = std::chrono::duration<double>(end - start).count();
    const double deals_per_second = static_cast<double>(samples) / seconds;

    std::printf("test_trump_constrained_sampler: throughput %.1f deals/sec\n", deals_per_second);
    if (deals_per_second < 500.0)
        fail("throughput fell below 500 deals/sec on a middle-game state");
}

void testPartnerTransparent() {
    // 0b.2: with partner_hand set, every sampled deal must keep both the
    // observer's and the partner's hands fixed, keep hands disjoint, and
    // preserve per-seat hand sizes — through both sampler layers.
    sampler::UniformDealSampler uniform_sampler;
    sampler::TrumpConstrainedDealSampler trump_sampler(1000, false);

    for (int trial = 0; trial < 30; ++trial) {
        const PublicStateFixture fixture = randomPublicState(-1, 0, 28);
        sampler::PublicState ps = fixture.public_state;
        const int partner = (ps.observer + 2) % 4;
        ps.partner_hand = fixture.game.handMask(partner);

        for (int i = 0; i < 100; ++i) {
            const sampler::Deal deal = (i % 2 == 0)
                ? uniform_sampler.sampleDeal(ps)
                : trump_sampler.sampleDeal(ps);
            if (deal.hands[partner] != ps.partner_hand)
                fail("partner-transparent sample changed the partner hand");
            if (deal.hands[ps.observer] != ps.observer_hand)
                fail("partner-transparent sample changed the observer hand");
            int64_t seen = 0;
            for (int p = 0; p < 4; ++p) {
                if ((seen & deal.hands[p]) != 0)
                    fail("partner-transparent sample produced overlapping hands");
                seen |= deal.hands[p];
                if (popcount(deal.hands[p]) != fixture.public_state.hand_sizes[p])
                    fail("partner-transparent sample changed a hand size");
            }
        }
    }
}

} // namespace

int main() {
    testMarginalConsistency();
    testDistributionShift();
    testAcceptanceRate();
    testThroughput();
    testPartnerTransparent();

    std::printf("test_trump_constrained_sampler: all checks passed\n");
    return 0;
}
