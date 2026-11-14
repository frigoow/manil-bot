#ifndef MANILLE_SAMPLER_UNIFORM_H
#define MANILLE_SAMPLER_UNIFORM_H

#include "manille_game.h"

#include <array>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

namespace sampler {

struct Deal {
    std::array<int64_t, 4> hands{};
};

struct PublicState {
    int observer = 0;
    int current_player = 0;
    int trump_suit = -1;
    int declarer = -1;
    int pts0 = 0;
    int pts1 = 0;
    int tricks_won0 = 0;
    int tricks_won1 = 0;
    int n_trick = 0;
    int n_played_cards = 0;
    int64_t observer_hand = 0;
    // Partner-transparent determinization (0b.2): when nonzero, the partner's
    // true hand is held fixed and only the two opponents' hands are sampled.
    int64_t partner_hand = 0;
    std::array<int, 4> hand_sizes{};
    std::array<int32_t, 4> trick_cards{};
    std::array<int32_t, 4> trick_players{};
    std::array<int32_t, 32> played_cards{};
    std::array<int32_t, 32> played_players{};

    static PublicState fromGame(const ManilleGame& game, int observer);
};

class UniformDealSampler {
public:
    Deal sampleDeal(const PublicState& public_state) const;
    // Every consistent deal exactly once (the support of sampleDeal), or an
    // empty vector when there are more than max_deals of them. Late in the
    // hand the support is small and listing it beats sampling copies of it.
    std::vector<Deal> enumerateDeals(const PublicState& public_state,
                                     size_t max_deals) const;

    // Reseed the shared thread_local sampling engine. BatchedISMCTS calls
    // this from begin() when constructed with a nonzero seed, making solver
    // seeds fully authoritative (deterministic determinizations for
    // reproducible tests and evals).
    static void seedPrng(uint64_t seed);

private:
    struct CardDomain {
        int card = 0;
        std::array<int, 3> players{};
        int count = 0;
    };

    struct SamplingProblem {
        std::vector<CardDomain> domains;
        std::array<int, 4> remaining{};
    };

    static SamplingProblem buildProblem(const PublicState& public_state);
    static std::array<int64_t, 4> historyImpossibleMasks(const PublicState& public_state);
    static uint64_t countAssignments(
        const SamplingProblem& problem,
        int index,
        std::array<int, 4>& remaining,
        std::unordered_map<uint64_t, uint64_t>& memo
    );
    static void sampleAssignments(
        const SamplingProblem& problem,
        int index,
        std::array<int, 4>& remaining,
        std::unordered_map<uint64_t, uint64_t>& memo,
        std::array<int64_t, 4>& hands
    );
    static void enumerateAssignments(
        const SamplingProblem& problem,
        int index,
        std::array<int, 4>& remaining,
        std::unordered_map<uint64_t, uint64_t>& memo,
        std::array<int64_t, 4>& hands,
        std::vector<Deal>& out
    );
    static uint64_t memoKey(int index, const std::array<int, 4>& remaining);
    static int popcount(int64_t mask);

    static std::mt19937_64& prng();
};

ManilleGame gameFromDeal(
    const PublicState& public_state,
    const Deal& deal,
    bool perfect_info = true
);

// Reseed the shared thread_local sampling engine. BatchedISMCTS calls this
// from begin() when constructed with a nonzero seed, making solver seeds
// fully authoritative (deterministic determinizations for reproducible
// tests and evals).
} // namespace sampler

#endif // MANILLE_SAMPLER_UNIFORM_H
