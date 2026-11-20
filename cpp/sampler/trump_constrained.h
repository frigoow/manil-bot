#ifndef MANILLE_SAMPLER_TRUMP_CONSTRAINED_H
#define MANILLE_SAMPLER_TRUMP_CONSTRAINED_H

#include "sampler/uniform.h"

#include <atomic>
#include <cstdint>
#include <stdexcept>

namespace sampler {

// The rejection loop ran out of budget on a state whose trump-consistent
// world set is legal but very thin. This is a "try again / take fewer worlds"
// condition, unlike the sampler's other runtime_errors, which signal a broken
// invariant. Callers must be able to tell the two apart before they retry.
class RejectionBudgetExhausted : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class TrumpConstrainedDealSampler {
public:
    struct SampleStats {
        int attempts = 0;
        int rejections = 0;
        double acceptance_rate = 1.0;
        bool used_constraint = false;
    };

    struct RollingStats {
        uint64_t attempts = 0;
        uint64_t accepted = 0;

        double acceptanceRate() const {
            return attempts > 0
                ? static_cast<double>(accepted) / static_cast<double>(attempts)
                : 1.0;
        }
    };

    explicit TrumpConstrainedDealSampler(int max_attempts = 1000, bool log_warnings = true);

    Deal sampleDeal(
        const PublicState& public_state,
        SampleStats* stats = nullptr
    ) const;
    // Draw up to `n` deals reusing the uniform sampler's DP table once.
    // Throws RejectionBudgetExhausted only if `allow_short` is false and a
    // draw exhausts the rejection budget, or if no deals survive at all.
    std::vector<Deal> sampleDeals(
        const PublicState& public_state,
        int n,
        bool allow_short = true
    ) const;
    // Every deal sampleDeal can return, exactly once (uniform support filtered
    // by the trump rule); empty when the unfiltered support exceeds max_deals.
    std::vector<Deal> enumerateDeals(const PublicState& public_state,
                                     size_t max_deals) const;

    RollingStats rollingStats() const;

    static int64_t reconstructOriginalDeclarerHand(
        const PublicState& public_state,
        const Deal& deal
    );

private:
    UniformDealSampler m_uniform_sampler;
    int m_max_attempts;
    bool m_log_warnings;
    mutable std::atomic<uint64_t> m_warning_count{0};
    mutable std::atomic<uint64_t> m_total_attempts{0};
    mutable std::atomic<uint64_t> m_total_accepted{0};
};

} // namespace sampler

#endif // MANILLE_SAMPLER_TRUMP_CONSTRAINED_H
