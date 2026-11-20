#include "sampler/trump_constrained.h"

#include "prior/trump_rule.h"

#include <cstdio>
#include <stdexcept>
#include "bitops.h"

namespace sampler {
namespace {

constexpr double kWarningAcceptanceRate = 0.05;
constexpr uint64_t kMaxWarningsPerSampler = 10;

bool shouldApplyConstraint(const PublicState& public_state) {
    return public_state.declarer >= 0
        && public_state.declarer < 4
        && public_state.observer != public_state.declarer
        && public_state.trump_suit >= 0
        && public_state.trump_suit < 4;
}

int popcount(int64_t mask) {
    return manille::popcount64(static_cast<unsigned long long>(mask));
}

bool maybeWarnLowAcceptance(
    const PublicState& public_state,
    const TrumpConstrainedDealSampler::SampleStats& stats
) {
    if (!stats.used_constraint || stats.acceptance_rate >= kWarningAcceptanceRate)
        return false;
    /*
    std::fprintf(
        stderr,
        "TrumpConstrainedDealSampler warning: low acceptance %.2f%% after %d attempts "
        "(observer=%d declarer=%d trump=%d played=%d partial=%d)\n",
        100.0 * stats.acceptance_rate,
        stats.attempts,
        public_state.observer,
        public_state.declarer,
        public_state.trump_suit,
        public_state.n_played_cards,
        public_state.n_trick
    );
    */
    return true;
}

} // namespace

TrumpConstrainedDealSampler::TrumpConstrainedDealSampler(int max_attempts, bool log_warnings)
    : m_max_attempts(max_attempts)
    , m_log_warnings(log_warnings)
{
    if (m_max_attempts <= 0) {
        throw std::invalid_argument(
            "TrumpConstrainedDealSampler: max_attempts must be positive."
        );
    }
}

Deal TrumpConstrainedDealSampler::sampleDeal(
    const PublicState& public_state,
    SampleStats* stats
) const {
    SampleStats local_stats;

    if (!shouldApplyConstraint(public_state)) {
        Deal deal = m_uniform_sampler.sampleDeal(public_state);
        local_stats.attempts = 1;
        if (stats != nullptr)
            *stats = local_stats;
        return deal;
    }

    local_stats.used_constraint = true;

    for (int attempt = 1; attempt <= m_max_attempts; ++attempt) {
        Deal deal = m_uniform_sampler.sampleDeal(public_state);
        local_stats.attempts = attempt;

        const int64_t original_hand = reconstructOriginalDeclarerHand(public_state, deal);
        if (prior::trump_consistency(original_hand, public_state.trump_suit) == 1.0) {
            local_stats.rejections = attempt - 1;
            local_stats.acceptance_rate = 1.0 / static_cast<double>(attempt);
            m_total_attempts.fetch_add(static_cast<uint64_t>(attempt), std::memory_order_relaxed);
            m_total_accepted.fetch_add(1, std::memory_order_relaxed);
            if (m_log_warnings
                && m_warning_count.load(std::memory_order_relaxed) < kMaxWarningsPerSampler
                && maybeWarnLowAcceptance(public_state, local_stats)) {
                m_warning_count.fetch_add(1, std::memory_order_relaxed);
            }
            if (stats != nullptr)
                *stats = local_stats;
            return deal;
        }
    }

    local_stats.attempts = m_max_attempts;
    local_stats.rejections = m_max_attempts;
    local_stats.acceptance_rate = 0.0;
    m_total_attempts.fetch_add(static_cast<uint64_t>(m_max_attempts), std::memory_order_relaxed);
    if (m_log_warnings
        && m_warning_count.load(std::memory_order_relaxed) < kMaxWarningsPerSampler
        && maybeWarnLowAcceptance(public_state, local_stats)) {
        m_warning_count.fetch_add(1, std::memory_order_relaxed);
    }
    if (stats != nullptr)
        *stats = local_stats;

    throw RejectionBudgetExhausted(
        "TrumpConstrainedDealSampler: exhausted rejection budget without a trump-consistent deal."
    );
}

std::vector<Deal> TrumpConstrainedDealSampler::sampleDeals(
    const PublicState& public_state,
    int n,
    bool allow_short
) const {
    if (n <= 0)
        return {};
    if (!shouldApplyConstraint(public_state))
        return m_uniform_sampler.sampleDeals(public_state, n, false);

    // Build the uniform DP table once by drawing chunks from sampleDeals().
    std::vector<Deal> accepted;
    accepted.reserve(static_cast<size_t>(n));
    const int max_draws = allow_short ? 2 * n : n;
    int draws_used = 0;
    while (draws_used < max_draws && static_cast<int>(accepted.size()) < n) {
        // Estimate chunk size: draw enough uniform proposals to likely finish.
        const int need = n - static_cast<int>(accepted.size());
        const int chunk = std::min(need * 4, m_max_attempts);
        auto proposals = m_uniform_sampler.sampleDeals(public_state, chunk, false);
        int local_attempts = 0;
        for (const Deal& deal : proposals) {
            ++local_attempts;
            const int64_t original_hand = reconstructOriginalDeclarerHand(public_state, deal);
            if (prior::trump_consistency(original_hand, public_state.trump_suit) == 1.0) {
                accepted.push_back(deal);
                m_total_attempts.fetch_add(static_cast<uint64_t>(local_attempts), std::memory_order_relaxed);
                m_total_accepted.fetch_add(1, std::memory_order_relaxed);
                local_attempts = 0;
                ++draws_used;
                if (static_cast<int>(accepted.size()) >= n)
                    break;
            } else if (local_attempts >= m_max_attempts) {
                m_total_attempts.fetch_add(static_cast<uint64_t>(local_attempts), std::memory_order_relaxed);
                local_attempts = 0;
                ++draws_used;
                if (!allow_short) {
                    throw RejectionBudgetExhausted(
                        "TrumpConstrainedDealSampler: exhausted rejection budget without a trump-consistent deal."
                    );
                }
                if (draws_used >= max_draws)
                    break;
            }
        }
        if (local_attempts > 0) {
            m_total_attempts.fetch_add(static_cast<uint64_t>(local_attempts), std::memory_order_relaxed);
            if (local_attempts >= m_max_attempts) {
                ++draws_used;
                if (!allow_short) {
                    throw RejectionBudgetExhausted(
                        "TrumpConstrainedDealSampler: exhausted rejection budget without a trump-consistent deal."
                    );
                }
            }
        }
    }
    if (accepted.empty() && !allow_short) {
        throw RejectionBudgetExhausted(
            "TrumpConstrainedDealSampler: exhausted rejection budget without a trump-consistent deal."
        );
    }
    return accepted;
}

TrumpConstrainedDealSampler::RollingStats TrumpConstrainedDealSampler::rollingStats() const {
    return RollingStats{
        m_total_attempts.load(std::memory_order_relaxed),
        m_total_accepted.load(std::memory_order_relaxed)
    };
}

std::vector<Deal> TrumpConstrainedDealSampler::enumerateDeals(
    const PublicState& public_state, size_t max_deals) const {
    std::vector<Deal> all = m_uniform_sampler.enumerateDeals(public_state, max_deals);
    if (!shouldApplyConstraint(public_state))
        return all;
    std::vector<Deal> out;
    out.reserve(all.size());
    for (const Deal& d : all)
        if (prior::trump_consistency(reconstructOriginalDeclarerHand(public_state, d),
                                     public_state.trump_suit) == 1.0)
            out.push_back(d);
    return out;
}

int64_t TrumpConstrainedDealSampler::reconstructOriginalDeclarerHand(
    const PublicState& public_state,
    const Deal& deal
) {
    if (public_state.declarer < 0 || public_state.declarer >= 4) {
        throw std::invalid_argument(
            "TrumpConstrainedDealSampler: declarer must be in range 0..3 to reconstruct."
        );
    }

    int64_t original_hand = deal.hands[public_state.declarer];

    for (int i = 0; i < public_state.n_played_cards; ++i) {
        if (public_state.played_players[i] == public_state.declarer
            && public_state.played_cards[i] >= 0) {
            original_hand |= int64_t(1) << public_state.played_cards[i];
        }
    }

    for (int i = 0; i < public_state.n_trick; ++i) {
        if (public_state.trick_players[i] == public_state.declarer
            && public_state.trick_cards[i] >= 0) {
            original_hand |= int64_t(1) << public_state.trick_cards[i];
        }
    }

    if (popcount(original_hand) != 8) {
        throw std::runtime_error(
            "TrumpConstrainedDealSampler: reconstructed declarer hand does not contain 8 cards."
        );
    }

    return original_hand;
}

} // namespace sampler
