#include "sampler/uniform.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include "bitops.h"

namespace sampler {
namespace {

constexpr int64_t FULL_DECK_MASK = int64_t(0xFFFFFFFF);

int popcountMask(int64_t mask) {
    return manille::popcount64(static_cast<unsigned long long>(mask));
}

int suitOf(int card_idx) {
    return card_idx / 8;
}

int rankOf(int card_idx) {
    return card_idx % 8;
}

int64_t suitMask(int suit) {
    return int64_t(0xFF) << (suit * 8);
}

int64_t nonTrumpMask(int trump_suit) {
    return trump_suit >= 0
        ? (~suitMask(trump_suit) & FULL_DECK_MASK)
        : FULL_DECK_MASK;
}

int findWinner(
    const std::array<int32_t, 4>& trick_cards,
    int trump_suit,
    int n_cards
) {
    int winning_card = trick_cards[0];
    int winning_idx = 0;
    for (int i = 1; i < n_cards; ++i) {
        const int card = trick_cards[i];
        const int card_suit = suitOf(card);
        const int winning_suit = suitOf(winning_card);
        if (card_suit == winning_suit) {
            if (rankOf(card) > rankOf(winning_card)) {
                winning_card = card;
                winning_idx = i;
            }
        } else if (card_suit == trump_suit) {
            winning_card = card;
            winning_idx = i;
        }
    }
    return winning_idx;
}

} // namespace

PublicState PublicState::fromGame(const ManilleGame& game, int observer) {
    PublicState public_state;
    public_state.observer = observer;
    public_state.current_player = static_cast<int>(game.currentPlayer());
    public_state.trump_suit = game.trumpSuit();
    public_state.declarer = game.declarer();
    public_state.pts0 = game.teamPoints(0);
    public_state.pts1 = game.teamPoints(1);
    public_state.tricks_won0 = game.tricksWon(0);
    public_state.tricks_won1 = game.tricksWon(1);
    public_state.n_trick = game.trickSize();
    public_state.n_played_cards = game.playedCardCount();
    public_state.observer_hand = game.handMask(observer);

    for (int player = 0; player < 4; ++player)
        public_state.hand_sizes[player] = popcountMask(game.handMask(player));

    for (int i = 0; i < public_state.n_trick; ++i) {
        public_state.trick_cards[i] = game.trickCard(i);
        public_state.trick_players[i] = game.trickPlayer(i);
    }

    for (int i = 0; i < public_state.n_played_cards; ++i) {
        public_state.played_cards[i] = game.playedCard(i);
        public_state.played_players[i] = game.playedPlayer(i);
    }

    return public_state;
}

Deal UniformDealSampler::sampleDeal(const PublicState& public_state) const {
    auto deals = sampleDeals(public_state, 1, false);
    return deals[0];
}

std::vector<Deal> UniformDealSampler::sampleDeals(
    const PublicState& public_state,
    int n,
    bool enumerate_if_small
) const {
    if (n <= 0)
        return {};
    SamplingProblem problem = buildProblem(public_state);

    std::unordered_map<uint64_t, uint64_t> memo;
    memo.reserve(problem.domains.size() * 64);

    std::array<int, 4> remaining = problem.remaining;
    const uint64_t completion_count = countAssignments(problem, 0, remaining, memo);
    if (completion_count == 0) {
        throw std::runtime_error(
            "UniformDealSampler: public state has no feasible hidden-card assignment."
        );
    }

    const int partner = (public_state.observer + 2) % 4;
    if (enumerate_if_small && completion_count <= static_cast<uint64_t>(n)) {
        std::vector<Deal> out;
        out.reserve(static_cast<size_t>(completion_count));
        std::array<int64_t, 4> hands{0, 0, 0, 0};
        hands[public_state.observer] = public_state.observer_hand;
        if (public_state.partner_hand != 0)
            hands[partner] = public_state.partner_hand;
        remaining = problem.remaining;
        enumerateAssignments(problem, 0, remaining, memo, hands, out);
        return out;
    }

    std::vector<Deal> out;
    out.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        Deal deal;
        deal.hands.fill(0);
        deal.hands[public_state.observer] = public_state.observer_hand;
        if (public_state.partner_hand != 0)
            deal.hands[partner] = public_state.partner_hand;
        remaining = problem.remaining;
        sampleAssignments(problem, 0, remaining, memo, deal.hands);
        out.push_back(deal);
    }
    return out;
}

std::vector<Deal> UniformDealSampler::sampleDealsBelief(
    const PublicState& public_state,
    int n,
    const float* belief,
    double belief_w,
    bool enumerate_if_small,
    bool* out_enumerated
) const {
    if (out_enumerated)
        *out_enumerated = false;
    if (n <= 0)
        return {};
    if (belief == nullptr || belief_w <= 0.0) {
        auto deals = sampleDeals(public_state, n, enumerate_if_small);
        if (out_enumerated && enumerate_if_small && static_cast<int>(deals.size()) < n)
            *out_enumerated = true;
        return deals;
    }

    SamplingProblem problem = buildProblem(public_state);
    const int M = static_cast<int>(problem.domains.size());
    const int obs = public_state.observer;
    const int partner = (obs + 2) % 4;

    if (enumerate_if_small && M <= 12) {
        std::unordered_map<uint64_t, uint64_t> memo;
        memo.reserve(static_cast<size_t>(M + 1) * 64);
        std::array<int, 4> rem = problem.remaining;
        const uint64_t cnt = countAssignments(problem, 0, rem, memo);
        if (cnt == 0)
            throw std::runtime_error("UniformDealSampler: public state has no feasible hidden-card assignment.");
        if (cnt <= static_cast<uint64_t>(n)) {
            std::vector<Deal> out;
            out.reserve(static_cast<size_t>(cnt));
            std::array<int64_t, 4> hands{0, 0, 0, 0};
            hands[obs] = public_state.observer_hand;
            if (public_state.partner_hand != 0)
                hands[partner] = public_state.partner_hand;
            rem = problem.remaining;
            enumerateAssignments(problem, 0, rem, memo, hands, out);
            if (out_enumerated)
                *out_enumerated = true;
            return out;
        }
    }

    const int s0 = (obs + 1) % 4;
    const int s1 = (obs + 2) % 4;
    const int s2 = (obs + 3) % 4;
    const int R0 = problem.remaining[s0];
    const int R1 = problem.remaining[s1];
    const int R2 = problem.remaining[s2];
    if (M > 24 || R0 > 8 || R1 > 8 || R2 > 8 || R0 + R1 + R2 != M)
        return sampleDeals(public_state, n, enumerate_if_small);

    // Precompute normalized per-card weights for exact belief-weighted DP.
    double cw[24][3]{};
    for (int idx = 0; idx < M; ++idx) {
        const CardDomain& dom = problem.domains[idx];
        double mx = 0.0;
        for (int k = 0; k < dom.count; ++k) {
            const int p = dom.players[k];
            const int rel = ((p - obs) % 4 + 4) % 4 - 1;  // 0, 1, 2
            const double w = std::pow(double(belief[rel * 32 + dom.card]) + 1e-9, belief_w);
            cw[idx][k] = w;
            if (w > mx) mx = w;
        }
        if (mx > 0.0) {
            for (int k = 0; k < dom.count; ++k)
                cw[idx][k] /= mx;
        }
    }

    // Bottom-up weighted DP: dp[idx][r0][r1] = partition function for domains[idx..M-1]
    // with remaining counts (r0, r1, r2 = (M - idx) - r0 - r1).
    double dp[25][9][9];
    std::memset(dp, 0, sizeof(dp));
    dp[M][0][0] = 1.0;

    for (int idx = M - 1; idx >= 0; --idx) {
        const CardDomain& dom = problem.domains[idx];
        const int rem_cards = M - idx;
        const int r0_max = std::min(R0, rem_cards);
        for (int r0 = 0; r0 <= r0_max; ++r0) {
            const int r1_max = std::min(R1, rem_cards - r0);
            const int r1_min = std::max(0, rem_cards - r0 - R2);
            for (int r1 = r1_min; r1 <= r1_max; ++r1) {
                const int r2 = rem_cards - r0 - r1;
                double acc = 0.0;
                for (int k = 0; k < dom.count; ++k) {
                    const int p = dom.players[k];
                    double child = 0.0;
                    if (p == s0) {
                        if (r0 > 0) child = dp[idx + 1][r0 - 1][r1];
                    } else if (p == s1) {
                        if (r1 > 0) child = dp[idx + 1][r0][r1 - 1];
                    } else {
                        if (r2 > 0) child = dp[idx + 1][r0][r1];
                    }
                    acc += cw[idx][k] * child;
                }
                dp[idx][r0][r1] = acc;
            }
        }
    }

    if (!(dp[0][R0][R1] > 0.0))
        throw std::runtime_error("UniformDealSampler: public state has no feasible hidden-card assignment.");

    std::uniform_real_distribution<double> u01(0.0, 1.0);
    auto& rng = prng();
    std::vector<Deal> out;
    out.reserve(static_cast<size_t>(n));

    for (int i = 0; i < n; ++i) {
        Deal deal;
        deal.hands.fill(0);
        deal.hands[obs] = public_state.observer_hand;
        if (public_state.partner_hand != 0)
            deal.hands[partner] = public_state.partner_hand;
        int r0 = R0, r1 = R1, r2 = R2;
        for (int idx = 0; idx < M; ++idx) {
            const CardDomain& dom = problem.domains[idx];
            double w[3] = {0.0, 0.0, 0.0};
            double tot = 0.0;
            for (int k = 0; k < dom.count; ++k) {
                const int p = dom.players[k];
                double child = 0.0;
                if (p == s0) {
                    if (r0 > 0) child = dp[idx + 1][r0 - 1][r1];
                } else if (p == s1) {
                    if (r1 > 0) child = dp[idx + 1][r0][r1 - 1];
                } else {
                    if (r2 > 0) child = dp[idx + 1][r0][r1];
                }
                w[k] = cw[idx][k] * child;
                tot += w[k];
            }
            double target = u01(rng) * tot;
            int chosen_k = dom.count - 1;
            for (int k = 0; k < dom.count; ++k) {
                if (w[k] <= 0.0) continue;
                chosen_k = k;
                if (target < w[k]) break;
                target -= w[k];
            }
            const int p = dom.players[chosen_k];
            deal.hands[p] |= int64_t(1) << dom.card;
            if (p == s0) --r0;
            else if (p == s1) --r1;
            else --r2;
        }
        out.push_back(deal);
    }
    return out;
}

std::vector<Deal> UniformDealSampler::enumerateDeals(
    const PublicState& public_state, size_t max_deals) const {
    SamplingProblem problem = buildProblem(public_state);
    std::unordered_map<uint64_t, uint64_t> memo;
    memo.reserve(problem.domains.size() * 64);
    std::array<int, 4> remaining = problem.remaining;
    const uint64_t total = countAssignments(problem, 0, remaining, memo);
    std::vector<Deal> out;
    if (total == 0 || total > max_deals)
        return out;
    out.reserve(static_cast<size_t>(total));
    std::array<int64_t, 4> hands{0, 0, 0, 0};
    hands[public_state.observer] = public_state.observer_hand;
    const int partner = (public_state.observer + 2) % 4;
    if (public_state.partner_hand != 0)
        hands[partner] = public_state.partner_hand;
    remaining = problem.remaining;
    enumerateAssignments(problem, 0, remaining, memo, hands, out);
    return out;
}

void UniformDealSampler::enumerateAssignments(
    const SamplingProblem& problem,
    int index,
    std::array<int, 4>& remaining,
    std::unordered_map<uint64_t, uint64_t>& memo,
    std::array<int64_t, 4>& hands,
    std::vector<Deal>& out
) {
    if (index == static_cast<int>(problem.domains.size())) {
        Deal d;
        d.hands = hands;
        out.push_back(d);
        return;
    }
    const CardDomain& domain = problem.domains[index];
    for (int i = 0; i < domain.count; ++i) {
        const int player = domain.players[i];
        if (remaining[player] == 0)
            continue;
        remaining[player]--;
        // prune branches with no completion (same counts sampleAssignments uses)
        if (countAssignments(problem, index + 1, remaining, memo) > 0) {
            hands[player] |= int64_t(1) << domain.card;
            enumerateAssignments(problem, index + 1, remaining, memo, hands, out);
            hands[player] &= ~(int64_t(1) << domain.card);
        }
        remaining[player]++;
    }
}

UniformDealSampler::SamplingProblem
UniformDealSampler::buildProblem(const PublicState& public_state) {
    SamplingProblem problem;
    problem.remaining = public_state.hand_sizes;

    const int observer_cards = popcount(public_state.observer_hand);
    problem.remaining[public_state.observer] -= observer_cards;
    if (problem.remaining[public_state.observer] != 0) {
        throw std::invalid_argument(
            "UniformDealSampler: observer hand size does not match public hand count."
        );
    }

    const int partner = (public_state.observer + 2) % 4;
    if (public_state.partner_hand != 0) {
        problem.remaining[partner] -= popcount(public_state.partner_hand);
        if (problem.remaining[partner] != 0) {
            throw std::invalid_argument(
                "UniformDealSampler: partner hand size does not match public hand count."
            );
        }
    }

    const auto impossible_masks = historyImpossibleMasks(public_state);

    int64_t unseen_mask = FULL_DECK_MASK & ~public_state.observer_hand
        & ~public_state.partner_hand;
    for (int i = 0; i < public_state.n_played_cards; ++i)
        unseen_mask &= ~(int64_t(1) << public_state.played_cards[i]);
    for (int i = 0; i < public_state.n_trick; ++i)
        unseen_mask &= ~(int64_t(1) << public_state.trick_cards[i]);

    for (int card = 0; card < 32; ++card) {
        if ((unseen_mask & (int64_t(1) << card)) == 0)
            continue;

        CardDomain domain;
        domain.card = card;
        for (int player = 0; player < 4; ++player) {
            if (problem.remaining[player] <= 0 || player == public_state.observer
                || (public_state.partner_hand != 0 && player == partner))
                continue;
            if ((impossible_masks[player] & (int64_t(1) << card)) != 0)
                continue;
            domain.players[domain.count++] = player;
        }

        if (domain.count == 0) {
            throw std::runtime_error(
                "UniformDealSampler: no eligible hidden hand for an unseen card."
            );
        }

        problem.domains.push_back(domain);
    }

    std::sort(problem.domains.begin(), problem.domains.end(),
              [](const CardDomain& lhs, const CardDomain& rhs) {
                  if (lhs.count != rhs.count)
                      return lhs.count < rhs.count;
                  return lhs.card < rhs.card;
              });

    return problem;
}

std::array<int64_t, 4>
UniformDealSampler::historyImpossibleMasks(const PublicState& public_state) {
    std::array<int64_t, 4> impossible{0, 0, 0, 0};

    auto mark_public = [&impossible](int card_idx) {
        const int64_t bit = int64_t(1) << card_idx;
        for (int player = 0; player < 4; ++player)
            impossible[player] |= bit;
    };

    auto add_history_inference = [&](const std::array<int32_t, 4>& trick_cards,
                                     const std::array<int32_t, 4>& trick_players,
                                     int n_before,
                                     int player,
                                     int card_idx) {
        if (n_before == 0)
            return;

        const int led_suit = suitOf(trick_cards[0]);
        const int card_suit = suitOf(card_idx);
        const int partner = (player + 2) % 4;
        const int winner_idx = findWinner(trick_cards, public_state.trump_suit, n_before);
        const int winning_card = trick_cards[winner_idx];
        const int winning_player = trick_players[winner_idx];
        const bool partner_winning = winning_player == partner;
        const int winning_suit = suitOf(winning_card);

        if (card_suit != led_suit) {
            impossible[player] |= suitMask(led_suit);

            if (partner_winning || public_state.trump_suit < 0)
                return;

            if (card_suit != public_state.trump_suit) {
                if (winning_suit != public_state.trump_suit) {
                    impossible[player] |= suitMask(public_state.trump_suit);
                } else {
                    for (int rank = rankOf(winning_card) + 1; rank < 8; ++rank)
                        impossible[player] |= int64_t(1) << (public_state.trump_suit * 8 + rank);
                }
                return;
            }

            if (winning_suit == public_state.trump_suit
                && rankOf(card_idx) < rankOf(winning_card)) {
                for (int rank = rankOf(winning_card) + 1; rank < 8; ++rank)
                    impossible[player] |= int64_t(1) << (public_state.trump_suit * 8 + rank);
                impossible[player] |= nonTrumpMask(public_state.trump_suit) & ~suitMask(led_suit);
            }
            return;
        }

        if (partner_winning)
            return;

        if (winning_suit == led_suit && rankOf(card_idx) < rankOf(winning_card)) {
            for (int rank = rankOf(winning_card) + 1; rank < 8; ++rank)
                impossible[player] |= int64_t(1) << (led_suit * 8 + rank);
        }
    };

    for (int i = 0; i < public_state.n_played_cards; i += 4) {
        std::array<int32_t, 4> trick_cards{};
        std::array<int32_t, 4> trick_players{};
        for (int j = 0; j < 4; ++j) {
            const int card_idx = public_state.played_cards[i + j];
            const int player = public_state.played_players[i + j];
            add_history_inference(trick_cards, trick_players, j, player, card_idx);
            trick_cards[j] = card_idx;
            trick_players[j] = player;
            mark_public(card_idx);
        }
    }

    std::array<int32_t, 4> partial_cards{};
    std::array<int32_t, 4> partial_players{};
    for (int i = 0; i < public_state.n_trick; ++i) {
        const int card_idx = public_state.trick_cards[i];
        const int player = public_state.trick_players[i];
        add_history_inference(partial_cards, partial_players, i, player, card_idx);
        partial_cards[i] = card_idx;
        partial_players[i] = player;
        mark_public(card_idx);
    }

    impossible[public_state.observer] |= ~public_state.observer_hand & FULL_DECK_MASK;
    for (int player = 0; player < 4; ++player) {
        if (player == public_state.observer)
            continue;
        impossible[player] |= public_state.observer_hand;
    }

    if (public_state.partner_hand != 0) {
        const int partner = (public_state.observer + 2) % 4;
        impossible[partner] |= ~public_state.partner_hand & FULL_DECK_MASK;
        for (int player = 0; player < 4; ++player) {
            if (player == public_state.observer || player == partner)
                continue;
            impossible[player] |= public_state.partner_hand;
        }
    }

    return impossible;
}

uint64_t UniformDealSampler::countAssignments(
    const SamplingProblem& problem,
    int index,
    std::array<int, 4>& remaining,
    std::unordered_map<uint64_t, uint64_t>& memo
) {
    int cards_left = 0;
    for (int player = 0; player < 4; ++player) {
        if (remaining[player] < 0)
            return 0;
        cards_left += remaining[player];
    }
    if (cards_left != static_cast<int>(problem.domains.size()) - index)
        return 0;
    if (index == static_cast<int>(problem.domains.size()))
        return 1;

    const uint64_t key = memoKey(index, remaining);
    auto it = memo.find(key);
    if (it != memo.end())
        return it->second;

    const CardDomain& domain = problem.domains[index];
    uint64_t total = 0;
    for (int i = 0; i < domain.count; ++i) {
        const int player = domain.players[i];
        if (remaining[player] == 0)
            continue;

        remaining[player]--;
        total += countAssignments(problem, index + 1, remaining, memo);
        remaining[player]++;
    }

    memo.emplace(key, total);
    return total;
}

void UniformDealSampler::sampleAssignments(
    const SamplingProblem& problem,
    int index,
    std::array<int, 4>& remaining,
    std::unordered_map<uint64_t, uint64_t>& memo,
    std::array<int64_t, 4>& hands
) {
    if (index == static_cast<int>(problem.domains.size()))
        return;

    const CardDomain& domain = problem.domains[index];
    std::array<uint64_t, 3> branch_counts{0, 0, 0};
    uint64_t total = 0;

    for (int i = 0; i < domain.count; ++i) {
        const int player = domain.players[i];
        if (remaining[player] == 0)
            continue;

        remaining[player]--;
        branch_counts[i] = countAssignments(problem, index + 1, remaining, memo);
        remaining[player]++;
        total += branch_counts[i];
    }

    if (total == 0) {
        throw std::runtime_error(
            "UniformDealSampler: reached an infeasible branch while sampling."
        );
    }

    std::uniform_int_distribution<uint64_t> dist(0, total - 1);
    uint64_t pick = dist(prng());

    for (int i = 0; i < domain.count; ++i) {
        const uint64_t branch = branch_counts[i];
        if (branch == 0)
            continue;

        if (pick < branch) {
            const int player = domain.players[i];
            remaining[player]--;
            hands[player] |= int64_t(1) << domain.card;
            sampleAssignments(problem, index + 1, remaining, memo, hands);
            return;
        }
        pick -= branch;
    }

    throw std::runtime_error("UniformDealSampler: failed to select a sampling branch.");
}

uint64_t UniformDealSampler::memoKey(int index, const std::array<int, 4>& remaining) {
    uint64_t key = static_cast<uint64_t>(index);
    for (int player = 0; player < 4; ++player)
        key = (key << 4) | static_cast<uint64_t>(remaining[player]);
    return key;
}

int UniformDealSampler::popcount(int64_t mask) {
    return manille::popcount64(static_cast<unsigned long long>(mask));
}

std::mt19937_64& UniformDealSampler::prng() {
    thread_local static std::mt19937_64 engine{std::random_device{}()};
    return engine;
}

void UniformDealSampler::seedPrng(uint64_t seed) {
    prng().seed(seed);
}

ManilleGame gameFromDeal(
    const PublicState& public_state,
    const Deal& deal,
    bool perfect_info
) {
    return ManilleGame(
        deal.hands,
        public_state.trick_cards,
        public_state.trick_players,
        public_state.n_trick,
        public_state.current_player,
        public_state.trump_suit,
        public_state.pts0,
        public_state.pts1,
        public_state.tricks_won0,
        public_state.tricks_won1,
        public_state.played_cards,
        public_state.played_players,
        public_state.n_played_cards,
        perfect_info,
        public_state.declarer
    );
}

} // namespace sampler
