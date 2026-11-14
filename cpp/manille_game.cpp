#include "manille_game.h"

#include "sampler/trump_constrained.h"

#include <algorithm>
#include <stdexcept>

static const int POINTS[] = {0, 0, 0, 1, 2, 3, 4, 5,   // suit 0
                              0, 0, 0, 1, 2, 3, 4, 5,   // suit 1
                              0, 0, 0, 1, 2, 3, 4, 5,   // suit 2
                              0, 0, 0, 1, 2, 3, 4, 5};  // suit 3

ManilleGame::ManilleGame(
    const std::array<int64_t, 4>& hands,
    const std::array<int32_t, 4>& trick_cards,
    const std::array<int32_t, 4>& trick_players,
    int n_trick,
    int current_player,
    int trump_suit,
    int pts0, int pts1,
    int tricks_won0, int tricks_won1,
    bool perfect_info,
    int declarer)
    : ManilleGame(
          hands,
          trick_cards,
          trick_players,
          n_trick,
          current_player,
          trump_suit,
          pts0,
          pts1,
          tricks_won0,
          tricks_won1,
          {},
          {},
          0,
          perfect_info,
          declarer)
{
}

ManilleGame::ManilleGame(
    const std::array<int64_t, 4>& hands,
    const std::array<int32_t, 4>& trick_cards,
    const std::array<int32_t, 4>& trick_players,
    int n_trick,
    int current_player,
    int trump_suit,
    int pts0, int pts1,
    int tricks_won0, int tricks_won1,
    const std::array<int32_t, 32>& played_cards,
    const std::array<int32_t, 32>& played_players,
    int n_played_cards,
    bool perfect_info,
    int declarer)
    : m_hands(hands)
    , m_trump(trump_suit)
    , m_declarer(declarer)
    , m_current_player(current_player)
    , m_trick_cards(trick_cards)
    , m_trick_players(trick_players)
    , m_n_trick(n_trick)
    , m_pts{pts0, pts1}
    , m_tricks_won{tricks_won0, tricks_won1}
    , m_finished(false)
    // Mull (no trump, trump_suit < 0) doubles the stake at trump selection;
    // the C++ constructor IS the trump selection. Announcements multiply
    // further via announceDouble/announceRedouble.
    , m_multiplier(trump_suit < 0 ? 2 : 1)
    , m_doubled(false)
    , m_redoubled(false)
    // The announcement window (Python's GamePhase.DOUBLING) is open only
    // before any card has been played.
    , m_doubling(n_trick == 0 && n_played_cards == 0)
    , m_perfect_info(perfect_info)
    , m_played_cards(played_cards)
    , m_played_players(played_players)
    , m_n_played_cards(n_played_cards)
{
}

ManilleGame::Clone ManilleGame::cloneAndRandomise(Player observer) const {
    auto clone = std::make_unique<ManilleGame>(*this);

    // Perfect info: all hands are known, no randomisation needed
    if (m_perfect_info)
        return clone;

    // Particle pool: draw a random particle and use its non-observer hands
    if (!m_particle_pool.empty()) {
        size_t particle_idx = 0;
        if (!m_particle_cumulative_weights.empty()) {
            std::uniform_real_distribution<double> dist(
                0.0,
                m_particle_cumulative_weights.back()
            );
            const double sample = dist(prng());
            auto it = std::lower_bound(
                m_particle_cumulative_weights.begin(),
                m_particle_cumulative_weights.end(),
                sample
            );
            particle_idx = (it == m_particle_cumulative_weights.end())
                ? (m_particle_pool.size() - 1)
                : static_cast<size_t>(std::distance(m_particle_cumulative_weights.begin(), it));
        } else {
            std::uniform_int_distribution<size_t> dist(0, m_particle_pool.size() - 1);
            particle_idx = dist(prng());
        }

        const auto& particle = m_particle_pool[particle_idx];
        for (int p = 0; p < 4; ++p) {
            if (p != static_cast<int>(observer))
                clone->m_hands[p] = particle[p];
        }
        return clone;
    }

    // Delegate to the shared trump-constrained sampler so every caller runs
    // through one code path. The sampler is stateless (its PRNG is thread_local).
    static const sampler::TrumpConstrainedDealSampler s_sampler;
    auto public_state = sampler::PublicState::fromGame(*this, static_cast<int>(observer));
    if (m_partner_transparent)
        public_state.partner_hand = m_hands[(static_cast<int>(observer) + 2) % 4];
    const sampler::Deal deal = s_sampler.sampleDeal(public_state);
    clone->m_hands = deal.hands;
    return clone;
}

ISMCTS::Game<Card>::Player ManilleGame::currentPlayer() const {
    return static_cast<Player>(m_current_player);
}

std::vector<ISMCTS::Game<Card>::Player> ManilleGame::players() const {
    return {0, 1, 2, 3};
}

int ManilleGame::findWinner(int n) const {
    return trickWinner(m_trick_cards, n, m_trump);
}

int ManilleGame::findWinner(
    const std::array<int32_t, 4>& trick_cards,
    const std::array<int32_t, 4>&,
    int n) const
{
    return trickWinner(trick_cards, n, m_trump);
}

int ManilleGame::trickWinner(
    const std::array<int32_t, 4>& trick_cards,
    int n,
    int trump)
{
    int winning_card = trick_cards[0];
    int winning_idx = 0;
    for (int i = 1; i < n; ++i) {
        int tc = trick_cards[i];
        int tc_suit = suitOf(tc);
        int wc_suit = suitOf(winning_card);
        if (tc_suit == wc_suit) {
            if (rankOf(tc) > rankOf(winning_card)) {
                winning_card = tc;
                winning_idx = i;
            }
        } else if (tc_suit == trump) {
            winning_card = tc;
            winning_idx = i;
        }
    }
    return winning_idx;
}

int64_t ManilleGame::legalMoveMask() const {
    int64_t hand = m_hands[m_current_player];

    // First to play or no trick started: any card
    if (m_n_trick == 0)
        return hand;

    int led_suit = suitOf(m_trick_cards[0]);
    int64_t suit_mask = suitMask(led_suit);
    int64_t following = hand & suit_mask;

    if (following) {
        // Can follow suit
        int winner_idx = findWinner(m_n_trick);
        int winning_card = m_trick_cards[winner_idx];
        int winning_player = m_trick_players[winner_idx];
        int partner_id = (m_current_player + 2) % 4;

        if (winning_player == partner_id)
            return following;

        // Must beat if possible (within led suit)
        // When following suit, we can only beat with a higher-ranked card
        // of the same suit. If the winning card is a trump of a different
        // suit, no following card can beat it.
        int64_t beating = 0;
        if (suitOf(winning_card) == led_suit) {
            for (int bit = led_suit * 8; bit < led_suit * 8 + 8; ++bit) {
                if (following & (int64_t(1) << bit)) {
                    if (rankOf(bit) > rankOf(winning_card))
                        beating |= (int64_t(1) << bit);
                }
            }
        }
        return beating ? beating : following;
    }

    // Cannot follow suit
    int partner_id = (m_current_player + 2) % 4;

    int winner_idx = findWinner(m_n_trick);
    int winning_card = m_trick_cards[winner_idx];
    int winning_player = m_trick_players[winner_idx];

    if (winning_player == partner_id)
        return hand;

    int64_t trump_mask = (m_trump >= 0) ? suitMask(m_trump) : 0;
    int64_t trumps = hand & trump_mask;
    int64_t non_trumps = hand & ~trump_mask;

    if (!trumps)
        return hand;

    bool winning_is_trump = (suitOf(winning_card) == m_trump);

    if (winning_is_trump) {
        // Must overtrump if possible
        int64_t higher_trumps = 0;
        for (int bit = m_trump * 8; bit < m_trump * 8 + 8; ++bit) {
            if (trumps & (int64_t(1) << bit)) {
                if (rankOf(bit) > rankOf(winning_card))
                    higher_trumps |= (int64_t(1) << bit);
            }
        }
        if (higher_trumps) return higher_trumps;
        if (non_trumps) return non_trumps;
        return trumps;  // forced to undertrump
    } else {
        return trumps;
    }
}

std::vector<Card> ManilleGame::validMoves() const {
    if (m_finished)
        return {};

    int64_t mask = legalMoveMask();
    std::vector<Card> moves;
    moves.reserve(8);
    for (int bit = 0; bit < 32; ++bit) {
        if (mask & (int64_t(1) << bit))
            moves.push_back(Card::from_index(bit));
    }
    return moves;
}

void ManilleGame::doMove(Card const move) {
    // Playing a card closes the announcement window (Python gates play on
    // GamePhase.PLAYING; the C++ engine transitions implicitly so the
    // solvers need no startPlaying call).
    m_doubling = false;

    int card_idx = move.index();
    m_hands[m_current_player] &= ~(int64_t(1) << card_idx);
    m_trick_cards[m_n_trick] = card_idx;
    m_trick_players[m_n_trick] = m_current_player;
    m_n_trick++;

    if (m_n_trick == 4) {
        resolveTrick();
    } else {
        m_current_player = (m_current_player + 1) % 4;
    }
}

void ManilleGame::resolveTrick() {
    for (int i = 0; i < 4; ++i) {
        m_played_cards[m_n_played_cards + i] = m_trick_cards[i];
        m_played_players[m_n_played_cards + i] = m_trick_players[i];
    }
    m_n_played_cards += 4;

    int trick_points = 0;
    for (int i = 0; i < 4; ++i)
        trick_points += POINTS[m_trick_cards[i]];

    int winner_idx = findWinner(4);
    int winning_player = m_trick_players[winner_idx];
    int winning_team = winning_player % 2;

    m_pts[winning_team] += trick_points;
    m_tricks_won[winning_team]++;

    // Check if game is over (all hands empty)
    if (m_hands[0] == 0 && m_hands[1] == 0 &&
        m_hands[2] == 0 && m_hands[3] == 0) {
        m_finished = true;
        return;
    }

    m_current_player = winning_player;
    m_n_trick = 0;
}

void ManilleGame::restoreStakes(bool doubled, bool redoubled, bool doubling_open) {
    m_doubled = doubled;
    m_redoubled = doubled && redoubled;
    m_multiplier = (m_trump < 0 ? 2 : 1) * (m_doubled ? 2 : 1) * (m_redoubled ? 2 : 1);
    m_doubling = doubling_open && m_n_trick == 0 && m_n_played_cards == 0;
    m_finished = m_hands[0] == 0 && m_hands[1] == 0 && m_hands[2] == 0 && m_hands[3] == 0;
}

bool ManilleGame::announceDouble(Player player) {
    // "Ik ga mee": the defending team doubles the stake. Only during the
    // announcement window, only the team that did NOT make trump, and only
    // once — it is a team call, so a second opponent cannot compound it.
    if (!m_doubling || m_doubled || m_declarer < 0)
        return false;
    if (teamOf(player) == teamOf(m_declarer))
        return false;

    m_doubled = true;
    m_multiplier *= 2;
    return true;
}

bool ManilleGame::announceRedouble(Player player) {
    // "Ik ga tegen": the trump-making team answers a meegaan. Only during
    // the announcement window, only the trump-making team, only in answer
    // to an actual meegaan, and only once. This MUST gate on the
    // doubled/redoubled flags, never on the multiplier: mull already sets
    // the multiplier to 2 on its own (game_logic.py c63fdbd).
    if (!m_doubling || !m_doubled || m_redoubled || m_declarer < 0)
        return false;
    if (teamOf(player) != teamOf(m_declarer))
        return false;

    m_redoubled = true;
    m_multiplier *= 2;
    return true;
}

int ManilleGame::handScore(int team) const {
    // game_logic._score_hand: only the winner scores, a 30-30 tie scores 0
    // (no carry-over). Terminal-state semantics, like getResult.
    if (m_pts[team] <= m_pts[1 - team])
        return 0;
    return (m_pts[team] - 30) * m_multiplier;
}

double ManilleGame::getResult(Player player) const {
    // MCTS utility: signed margin, this class's long-standing objective.
    // It is a better model of true match scoring than winner-only-flat:
    // in a race to 101 the losing side must still DENY the winner's
    // margin — every point past 30 is real match progress (× multiplier),
    // but winner-only flattens every loss to a constant 0, and the
    // 2026-07-27 variant measurably weakened the ISMCTS anchor (+4.5 vs
    // +2.6 at 150 deals, same deals/seed). Match accounting lives in
    // handScore; rollout values stay here.
    return (m_pts[teamOf(player)] - 30.0) / 30.0;
}
