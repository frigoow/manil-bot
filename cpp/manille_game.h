#ifndef MANILLE_GAME_H
#define MANILLE_GAME_H

#include "card.h"
#include <ismcts/game.h>

#include <array>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

class ManilleGame : public ISMCTS::POMGame<Card> {
public:
    ManilleGame(
        const std::array<int64_t, 4>& hands,
        const std::array<int32_t, 4>& trick_cards,
        const std::array<int32_t, 4>& trick_players,
        int n_trick,
        int current_player,
        int trump_suit,
        int pts0, int pts1,
        int tricks_won0, int tricks_won1,
        bool perfect_info = false,
        int declarer = -1
    );

    ManilleGame(
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
        bool perfect_info = false,
        int declarer = -1
    );

    // ISMCTS interface
    Clone cloneAndRandomise(Player observer) const override;
    Player currentPlayer() const override;
    std::vector<Player> players() const override;
    std::vector<Card> validMoves() const override;
    void doMove(Card const move) override;
    double getResult(Player player) const override;

    // Stakes (match plan M0, mirrors manille_env/game_logic.py): the deal
    // score is (winner_points - 30) * multiplier, ONLY the winner scores,
    // and a 30-30 tie scores 0. Multiplier: base 1, mull x2 (set at
    // construction, the C++ equivalent of trump selection), meegaan x2,
    // terug meegaan x2 -> max 8. One call per side.
    //
    // The announcement window (Python's GamePhase.DOUBLING) opens at
    // construction when no card has been played yet and closes on
    // startPlaying() or the first doMove. It gates ONLY the announce
    // methods; validMoves/doMove are not phase-gated so the solvers, which
    // never announce, run unchanged.
    bool announceDouble(Player player);
    bool announceRedouble(Player player);
    void startPlaying() { m_doubling = false; }
    // Saved-game restore: the stake state the state constructor cannot know.
    // The multiplier is recomputed exactly as the announcements build it
    // (mull x2, meegaan x2, terug x2); the window can only be open before
    // the first card; a deal whose hands are all empty is finished.
    void restoreStakes(bool doubled, bool redoubled, bool doubling_open);
    int multiplier() const { return m_multiplier; }
    bool doubled() const { return m_doubled; }
    bool redoubled() const { return m_redoubled; }
    bool doublingPhase() const { return m_doubling; }
    bool finished() const { return m_finished; }
    // Raw match points this deal adds to a team's score (terminal states):
    // (winner_points - 30) * multiplier for the winner, 0 otherwise.
    int handScore(int team) const;

    // Public accessors for solver wrapper
    int teamOf(Player p) const { return p % 2; }
    int teamPoints(int team) const { return m_pts[team]; }
    int tricksWon(int team) const { return m_tricks_won[team]; }
    int64_t handMask(int player) const { return m_hands[player]; }
    // Partner-transparent determinization (0b.2, eval-only): when enabled,
    // cloneAndRandomise fixes the observer's partner's true hand and samples
    // only the two opponents' hands.
    void setPartnerTransparent(bool on) { m_partner_transparent = on; }
    bool partnerTransparent() const { return m_partner_transparent; }
    int trumpSuit() const { return m_trump; }
    int declarer() const { return m_declarer; }
    int trickSize() const { return m_n_trick; }
    int trickCard(int idx) const { return m_trick_cards[idx]; }
    int trickPlayer(int idx) const { return m_trick_players[idx]; }
    int playedCardCount() const { return m_n_played_cards; }
    int playedCard(int idx) const { return m_played_cards[idx]; }
    int playedPlayer(int idx) const { return m_played_players[idx]; }

    // Index (into trick_cards) of the card currently winning among the first
    // n cards of a trick. Exposed so feature encoders reuse the engine's
    // winner logic instead of reimplementing it.
    static int trickWinner(
        const std::array<int32_t, 4>& trick_cards,
        int n,
        int trump
    );

    // Particle pool: if non-empty, cloneAndRandomise draws from this pool
    // instead of running the CSP sampler. Optional weights let the solver
    // sample worlds according to the PF posterior without duplicating rows.
    void setParticlePool(
        std::vector<std::array<int64_t, 4>> pool,
        std::vector<double> weights = {}
    ) {
        m_particle_pool = std::move(pool);
        m_particle_cumulative_weights.clear();
        if (weights.empty())
            return;

        if (weights.size() != m_particle_pool.size()) {
            throw std::invalid_argument(
                "Particle weight count must match particle pool size."
            );
        }

        m_particle_cumulative_weights.reserve(weights.size());
        double cumulative = 0.0;
        for (double weight : weights) {
            if (weight < 0.0)
                throw std::invalid_argument("Particle weights must be non-negative.");
            cumulative += weight;
            m_particle_cumulative_weights.push_back(cumulative);
        }
        if (cumulative <= 0.0)
            throw std::invalid_argument("Particle weights must sum to a positive value.");
    }

private:
    std::array<int64_t, 4> m_hands;       // bitmask per player
    int m_trump;                           // -1 for no trump
    int m_declarer;                        // -1 when unknown / disabled
    int m_current_player;
    std::array<int32_t, 4> m_trick_cards;
    std::array<int32_t, 4> m_trick_players;
    int m_n_trick;
    std::array<int, 2> m_pts;             // team points: {0,2} vs {1,3}
    std::array<int, 2> m_tricks_won;
    bool m_finished;
    int m_multiplier;                      // 1 base, x2 mull, x2 meegaan, x2 terug
    bool m_doubled;                        // defending team said "ik ga mee"
    bool m_redoubled;                      // trump team answered "ik ga tegen"
    bool m_doubling;                       // announcement window still open
    bool m_perfect_info;
    bool m_partner_transparent = false;
    std::array<int32_t, 32> m_played_cards;
    std::array<int32_t, 32> m_played_players;
    int m_n_played_cards;
    std::vector<std::array<int64_t, 4>> m_particle_pool;
    std::vector<double> m_particle_cumulative_weights;

    int64_t legalMoveMask() const;
    void resolveTrick();
    int findWinner(int n) const;
    int findWinner(
        const std::array<int32_t, 4>& trick_cards,
        const std::array<int32_t, 4>& trick_players,
        int n
    ) const;

    static int suitOf(int card_idx) { return card_idx / 8; }
    static int rankOf(int card_idx) { return card_idx % 8; }
    static int64_t suitMask(int suit) { return int64_t(0xFF) << (suit * 8); }

    static std::mt19937& prng() {
        thread_local static std::mt19937 engine{std::random_device{}()};
        return engine;
    }
};

#endif // MANILLE_GAME_H
