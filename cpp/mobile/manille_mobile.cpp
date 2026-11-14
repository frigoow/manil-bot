#include "mobile/manille_mobile.h"

#include "encoding.h"
#include "manille_game.h"
#include "prior/trump_rule.h"
#include "rule_agent.h"
#include "sampler/trump_constrained.h"

#include <algorithm>
#include <cmath>
#include <array>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct ManilleMobileGame {
    explicit ManilleMobileGame(ManilleGame value) : game(std::move(value)) {}
    ManilleGame game;
};

namespace {

// The C ABI has to spell the dimensions out as literals, so pin them to the
// encoder here: a layout change that forgets the header is a compile error
// instead of a silently mis-shaped tensor on a phone.
static_assert(MANILLE_MOBILE_OBS_DIM == encoding::OBS_DIM, "OBS_DIM drift");
static_assert(MANILLE_MOBILE_ACT_DIM == encoding::ACT_DIM, "ACT_DIM drift");
static_assert(MANILLE_MOBILE_RAW_TOKEN_DIM == encoding::PLY_DIM,
              "RAW_TOKEN_DIM drift");
static_assert(MANILLE_MOBILE_GLOBAL_DIM == 50 + encoding::TACTICAL_DIM,
              "GLOBAL_DIM drift");

thread_local std::string last_error;

int fail(int code, const char* message) {
    last_error = message;
    return code;
}

template <typename Fn>
int guarded(Fn fn) {
    try {
        last_error.clear();
        return fn();
    } catch (const std::invalid_argument& error) {
        return fail(MANILLE_MOBILE_INVALID_ARGUMENT, error.what());
    } catch (const std::exception& error) {
        return fail(MANILLE_MOBILE_ERROR, error.what());
    } catch (...) {
        return fail(MANILLE_MOBILE_ERROR, "Unknown native engine error.");
    }
}

bool valid_player(int32_t player) {
    return player >= 0 && player < 4;
}

int new_random_deal(int32_t dealer, uint64_t seed, int32_t trump_suit,
                    ManilleMobileGame** out_game) {
    if (!valid_player(dealer) || out_game == nullptr)
        return fail(MANILLE_MOBILE_INVALID_ARGUMENT,
                    "Dealer must be 0..3 and out_game must be non-null.");
    if (trump_suit < -1 || trump_suit > 3)
        return fail(MANILLE_MOBILE_INVALID_ARGUMENT,
                    "Trump must be -1 (no trump) or 0..3.");

    std::array<int32_t, 32> deck{};
    for (int card = 0; card < 32; ++card)
        deck[card] = card;
    std::mt19937_64 random(seed);
    std::shuffle(deck.begin(), deck.end(), random);

    std::array<int64_t, 4> hands{};
    for (int player = 0; player < 4; ++player)
        for (int offset = 0; offset < 8; ++offset)
            hands[player] |= int64_t(1) << deck[player * 8 + offset];

    const std::array<int32_t, 4> no_trick{-1, -1, -1, -1};
    const std::array<int32_t, 32> no_history = [] {
        std::array<int32_t, 32> values{};
        values.fill(-1);
        return values;
    }();
    *out_game = new ManilleMobileGame(ManilleGame(
        hands, no_trick, no_trick, 0, (dealer + 1) % 4, trump_suit,
        0, 0, 0, 0, no_history, no_history, 0, false, dealer));
    return MANILLE_MOBILE_OK;
}

void fill_state(const ManilleGame& game, ManilleMobileState* out) {
    std::memset(out, 0, sizeof(*out));
    std::fill(std::begin(out->trick_cards), std::end(out->trick_cards), -1);
    std::fill(std::begin(out->trick_players), std::end(out->trick_players), -1);
    std::fill(std::begin(out->played_cards), std::end(out->played_cards), -1);
    std::fill(std::begin(out->played_players), std::end(out->played_players), -1);

    out->current_player = static_cast<int32_t>(game.currentPlayer());
    out->trump_suit = game.trumpSuit();
    out->declarer = game.declarer();
    out->pts0 = game.teamPoints(0);
    out->pts1 = game.teamPoints(1);
    out->tricks_won0 = game.tricksWon(0);
    out->tricks_won1 = game.tricksWon(1);
    out->multiplier = game.multiplier();
    out->doubled = game.doubled();
    out->redoubled = game.redoubled();
    out->doubling_phase = game.doublingPhase();
    out->finished = game.finished();
    out->n_trick = game.trickSize();
    out->n_played_cards = game.playedCardCount();
    for (int index = 0; index < out->n_trick; ++index) {
        out->trick_cards[index] = game.trickCard(index);
        out->trick_players[index] = game.trickPlayer(index);
    }
    for (int index = 0; index < out->n_played_cards; ++index) {
        out->played_cards[index] = game.playedCard(index);
        out->played_players[index] = game.playedPlayer(index);
    }
}

int64_t move_mask(const ManilleGame& game) {
    int64_t mask = 0;
    for (const Card& card : game.validMoves())
        mask |= int64_t(1) << card.index();
    return mask;
}

int encode_candidates(
    const ManilleGame& game,
    float* act,
    int32_t* legal,
    int32_t* out_n_legal
) {
    std::fill(act, act + MANILLE_MOBILE_MAX_LEGAL * MANILLE_MOBILE_ACT_DIM, 0.0f);
    std::fill(legal, legal + MANILLE_MOBILE_MAX_LEGAL, -1);
    int32_t count = 0;
    for (const Card& card : game.validMoves()) {
        encoding::encode_action(
            game, card.index(),
            act + count * MANILLE_MOBILE_ACT_DIM);
        legal[count++] = card.index();
    }
    *out_n_legal = count;
    return MANILLE_MOBILE_OK;
}

}  // namespace

extern "C" {

int manille_mobile_new_random_deal(
    int32_t dealer,
    uint64_t seed,
    ManilleMobileGame** out_game
) {
    return guarded([&]() -> int {
        if (!valid_player(dealer) || out_game == nullptr)
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT,
                        "Dealer must be 0..3 and out_game must be non-null.");
        std::array<int32_t, 32> deck{};
        for (int card = 0; card < 32; ++card)
            deck[card] = card;
        std::mt19937_64 random(seed);
        std::shuffle(deck.begin(), deck.end(), random);
        std::array<int64_t, 4> hands{};
        for (int player = 0; player < 4; ++player)
            for (int offset = 0; offset < 8; ++offset)
                hands[player] |= int64_t(1) << deck[player * 8 + offset];
        return new_random_deal(dealer, seed, prior::apply_trump_rule(hands[dealer]), out_game);
    });
}

int manille_mobile_new_random_deal_with_trump(
    int32_t dealer, uint64_t seed, int32_t trump_suit, ManilleMobileGame** out_game
) {
    return guarded([&]() -> int { return new_random_deal(dealer, seed, trump_suit, out_game); });
}

int manille_mobile_create(
    const int64_t hands[4],
    const ManilleMobileState* state,
    ManilleMobileGame** out_game
) {
    return guarded([&]() -> int {
        if (hands == nullptr || state == nullptr || out_game == nullptr)
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT,
                        "Hands, state, and out_game must be non-null.");
        if (!valid_player(state->current_player) || !valid_player(state->declarer)
            || state->n_trick < 0 || state->n_trick > 4
            || state->n_played_cards < 0 || state->n_played_cards > 32)
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT, "Invalid game state.");

        std::array<int64_t, 4> hand_array{};
        std::array<int32_t, 4> trick_cards{};
        std::array<int32_t, 4> trick_players{};
        std::array<int32_t, 32> played_cards{};
        std::array<int32_t, 32> played_players{};
        std::copy(hands, hands + 4, hand_array.begin());
        std::copy(state->trick_cards, state->trick_cards + 4, trick_cards.begin());
        std::copy(state->trick_players, state->trick_players + 4, trick_players.begin());
        std::copy(state->played_cards, state->played_cards + 32, played_cards.begin());
        std::copy(state->played_players, state->played_players + 32, played_players.begin());
        auto restored = std::make_unique<ManilleMobileGame>(ManilleGame(
            hand_array, trick_cards, trick_players, state->n_trick,
            state->current_player, state->trump_suit, state->pts0, state->pts1,
            state->tricks_won0, state->tricks_won1, played_cards, played_players,
            state->n_played_cards, false, state->declarer));
        restored->game.restoreStakes(state->doubled != 0, state->redoubled != 0,
                                     state->doubling_phase != 0);
        *out_game = restored.release();
        return MANILLE_MOBILE_OK;
    });
}

void manille_mobile_destroy(ManilleMobileGame* game) {
    delete game;
}

const char* manille_mobile_last_error(void) {
    return last_error.c_str();
}

int manille_mobile_get_state(const ManilleMobileGame* game, ManilleMobileState* out_state) {
    return guarded([&]() -> int {
        if (game == nullptr || out_state == nullptr)
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT, "Game and out_state must be non-null.");
        fill_state(game->game, out_state);
        return MANILLE_MOBILE_OK;
    });
}

int manille_mobile_hand_mask(const ManilleMobileGame* game, int32_t player, int64_t* out_mask) {
    return guarded([&]() -> int {
        if (game == nullptr || out_mask == nullptr || !valid_player(player))
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT, "Invalid hand-mask request.");
        *out_mask = game->game.handMask(player);
        return MANILLE_MOBILE_OK;
    });
}

int manille_mobile_legal_mask(const ManilleMobileGame* game, int64_t* out_mask) {
    return guarded([&]() -> int {
        if (game == nullptr || out_mask == nullptr)
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT, "Game and out_mask must be non-null.");
        *out_mask = move_mask(game->game);
        return MANILLE_MOBILE_OK;
    });
}

int manille_mobile_apply_card(ManilleMobileGame* game, int32_t card) {
    return guarded([&]() -> int {
        if (game == nullptr || card < 0 || card >= 32)
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT, "Card must be 0..31.");
        if ((move_mask(game->game) & (int64_t(1) << card)) == 0)
            return fail(MANILLE_MOBILE_ILLEGAL_MOVE, "Card is not legal in this state.");
        game->game.doMove(Card::from_index(card));
        return MANILLE_MOBILE_OK;
    });
}

int manille_mobile_announce_double(ManilleMobileGame* game, int32_t player) {
    return guarded([&]() -> int {
        if (game == nullptr || !valid_player(player))
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT, "Invalid double request.");
        return game->game.announceDouble(player) ? MANILLE_MOBILE_OK
                                                 : MANILLE_MOBILE_ILLEGAL_MOVE;
    });
}

int manille_mobile_announce_redouble(ManilleMobileGame* game, int32_t player) {
    return guarded([&]() -> int {
        if (game == nullptr || !valid_player(player))
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT, "Invalid redouble request.");
        return game->game.announceRedouble(player) ? MANILLE_MOBILE_OK
                                                   : MANILLE_MOBILE_ILLEGAL_MOVE;
    });
}

int manille_mobile_start_playing(ManilleMobileGame* game) {
    return guarded([&]() -> int {
        if (game == nullptr)
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT, "Game must be non-null.");
        game->game.startPlaying();
        return MANILLE_MOBILE_OK;
    });
}

int manille_mobile_hand_score(const ManilleMobileGame* game, int32_t team, int32_t* out_score) {
    return guarded([&]() -> int {
        if (game == nullptr || out_score == nullptr || team < 0 || team > 1)
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT, "Invalid hand-score request.");
        *out_score = game->game.handScore(team);
        return MANILLE_MOBILE_OK;
    });
}

int manille_mobile_encode_q(
    const ManilleMobileGame* game,
    int32_t observer,
    float* obs,
    float* act,
    int32_t* legal,
    int32_t* out_n_legal
) {
    return guarded([&]() -> int {
        if (game == nullptr || obs == nullptr || act == nullptr || legal == nullptr
            || out_n_legal == nullptr || !valid_player(observer))
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT, "Invalid Q encoding request.");
        if (observer != static_cast<int32_t>(game->game.currentPlayer()))
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT,
                        "Only the current player's Q observation can be encoded.");

        encoding::encode(game->game, observer, obs);
        return encode_candidates(game->game, act, legal, out_n_legal);
    });
}

int manille_mobile_encode_gru(
    const ManilleMobileGame* game,
    int32_t observer,
    float* raw,
    int64_t* out_n_raw,
    float* glob,
    float* act,
    int32_t* legal,
    int32_t* out_n_legal
) {
    return guarded([&]() -> int {
        if (game == nullptr || raw == nullptr || out_n_raw == nullptr || glob == nullptr
            || act == nullptr || legal == nullptr || out_n_legal == nullptr
            || !valid_player(observer))
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT, "Invalid GRU encoding request.");
        if (observer != static_cast<int32_t>(game->game.currentPlayer()))
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT,
                        "Only the current player's GRU observation can be encoded.");

        std::array<float, encoding::OBS_DIM> obs{};
        encoding::encode(game->game, observer, obs.data());
        std::copy_n(
            obs.data() + encoding::HISTORY_OFF,
            MANILLE_MOBILE_MAX_PLAYS * MANILLE_MOBILE_RAW_TOKEN_DIM,
            raw);
        // Every card played so far, the current trick included -- the
        // history tokens hold them all (learn/dmc_agent.py: n_plays =
        // played_cards + current_trick). Counting completed tricks only hid
        // the cards on the table from the GRU at seats 2-4 of every trick.
        *out_n_raw = std::min<int>(
            game->game.playedCardCount() + game->game.trickSize(),
            MANILLE_MOBILE_MAX_PLAYS);

        std::fill(glob, glob + MANILLE_MOBILE_GLOBAL_DIM, 0.0f);
        std::copy_n(obs.data() + encoding::OWN_HAND_OFF, 32, glob);
        std::copy_n(obs.data() + encoding::TRUMP_OFF, 4, glob + 32);
        std::copy_n(obs.data() + encoding::TRICK_NUM_OFF, 8, glob + 36);
        std::copy_n(obs.data() + encoding::TEAM_PTS_OFF, 2, glob + 44);
        std::copy_n(obs.data() + encoding::DECLARER_OFF, 4, glob + 46);
        // Tactical block (plan 1.3). The GRU reads only glob + history
        // tokens, so a feature that never reaches the global token is
        // invisible to it no matter what the observation holds.
        std::copy_n(obs.data() + encoding::MASTER_OFF, encoding::TACTICAL_DIM,
                    glob + 50);
        return encode_candidates(game->game, act, legal, out_n_legal);
    });
}

int manille_mobile_clone(const ManilleMobileGame* game, ManilleMobileGame** out_game) {
    return guarded([&]() -> int {
        if (game == nullptr || out_game == nullptr)
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT, "game and out_game must be non-null.");
        *out_game = new ManilleMobileGame(game->game);
        return MANILLE_MOBILE_OK;
    });
}

namespace {

rollout::History history_of(const ManilleGame& g) {
    rollout::History history;
    for (int i = 0; i < g.playedCardCount(); ++i)
        history.emplace_back(g.playedPlayer(i), g.playedCard(i));
    for (int i = 0; i < g.trickSize(); ++i)
        history.emplace_back(g.trickPlayer(i), g.trickCard(i));
    return history;
}

// Determinizations for the player to move, current hands by absolute seat.
// The sampler's declarer is the dealer only when trump came from the C++
// rule (it then keeps the declarer hands the rule would have declared);
// unconstrained otherwise -- the rule constraint would be wrong for a
// chooser's or a human's trump.
std::vector<rollout::World> sample_worlds(const ManilleGame& g, bool trump_rule_known, int n) {
    const int me = static_cast<int>(g.currentPlayer());
    std::array<int64_t, 4> hands{};
    for (int p = 0; p < 4; ++p)
        hands[p] = g.handMask(p);
    std::array<int32_t, 4> tc{-1, -1, -1, -1}, tp{-1, -1, -1, -1};
    for (int i = 0; i < g.trickSize(); ++i) {
        tc[i] = g.trickCard(i);
        tp[i] = g.trickPlayer(i);
    }
    std::array<int32_t, 32> pc{}, pp{};
    pc.fill(-1);
    pp.fill(-1);
    for (int i = 0; i < g.playedCardCount(); ++i) {
        pc[i] = g.playedCard(i);
        pp[i] = g.playedPlayer(i);
    }
    ManilleGame sampling(hands, tc, tp, g.trickSize(), me, g.trumpSuit(),
                         g.teamPoints(0), g.teamPoints(1), g.tricksWon(0),
                         g.tricksWon(1), pc, pp, g.playedCardCount(), false,
                         trump_rule_known ? g.declarer() : -1);
    std::vector<rollout::World> worlds;
    worlds.reserve(static_cast<size_t>(n));
    for (int draw = 0; draw < 2 * n && static_cast<int>(worlds.size()) < n; ++draw) {
        ManilleGame::Clone clone;
        try {
            clone = sampling.cloneAndRandomise(static_cast<unsigned int>(me));
        } catch (const sampler::RejectionBudgetExhausted&) {
            continue;
        }
        auto* det = dynamic_cast<ManilleGame*>(clone.get());
        if (det == nullptr)
            throw std::runtime_error("Determinization failed.");
        rollout::World w;
        for (int p = 0; p < 4; ++p)
            w[p] = det->handMask(p);
        worlds.push_back(w);
    }
    return worlds;
}

}  // namespace

int manille_mobile_sample_worlds(const ManilleMobileGame* game, int32_t trump_rule_known,
                                 int32_t n_worlds, int64_t* out_current,
                                 int64_t* out_initial, int32_t* out_n) {
    return guarded([&]() -> int {
        if (game == nullptr || out_current == nullptr || out_initial == nullptr
                || out_n == nullptr || n_worlds < 0)
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT, "Invalid world sampling request.");
        const ManilleGame& g = game->game;
        if (g.finished())
            return fail(MANILLE_MOBILE_ILLEGAL_MOVE, "The deal is finished.");
        auto worlds = sample_worlds(g, trump_rule_known != 0, n_worlds);
        int64_t played[4] = {0, 0, 0, 0};
        for (auto& sc : history_of(g))
            played[sc.first] |= int64_t(1) << sc.second;
        for (size_t i = 0; i < worlds.size(); ++i)
            for (int p = 0; p < 4; ++p) {
                out_current[4 * i + p] = worlds[i][p];
                out_initial[4 * i + p] = worlds[i][p] | played[p];
            }
        *out_n = static_cast<int32_t>(worlds.size());
        return MANILLE_MOBILE_OK;
    });
}

struct ManilleFastNet {
    fastpolicy::Net net;
    // Net-guided plies before the rule policy takes over. 4 is the setting
    // that fits the phone budget (~0.1 s/move on a desktop core vs ~0.55 s
    // for the full net play-out) and still kept most of the gain; 0 = net
    // for the whole play-out.
    int plies = 4;
};

int manille_mobile_fastnet_load(const char* path, ManilleFastNet** out_net) {
    return guarded([&]() -> int {
        if (path == nullptr || out_net == nullptr)
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT, "path and out_net must be non-null.");
        auto* n = new ManilleFastNet{fastpolicy::load_binary(path)};
        *out_net = n;
        return MANILLE_MOBILE_OK;
    });
}

void manille_mobile_fastnet_free(ManilleFastNet* net) {
    delete net;
}

int manille_mobile_fastnet_set_plies(ManilleFastNet* net, int32_t plies) {
    return guarded([&]() -> int {
        if (net == nullptr || plies < 0)
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT, "net must be non-null, plies >= 0.");
        net->plies = plies;
        return MANILLE_MOBILE_OK;
    });
}

int manille_mobile_rule_choose_worlds(const ManilleMobileGame* game, const int64_t* worlds,
                                      int32_t n_worlds, const float* belief, const float* q,
                                      const double* extra, int32_t* out_card,
                                      double* out_values) {
    return manille_mobile_rule_choose_worlds_net(game, worlds, n_worlds, belief, q, extra,
                                                 nullptr, out_card, out_values);
}

int manille_mobile_rule_choose_worlds_net(const ManilleMobileGame* game, const int64_t* worlds,
                                          int32_t n_worlds, const float* belief, const float* q,
                                          const double* extra, const ManilleFastNet* net,
                                          int32_t* out_card, double* out_values) {
    return guarded([&]() -> int {
        if (game == nullptr || out_card == nullptr || n_worlds < 0
                || (n_worlds > 0 && worlds == nullptr))
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT, "Invalid rule agent request.");
        const ManilleGame& g = game->game;
        if (g.finished())
            return fail(MANILLE_MOBILE_ILLEGAL_MOVE, "The deal is finished.");
        std::vector<rollout::World> ws(static_cast<size_t>(n_worlds));
        for (int i = 0; i < n_worlds; ++i)
            for (int p = 0; p < 4; ++p)
                ws[i][p] = worlds[4 * i + p];
        rule::Params params;
        if (net)
            params.playout_net_plies = net->plies;
        rule::Decision d = rule::choose(g, static_cast<int>(g.currentPlayer()), g.declarer(),
                                        ws, history_of(g), belief, q, params, extra,
                                        net ? &net->net : nullptr);
        *out_card = d.card;
        if (out_values != nullptr) {
            std::fill(out_values, out_values + 32, std::nan(""));
            for (size_t i = 0; i < d.value_cards.size(); ++i)
                out_values[d.value_cards[i]] = d.values[i];
        }
        return MANILLE_MOBILE_OK;
    });
}

int manille_mobile_rule_choose(const ManilleMobileGame* game, const float* belief,
                               const float* q, int32_t trump_rule_known,
                               int32_t n_worlds, int32_t* out_card) {
    return guarded([&]() -> int {
        if (game == nullptr || out_card == nullptr || n_worlds < 0)
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT,
                        "game and out_card must be non-null, n_worlds >= 0.");
        const ManilleGame& g = game->game;
        if (g.finished())
            return fail(MANILLE_MOBILE_ILLEGAL_MOVE, "The deal is finished.");
        auto ws = sample_worlds(g, trump_rule_known != 0, n_worlds);
        rule::Params params;
        rule::Decision d = rule::choose(g, static_cast<int>(g.currentPlayer()), g.declarer(),
                                        ws, history_of(g), belief, q, params);
        *out_card = d.card;
        return MANILLE_MOBILE_OK;
    });
}

int manille_mobile_trick_winner(const int32_t cards[4], int32_t n, int32_t trump, int32_t* out_index) {
    return guarded([&]() -> int {
        if (cards == nullptr || out_index == nullptr || n < 1 || n > 4 || trump < -1 || trump > 3)
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT, "Invalid trick.");
        std::array<int32_t, 4> tc{-1, -1, -1, -1};
        for (int i = 0; i < n; ++i) {
            if (cards[i] < 0 || cards[i] > 31)
                return fail(MANILLE_MOBILE_INVALID_ARGUMENT, "Invalid card.");
            tc[i] = cards[i];
        }
        *out_index = ManilleGame::trickWinner(tc, n, trump);
        return MANILLE_MOBILE_OK;
    });
}

int manille_mobile_apply_trump_rule(int64_t hand_mask, int32_t* out_suit) {
    return guarded([&]() -> int {
        if (out_suit == nullptr)
            return fail(MANILLE_MOBILE_INVALID_ARGUMENT, "out_suit must be non-null.");
        *out_suit = prior::apply_trump_rule(hand_mask);
        return MANILLE_MOBILE_OK;
    });
}

}  // extern "C"
