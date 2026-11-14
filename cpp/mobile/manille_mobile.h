#ifndef MANILLE_MOBILE_H
#define MANILLE_MOBILE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    MANILLE_MOBILE_OK = 0,
    MANILLE_MOBILE_INVALID_ARGUMENT = 1,
    MANILLE_MOBILE_ILLEGAL_MOVE = 2,
    MANILLE_MOBILE_ERROR = 3,
};

enum {
    MANILLE_MOBILE_OBS_DIM = 2492,
    MANILLE_MOBILE_ACT_DIM = 40,
    MANILLE_MOBILE_MAX_LEGAL = 8,
    MANILLE_MOBILE_RAW_TOKEN_DIM = 58,
    MANILLE_MOBILE_MAX_PLAYS = 31,
    MANILLE_MOBILE_GLOBAL_DIM = 130,
};

typedef struct ManilleMobileGame ManilleMobileGame;
typedef struct ManilleFastNet ManilleFastNet;

// POD snapshot intended for language bindings. Cards are the canonical 0..31
// indices; unused trick/played slots are -1. Team 0 is players 0 and 2.
typedef struct {
    int32_t current_player;
    int32_t trump_suit;
    int32_t declarer;
    int32_t pts0;
    int32_t pts1;
    int32_t tricks_won0;
    int32_t tricks_won1;
    int32_t multiplier;
    int32_t doubled;
    int32_t redoubled;
    int32_t doubling_phase;
    int32_t finished;
    int32_t n_trick;
    int32_t n_played_cards;
    int32_t trick_cards[4];
    int32_t trick_players[4];
    int32_t played_cards[32];
    int32_t played_players[32];
} ManilleMobileState;

// Starts a complete deal. `dealer` is also the declarer; trump is selected
// with the canonical C++ rule and the player after the dealer leads.
int manille_mobile_new_random_deal(
    int32_t dealer,
    uint64_t seed,
    ManilleMobileGame** out_game
);

// Same deterministic deal as manille_mobile_new_random_deal, but uses the
// supplied declaration. trump_suit is 0..3 or -1 for no-trump (mull).
// This lets a mobile client show the dealer's cards before committing the
// human declaration without maintaining a second rules implementation.
int manille_mobile_new_random_deal_with_trump(
    int32_t dealer,
    uint64_t seed,
    int32_t trump_suit,
    ManilleMobileGame** out_game
);

// Restores a public/private engine state. All arrays have their fixed sizes
// from ManilleMobileState. This is for saved games and parity tests: the
// stakes (doubled, redoubled, the announcement window, the multiplier) and
// the finished flag come back too.
int manille_mobile_create(
    const int64_t hands[4],
    const ManilleMobileState* state,
    ManilleMobileGame** out_game
);

void manille_mobile_destroy(ManilleMobileGame* game);

const char* manille_mobile_last_error(void);

int manille_mobile_get_state(
    const ManilleMobileGame* game,
    ManilleMobileState* out_state
);

int manille_mobile_hand_mask(
    const ManilleMobileGame* game,
    int32_t player,
    int64_t* out_mask
);

int manille_mobile_legal_mask(
    const ManilleMobileGame* game,
    int64_t* out_mask
);

int manille_mobile_apply_card(ManilleMobileGame* game, int32_t card);
int manille_mobile_announce_double(ManilleMobileGame* game, int32_t player);
int manille_mobile_announce_redouble(ManilleMobileGame* game, int32_t player);
int manille_mobile_start_playing(ManilleMobileGame* game);
int manille_mobile_hand_score(const ManilleMobileGame* game, int32_t team, int32_t* out_score);

// Encodes exactly the model inputs for the player currently to move. `obs`
// must hold MANILLE_MOBILE_OBS_DIM floats; `act` holds MAX_LEGAL x ACT_DIM
// floats; `legal` holds MAX_LEGAL card indices. Unused rows are zero/-1.
int manille_mobile_encode_q(
    const ManilleMobileGame* game,
    int32_t observer,
    float* obs,
    float* act,
    int32_t* legal,
    int32_t* out_n_legal
);

// Encodes the recurrent QGRU inputs from the same canonical C++ observation.
// `raw` holds MAX_PLAYS x RAW_TOKEN_DIM floats, `glob` holds GLOBAL_DIM, and
// candidates use the same padded action/legal outputs as encode_q.
int manille_mobile_encode_gru(
    const ManilleMobileGame* game,
    int32_t observer,
    float* raw,
    int64_t* out_n_raw,
    float* glob,
    float* act,
    int32_t* legal,
    int32_t* out_n_legal
);

int manille_mobile_apply_trump_rule(int64_t hand_mask, int32_t* out_suit);

// Index (0..n-1) of the card winning the first n cards of a trick, by the
// engine's own rule (ManilleGame::trickWinner). trump -1 = no trump.
int manille_mobile_trick_winner(const int32_t cards[4], int32_t n, int32_t trump, int32_t* out_index);

// Rule agent (cpp/rule_agent.h, the deployed desktop config) for the player
// to move. The nets run outside, on the same observation:
//   belief: 3 x 32 floats from manille_belief_v1.onnx (no trump:
//           manille_belief_mull_v1.onnx) fed the `obs` of
//           manille_mobile_encode_q for the player to move -- P(seat
//           me+1+r holds card c); NULL = rule likelihood only
//   q:      32 floats indexed by card, the QGRU value of each legal card
//           (manille_qgru_v1 / manille_qgru_mull_v1); NULL = no blend
// trump_rule_known: 1 when trump came from manille_mobile_apply_trump_rule,
// 0 when a chooser or a human declared it. n_worlds: 400 on the desktop.
// Independent copy of a game (for a worker thread: the copy is never touched
// by the UI). Destroy it with manille_mobile_destroy.
int manille_mobile_clone(const ManilleMobileGame* game, ManilleMobileGame** out_game);

// The rule agent in two steps, so the caller can weight the worlds with
// nets it runs itself (the meegaan/terug likelihood under the announce net):
//  1. manille_mobile_sample_worlds: up to n_worlds determinizations for the
//     player to move. out_current / out_initial hold 4 x n_worlds int64 hand
//     masks by absolute seat (current hands, and current | already played
//     = the hand each seat was dealt in that world); *out_n = worlds drawn.
//  2. manille_mobile_rule_choose_worlds: the card, given those worlds (the
//     out_current rows), the belief / q inputs as for
//     manille_mobile_rule_choose, and an optional extra log-weight per world
//     (NULL = none). out_values (optional, 32 doubles by card) receives the
//     play-out value of each legal card (expected final team points, the
//     hint numbers); NaN for the other cards, and for every card when the
//     play-out did not run.
int manille_mobile_sample_worlds(
    const ManilleMobileGame* game,
    int32_t trump_rule_known,
    int32_t n_worlds,
    int64_t* out_current,
    int64_t* out_initial,
    int32_t* out_n
);
int manille_mobile_rule_choose_worlds(
    const ManilleMobileGame* game,
    const int64_t* worlds,
    int32_t n_worlds,
    const float* belief,
    const float* q,
    const double* extra,
    int32_t* out_card,
    double* out_values
);

// The fast learned play-out (cpp/fast_policy.h) for the rule agent: load the
// manille_fast_policy_v1.bin asset once, pass it to
// manille_mobile_rule_choose_worlds_net (same as rule_choose_worlds, with
// every seat of the play-out following the net for the first `plies` plies,
// default 4, then the rule policy; 0 = the net throughout), free it at
// shutdown. The net is read-only after loading, so worker threads may share
// it (set the plies before sharing).
int manille_mobile_fastnet_load(const char* path, ManilleFastNet** out_net);
void manille_mobile_fastnet_free(ManilleFastNet* net);
int manille_mobile_fastnet_set_plies(ManilleFastNet* net, int32_t plies);
// Nonzero: when its plies run out the net still finishes the trick it is in.
int manille_mobile_fastnet_set_trick_end(ManilleFastNet* net, int32_t on);
int manille_mobile_rule_choose_worlds_net(
    const ManilleMobileGame* game,
    const int64_t* worlds,
    int32_t n_worlds,
    const float* belief,
    const float* q,
    const double* extra,
    const ManilleFastNet* net,
    int32_t* out_card,
    double* out_values
);

int manille_mobile_rule_choose(
    const ManilleMobileGame* game,
    const float* belief,
    const float* q,
    int32_t trump_rule_known,
    int32_t n_worlds,
    int32_t* out_card
);

#ifdef __cplusplus
}
#endif

#endif  // MANILLE_MOBILE_H
