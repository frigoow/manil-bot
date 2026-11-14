#ifndef MANILLE_ENCODING_H
#define MANILLE_ENCODING_H

#include "manille_game.h"

// Canonical observation encoding for the learning stack (plan §4/§5.3).
// Implemented once here; the Python side mirrors it only for golden parity
// tests. All seats are encoded relative to the observer:
//   0 = self, 1 = next player, 2 = partner, 3 = previous player.
//
// Everything in the vector is computable from public history plus the
// observer's own hand — never from the true hidden hands — so the same
// encoder works for training data and for live inference.

namespace encoding {

// --- static block ---
constexpr int OWN_HAND_OFF = 0;     // 32 binary, observer's hand
constexpr int PLAYED_OFF = 32;      // 4x32 binary, cards played per relative seat
constexpr int TRICK_OFF = 160;      // 4x32 one-hot, card each seat has on the table
constexpr int TRICK_LEADER_OFF = 288; // 4 one-hot, relative seat of trick leader
constexpr int TRICK_POS_OFF = 292;  // 4 one-hot, cards already on the table
constexpr int TRUMP_OFF = 296;      // 4 one-hot
constexpr int DECLARER_OFF = 300;   // 4 one-hot, relative seat of trump chooser
constexpr int TRICK_NUM_OFF = 304;  // 8 one-hot, current trick index
constexpr int TEAM_PTS_OFF = 312;   // 2, own/opp team points (/60)
constexpr int TRICK_PTS_OFF = 314;  // 1, points on the table (/20)
constexpr int VOID_OFF = 315;       // 4x4, established voids per relative seat
constexpr int LEGAL_MASK_OFF = 331; // 32 binary, observer's legal moves
constexpr int LEGAL_COUNT_OFF = 363;  // 1, |legal| / 8
constexpr int STATIC_DIM = 364;

// --- history block: 32 play slots, zero-padded past the current ply ---
// Per-play layout (PLY_DIM = 58):
//   [0:4]   actor seat one-hot (relative to observer)
//   [4:36]  card one-hot
//   [36:40] led suit one-hot
//   [40:44] position within trick one-hot
//   [44:48] winning-so-far seat one-hot (all zero for the lead card)
//   [48:58] public context flags (see Flag enum below)
constexpr int HISTORY_OFF = 364;
constexpr int PLY_DIM = 58;
constexpr int PLY_FLAG_OFF = 48;
constexpr int HISTORY_DIM = 32 * PLY_DIM;

// Public context flags: facts an observer can derive with certainty from the
// trick state and the played card alone. This is the §5.3 "legal-set summary":
// forced/obliged plays are visibly marked without leaking hidden holdings.
enum Flag {
    F_FOLLOWED = 0,       // played the led suit
    F_VOID_ESTABLISHED,   // did not follow -> hard public void in led suit
    F_OBLIGATION,         // opponent was winning -> must-beat/trump obligation
    F_FOLLOWED_NO_BEAT,   // followed under obligation without beating;
                          // hard info (holds no higher card of led suit) only
                          // when the current winner is in the led suit — when
                          // the winner is a trump it implies nothing
    F_FOLLOWED_BEAT,      // followed and took the lead
    F_TRUMPED,            // played trump off-suit
    F_OVERTRUMPED,        // played a higher trump than the winning trump
    F_DUCKED_TRUMP,       // failed to beat an opponent's winning trump off-suit
                          // -> hard info: holds no higher trump
    F_PARTNER_WINNING,    // partner was winning -> free choice, no obligation
    F_TRICK_COMPLETED,    // this play was the last card of its trick
    N_FLAGS = 10
};

// --- belief block ---
// Slots [r][c] with r in {0,1,2} = hidden seats next/partner/prev, c = card.
constexpr int BELIEF_PRIOR_OFF = HISTORY_OFF + HISTORY_DIM; // 3x32 constrained-uniform prior marginals
constexpr int BELIEF_MASK_OFF = BELIEF_PRIOR_OFF + 96;      // 3x32 hard validity mask

// --- tactical block (plan 1.3) ---
// Exact facts about the CURRENT trick-taking situation that the blocks above
// only imply. Everything here is derived from the observer's own hand plus
// the public play record (played cards, cards on the table, established
// voids) -- never from a hidden hand -- and every rules question is answered
// by the engine, never re-derived here.
//
// "Unseen" = not in the observer's hand, not played, not on the table.
// MASTER is per-suit ranking only: the highest unseen card of its own suit
// can still be trumped, and SEAT_CAN_HOLD on the trump suit is what says
// whether that is possible.
constexpr int MASTER_OFF = BELIEF_MASK_OFF + 96;        // 32, own cards only
constexpr int HIGHER_UNSEEN_OFF = MASTER_OFF + 32;      // 32, count/7, own cards
constexpr int UNSEEN_SUIT_OFF = HIGHER_UNSEEN_OFF + 32; // 4, count/8 per suit
constexpr int SEAT_CAN_HOLD_OFF = UNSEEN_SUIT_OFF + 4;  // 3 hidden seats x 4 suits
constexpr int TACTICAL_DIM = 32 + 32 + 4 + 12;

constexpr int OBS_DIM = SEAT_CAN_HOLD_OFF + 12;

// --- action block (DMC Q(s,a): the candidate card is an input, not an
// output head). One ACT_DIM vector per legal candidate. Layout:
//   [0:32]  card one-hot
//   [32]    card is a trump
//   [33]    card follows the led suit (0 when leading the trick)
//   [34]    rank / 7
//   [35]    point value / 5
//   [36]    card would be winning the trick if played now (1 when leading)
constexpr int ACT_IS_TRUMP_OFF = 32;
constexpr int ACT_IS_LED_OFF = 33;
constexpr int ACT_RANK_OFF = 34;
constexpr int ACT_POINTS_OFF = 35;
constexpr int ACT_WINS_OFF = 36;
// Tactical candidate features (plan 1.3). ACT_WINS_OFF says the candidate
// beats what is on the table NOW; these say what the seats still to play can
// do about it. Both are answered by asking the engine for the legal moves of
// a hypothetical hand holding every unseen card in the suits that seat is not
// publicly void in -- an upper bound on what it can hold, so "cannot beat" is
// a certainty and "can beat" is a possibility.
constexpr int ACT_SURE_WIN_OFF = 37;        // wins the trick outright
constexpr int ACT_OPP_CAN_BEAT_OFF = 38;    // an opponent still to play may beat it
constexpr int ACT_PARTNER_WINNING_OFF = 39; // partner holds the trick right now
constexpr int ACT_DIM = 40;
constexpr int MAX_LEGAL = 8;  // hand size; a player never has more legal moves

// Fill `out` (ACT_DIM floats) with the candidate-action features of `card`
// in `game`. `card` must be a legal move of the current player; legality
// itself is not encoded (candidates are enumerated from validMoves()).
// Candidates are always encoded for the seat to move, so that seat is the
// observer for the tactical features -- every call site enumerates
// game.validMoves(), which is the current player's legal set.
void encode_action(const ManilleGame& game, int card, float* out);

// Fill `out` (OBS_DIM floats, zero-initialized by caller or here) with the
// observation of `game` from `observer`'s perspective.
void encode(const ManilleGame& game, int observer, float* out);

} // namespace encoding

#endif // MANILLE_ENCODING_H
