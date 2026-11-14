#ifndef MANILLE_RULE_PLAYOUT_H
#define MANILLE_RULE_PLAYOUT_H

// Multi-trick mental play-out for the rule agent (learn/rule_agent.py).
//
// C++ port of learn/rule_playout.py, which stays the readable reference;
// tests/test_rule_playout_parity.py pins the two to identical outputs. For
// each candidate card the hand is played to the end in every sampled world,
// all four seats following one fast human-style rule policy. Legality and
// card ordering come from the engine (ManilleGame::validMoves, Card::beats).
// No minimax: nobody optimises against anyone.

#include "manille_game.h"

#include <array>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace rollout {

struct Options {
    // 0 = clairvoyant (seats see the world's real hands behind them),
    // 1 = honest without void tracking, 2 = honest with public voids.
    int honest = 0;
    // play-out leads follow the rule agent's planner tiers
    bool agent_like = false;
    // a ruff behind counts as a threat once this few led-suit cards are
    // left outside the seat's own hand
    int ruff_short = 2;
    // No-trump card play, bitmask (ignored when there is a trump suit):
    //   1  a seat that wins the trick for certain wins it with its highest
    //      honour an opponent could still capture, not its cheapest winner
    //      -- unless enough small cards guard that honour
    //   2  partner's card holds and the opponents are out of the suit: the
    //      hand with fewer cards left in it plays high (unblocks, overtaking
    //      if need be), the longer hand plays low and overtakes partner's
    //      last card of the suit
    //   4  a suit both opponents have shown out of is cashed like a master
    //      suit
    //   8  discards keep the cards no opponent can beat (masters and cards
    //      of a suit the opponents are out of: they are tricks and the link
    //      to partner's winners); a loser goes instead, unless every loser
    //      carries 3+ points
    //  16  a discard on an opponents' trick also keeps a guarded honour and
    //      its guards (ace and a small card under the outstanding manille:
    //      the small card ducks, the ace then wins). With only winners and
    //      guarded honours left, a cheap low card of a three-winner suit
    //      goes. On partner's trick the honour is still vetted: banking it
    //      is the percentage play
    int nt_rules = 0;
    // Trump card play, bitmask (ignored at no trump; review of the play-out
    // policy against the blueprint, learn/rule_playout.py has the text):
    //   1  trump through to the declaring partner needs 2+ trumps
    //   2  partner's safe trick, side suit, opponent still to play: vet the
    //      biggest card even if it is a master
    //   4  no plan lead: holding trumps, lead a low non-master side singleton
    //   8  discard on an opponent's trick partner can take: biggest
    //      non-master side card
    //  16  a sure winner is taken with the honour that would be captured
    //  32  ruff convention: partner opened the deal with a side suit and has
    //      not played it since -- return it while partner can ruff
    int tr_rules = 0;
};

using History = std::vector<std::pair<int, int>>;  // (seat, card) in play order
using World = std::array<int64_t, 4>;              // current hands, absolute seats

// A learned continuation (cpp/fast_policy.h): the card for `seat` to move.
using NetChoose = std::function<int(const ManilleGame&, int)>;

// Final team points of `me`'s team, [candidate][world]. With `net`, the first
// `net_plies` plies after the candidate follow the net, the rest the rule
// policy (whose public record still sees every play). With
// `net_to_trick_end` the net also finishes the trick it is in when the count
// runs out, so the hand-over always falls on a trick boundary.
std::vector<std::vector<double>> evaluate_candidates(
    int trump, int declarer, int me,
    const std::vector<int>& candidates,
    const std::vector<World>& worlds,
    const History& history,
    const Options& opt,
    const NetChoose* net = nullptr, int net_plies = 0, bool net_to_trick_end = false);

// Likelihood weight per world: each play by another seat that the policy
// would not have made in that world multiplies the weight by eps.
std::vector<double> world_weights(
    int trump, int declarer, int me,
    const std::vector<World>& worlds,
    const History& history,
    double eps,
    const Options& opt);

// The same replay, split by seat: per world, how many plays of each seat
// the policy fails to explain ([world][seat]; `me` stays 0). Lets callers
// give each seat its own error rate -- a partner whose policy we know and
// an opponent we only approximate deserve different eps.
std::vector<std::array<int, 4>> world_mismatches(
    int trump, int declarer, int me,
    const std::vector<World>& worlds,
    const History& history,
    const Options& opt);

} // namespace rollout

#endif // MANILLE_RULE_PLAYOUT_H
