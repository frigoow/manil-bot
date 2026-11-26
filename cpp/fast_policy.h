#ifndef MANILLE_FAST_POLICY_H
#define MANILLE_FAST_POLICY_H

// Fast learned play-out policy for the rule agent's play-out
// (cpp/rule_playout.h): a small MLP that imitates a strong player's card
// choice from the acting seat's OWN information -- its hand, the public
// record and a few tactical bits the engine answers -- and replaces the
// hand-written rule policy as the continuation in every simulated seat.
// AlphaGo's fast rollout policy, for Manille: cheap enough to run for every
// seat of every world of every candidate inside the phone budget.
//
// The feature layout lives here only. Training data comes from the same
// function through the Python binding (fast_policy_features), so there is no
// second implementation to drift. Legality and the trick winner come from
// the engine.

#include "manille_game.h"
#include "rule_playout.h"

#include <cstdint>
#include <string>
#include <vector>

namespace fastpolicy {

// Sparse binary features (indices that are 1) + 2 dense values.
constexpr int F_HAND = 0;           // 32 own hand
constexpr int F_LEGAL = 32;         // 32 legal cards
constexpr int F_PLAYED = 64;        // 4 x 32, cards played by relative seat (0 = me)
constexpr int F_TRICK = 192;        // 3 x 32, card on the table by relative seat 1..3
constexpr int F_TRUMP = 288;        // 4 one-hot, 292 = no trump
constexpr int F_DECLARER = 293;     // 4 one-hot, relative seat of the declarer
constexpr int F_VOID = 297;         // 3 x 4, public voids of relative seats 1..3
constexpr int F_TRICK_INDEX = 309;  // 8 one-hot
constexpr int F_POS = 317;          // 4 one-hot, cards on the table
constexpr int F_MASTER = 321;       // 32, own cards no unseen card outranks in suit
constexpr int F_WINS = 353;         // 32, legal cards that take the lead now
constexpr int F_PARTNER_WINS = 385; // 1
constexpr int F_BINARY = 386;
constexpr int F_DENSE = 2;          // my team's points / 60, the other team's
constexpr int F_DIM = F_BINARY + F_DENSE;

// Features of `seat`, the player to move in `game`. `active` receives the
// indices of the binary features that are 1 (ascending), `dense` the two
// dense values. Returns the legal mask.
int64_t features(const ManilleGame& game, int seat, std::vector<int>& active,
                 float dense[F_DENSE]);

// MLP F_DIM -> h1 -> h2 -> 32 (ReLU), logits over cards. Weights row-major
// [out][in], as torch.nn.Linear stores them.
struct Net {
    int h1 = 0, h2 = 0;
    std::vector<float> w1, b1, w2, b2, w3, b3;
    std::vector<float> w1t;  // [in][h1]: first layer by input column, for the sparse sum
    std::vector<float> w2t;  // [h1][h2]: transposed layer 2 for sparse-ReLU SAXPY
    std::vector<float> w3t;  // [h2][32]: transposed layer 3 for sparse-ReLU SAXPY
    void finalize();         // builds w1t/w2t/w3t; throws on a shape mismatch
    bool empty() const { return h1 == 0; }
};

// Weights from the deploy file learn/fast_policy.py::export_binary writes
// (little-endian): "MFPN", int32 version 1, int32 F_DIM, h1, h2, then
// float32 w1 [h1][F_DIM], b1, w2 [h2][h1], b2, w3 [32][h2], b3. Throws on a
// bad file or a layout that does not match this build's F_DIM.
Net load_binary(const std::string& path);

// The net's card for `seat` to move: argmax logit over the legal cards.
int choose(const Net& net, const ManilleGame& game, int seat);

// Softmax probabilities over `legal` cards for `seat` to move.
void legal_probs(const Net& net, const ManilleGame& game, int seat,
                 const std::vector<int>& legal, std::vector<double>& probs_out);

// Honest soft action log-likelihood over `history` for each sampled world:
// replays the public history once, precomputes public layer-1 activations per
// ply, and sums weight * log_softmax(logits / tau)[actual_card] over non-forced
// plays by seats other than `me` (`partner_w` for me+2, `opp_w` for me+1/me+3).
std::vector<double> history_log_likelihood(
    const Net& net, int trump, int declarer, int me,
    const std::vector<rollout::World>& worlds,
    const rollout::History& history,
    double tau, double partner_w, double opp_w);

// rollout::evaluate_candidates with every seat following the net: my team's
// final points, [candidate][world]. When endgame_ab_cards > 0, hands off to
// exact double-dummy alpha-beta at trick boundaries once <= endgame_ab_cards
// tricks remain.
std::vector<std::vector<double>> evaluate_candidates(
    const Net& net, int trump, int declarer, int me,
    const std::vector<int>& candidates,
    const std::vector<rollout::World>& worlds,
    const rollout::History& history,
    int endgame_ab_cards = 0);

} // namespace fastpolicy

#endif // MANILLE_FAST_POLICY_H
