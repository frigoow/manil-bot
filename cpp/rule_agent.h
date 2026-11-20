#ifndef MANILLE_RULE_AGENT_H
#define MANILLE_RULE_AGENT_H

// Native rule agent: the deploy-config path of learn/rule_agent.py
// (RuleAgent.choose_card) in C++, for the phone. The Python agent stays the
// readable reference; tests/test_rule_agent_native.py pins the two to the
// same card on recorded positions given the same sampled worlds.
//
// Nets stay outside: the belief net's per-card holder probabilities and the
// Q net's per-card value depend only on the information set, so the caller
// evaluates them once (torch on the desktop, ONNX on the phone) and passes
// the numbers in. Everything that depends on the sampled worlds -- the
// Knowledge marginals, the rule tiers, the play-out, the world weights and
// the paired overrule test -- runs here. Legality, trick order and scoring
// come from the engine (ManilleGame), never from local rules code.
//
// Only the configuration the deployed agent runs is ported: no signals, no
// soft vet inference, no declaration tilt, no SPARTA weights, no exact
// enumeration, no match-equity objective. Python callers fall back to the
// Python path for anything else (RuleAgent.native_supported).

#include "manille_game.h"
#include "fast_policy.h"
#include "rule_playout.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace rule {

// learn/rule_agent.py DEFAULTS, the fields the deployed path reads.
struct Params {
    double trick_rest = 1.5;
    double keep_win_value = 7.0;
    double keep_pts_weight = 0.7;
    double win_own_pts_w = 1.0;
    double partner_after_f = 0.6;
    int sim_worlds = 40;
    bool sim_vet = false;
    double guard_power = 0.5;
    double trump_power = 0.85;
    double trump_power_slope = 0.3;
    double ruff_future = 1.3;
    double trump_keep_bonus = 2.0;
    int min_worlds = 20;
    // planner (_lead_plan); plan_skip as a bitmask of tier numbers
    int plan_skip_mask = 1 << 3;
    double plan_pull_min_opp = 0.3;
    double plan_majority_margin = 0.0;
    bool plan_to_declarer = true;
    double plan_ruff = 0.5;
    bool plan_force_trump = false;
    double plan_cash = 0.3;
    double plan_cash_risk = 0.0;
    double plan_est_ruff = 0.4;
    bool plan_est_v2 = false;
    // no trump: establish from a touching run with its lowest card
    bool nt_est_low_run = false;
    // no trump, from this trick on (0 = off): the play-out may only swap a
    // "cash master" lead for another master
    int nt_lock_cash_from_trick = 0;
    // defending a suit: the play-out may not turn a side-suit lead into a
    // trump lead (user feedback: weak trump leads by defenders)
    bool def_trump_lock = false;
    // ... only from a weak trump holding: at most 2 trumps, at most 3 points
    bool def_trump_lock_weak = false;
    // softer than the lock (0 = off): such a trump lead must beat the lead
    // tier's card by this many points before the play-out may switch to it
    double def_trump_margin = 0.0;
    double plan_est_v2_ruff = 0.25;
    double plan_partner_top = 0.45;
    bool plan_exit_sim = true;
    bool nt_tiers = false;
    double trump_lead_pen = 3.0;
    double to_declarer_bonus = 3.0;
    double bare_honour_lead_pen = 0.0;
    // fallback leads when too few worlds survive (_lead_old / _lead_low)
    bool pull_trumps = true;
    int pull_min_trumps = 2;
    int pull_min_trumps_def = 3;
    bool declarer_pull = true;
    int declarer_pull_min = 3;
    bool endgame_master_trump = true;
    double cash_threshold = 0.15;
    double lead_pts_w = 1.0;
    int low_pull_min = 3;
    double low_pull_bonus = 2.0;
    double guard_lead_bonus = 1.0;
    double bare_honour_pen = 2.0;
    double partner_top_w = 2.0;
    double opp_ruff_pen = 3.0;
    double partner_ruff_bonus = 3.0;
    double singleton_bonus = 1.0;
    double length_w = 0.2;
    // last to play with the highest trump left: take the trick with the
    // lowest trump that also wins instead
    bool last_ruff_low = true;
    // trump through to a declaring partner needs 2+ trumps (a lone trump is
    // my ruff)
    bool through_fix = true;
    // ruff convention: the opening leader (defending, holding a trump) leads a
    // side singleton at trick 1; partner returns that suit while the opener
    // can ruff. Also modelled in the play-out (rollout tr_rules bit 32).
    bool ruff_convention = false;
    // play-out over the sampled worlds
    bool playout = true;
    int playout_worlds = 384;
    int playout_max_cards = 8;
    double playout_margin = 0.0;
    double playout_z = 1.0;
    int playout_nt_rules = 31;
    // trump card play in the play-out (cpp/rule_playout.h bits). Off: +1.13
    // pts/deal of play-out fidelity offline, but -0.00 [-0.30, +0.30] in a
    // 2000-deal declared match (seed 99)
    int playout_tr_rules = 0;
    // world weights: belief exponent and per-seat rule likelihood
    double lik_belief_w = 0.75;
    double lik_partner_eps = 0.5;
    double lik_opp_eps = 0.7;
    // root Q blend; mull_root_q_w < 0 = use root_q_w at no trump too
    double root_q_w = 0.4;
    double mull_root_q_w = -1.0;
    // weight of the fast learned play-out (cpp/fast_policy.h) against the
    // rule play-out when choose() is given a net: +0.55 [+0.21, +0.90]
    // pts/deal at 1.0 in a 2000-deal declared match (seed 96), +0.43 at 0.5
    double playout_net_mix = 1.0;
    // > 0: hybrid continuation instead -- the net plays the first this many
    // plies after the candidate, the rule policy the rest (the full net is
    // ~70x the rule play-out's cost, far over the phone budget)
    int playout_net_plies = 0;
    // the net also finishes the trick in progress when its plies run out
    bool playout_net_trick_end = false;
    // weight-first deduplication + deterministic stratified resampling to
    // <= playout_worlds unique worlds before running the play-out
    bool playout_sir = false;
    // honest soft action log-likelihood from the fast policy net on suit deals
    // (replaces rule world_mismatches when > 0, net is provided, and trump >= 0)
    double lik_net_partner_w = 0.0;
    double lik_net_opp_w = 0.0;
    double lik_net_tau = 1.0;
    // exact double-dummy alpha-beta solver on mull (no-trump) deals when hand
    // size <= mull_endgame_ab_cards (0 = off); < 0 falls back to normal qw/z
    int mull_endgame_ab_cards = 0;
    double mull_endgame_q_w = -1.0;
    double mull_endgame_z = -1.0;
    // paired-test z on suit deals when the fast policy net plays the
    // continuation (< 0 = use playout_z)
    double suit_net_z = -1.0;
    // two-stage candidate racing: evaluate all legal cards on playout_race_k
    // worlds first, prune cards > playout_race_z SE behind the leader, and
    // run the remaining worlds only on surviving contenders (0 = off)
    int playout_race_k = 0;
    double playout_race_z = 2.5;
    // optional late-deal Q blend weight on suit deals at trick 7..8 (hand size
    // <= 2): when >= 0 and trump >= 0, linearly tapers base_qw from trick 1
    // (hand size 8) to trick 7 (hand size 2)
    double suit_root_q_w_late = -1.0;
};

struct Decision {
    int card = -1;
    std::string why;
    // play-out value per legal card (team points, blended), empty when the
    // play-out did not run
    std::vector<int> value_cards;
    std::vector<double> values;
};

// Choose a card for `me`, the player to move in `game`.
//   worlds   : sampled determinizations, current hands by ABSOLUTE seat
//              (sample_determinizations output, the n_worlds draw)
//   history  : every card played so far incl. the current trick, (seat, card)
//   belief   : 3x32 holder probabilities for seats me+1, me+2, me+3 (the
//              belief net's output for this information set); may be null,
//              then worlds are weighted by the rule likelihood only
//   q        : 32 floats, the Q net's value ((pts-30)/30) for each legal
//              card, indexed by card; may be null (no blend)
//   extra    : optional extra log-weight per entry of `worlds` (e.g. the
//              likelihood of a called meegaan under the decision layer,
//              learn/rule_agent.py::decl_log_weights); may be null
//   dealer   : the declarer (encoder's DECLARER feature, play-out declarer)
Decision choose(const ManilleGame& game, int me, int dealer,
                const std::vector<rollout::World>& worlds,
                const rollout::History& history,
                const float* belief, const float* q,
                const Params& p, const double* extra = nullptr,
                const fastpolicy::Net* net = nullptr);

} // namespace rule

#endif // MANILLE_RULE_AGENT_H
