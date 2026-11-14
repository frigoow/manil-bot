#ifndef MANILLE_PRIOR_TRUMP_RULE_H
#define MANILLE_PRIOR_TRUMP_RULE_H

#include <cstdint>

namespace prior {

// Suits follow the codebase ordering from card.h:
// 0=Diamonds, 1=Hearts, 2=Clubs, 3=Spades.
// This baseline rule never chooses no-trump.
int apply_trump_rule(int64_t hand_mask);

// Returns exactly 1.0 when the observed suit matches the rule, else 0.0.
// Kept as double so future stochastic policies can plug in without API churn.
double trump_consistency(int64_t hand_mask, int observed_suit);

} // namespace prior

#endif // MANILLE_PRIOR_TRUMP_RULE_H
