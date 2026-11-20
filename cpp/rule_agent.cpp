#include "rule_agent.h"

#include "encoding.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <tuple>

// Mirrors learn/rule_agent.py. Every loop runs in the Python's iteration
// order and keeps the FIRST best element (Python's max/min), sums run in the
// same order, so the two make identical choices on the same worlds -- up to
// floating-point summation order in the world weights (numpy sums pairwise),
// which only matters on razor-edge overrule tests.

namespace rule {
namespace {

const int PTS[8] = {0, 0, 0, 1, 2, 3, 4, 5};
const char* RANK_STR[8] = {"7", "8", "9", "J", "Q", "K", "A", "10"};
const char* SUIT_STR[4] = {"D", "H", "C", "S"};

inline int suitOf(int c) { return c >> 3; }
inline int rankOf(int c) { return c & 7; }
inline int64_t bit(int c) { return int64_t(1) << c; }

std::string cstr(int c) { return std::string(RANK_STR[rankOf(c)]) + SUIT_STR[suitOf(c)]; }

std::string fmt(const char* f, double a) {
    char buf[64];
    std::snprintf(buf, sizeof buf, f, a);
    return buf;
}

using Trick = std::vector<std::pair<int, int>>;  // (seat, card)

// Ordering for probability estimates only (rule_agent.py::outranks).
bool outranks(int u, int x, int trump) {
    if (trump >= 0 && suitOf(u) == trump && suitOf(x) != trump)
        return true;
    return suitOf(u) == suitOf(x) && rankOf(u) > rankOf(x);
}

bool beats(int a, int b, int trump) {
    return Card::from_index(a).beats(Card::from_index(b), trump);
}

// Winner of a (partial) trick, via the engine's Card::beats (_winner).
std::pair<int, int> winnerOf(const Trick& t, int trump) {
    int ws = t[0].first, wc = t[0].second;
    for (size_t i = 1; i < t.size(); ++i)
        if (beats(t[i].second, wc, trump)) {
            ws = t[i].first;
            wc = t[i].second;
        }
    return {ws, wc};
}

// Legal cards of `seat` holding `hand` if it played next on trick `t`: asked
// of the engine (a state with that seat to move), never re-derived here.
std::vector<int> legalFor(int64_t hand, const Trick& t, int seat, int trump) {
    std::array<int64_t, 4> hands{0, 0, 0, 0};
    hands[seat] = hand;
    std::array<int32_t, 4> tc{-1, -1, -1, -1}, tp{-1, -1, -1, -1};
    for (size_t i = 0; i < t.size(); ++i) {
        tp[i] = t[i].first;
        tc[i] = t[i].second;
    }
    ManilleGame g(hands, tc, tp, static_cast<int>(t.size()), seat, trump, 0, 0, 0, 0);
    std::vector<int> out;
    for (const Card& c : g.validMoves())
        out.push_back(c.index());
    return out;
}

// --------------------------------------------------------------------------
// Knowledge: rule_agent.py::Knowledge. Seats relative: 0 me, 1 next,
// 2 partner, 3 previous.
struct Knowledge {
    int trump = -1;
    int declarer = -1;  // relative seat, -1 unknown
    std::vector<int> hand;
    int cap[4][4];
    int played_by[4] = {0, 0, 0, 0};
    Trick trick;
    std::vector<int> unseen;
    int size[4];
    std::vector<int> legal;
    std::vector<std::array<float, encoding::ACT_DIM>> act;  // per legal index
    std::vector<std::vector<int>> poss;                     // [card]
    double prob[32][4];
    bool haveW = false;
    std::vector<std::array<int64_t, 4>> W;  // relative seats
    std::vector<int> Widx;                  // index of each W row in the input worlds
    // the deal's opening lead (relative seat, card) and whether that seat
    // played the suit again since
    int first_seat = -1, first_card = -1;
    bool first_followed = false;

    const float* actOf(int c) const {
        for (size_t i = 0; i < legal.size(); ++i)
            if (legal[i] == c)
                return act[i].data();
        return nullptr;
    }

    double p_holds(int c, int p) const { return prob[c][p]; }

    double p_none(const std::vector<int>& cards, int p) const {
        if (haveW) {
            int64_t m = 0;
            for (int c : cards)
                m |= bit(c);
            if (!m)
                return 1.0;
            long n = 0;
            for (auto& w : W)
                n += (w[p] & m) == 0;
            return double(n) / double(W.size());
        }
        double q = 1.0;
        for (int c : cards)
            q *= 1.0 - p_holds(c, p);
        return q;
    }

    std::vector<int> unseen_suit(int s) const {
        std::vector<int> out;
        for (int c : unseen)
            if (suitOf(c) == s)
                out.push_back(c);
        return out;
    }

    std::vector<int> higher_unseen(int c) const {
        std::vector<int> out;
        for (int u : unseen)
            if (suitOf(u) == suitOf(c) && rankOf(u) > rankOf(c))
                out.push_back(u);
        return out;
    }

    bool is_master(int c) const { return higher_unseen(c).empty(); }

    bool is_boss(int c) const {
        int s = suitOf(c), r = rankOf(c);
        return is_master(c) || (cap[1][s] < r && cap[3][s] < r);
    }

    double p_ruff(int s, int o) const {
        if (trump < 0 || s == trump)
            return 0.0;
        int64_t sm = 0, tm = 0;
        for (int c : unseen_suit(s))
            sm |= bit(c);
        for (int c : unseen_suit(trump))
            tm |= bit(c);
        if (!tm)
            return 0.0;
        if (haveW) {
            long n = 0;
            for (auto& w : W)
                n += ((w[o] & sm) == 0) && ((w[o] & tm) != 0);
            return double(n) / double(W.size());
        }
        return p_none(unseen_suit(s), o) * (1.0 - p_none(unseen_suit(trump), o));
    }

    double partner_forced_pts(int s) const {
        if (!haveW)
            return 0.0;
        std::vector<int> cards = unseen_suit(s);
        if (cards.empty())
            return 0.0;
        double tot = 0.0;
        for (auto& w : W) {
            int lo = -1;
            for (int c : cards)
                if ((w[2] >> c) & 1)
                    lo = lo < 0 ? PTS[rankOf(c)] : std::min(lo, PTS[rankOf(c)]);
            tot += lo < 0 ? 0 : lo;
        }
        return tot / double(W.size());
    }

    double p_beats(int x, int o, int led) const {
        int sx = suitOf(x);
        std::vector<int> higher = higher_unseen(x);
        if (sx == led) {
            double p_high = 1.0 - p_none(higher, o);
            if (trump >= 0 && sx != trump) {
                double p_void = p_none(unseen_suit(led), o);
                double p_trump = 1.0 - p_none(unseen_suit(trump), o);
                return std::min(1.0, p_high + p_void * p_trump);
            }
            return p_high;
        }
        if (trump >= 0 && sx == trump) {
            double p_void = p_none(unseen_suit(led), o);
            return p_void * (1.0 - p_none(higher, o));
        }
        return 1.0;
    }

    std::pair<int, int> current_winner() const {
        if (trick.empty())
            return {-1, -1};
        int ws = trick[0].first, wc = trick[0].second;
        for (size_t i = 1; i < trick.size(); ++i)
            if (outranks(trick[i].second, wc, trump)) {
                ws = trick[i].first;
                wc = trick[i].second;
            }
        return {ws, wc};
    }
};

int argmax(const float* v, int n) {
    int b = 0;
    for (int i = 1; i < n; ++i)
        if (v[i] > v[b])
            b = i;
    return b;
}

float maxOf(const float* v, int n) {
    float m = v[0];
    for (int i = 1; i < n; ++i)
        m = std::max(m, v[i]);
    return m;
}

void readHistory(Knowledge& k, const float* obs, std::vector<bool>& seen) {
    using namespace encoding;
    Trick trick;
    for (int i = 0; i < 32; ++i) {
        const float* p = obs + HISTORY_OFF + i * PLY_DIM;
        if (maxOf(p, 4) == 0)
            break;
        int seat = argmax(p, 4);
        int card = argmax(p + 4, 32);
        int led = argmax(p + 36, 4);
        int pos = argmax(p + 40, 4);
        const float* f = p + PLY_FLAG_OFF;
        if (pos == 0)
            trick.clear();
        int winner_card = -1;
        if (pos > 0 && maxOf(p + 44, 4) > 0) {
            int wseat = argmax(p + 44, 4);
            for (auto& sc : trick)
                if (sc.first == wseat) {
                    winner_card = sc.second;
                    break;
                }
        }
        if (winner_card >= 0) {
            if (f[F_FOLLOWED_NO_BEAT] != 0 && suitOf(winner_card) == led)
                k.cap[seat][led] = std::min(k.cap[seat][led], rankOf(winner_card));
            if (f[F_DUCKED_TRUMP] != 0 && k.trump >= 0)
                k.cap[seat][k.trump] = std::min(k.cap[seat][k.trump], rankOf(winner_card));
            if (f[F_OBLIGATION] != 0 && f[F_VOID_ESTABLISHED] != 0 && f[F_TRUMPED] == 0
                    && k.trump >= 0 && suitOf(winner_card) != k.trump)
                k.cap[seat][k.trump] = -1;
        }
        if (i == 0) {
            k.first_seat = seat;
            k.first_card = card;
        } else if (seat == k.first_seat && suitOf(card) == suitOf(k.first_card)) {
            k.first_followed = true;
        }
        trick.emplace_back(seat, card);
        seen[card] = true;
        k.played_by[seat] += 1;
        if (f[F_TRICK_COMPLETED] != 0)
            trick.clear();
    }
    k.trick = trick;
    for (int s = 0; s < 4; ++s)
        k.cap[0][s] = 7;
}

void locate(Knowledge& k, const std::vector<rollout::World>& worlds, int me, int min_worlds) {
    const int others[3] = {1, 2, 3};
    k.poss.assign(32, {});
    for (int c : k.unseen)
        for (int p : others)
            if (k.cap[p][suitOf(c)] >= rankOf(c))
                k.poss[c].push_back(p);
    for (int it = 0; it < 4; ++it) {
        bool changed = false;
        for (int p : others) {
            std::vector<int> mine, fixed;
            for (int c : k.unseen)
                if (std::find(k.poss[c].begin(), k.poss[c].end(), p) != k.poss[c].end())
                    mine.push_back(c);
            for (int c : mine)
                if (k.poss[c].size() == 1 && k.poss[c][0] == p)
                    fixed.push_back(c);
            if (static_cast<int>(mine.size()) == k.size[p] && fixed.size() < mine.size()) {
                for (int c : mine)
                    if (!(k.poss[c].size() == 1 && k.poss[c][0] == p)) {
                        k.poss[c] = {p};
                        changed = true;
                    }
            } else if (static_cast<int>(fixed.size()) == k.size[p] && mine.size() > fixed.size()) {
                for (int c : mine)
                    if (!(k.poss[c].size() == 1 && k.poss[c][0] == p)) {
                        std::vector<int> q;
                        for (int x : k.poss[c])
                            if (x != p)
                                q.push_back(x);
                        k.poss[c] = q;
                        changed = true;
                    }
            }
        }
        if (!changed)
            break;
    }
    for (int c = 0; c < 32; ++c)
        for (int p = 0; p < 4; ++p)
            k.prob[c][p] = 0.0;
    if (!worlds.empty()) {
        // _locate_from_worlds: keep the worlds that respect the rank caps
        std::vector<std::array<int64_t, 4>> W;
        std::vector<int> idx;
        int64_t forbid[4] = {0, 0, 0, 0};
        for (int r = 1; r < 4; ++r)
            for (int su = 0; su < 4; ++su)
                for (int rk = k.cap[r][su] + 1; rk < 8; ++rk)
                    forbid[r] |= bit(su * 8 + rk);
        for (size_t wi = 0; wi < worlds.size(); ++wi) {
            const auto& w = worlds[wi];
            std::array<int64_t, 4> rel;
            for (int r = 0; r < 4; ++r)
                rel[r] = w[(me + r) % 4];
            bool keep = true;
            for (int r = 1; r < 4; ++r)
                if (forbid[r] && (rel[r] & forbid[r]) != 0)
                    keep = false;
            if (keep) {
                W.push_back(rel);
                idx.push_back(static_cast<int>(wi));
            }
        }
        if (static_cast<int>(W.size()) >= min_worlds
                || (!W.empty() && W.size() == worlds.size())) {
            k.W = std::move(W);
            k.Widx = std::move(idx);
            k.haveW = true;
            for (int c : k.unseen)
                for (int p = 1; p < 4; ++p) {
                    long n = 0;
                    for (auto& w : k.W)
                        n += (w[p] & bit(c)) != 0;
                    k.prob[c][p] = double(n) / double(k.W.size());
                }
        }
    }
    if (!k.haveW) {
        for (int c : k.unseen) {
            double z = 0.0;
            for (int p : k.poss[c])
                z += std::max(double(k.size[p]), 1e-3);
            if (z == 0.0)
                z = 1.0;
            for (int p : k.poss[c])
                k.prob[c][p] = std::max(double(k.size[p]), 1e-3) / z;
        }
    }
}

Knowledge buildKnowledge(const ManilleGame& game, int me,
                         const std::vector<rollout::World>& worlds, int min_worlds) {
    using namespace encoding;
    Knowledge k;
    std::vector<float> obs(OBS_DIM, 0.0f);
    encode(game, me, obs.data());
    for (const Card& c : game.validMoves()) {
        k.legal.push_back(c.index());
        std::array<float, ACT_DIM> a{};
        encode_action(game, c.index(), a.data());
        k.act.push_back(a);
    }
    const float* o = obs.data();
    k.trump = maxOf(o + TRUMP_OFF, 4) > 0 ? argmax(o + TRUMP_OFF, 4) : -1;
    k.declarer = maxOf(o + DECLARER_OFF, 4) > 0 ? argmax(o + DECLARER_OFF, 4) : -1;
    std::vector<bool> seen(32, false);
    for (int c = 0; c < 32; ++c)
        if (o[OWN_HAND_OFF + c] > 0) {
            k.hand.push_back(c);
            seen[c] = true;
        }
    for (int r = 0; r < 4; ++r)
        for (int s = 0; s < 4; ++s)
            k.cap[r][s] = o[VOID_OFF + r * 4 + s] > 0 ? -1 : 7;
    readHistory(k, o, seen);
    for (int c = 0; c < 32; ++c)
        if (!seen[c])
            k.unseen.push_back(c);
    k.size[0] = static_cast<int>(k.hand.size());
    for (int r = 1; r < 4; ++r)
        k.size[r] = 8 - k.played_by[r];
    locate(k, worlds, me, min_worlds);
    return k;
}

// --------------------------------------------------------------------------
// RuleAgent methods
class Agent {
public:
    Agent(const Knowledge& k, const Params& p) : k(k), P(p) {}

    std::pair<int, std::string> follow() const;
    std::pair<int, std::string> lead() const;

private:
    const Knowledge& k;
    const Params& P;

    static bool inSkip(int mask, int tier) { return (mask >> tier) & 1; }

    double power(int c) const {
        const int trump = k.trump;
        const int s = suitOf(c);
        std::vector<int> higher = k.higher_unseen(c);
        const int opps[2] = {1, 3};
        if (trump >= 0 && s == trump) {
            if (higher.empty())
                return 0.95;
            double n_opp = 0.0;
            for (int u : higher)
                for (int o : opps)
                    n_opp += k.p_holds(u, o);
            return std::max(0.15, P.trump_power - P.trump_power_slope * n_opp);
        }
        double n_higher_opp = 0.0;
        for (int u : higher)
            for (int o : opps)
                n_higher_opp += k.p_holds(u, o);
        int mine_in_suit = 0;
        for (int h : k.hand)
            mine_in_suit += suitOf(h) == s;
        double ruff_risk = 0.0;
        if (trump >= 0) {
            double safe = 1.0;
            for (int o : opps)
                safe *= 1.0 - k.p_ruff(s, o);
            ruff_risk = std::min(1.0, (1.0 - safe) * P.ruff_future);
        }
        if (higher.empty())
            return 0.9 * (1 - ruff_risk);
        if (mine_in_suit - 1 >= static_cast<int>(higher.size()))
            return P.guard_power * (1 - ruff_risk) / (1 + 0.5 * n_higher_opp);
        return 0.1 / (1 + n_higher_opp);
    }

    double keep_value(int c) const {
        double pw = power(c);
        double v = pw * P.keep_win_value + (2 * pw - 1) * PTS[rankOf(c)] * P.keep_pts_weight;
        if (k.trump >= 0 && suitOf(c) == k.trump)
            v += P.trump_keep_bonus;
        return v;
    }

    // rule_agent.py::_respond
    int respond(const Trick& trick, int seat, const std::array<int64_t, 4>& hands,
                const std::vector<int>& later, bool vet) const {
        const int trump = k.trump;
        std::vector<int> legal = legalFor(hands[seat], trick, seat, trump);
        if (legal.empty())
            return -1;
        auto opp_behind_beats = [&](const Trick& t) {
            auto w = winnerOf(t, trump);
            for (int o : later) {
                if (o % 2 == w.first % 2)
                    continue;
                for (int c : legalFor(hands[o], t, o, trump))
                    if (beats(c, w.second, trump))
                        return true;
            }
            return false;
        };
        auto isTrump = [&](int c) { return trump >= 0 && suitOf(c) == trump; };
        auto cheapest = [&](const std::vector<int>& cs) {
            int best = -1;
            std::tuple<int, int, int> bk;
            for (int c : cs) {
                auto key = std::make_tuple(PTS[rankOf(c)], isTrump(c) ? 1 : 0, rankOf(c));
                if (best < 0 || key < bk) {
                    best = c;
                    bk = key;
                }
            }
            return best;
        };
        auto lowestWinner = [&](const std::vector<int>& cs) {
            int best = -1;
            std::tuple<int, int> bk;
            for (int c : cs) {
                auto key = std::make_tuple(isTrump(c) ? 1 : 0, rankOf(c));
                if (best < 0 || key < bk) {
                    best = c;
                    bk = key;
                }
            }
            return best;
        };
        auto w = winnerOf(trick, trump);
        if (w.first % 2 == seat % 2 && !opp_behind_beats(trick)) {
            if (vet) {
                int best = -1;
                std::tuple<int, int, int> bk;
                for (int c : legal) {
                    auto key = std::make_tuple(isTrump(c) ? 0 : 1, PTS[rankOf(c)], -rankOf(c));
                    if (best < 0 || key > bk) {
                        best = c;
                        bk = key;
                    }
                }
                return best;
            }
            return cheapest(legal);
        }
        std::vector<int> secure;
        for (int c : legal) {
            Trick t = trick;
            t.emplace_back(seat, c);
            auto tw = winnerOf(t, trump);
            if (tw.first % 2 == seat % 2 && !opp_behind_beats(t))
                secure.push_back(c);
        }
        if (!secure.empty())
            return lowestWinner(secure);
        std::vector<int> winners;
        for (int c : legal)
            if (beats(c, w.second, trump))
                winners.push_back(c);
        if (w.first % 2 != seat % 2 && !winners.empty())
            return lowestWinner(winners);
        return cheapest(legal);
    }

    // rule_agent.py::simulate_trick -> (P(our side wins), E[signed points])
    std::pair<double, double> simulate_trick(int cand) const {
        const int n = std::min(static_cast<int>(k.W.size()), P.sim_worlds);
        Trick base = k.trick;
        base.emplace_back(0, cand);
        const int leader = base[0].first;
        std::vector<int> after;
        for (int i = static_cast<int>(base.size()); i < 4; ++i)
            after.push_back((leader + i) % 4);
        int wins = 0;
        double signed_pts = 0.0;
        for (int wi = 0; wi < n; ++wi) {
            Trick trick = base;
            std::array<int64_t, 4> hands{0, 0, 0, 0};
            for (int seat : after)
                hands[seat] = k.W[wi][seat];
            for (size_t j = 0; j < after.size(); ++j) {
                int seat = after[j];
                std::vector<int> later(after.begin() + j + 1, after.end());
                int play = respond(trick, seat, hands, later, P.sim_vet);
                if (play < 0)
                    continue;
                trick.emplace_back(seat, play);
                hands[seat] &= ~bit(play);
            }
            auto w = winnerOf(trick, k.trump);
            int pts = 0;
            for (auto& sc : trick)
                pts += PTS[rankOf(sc.second)];
            if (w.first % 2 == 0) {
                wins += 1;
                signed_pts += pts;
            } else {
                signed_pts -= pts;
            }
        }
        return {double(wins) / n, signed_pts / n};
    }

    struct SuitView {
        std::vector<int> mine, unseen, masters;
        double p_opp_ruff, p_partner_ruff, p_partner_top, opp_len;
    };

    SuitView suit_view(int s) const {
        SuitView v;
        for (int c : k.hand)
            if (suitOf(c) == s)
                v.mine.push_back(c);  // ascending index = ascending rank
        v.unseen = k.unseen_suit(s);
        v.p_opp_ruff = 1.0 - (1 - k.p_ruff(s, 1)) * (1 - k.p_ruff(s, 3));
        for (int c : v.mine)
            if (k.trump < 0 && P.nt_tiers ? k.is_boss(c) : k.is_master(c))
                v.masters.push_back(c);
        int top = v.unseen.empty() ? -1 : v.unseen.back();
        v.p_partner_ruff = k.p_ruff(s, 2) * (1 - k.p_ruff(s, 3));
        v.p_partner_top = top >= 0 ? k.p_holds(top, 2) : 0.0;
        v.opp_len = 0.0;
        for (int u : v.unseen)
            v.opp_len += k.p_holds(u, 1) + k.p_holds(u, 3);
        return v;
    }

    std::pair<int, std::string> lead_plan() const;
    std::pair<int, std::string> lead_sim() const;
    std::pair<int, std::string> lead_old() const;
    std::pair<int, std::string> lead_low() const;
};

std::pair<int, std::string> Agent::follow() const {
    const int led = suitOf(k.trick[0].second);
    const int leader = k.trick[0].first;
    const int n = static_cast<int>(k.trick.size());
    std::vector<int> after, opps_after;
    for (int i = n + 1; i < 4; ++i)
        after.push_back((leader + i) % 4);
    bool partner_after = false;
    for (int s : after) {
        if (s == 1 || s == 3)
            opps_after.push_back(s);
        if (s == 2)
            partner_after = true;
    }
    auto cw = k.current_winner();
    int trick_pts = 0;
    for (auto& sc : k.trick)
        trick_pts += PTS[rankOf(sc.second)];
    const double rest = after.size() * P.trick_rest;

    int best = -1;
    double best_ev = 0.0;
    std::string why;
    for (int c : k.legal) {
        const float* a = k.actOf(c);
        bool wins_now = a[encoding::ACT_WINS_OFF] > 0;
        bool partner_winning = a[encoding::ACT_PARTNER_WINNING_OFF] > 0;
        double p_win;
        if (wins_now || partner_winning) {
            int win_card = wins_now ? c : cw.second;
            double p_safe;
            if (wins_now && a[encoding::ACT_SURE_WIN_OFF] > 0) {
                p_safe = 1.0;
            } else if (wins_now && a[encoding::ACT_OPP_CAN_BEAT_OFF] == 0) {
                p_safe = 1.0;
            } else {
                p_safe = 1.0;
                for (int o : opps_after)
                    p_safe *= 1.0 - k.p_beats(win_card, o, led);
            }
            p_win = p_safe + (partner_after ? (1 - p_safe) * 0.3 : 0.0);
        } else {
            p_win = 0.0;
            if (partner_after) {
                double p_part = k.p_beats(cw.second, 2, led);
                p_win = p_part * std::pow(P.partner_after_f, double(opps_after.size()));
            }
        }
        double own = PTS[rankOf(c)] * (wins_now ? P.win_own_pts_w : 1.0);
        double stake = trick_pts + own + rest;
        double ev;
        if (k.haveW) {
            auto sim = simulate_trick(c);
            p_win = sim.first;
            ev = sim.second - keep_value(c);
        } else {
            ev = (2 * p_win - 1) * stake - keep_value(c);
        }
        if (best < 0 || ev > best_ev + 1e-9) {
            best = c;
            best_ev = ev;
            why = "p_win=" + fmt("%.2f", p_win) + " stake=" + fmt("%.1f", stake)
                + " ev=" + fmt("%.2f", ev);
        }
    }
    return {best, why};
}

std::pair<int, std::string> Agent::lead() const {
    if (k.haveW)
        return lead_plan();
    return lead_old();
}

std::pair<int, std::string> Agent::lead_plan() const {
    const int trump = k.trump;
    const std::vector<int>& legal = k.legal;
    std::vector<int> side;
    for (int s = 0; s < 4; ++s) {
        if (s == trump)
            continue;
        for (int c : legal)
            if (suitOf(c) == s) {
                side.push_back(s);
                break;
            }
    }
    std::array<SuitView, 4> view;
    for (int s : side)
        view[s] = suit_view(s);
    std::vector<int> my_tr;
    if (trump >= 0)
        for (int c : legal)
            if (suitOf(c) == trump)
                my_tr.push_back(c);
    double opp_tr = 0.0, par_tr = 0.0;
    if (trump >= 0)
        for (int u : k.unseen_suit(trump)) {
            opp_tr += k.p_holds(u, 1) + k.p_holds(u, 3);
            par_tr += k.p_holds(u, 2);
        }
    const int skip = P.plan_skip_mask;
    const int nh = static_cast<int>(k.hand.size());

    // 0. ruff convention: the opener leads a side singleton at trick 1,
    //    partner returns that suit while the opener can still ruff
    if (P.ruff_convention && trump >= 0) {
        if (k.first_seat < 0 && k.declarer != 0 && k.declarer != 2 && !my_tr.empty()) {
            int best = -1;
            for (int c : legal) {
                if (suitOf(c) == trump)
                    continue;
                int n_s = 0;
                for (int h : k.hand)
                    n_s += suitOf(h) == suitOf(c);
                if (n_s == 1 && (best < 0 || std::make_pair(PTS[rankOf(c)], rankOf(c))
                                             < std::make_pair(PTS[rankOf(best)], rankOf(best))))
                    best = c;
            }
            if (best >= 0)
                return {best, "plan: ruff convention, singleton lead"};
        }
        if (k.first_seat == 2 && !k.first_followed && suitOf(k.first_card) != trump
                && k.cap[2][trump] >= 0) {
            int low = -1;
            for (int c : legal)
                if (suitOf(c) == suitOf(k.first_card) && (low < 0 || rankOf(c) < rankOf(low)))
                    low = c;
            if (low >= 0)
                return {low, "plan: ruff convention, return partner's suit"};
        }
    }

    // 1. endgame: a master trump keeps control of the last tricks
    if (!inSkip(skip, 1) && !my_tr.empty() && opp_tr > 0.2 && nh <= 3
            && k.is_master(my_tr.back()))
        return {my_tr.back(), "plan: endgame master trump"};

    // 2. draw trumps -- only with a majority and side winners to protect
    if (!inSkip(skip, 2) && !my_tr.empty() && opp_tr > P.plan_pull_min_opp) {
        bool majority = my_tr.size() + par_tr > opp_tr + P.plan_majority_margin;
        bool winners = false;
        for (int s : side)
            if (!view[s].masters.empty())
                winners = true;
        for (int s : side)
            if (view[s].mine.size() >= 3)
                winners = true;
        bool we_declare = k.declarer == 0 || k.declarer == 2;
        if (majority && winners && (we_declare || my_tr.size() >= 3)) {
            if (k.is_master(my_tr.back()))
                return {my_tr.back(), "plan: draw trumps (master) opp~" + fmt("%.1f", opp_tr)};
            if (k.declarer == 0 && my_tr.size() >= 3)
                return {my_tr.front(), "plan: declarer draws low opp~" + fmt("%.1f", opp_tr)};
        }
        if (k.declarer == 2 && P.plan_to_declarer && !k.is_master(my_tr.front())
                && PTS[rankOf(my_tr.front())] <= 2) {
            if (!P.through_fix || my_tr.size() >= 2)
                return {my_tr.front(), "plan: trump through to declaring partner"};
        }
    }

    // 3. partner ruff
    if (!inSkip(skip, 3) && P.plan_ruff != 0.0 && par_tr > 0) {
        double bp = -1.0;
        int bc = -1;
        for (int s : side) {
            const SuitView& v = view[s];
            if (v.p_partner_ruff >= P.plan_ruff && (bc < 0 || v.p_partner_ruff > bp)) {
                bp = v.p_partner_ruff;
                bc = v.mine[0];
            }
        }
        if (bc >= 0)
            return {bc, "plan: partner ruff p=" + fmt("%.2f", bp)};
    }

    // 3b. one enemy trump left, opponents void in a side suit
    if (P.plan_force_trump && trump >= 0) {
        std::vector<int> ut = k.unseen_suit(trump);
        if (ut.size() == 1 && (k.p_holds(ut[0], 1) + k.p_holds(ut[0], 3)) > 0.7
                && (!my_tr.empty() || par_tr > 0.5)) {
            int best = -1;
            for (int s : side) {
                const SuitView& v = view[s];
                int low = v.mine[0];
                bool both_void = v.unseen.empty()
                    || k.p_none(v.unseen, 1) * k.p_none(v.unseen, 3) > 0.9;
                if (both_void && PTS[rankOf(low)] <= 1 && !k.is_master(low))
                    if (best < 0 || PTS[rankOf(low)] < PTS[rankOf(best)])
                        best = low;
            }
            if (best >= 0)
                return {best, "plan: force out the last enemy trump"};
        }
    }

    // 4. cash safe masters, biggest first
    {
        bool have = false;
        std::tuple<double, int, double> bestT;
        for (int s : side) {
            const SuitView& v = view[s];
            if (v.masters.empty())
                continue;
            int c = v.masters.back();
            double p_safe = 1.0 - v.p_opp_ruff;
            double risk = PTS[rankOf(c)] + k.partner_forced_pts(s);
            double need = std::min(0.95, P.plan_cash + P.plan_cash_risk * (risk - 4));
            if (p_safe >= need) {
                auto t = std::make_tuple(p_safe * (PTS[rankOf(c)] + 3), c, p_safe);
                if (!have || t > bestT) {
                    bestT = t;
                    have = true;
                }
            }
        }
        if (have && !inSkip(skip, 4))
            return {std::get<1>(bestT), "plan: cash master p_safe=" + fmt("%.2f", std::get<2>(bestT))};
    }

    // 5. establish: honour behind missing higher cards, with guards
    {
        bool have = false;
        std::tuple<double, int, int> bestT;
        for (int s : side) {
            const SuitView& v = view[s];
            const std::vector<int>& mine = v.mine;
            if (mine.size() < 2 || v.p_opp_ruff > P.plan_est_ruff)
                continue;
            std::vector<int> hon;
            for (int c : mine)
                if (PTS[rankOf(c)] >= 2 && !k.is_master(c))
                    hon.push_back(c);
            if (hon.empty())
                continue;
            int h = hon.back();
            int n_higher = static_cast<int>(k.higher_unseen(h).size());
            if (static_cast<int>(mine.size()) - 1 < n_higher)
                continue;
            std::vector<int> run{h};
            for (int r = rankOf(h) - 1; r >= 0; --r) {
                int c = (s << 3) | r;
                if (std::find(mine.begin(), mine.end(), c) == mine.end())
                    break;
                run.push_back(c);
            }
            int lead;
            if (P.plan_est_v2) {
                if (n_higher != 1 || v.p_opp_ruff > P.plan_est_v2_ruff)
                    continue;
                bool entry = false;
                for (int c : k.hand)
                    if (std::find(mine.begin(), mine.end(), c) == mine.end()
                            && (k.is_master(c) || (trump >= 0 && suitOf(c) == trump)))
                        entry = true;
                if (!entry)
                    continue;
                lead = run.size() >= 2 ? run.back() : mine[0];
            } else if (trump < 0 && P.nt_est_low_run) {
                // Q from A-K-Q forces the manille as well and costs less
                lead = run.size() >= 2 ? run.back() : mine[0];
            } else {
                lead = run.size() >= 2 ? h : mine[0];
            }
            double score = double(PTS[rankOf(h)] + static_cast<int>(mine.size()) - 2 * n_higher)
                - 3 * v.p_opp_ruff;
            auto t = std::make_tuple(score, lead, h);
            if (!have || t > bestT) {
                bestT = t;
                have = true;
            }
        }
        if (have && !inSkip(skip, 5))
            return {std::get<1>(bestT), "plan: establish " + cstr(std::get<2>(bestT))};
    }

    // 6. toward partner's strength
    {
        bool have = false;
        std::tuple<double, int> bestT;
        for (int s : side) {
            const SuitView& v = view[s];
            if (v.p_partner_top >= P.plan_partner_top && v.p_opp_ruff < 0.5) {
                if (trump < 0 && P.nt_tiers && v.p_partner_top < 0.9
                        && ((k.cap[1][s] < 0) != (k.cap[3][s] < 0)))
                    continue;
                int c = v.mine[0];
                if (PTS[rankOf(c)] <= 1 || v.mine.size() == 1) {
                    auto t = std::make_tuple(v.p_partner_top - 0.2 * PTS[rankOf(c)], c);
                    if (!have || t > bestT) {
                        bestT = t;
                        have = true;
                    }
                }
            }
        }
        if (have && !inSkip(skip, 6))
            return {std::get<1>(bestT), "plan: toward partner p_top=" + fmt("%.2f", std::get<0>(bestT))};
    }

    // 7. safe exit
    if (P.plan_exit_sim)
        return lead_sim();
    int best = -1;
    double best_sc = 0.0;
    for (int c : legal) {
        int s = suitOf(c);
        double sc;
        if (trump >= 0 && s == trump) {
            sc = -P.trump_lead_pen - PTS[rankOf(c)] - 2 * power(c);
        } else {
            const SuitView& v = view[s];
            const std::vector<int>& mine = v.mine;
            sc = -PTS[rankOf(c)] * 1.5;
            sc -= 4 * v.p_opp_ruff;
            if (c != mine[0])
                sc -= 1.0;
            if (mine.size() == 2 && c == mine[0] && PTS[rankOf(mine[1])] >= 3
                    && !k.is_master(mine[1]))
                sc -= 1.5;
            sc += 0.3 * v.opp_len;
            sc -= 0.5 * power(c) * 3;
        }
        if (best < 0 || sc > best_sc) {
            best = c;
            best_sc = sc;
        }
    }
    return {best, "plan: safe exit sc=" + fmt("%.2f", best_sc)};
}

std::pair<int, std::string> Agent::lead_sim() const {
    const int trump = k.trump;
    int best = -1;
    double best_ev = 0.0;
    std::string why;
    for (int c : k.legal) {
        auto sim = simulate_trick(c);
        double ev = sim.second - keep_value(c);
        const int s_ = suitOf(c);
        std::vector<int> mine;
        for (int h : k.hand)
            if (suitOf(h) == s_)
                mine.push_back(h);
        std::string tag;
        if (trump >= 0 && s_ == trump) {
            if (k.declarer == 2 && c == mine[0]
                    && (!P.through_fix || mine.size() >= 2)) {
                bool any = false;
                for (int u : k.unseen_suit(trump))
                    for (int o : {1, 3})
                        if (k.p_holds(u, o) > 0)
                            any = true;
                if (any) {
                    ev += P.to_declarer_bonus;
                    tag = " trump-to-declarer";
                }
            }
        } else {
            bool seq_top = false;
            if (rankOf(c) > 0) {
                int below = (s_ << 3) | (rankOf(c) - 1);
                seq_top = std::find(k.hand.begin(), k.hand.end(), below) != k.hand.end();
            }
            if (PTS[rankOf(c)] >= 2 && !k.is_master(c) && mine.size() <= 2 && !seq_top) {
                ev -= P.bare_honour_lead_pen;
                tag = " unsupported-honour";
            }
        }
        if (best < 0 || ev > best_ev) {
            best = c;
            best_ev = ev;
            why = "sim lead p_win=" + fmt("%.2f", sim.first) + " ev=" + fmt("%.2f", ev) + tag;
        }
    }
    return {best, why};
}

std::pair<int, std::string> Agent::lead_old() const {
    const int trump = k.trump;
    const int opps[2] = {1, 3};
    if (trump >= 0 && P.pull_trumps) {
        std::vector<int> my_trumps;
        for (int c : k.legal)
            if (suitOf(c) == trump)
                my_trumps.push_back(c);
        bool opp_may_trump = false;
        for (int u : k.unseen_suit(trump))
            for (int o : opps)
                if (k.p_holds(u, o) > 0)
                    opp_may_trump = true;
        bool we_declare = k.declarer == 0 || k.declarer == 2;
        int need = we_declare ? P.pull_min_trumps : P.pull_min_trumps_def;
        int nt = static_cast<int>(my_trumps.size());
        if (nt && opp_may_trump && k.is_master(my_trumps.back()) && nt >= need)
            return {my_trumps.back(), "pull trumps with master"};
        if (P.declarer_pull && k.declarer == 0 && opp_may_trump && nt >= P.declarer_pull_min)
            return {my_trumps.front(), "declarer pulls trumps low"};
        if (P.endgame_master_trump && nt && opp_may_trump && k.hand.size() <= 3
                && k.is_master(my_trumps.back()))
            return {my_trumps.back(), "endgame master trump"};
    }
    bool have = false;
    std::tuple<double, int, double> bestT;
    for (int c : k.legal) {
        if (trump >= 0 && suitOf(c) == trump)
            continue;
        if (!k.is_master(c))
            continue;
        double p_safe = 1.0;
        for (int o : opps)
            p_safe *= 1.0 - k.p_beats(c, o, suitOf(c));
        if (p_safe >= P.cash_threshold) {
            auto t = std::make_tuple(PTS[rankOf(c)] + 2 * p_safe, c, p_safe);
            if (!have || t > bestT) {
                bestT = t;
                have = true;
            }
        }
    }
    if (have)
        return {std::get<1>(bestT), "cash master p_safe=" + fmt("%.2f", std::get<2>(bestT))};
    if (k.haveW)
        return lead_sim();
    return lead_low();
}

std::pair<int, std::string> Agent::lead_low() const {
    const int trump = k.trump;
    const int opps[2] = {1, 3};
    const bool we_declare = k.declarer == 0 || k.declarer == 2;
    // suits in order of first appearance in legal (dict insertion order)
    std::vector<int> order;
    std::array<std::vector<int>, 4> by_suit;
    for (int c : k.legal) {
        int s = suitOf(c);
        if (by_suit[s].empty())
            order.push_back(s);
        by_suit[s].push_back(c);
    }
    int best = -1;
    double best_sc = 0.0;
    std::string why;
    for (int s : order) {
        std::vector<int> cards = by_suit[s];
        std::sort(cards.begin(), cards.end());
        int c = cards[0];
        int n_mine = static_cast<int>(cards.size());
        std::vector<int> unseen_s = k.unseen_suit(s);
        double sc = -P.lead_pts_w * PTS[rankOf(c)];
        std::vector<std::string> reasons;
        if (trump >= 0 && s == trump) {
            bool opp_trumps = false;
            for (int u : unseen_s)
                for (int o : opps)
                    if (k.p_holds(u, o) > 0)
                        opp_trumps = true;
            if (we_declare && n_mine >= P.low_pull_min && opp_trumps) {
                sc += P.low_pull_bonus;
                reasons.push_back("pull");
            } else if (k.declarer == 2 && opp_trumps && P.to_declarer_bonus != 0.0
                       && (!P.through_fix || n_mine >= 2)) {
                sc += P.to_declarer_bonus - P.trump_lead_pen * 0;
                reasons.push_back("trump-to-declarer");
            } else {
                sc -= P.trump_lead_pen;
            }
        } else {
            bool honour_above = false;
            for (size_t i = 1; i < cards.size(); ++i)
                if (PTS[rankOf(cards[i])] >= 3)
                    honour_above = true;
            if (n_mine >= 2 && honour_above) {
                sc += P.guard_lead_bonus;
                reasons.push_back("draw");
            }
            if (n_mine == 1 && PTS[rankOf(c)] >= 3)
                sc -= P.bare_honour_pen;
            int top = unseen_s.empty() ? -1 : unseen_s.back();
            if (top >= 0 && !k.is_master(cards.back()))
                sc += P.partner_top_w * k.p_holds(top, 2);
            if (trump >= 0) {
                double p_opp = 1.0 - (1 - k.p_ruff(s, 1)) * (1 - k.p_ruff(s, 3));
                double p_par = k.p_ruff(s, 2);
                sc -= P.opp_ruff_pen * p_opp;
                sc += P.partner_ruff_bonus * p_par * (1 - k.p_ruff(s, 3));
                if (p_opp > 0.3)
                    reasons.push_back("opp-ruff" + fmt("%.1f", p_opp));
                if (p_par > 0.3)
                    reasons.push_back("partner-ruff" + fmt("%.1f", p_par));
            }
            if (trump >= 0) {
                int my_trumps = 0;
                for (int h : k.hand)
                    my_trumps += suitOf(h) == trump;
                if (n_mine == 1 && my_trumps > 0 && PTS[rankOf(c)] <= 1) {
                    sc += P.singleton_bonus;
                    reasons.push_back("make-void");
                }
            }
            sc += P.length_w * unseen_s.size();
        }
        if (best < 0 || sc > best_sc) {
            best = c;
            best_sc = sc;
            std::string r;
            for (size_t i = 0; i < reasons.size(); ++i)
                r += (i ? "," : "") + reasons[i];
            why = "low lead " + r + " sc=" + fmt("%.2f", sc);
        }
    }
    return {best, why};
}

// rule_agent.py::belief_world_weights (dmc_belief_log_weights + per-seat
// rule likelihood or honest fast-policy soft action likelihood on suit deals),
// worlds by absolute seat.
std::vector<double> worldWeights(int trump, int dealer, int me, int64_t seen,
                                 const std::vector<rollout::World>& worlds,
                                 const rollout::History& history,
                                 const float* belief, const Params& P,
                                 const double* extra, const std::vector<int>& widx,
                                 const std::vector<int>* mults = nullptr,
                                 const fastpolicy::Net* net = nullptr) {
    const size_t n = worlds.size();
    std::vector<double> logw(n, 0.0);
    if (belief) {
        float lp[3][32];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 32; ++c)
                lp[r][c] = std::log(belief[r * 32 + c] + 1e-9f);
        for (size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (int r = 0; r < 3; ++r) {
                const int64_t h = worlds[i][(me + r + 1) % 4];
                double t = 0.0;
                for (int c = 0; c < 32; ++c)
                    if (!((seen >> c) & 1) && ((h >> c) & 1))
                        t += double(lp[r][c]);
                s += t;
            }
            logw[i] = P.lik_belief_w * s;
        }
    }
    if (!history.empty()) {
        const bool use_net_lik = (trump >= 0) && net && !net->empty()
            && (P.lik_net_partner_w > 0.0 || P.lik_net_opp_w > 0.0);
        if (use_net_lik) {
            auto fpl = fastpolicy::history_log_likelihood(
                *net, trump, dealer, me, worlds, history,
                P.lik_net_tau, P.lik_net_partner_w, P.lik_net_opp_w);
            for (size_t i = 0; i < n; ++i)
                logw[i] += fpl[i];
        } else {
            rollout::Options opt;
            opt.honest = 0;
            opt.agent_like = true;
            opt.nt_rules = 0;
            auto mm = rollout::world_mismatches(trump, dealer, me, worlds, history, opt);
            const int partner = (me + 2) % 4, o1 = (me + 1) % 4, o3 = (me + 3) % 4;
            const double lp = std::log(P.lik_partner_eps), lo = std::log(P.lik_opp_eps);
            for (size_t i = 0; i < n; ++i)
                logw[i] = logw[i] + mm[i][partner] * lp + (mm[i][o1] + mm[i][o3]) * lo;
        }
    }
    if (extra)
        for (size_t i = 0; i < n; ++i)
            logw[i] = logw[i] + extra[widx[i]];
    if (mults)
        for (size_t i = 0; i < n; ++i)
            if ((*mults)[i] > 1)
                logw[i] += std::log(double((*mults)[i]));
    double mx = logw.empty() ? 0.0 : *std::max_element(logw.begin(), logw.end());
    std::vector<double> w(n);
    for (size_t i = 0; i < n; ++i)
        w[i] = std::exp(logw[i] - mx);
    return w;
}

} // namespace

Decision choose(const ManilleGame& game, int me, int dealer,
                const std::vector<rollout::World>& worlds,
                const rollout::History& history,
                const float* belief, const float* q,
                const Params& P, const double* extra,
                const fastpolicy::Net* net) {
    Decision d;
    std::vector<int> legal;
    for (const Card& c : game.validMoves())
        legal.push_back(c.index());
    if (legal.size() == 1) {
        d.card = legal[0];
        return d;
    }
    Knowledge k = buildKnowledge(game, me, worlds, P.min_worlds);
    Agent agent(k, P);
    auto pick = k.trick.empty() ? agent.lead() : agent.follow();
    int choice = pick.first;
    d.why = pick.second;

    if (P.playout && k.haveW && k.legal.size() > 1
            && static_cast<int>(k.hand.size()) <= P.playout_max_cards) {
        const bool mull = k.trump < 0;
        const bool use_ab = mull && P.mull_endgame_ab_cards > 0
            && static_cast<int>(k.hand.size()) <= P.mull_endgame_ab_cards;
        const bool use_sir = P.playout_sir;
        const size_t raw_n = use_sir
            ? k.W.size()
            : std::min(k.W.size(), static_cast<size_t>(P.playout_worlds));

        std::vector<rollout::World> wabs;
        std::vector<int> widx;
        std::vector<int> mults;
        if (use_sir || use_ab) {
            // Deduplicate worlds preserving exact multiplicity and extra index
            std::map<rollout::World, size_t> wmap;
            wabs.reserve(raw_n);
            widx.reserve(raw_n);
            mults.reserve(raw_n);
            for (size_t i = 0; i < raw_n; ++i) {
                rollout::World abs_w{};
                for (int s = 0; s < 4; ++s)
                    abs_w[s] = k.W[i][((s - me) % 4 + 4) % 4];
                auto it = wmap.find(abs_w);
                if (it == wmap.end()) {
                    wmap.emplace(abs_w, wabs.size());
                    wabs.push_back(abs_w);
                    widx.push_back(k.Widx[i]);
                    mults.push_back(1);
                } else {
                    mults[it->second] += 1;
                }
            }
        } else {
            wabs.resize(raw_n);
            widx.resize(raw_n);
            for (size_t i = 0; i < raw_n; ++i) {
                for (int s = 0; s < 4; ++s)
                    wabs[i][s] = k.W[i][((s - me) % 4 + 4) % 4];
                widx[i] = k.Widx[i];
            }
        }

        int64_t seen = 0;
        for (int c : k.hand)
            seen |= bit(c);
        for (auto& sc : history)
            seen |= bit(sc.second);
        std::vector<double> wts = worldWeights(k.trump, dealer, me, seen, wabs, history,
                                               belief, P, extra, widx,
                                               mults.empty() ? nullptr : &mults, net);
        double tot = 0.0;
        for (double w : wts)
            tot += w;
        for (double& w : wts)
            w /= tot;

        double n_eff_override = 0.0;
        if (use_sir && wabs.size() > 1) {
            double ww = 0.0;
            for (size_t i = 0; i < wabs.size(); ++i) {
                const double w_single = wts[i] / double(mults[i]);
                ww += double(mults[i]) * w_single * w_single;
            }
            const int K = std::min<int>(P.playout_worlds, static_cast<int>(raw_n));
            n_eff_override = std::min(1.0 / ww, double(K));
            std::vector<int> counts(wabs.size(), 0);
            double cdf = 0.0;
            size_t idx = 0;
            for (int j = 0; j < K; ++j) {
                const double u = (j + 0.5) / double(K);
                while (idx + 1 < wabs.size() && cdf + wts[idx] < u) {
                    cdf += wts[idx];
                    ++idx;
                }
                counts[idx] += 1;
            }
            std::vector<rollout::World> nwabs;
            std::vector<double> nwts;
            nwabs.reserve(wabs.size());
            nwts.reserve(wabs.size());
            for (size_t i = 0; i < wabs.size(); ++i) {
                if (counts[i] > 0) {
                    nwabs.push_back(wabs[i]);
                    nwts.push_back(double(counts[i]) / double(K));
                }
            }
            wabs.swap(nwabs);
            wts.swap(nwts);
            mults.clear();
        }

        const size_t n = wabs.size();
        const bool has_net = net && !net->empty();
        const bool net_whole = has_net && ((P.playout_net_plies >= 32)
                                           || (P.playout_net_plies == 0 && P.playout_net_mix >= 1.0));
        const bool hybrid = has_net && P.playout_net_plies > 0 && !net_whole;

        auto run_playout = [&](const std::vector<int>& cands,
                               const std::vector<rollout::World>& ws_sub) {
            if (use_ab)
                return rollout::evaluate_endgame(game, me, cands, ws_sub);
            if (net_whole)
                return fastpolicy::evaluate_candidates(*net, k.trump, dealer, me, cands, ws_sub, history);
            rollout::Options opt;
            opt.honest = 0;
            opt.agent_like = true;
            opt.nt_rules = P.playout_nt_rules;
            opt.tr_rules = P.playout_tr_rules | (P.ruff_convention ? 32 : 0);
            rollout::NetChoose net_choose = [net](const ManilleGame& g, int seat) {
                return fastpolicy::choose(*net, g, seat);
            };
            auto res = rollout::evaluate_candidates(k.trump, dealer, me, cands, ws_sub, history, opt,
                                                    hybrid ? &net_choose : nullptr,
                                                    P.playout_net_plies, P.playout_net_trick_end);
            if (!hybrid && has_net && P.playout_net_mix > 0) {
                auto pn = fastpolicy::evaluate_candidates(*net, k.trump, dealer, me, cands,
                                                          ws_sub, history);
                const double nm = P.playout_net_mix;
                for (size_t ci = 0; ci < res.size(); ++ci)
                    for (size_t i = 0; i < res[ci].size(); ++i)
                        res[ci][i] = (1 - nm) * res[ci][i] + nm * pn[ci][i];
            }
            return res;
        };

        double base_qw = mull && P.mull_root_q_w >= 0 ? P.mull_root_q_w : P.root_q_w;
        if (!mull && P.suit_root_q_w_late >= 0 && !use_ab) {
            // Linearly taper from hand size 8 (trick 1) to hand size 2 (trick 7)
            const double t = std::clamp((8.0 - double(k.hand.size())) / 6.0, 0.0, 1.0);
            base_qw = base_qw + t * (P.suit_root_q_w_late - base_qw);
        }
        const double qw = (use_ab && P.mull_endgame_q_w >= 0) ? P.mull_endgame_q_w : base_qw;

        // Precompute lock / def_trump penalties so racing uses the exact same objective.
        const bool lock = P.nt_lock_cash_from_trick > 0 && k.trump < 0 && k.trick.empty()
            && d.why.rfind("plan: cash master", 0) == 0
            && 9 - static_cast<int>(k.hand.size()) >= P.nt_lock_cash_from_trick;
        int my_tr = 0, my_tr_pts = 0;
        for (int c : k.hand)
            if (suitOf(c) == k.trump) {
                ++my_tr;
                my_tr_pts += PTS[rankOf(c)];
            }
        const bool def_case = (P.def_trump_lock || P.def_trump_margin > 0) && k.trump >= 0
            && k.trick.empty() && k.declarer != 0 && k.declarer != 2 && suitOf(choice) != k.trump
            && (!P.def_trump_lock_weak || (my_tr <= 2 && my_tr_pts <= 3));
        const bool def_lock = def_case && P.def_trump_lock;
        const double def_pen = def_case && !def_lock ? P.def_trump_margin : 0.0;

        size_t ci_choice = 0;
        for (size_t ci = 0; ci < k.legal.size(); ++ci)
            if (k.legal[ci] == choice)
                ci_choice = ci;

        std::vector<std::vector<double>> per;
        std::vector<double> vals(k.legal.size(), 0.0);
        std::vector<bool> active_cand(k.legal.size(), true);

        const bool do_race = P.playout_race_k > 0 && !use_ab && k.legal.size() > 2
            && n > static_cast<size_t>(P.playout_race_k);

        if (do_race) {
            // Deterministic unbiased permutation of wabs so Stage 1 is a uniform subsample
            std::vector<size_t> perm(n);
            for (size_t i = 0; i < n; ++i) perm[i] = i;
            auto h64 = [](size_t x) -> uint64_t {
                uint64_t z = (static_cast<uint64_t>(x) + 1ULL) * 0x9e3779b97f4a7c15ULL;
                z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
                z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
                return z ^ (z >> 31);
            };
            std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b) { return h64(a) < h64(b); });
            std::vector<rollout::World> pwabs(n);
            std::vector<double> pwts(n);
            std::vector<int> pmults;
            if (!mults.empty()) pmults.resize(n);
            for (size_t i = 0; i < n; ++i) {
                pwabs[i] = wabs[perm[i]];
                pwts[i] = wts[perm[i]];
                if (!mults.empty()) pmults[i] = mults[perm[i]];
            }
            wabs.swap(pwabs);
            wts.swap(pwts);
            if (!mults.empty()) mults.swap(pmults);

            const size_t k0 = static_cast<size_t>(P.playout_race_k);
            std::vector<rollout::World> w1(wabs.begin(), wabs.begin() + k0);
            auto per1 = run_playout(k.legal, w1);
            if (qw > 0 && q) {
                for (size_t ci = 0; ci < k.legal.size(); ++ci) {
                    const double qv = 30.0 * double(q[k.legal[ci]]) + 30.0;
                    for (auto& v : per1[ci])
                        v = (1 - qw) * v + qw * qv;
                }
            }
            double sw1 = 0.0, sww1 = 0.0;
            for (size_t i = 0; i < k0; ++i) {
                sw1 += wts[i];
                sww1 += wts[i] * wts[i];
            }
            const double inv_sw1 = sw1 > 0.0 ? 1.0 / sw1 : 1.0;
            const double neff1 = sww1 > 0.0 ? (sw1 * sw1) / sww1 : double(k0);
            std::vector<double> v1(k.legal.size(), 0.0);
            for (size_t ci = 0; ci < k.legal.size(); ++ci) {
                double s = 0.0;
                for (size_t i = 0; i < k0; ++i)
                    s += wts[i] * per1[ci][i];
                v1[ci] = s * inv_sw1;
            }
            auto adj1 = [&](size_t ci) {
                return v1[ci] - (suitOf(k.legal[ci]) == k.trump ? def_pen : 0.0);
            };
            size_t best1 = ci_choice;
            bool have_b1 = false;
            for (size_t ci = 0; ci < k.legal.size(); ++ci) {
                if (lock && k.legal[ci] != choice && !k.is_master(k.legal[ci])) continue;
                if (def_lock && suitOf(k.legal[ci]) == k.trump) continue;
                if (!have_b1 || adj1(ci) > adj1(best1)) {
                    best1 = ci;
                    have_b1 = true;
                }
            }
            std::vector<size_t> surv_idx;
            std::vector<int> surv_cards;
            for (size_t ci = 0; ci < k.legal.size(); ++ci) {
                bool keep = (ci == ci_choice || ci == best1);
                if (!keep) {
                    const bool elig = !(lock && k.legal[ci] != choice && !k.is_master(k.legal[ci]))
                                   && !(def_lock && suitOf(k.legal[ci]) == k.trump);
                    if (elig) {
                        const double gap = adj1(best1) - adj1(ci);
                        double mean_d = 0.0;
                        for (size_t i = 0; i < k0; ++i)
                            mean_d += (wts[i] * inv_sw1) * (per1[best1][i] - per1[ci][i]);
                        double var_d = 0.0;
                        for (size_t i = 0; i < k0; ++i) {
                            const double e = (per1[best1][i] - per1[ci][i]) - mean_d;
                            var_d += (wts[i] * inv_sw1) * e * e;
                        }
                        const double se_d = std::sqrt(std::max(var_d, 1e-12) / neff1);
                        if (gap <= P.playout_race_z * se_d)
                            keep = true;
                    }
                }
                if (keep) {
                    surv_idx.push_back(ci);
                    surv_cards.push_back(k.legal[ci]);
                } else {
                    active_cand[ci] = false;
                    vals[ci] = v1[ci];
                }
            }
            std::vector<rollout::World> w2(wabs.begin() + k0, wabs.end());
            auto per2 = run_playout(surv_cards, w2);
            per.resize(k.legal.size());
            for (size_t si = 0; si < surv_idx.size(); ++si) {
                const size_t ci = surv_idx[si];
                if (qw > 0 && q) {
                    const double qv = 30.0 * double(q[k.legal[ci]]) + 30.0;
                    for (auto& v : per2[si])
                        v = (1 - qw) * v + qw * qv;
                }
                per[ci].resize(n);
                std::copy(per1[ci].begin(), per1[ci].end(), per[ci].begin());
                std::copy(per2[si].begin(), per2[si].end(), per[ci].begin() + k0);
                double s = 0.0;
                for (size_t i = 0; i < n; ++i)
                    s += wts[i] * per[ci][i];
                vals[ci] = s;
            }
        } else {
            per = run_playout(k.legal, wabs);
            if (qw > 0 && q) {
                for (size_t ci = 0; ci < k.legal.size(); ++ci) {
                    const double qv = 30.0 * double(q[k.legal[ci]]) + 30.0;
                    for (auto& v : per[ci])
                        v = (1 - qw) * v + qw * qv;
                }
            }
            for (size_t ci = 0; ci < k.legal.size(); ++ci) {
                double s = 0.0;
                for (size_t i = 0; i < n; ++i)
                    s += wts[i] * per[ci][i];
                vals[ci] = s;
            }
        }
        d.value_cards = k.legal;
        d.values = vals;
        auto adj = [&](size_t ci) {
            return vals[ci] - (suitOf(k.legal[ci]) == k.trump ? def_pen : 0.0);
        };
        size_t bi = 0;
        bool have_bi = false;
        for (size_t ci = 0; ci < vals.size(); ++ci) {
            if (!active_cand[ci])
                continue;
            if (lock && k.legal[ci] != choice && !k.is_master(k.legal[ci]))
                continue;
            if (def_lock && suitOf(k.legal[ci]) == k.trump)
                continue;
            if (!have_bi || adj(ci) > adj(bi)) {
                bi = ci;
                have_bi = true;
            }
        }
        const bool used_net_playout = has_net && (P.playout_net_plies > 0 || P.playout_net_mix > 0);
        double z_eff = P.playout_z;
        if (use_ab && P.mull_endgame_z >= 0)
            z_eff = P.mull_endgame_z;
        else if (!mull && used_net_playout && P.suit_net_z >= 0)
            z_eff = P.suit_net_z;
        if (k.legal[bi] != choice) {
            double m = 0.0, ww = 0.0;
            for (size_t i = 0; i < n; ++i) {
                m += wts[i] * (per[bi][i] - per[ci_choice][i]);
                if (!mults.empty()) {
                    const double w_single = wts[i] / double(mults[i]);
                    ww += double(mults[i]) * w_single * w_single;
                } else {
                    ww += wts[i] * wts[i];
                }
            }
            const double n_eff = n_eff_override > 0.0 ? n_eff_override : (1.0 / ww);
            double var = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const double e = (per[bi][i] - per[ci_choice][i]) - m;
                var += wts[i] * e * e;
            }
            const double se = std::sqrt(std::max(var, 1e-12) / n_eff);
            const double need = P.playout_margin + (suitOf(k.legal[bi]) == k.trump ? def_pen : 0.0);
            if (m > need && (z_eff <= 0 || m > z_eff * se)) {
                d.why += "; play-out prefers " + cstr(k.legal[bi]) + " " + fmt("%.1f", vals[bi])
                    + " vs " + fmt("%.1f", vals[ci_choice]) + " (+" + fmt("%.2f", m)
                    + "\xc2\xb1" + fmt("%.2f", se) + ")";
                choice = k.legal[bi];
            }
        }
    }
    if (P.last_ruff_low && k.trick.size() == 3 && k.trump >= 0
            && suitOf(choice) == k.trump && k.is_master(choice)) {
        // last to play, about to spend the highest trump left in the game: a
        // lower trump that also takes the trick saves it for later
        int low = -1;
        for (int c : k.legal) {
            const float* a = k.actOf(c);
            if (suitOf(c) == k.trump && rankOf(c) < rankOf(choice) && a[encoding::ACT_WINS_OFF] > 0
                    && (low < 0 || rankOf(c) < rankOf(low)))
                low = c;
        }
        if (low >= 0) {
            d.why += "; " + cstr(low) + " wins too, master trump kept";
            choice = low;
        }
    }
    d.card = choice;
    return d;
}

} // namespace rule
