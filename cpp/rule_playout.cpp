#include "rule_playout.h"

#include <algorithm>
#include <tuple>
#include "bitops.h"

namespace rollout {
namespace {

const int PTS[8] = {0, 0, 0, 1, 2, 3, 4, 5};

inline int64_t suitMask(int s) { return int64_t(0xFF) << (8 * s); }
inline int popcount(int64_t m) {
    return manille::popcount64(static_cast<unsigned long long>(m));
}
inline int suitOf(int c) { return c >> 3; }
inline int rankOf(int c) { return c & 7; }

// Mirrors learn/rule_playout.py::Playout. Every iteration runs in ascending
// card order and keeps the FIRST best element, like Python's min/max, so the
// two implementations make identical choices.
class Policy {
public:
    Policy(int trump, int declarer, const Options& opt)
        : m_trump(trump), m_declarer(declarer), m_opt(opt),
          m_nt(trump < 0 ? opt.nt_rules : 0), m_tr(trump >= 0 ? opt.tr_rules : 0) {}

    std::array<int, 4> voids{0, 0, 0, 0};  // public voids, suit bitmask

    // the deal's opening lead, and whether its leader played that suit again
    int first_seat = -1, first_card = -1;
    bool first_followed = false;

    void note(int seat, int card, const std::vector<std::pair<int, int>>& trick) {
        if (!trick.empty() && suitOf(card) != suitOf(trick[0].second))
            voids[seat] |= 1 << suitOf(trick[0].second);
        if (first_seat < 0) {
            first_seat = seat;
            first_card = card;
        } else if (seat == first_seat && suitOf(card) == suitOf(first_card)) {
            first_followed = true;
        }
    }

    int choose(int seat, const std::vector<int>& legal, const World& hands,
               const std::vector<std::pair<int, int>>& trick) const {
        if (legal.size() == 1)
            return legal[0];
        if (trick.empty())
            return lead(seat, legal, hands);
        return follow(seat, legal, hands, trick);
    }

private:
    int m_trump;
    int m_declarer;
    Options m_opt;
    int m_nt;  // no-trump rule bits (Options::nt_rules), 0 with a trump suit
    int m_tr;  // trump rule bits (Options::tr_rules), 0 at no trump

    bool beats(int a, int b) const {
        return Card::from_index(a).beats(Card::from_index(b), m_trump);
    }
    bool isVoid(int seat, int suit) const { return (voids[seat] >> suit) & 1; }
    bool isTrump(int c) const { return m_trump >= 0 && suitOf(c) == m_trump; }

    static int64_t remainingOf(const World& h) { return h[0] | h[1] | h[2] | h[3]; }

    static bool isMaster(int c, int64_t hand, int64_t remaining) {
        // no card I cannot see outranks c in its suit
        int64_t above = ~((int64_t(1) << (c + 1)) - 1);
        int64_t higher = remaining & suitMask(suitOf(c)) & above;
        return (higher & ~hand) == 0;
    }

    // No trump: a master is not cashed while the suit is still long outside
    // (5+ cards, or 4+ when I hold at most two). The others just follow
    // small under it; kept, it captures one of their honours later.
    static bool holdMaster(int s, int64_t hand, int64_t outside) {
        const int n_out = popcount(outside & suitMask(s));
        return n_out >= 5 || (n_out >= 4 && popcount(hand & suitMask(s)) <= 2);
    }

    // No opponent can beat c in its suit: clairvoyant seats look at the
    // opponents' real hands, honest ones need their own master or both
    // opponents publicly void.
    bool isBoss(int c, int seat, const World& hands) const {
        const int s = suitOf(c);
        const int o1 = (seat + 1) % 4, o2 = (seat + 3) % 4;
        if (m_opt.honest)
            return isMaster(c, hands[seat], remainingOf(hands))
                || (isVoid(o1, s) && isVoid(o2, s));
        int64_t above = ~((int64_t(1) << (c + 1)) - 1);
        return ((hands[o1] | hands[o2]) & suitMask(s) & above) == 0;
    }

    // For a discard (legal = the whole hand): free cards and bosses. Per
    // suit, the top honour below the bosses is protected with its guards
    // when I hold at least as many smaller cards as there are higher cards
    // out against it; free cards are the rest. Mirrors _discard_classes.
    void discardClasses(int seat, const std::vector<int>& legal, const World& hands,
                        std::vector<int>& free, std::vector<int>& boss) const {
        const int64_t hand = hands[seat];
        const int64_t against = m_opt.honest
            ? (remainingOf(hands) & ~hand)
            : (hands[(seat + 1) % 4] | hands[(seat + 3) % 4]);
        int64_t keep = 0;
        for (int c : legal)
            if (isBoss(c, seat, hands)) {
                boss.push_back(c);
                keep |= int64_t(1) << c;
            }
        for (int s = 0; s < 4; ++s) {
            int h = -1;
            for (int c : legal)
                if (suitOf(c) == s && !((keep >> c) & 1) && PTS[rankOf(c)] > 0)
                    h = c;  // legal is ascending: the last one is the top
            if (h < 0)
                continue;
            const int n_hi = popcount(against & suitMask(s) & ~((int64_t(1) << (h + 1)) - 1));
            std::vector<int> low;
            for (int c : legal)
                if (suitOf(c) == s && !((keep >> c) & 1) && c < h)
                    low.push_back(c);
            if (n_hi >= 1 && n_hi <= static_cast<int>(low.size())) {
                keep |= int64_t(1) << h;
                for (int i = 0; i < n_hi; ++i)
                    keep |= int64_t(1) << low[i];
            }
        }
        for (int c : legal)
            if (!((keep >> c) & 1))
                free.push_back(c);
    }

    // Lowest card of the suit holding the most bosses (first suit on ties),
    // -1 when no suit has min_len of them.
    static int lowestOfLongest(const std::vector<int>& boss, int min_len) {
        int n[4] = {0, 0, 0, 0};
        for (int c : boss)
            ++n[suitOf(c)];
        int s = 0;
        for (int x = 1; x < 4; ++x)
            if (n[x] > n[s])
                s = x;
        if (n[s] < min_len)
            return -1;
        int lo = -1;
        for (int c : boss)
            if (suitOf(c) == s && (lo < 0 || c < lo))
                lo = c;
        return lo;
    }

    static int cheapest(const std::vector<int>& cards) {
        int best = -1;
        std::pair<int, int> bk;
        for (int c : cards) {
            auto k = std::make_pair(PTS[rankOf(c)], rankOf(c));
            if (best < 0 || k < bk) {
                best = c;
                bk = k;
            }
        }
        return best;
    }

    std::pair<int, int> winner(const std::vector<std::pair<int, int>>& trick) const {
        int ws = trick[0].first, wc = trick[0].second;
        for (size_t i = 1; i < trick.size(); ++i)
            if (beats(trick[i].second, wc)) {
                ws = trick[i].first;
                wc = trick[i].second;
            }
        return {ws, wc};
    }

    bool canBeat(int64_t hand, int x, int led) const {
        int64_t follow = hand & suitMask(led);
        int64_t pool = follow ? follow : hand;
        for (int y = 0; y < 32; ++y)
            if ((pool >> y) & 1 && beats(y, x))
                return true;
        return false;
    }

    // ---- leads -------------------------------------------------------
    int leadLowLongest(const std::vector<int>& pool, int64_t hand) const {
        int best = -1;
        std::tuple<int, int, int> bk;
        for (int c : pool) {
            auto k = std::make_tuple(-popcount(hand & suitMask(suitOf(c))),
                                     PTS[rankOf(c)], rankOf(c));
            if (best < 0 || k < bk) {
                best = c;
                bk = k;
            }
        }
        return best;
    }

    int lead(int seat, const std::vector<int>& legal, const World& hands) const {
        if (m_opt.agent_like)
            return leadAgent(seat, legal, hands);
        const int64_t remaining = remainingOf(hands);
        int best = -1;
        for (int c : legal) {
            if (isTrump(c))
                continue;
            int64_t above = ~((int64_t(1) << (c + 1)) - 1);
            int64_t higher = remaining & suitMask(suitOf(c)) & above;
            if (higher & ~hands[seat])
                continue;
            if (best < 0 || PTS[rankOf(c)] > PTS[rankOf(best)])
                best = c;
        }
        if (best >= 0)
            return best;
        std::vector<int> my_tr;
        for (int c : legal)
            if (isTrump(c))
                my_tr.push_back(c);
        if (!my_tr.empty() && m_declarer % 2 == seat % 2) {
            int64_t opp_tr;
            if (m_opt.honest)
                opp_tr = remaining & ~hands[seat] & suitMask(m_trump);
            else
                opp_tr = (hands[(seat + 1) % 4] | hands[(seat + 3) % 4]) & suitMask(m_trump);
            if (opp_tr) {
                int top = my_tr.back();
                bool top_safe = true;
                for (int o = 0; o < 32; ++o)
                    if ((opp_tr >> o) & 1 && beats(o, top)) {
                        top_safe = false;
                        break;
                    }
                return top_safe ? top : my_tr.front();
            }
        }
        std::vector<int> side;
        for (int c : legal)
            if (!isTrump(c))
                side.push_back(c);
        return leadLowLongest(side.empty() ? legal : side, hands[seat]);
    }

    int leadAgent(int seat, const std::vector<int>& legal, const World& hands) const {
        const int64_t hand = hands[seat];
        const int64_t remaining = remainingOf(hands);
        const int64_t outside = remaining & ~hand;
        const int opps[2] = {(seat + 1) % 4, (seat + 3) % 4};
        const int partner = (seat + 2) % 4;
        std::vector<int> my_tr, side;
        for (int c : legal) {
            if (isTrump(c))
                my_tr.push_back(c);
            else
                side.push_back(c);
        }
        const int out_tr = m_trump >= 0 ? popcount(outside & suitMask(m_trump)) : 0;
        const bool opp_may_trump = out_tr > 0
            && !(isVoid(opps[0], m_trump) && isVoid(opps[1], m_trump));
        const int n_cards = popcount(hand);

        // 0. ruff convention: partner opened the deal with a side suit and has
        // not played it since -- return it while partner can still ruff
        if ((m_tr & 32) && first_seat == partner && !first_followed
                && suitOf(first_card) != m_trump) {
            const int s = suitOf(first_card);
            const bool can_ruff = m_opt.honest
                ? !isVoid(partner, m_trump)
                : ((hands[partner] & suitMask(s)) == 0 && (hands[partner] & suitMask(m_trump)) != 0);
            int low = -1;
            for (int c : side)
                if (suitOf(c) == s && (low < 0 || c < low))
                    low = c;
            if (can_ruff && low >= 0)
                return low;
        }
        // 1. endgame master trump
        if (!my_tr.empty() && opp_may_trump && n_cards <= 3
                && isMaster(my_tr.back(), hand, remaining))
            return my_tr.back();
        // 2. draw trumps with a likely majority and side winners to protect
        if (!my_tr.empty() && opp_may_trump) {
            double partner_share = isVoid(partner, m_trump) ? 0.0 : out_tr / 3.0;
            bool majority = my_tr.size() + partner_share > out_tr - partner_share;
            bool winners = false;
            for (int c : side)
                if (isMaster(c, hand, remaining)) {
                    winners = true;
                    break;
                }
            if (!winners)
                for (int s = 0; s < 4; ++s)
                    if (s != m_trump && popcount(hand & suitMask(s)) >= 3) {
                        winners = true;
                        break;
                    }
            bool we_declare = m_declarer % 2 == seat % 2;
            if (majority && winners && (we_declare || my_tr.size() >= 3)) {
                if (isMaster(my_tr.back(), hand, remaining))
                    return my_tr.back();
                if (m_declarer == seat && my_tr.size() >= 3)
                    return my_tr.front();
            }
            // 3. trump through to the declaring partner
            if (m_declarer == partner && n_cards > 2
                    && (my_tr.size() >= 2 || !(m_tr & 1))
                    && !isMaster(my_tr.front(), hand, remaining)
                    && PTS[rankOf(my_tr.front())] <= 2)
                return my_tr.front();
        }
        // 4. cash a side master unless a known-void opponent can ruff it
        int best = -1;
        for (int c : side) {
            int s = suitOf(c);
            // bit 128 (clairvoyant): a team boss -- partner may hold the
            // master, the opponents cannot beat it -- is cashed too
            if (!isMaster(c, hand, remaining)
                    && !((m_nt & 4) && isVoid(opps[0], s) && isVoid(opps[1], s))
                    && !((m_nt & 128) && !m_opt.honest && isBoss(c, seat, hands)))
                continue;
            if ((m_nt & 32) && holdMaster(s, hand, outside))
                continue;
            if (out_tr) {
                bool ruffable = false;
                for (int o : opps)
                    if (isVoid(o, s) && !isVoid(o, m_trump))
                        ruffable = true;
                if (ruffable)
                    continue;
            }
            if (best < 0 || PTS[rankOf(c)] > PTS[rankOf(best)])
                best = c;
        }
        if (best >= 0)
            return best;
        // 4b. no trump: back to the suit partner opened the deal with
        if ((m_nt & 64) && first_seat == partner) {
            const int s = suitOf(first_card);
            int low = -1;
            for (int c : side)
                if (suitOf(c) == s && (low < 0 || c < low))
                    low = c;
            if (low >= 0)
                return low;
        }
        // 4c. no trump, clairvoyant: nothing to cash -- cross to partner's
        // bosses (the suit where partner holds the most), cheapest card,
        // so partner gets in to cash them
        if ((m_nt & 256) && !m_opt.honest) {
            int cross = -1;
            std::tuple<int, int, int> ck;
            for (int c : side) {
                int n_boss = 0;
                for (int pc = suitOf(c) * 8; pc < suitOf(c) * 8 + 8; ++pc)
                    if ((hands[partner] >> pc) & 1 && isBoss(pc, partner, hands))
                        ++n_boss;
                if (n_boss == 0)
                    continue;
                auto k = std::make_tuple(-n_boss, PTS[rankOf(c)], rankOf(c));
                if (cross < 0 || k < ck) {
                    cross = c;
                    ck = k;
                }
            }
            if (cross >= 0)
                return cross;
        }
        // 5. establish: guarded honour, nobody known void to ruff
        bool have_est = false;
        int est_score = 0, est_lead = -1;
        for (int s = 0; s < 4; ++s) {
            if (s == m_trump)
                continue;
            std::vector<int> mine;
            for (int c : side)
                if (suitOf(c) == s)
                    mine.push_back(c);
            if (mine.size() < 2)
                continue;
            if (out_tr && (isVoid(opps[0], s) || isVoid(opps[1], s)))
                continue;
            int h = -1;
            for (int c : mine)
                if (PTS[rankOf(c)] >= 2 && !isMaster(c, hand, remaining))
                    h = c;  // highest such honour (mine is ascending)
            if (h < 0)
                continue;
            int64_t above = ~((int64_t(1) << (h + 1)) - 1);
            int n_higher = popcount(outside & suitMask(s) & above);
            if (static_cast<int>(mine.size()) - 1 < n_higher)
                continue;
            bool below_mine = false;
            for (int c : mine)
                if (c == h - 1)
                    below_mine = true;
            int lead_card = (below_mine && rankOf(h) > 0) ? h : mine.front();
            if ((m_nt & 512) && lead_card == h) {
                // the lowest card of the touching run forces the higher
                // cards out just as well under must-beat, and costs less
                // when it is captured (Q from A-K-Q)
                while (rankOf(lead_card) > 0
                        && std::find(mine.begin(), mine.end(), lead_card - 1) != mine.end())
                    --lead_card;
            }
            int score = PTS[rankOf(h)] + static_cast<int>(mine.size()) - 2 * n_higher;
            if (!have_est || score > est_score) {
                have_est = true;
                est_score = score;
                est_lead = lead_card;
            }
        }
        if (have_est)
            return est_lead;
        // 6. low from the longest side suit
        if ((m_tr & 4) && !my_tr.empty()) {
            // a low non-master side singleton first: the void makes a ruff
            int best_s = -1;
            for (int c : side)
                if (popcount(hand & suitMask(suitOf(c))) == 1 && PTS[rankOf(c)] <= 1
                        && !isMaster(c, hand, remaining)
                        && (best_s < 0 || std::make_pair(PTS[rankOf(c)], rankOf(c))
                                          < std::make_pair(PTS[rankOf(best_s)], rankOf(best_s))))
                    best_s = c;
            if (best_s >= 0)
                return best_s;
        }
        return leadLowLongest(side.empty() ? legal : side, hand);
    }

    // ---- following ---------------------------------------------------
    int follow(int seat, const std::vector<int>& legal, const World& hands,
               const std::vector<std::pair<int, int>>& trick) const {
        const int led = suitOf(trick[0].second);
        auto w = winner(trick);
        const int ws = w.first, wc = w.second;
        const int n = static_cast<int>(trick.size());
        std::vector<int> opp_behind;
        for (int i = n + 1; i < 4; ++i) {
            int o = (trick[0].first + i) % 4;
            if (o % 2 != seat % 2)
                opp_behind.push_back(o);
        }

        int64_t outside = remainingOf(hands) & ~hands[seat];
        for (auto& sc : trick)
            outside &= ~(int64_t(1) << sc.second);

        auto safe = [&](int x) -> bool {
            if (m_opt.honest == 2) {
                const bool led_short = popcount(outside & suitMask(led)) <= m_opt.ruff_short;
                const bool out_tr = m_trump >= 0 && (outside & suitMask(m_trump)) != 0;
                auto can_ruff = [&](int o) {
                    return out_tr && led != m_trump && !isVoid(o, m_trump)
                        && (isVoid(o, led) || led_short);
                };
                const int xs = suitOf(x);
                for (int o : opp_behind) {
                    if (xs == led) {
                        if (!isVoid(o, xs)) {
                            int64_t pool = outside & suitMask(xs);
                            for (int y = 0; y < 32; ++y)
                                if ((pool >> y) & 1 && beats(y, x))
                                    return false;
                        }
                        if (xs != m_trump && can_ruff(o))
                            return false;
                    } else if (m_trump >= 0 && xs == m_trump && can_ruff(o)) {
                        int64_t pool = outside & suitMask(m_trump);
                        for (int y = 0; y < 32; ++y)
                            if ((pool >> y) & 1 && beats(y, x))
                                return false;
                    }
                }
                return true;
            }
            if (m_opt.honest) {
                if (opp_behind.empty())
                    return true;
                const bool ruff_threat = m_trump >= 0 && led != m_trump
                    && popcount(outside & suitMask(led)) <= m_opt.ruff_short
                    && (outside & suitMask(m_trump)) != 0;
                if (ruff_threat && suitOf(x) != m_trump)
                    return false;
                for (int y = 0; y < 32; ++y)
                    if ((outside >> y) & 1 && beats(y, x) && suitOf(y) == suitOf(x))
                        return false;
                return true;
            }
            for (int o : opp_behind)
                if (canBeat(hands[o], x, led))
                    return false;
            return true;
        };

        if (ws % 2 == seat % 2 && safe(wc)) {
            if ((m_nt & 2) && suitOf(legal[0]) == led) {
                const int o1 = (seat + 1) % 4, o2 = (seat + 3) % 4;
                bool ours;
                int p_n;
                if (m_opt.honest) {
                    ours = isVoid(o1, led) && isVoid(o2, led);
                    p_n = popcount(remainingOf(hands) & ~hands[seat] & suitMask(led));
                } else {
                    ours = ((hands[o1] | hands[o2]) & suitMask(led)) == 0;
                    p_n = popcount(hands[(seat + 2) % 4] & suitMask(led));
                }
                if (ours) {
                    int lo = legal[0], hi = legal[0];
                    for (int c : legal) {
                        lo = std::min(lo, c);
                        hi = std::max(hi, c);
                    }
                    if (p_n == 0) {
                        // partner led his last card of our suit: take the
                        // lead over (cheaply) if I have more of it to cash
                        int over = -1;
                        for (int c : legal)
                            if (beats(c, wc) && (over < 0 || c < over))
                                over = c;
                        return (over >= 0 && legal.size() >= 2) ? over : lo;
                    }
                    return static_cast<int>(legal.size()) <= p_n ? hi : lo;
                }
            }
            std::vector<int> side;
            for (int c : legal)
                if (!isTrump(c))
                    side.push_back(c);
            if (side.empty())
                side = legal;
            const bool vet_master = (m_tr & 2) && led != m_trump
                && suitOf(legal[0]) == led && !opp_behind.empty();
            if (m_opt.agent_like && !vet_master) {
                const int64_t rem = remainingOf(hands);
                std::vector<int> keepers;
                const bool keep_bosses = (m_nt & 8) && suitOf(legal[0]) != led;
                for (int c : side)
                    if (keep_bosses ? !isBoss(c, seat, hands)
                                    : !isMaster(c, hands[seat], rem))
                        keepers.push_back(c);
                if (!keepers.empty())
                    side = keepers;
            }
            int best = -1;
            std::pair<int, int> bk;
            for (int c : side) {
                auto k = std::make_pair(PTS[rankOf(c)], -rankOf(c));
                if (best < 0 || k > bk) {
                    best = c;
                    bk = k;
                }
            }
            return best;
        }
        int best_secure = -1, best_win = -1;
        std::pair<int, int> ks, kw;
        for (int c : legal) {
            if (!beats(c, wc))
                continue;
            auto k = std::make_pair(isTrump(c) ? 1 : 0, rankOf(c));
            if (best_win < 0 || k < kw) {
                best_win = c;
                kw = k;
            }
            if (safe(c) && (best_secure < 0 || k < ks)) {
                best_secure = c;
                ks = k;
            }
        }
        if (best_secure >= 0) {
            if ((m_nt & 1) || (m_tr & 16)) {
                // highest honour among the sure winners that winning
                // cheaply would leave to be captured: fewer small cards
                // beside it than higher cards still out against it (in its
                // own suit: the led suit, or trump on a ruff)
                int h = -1;
                for (int c : legal)
                    if (beats(c, wc) && safe(c) && PTS[rankOf(c)] > 0
                            && !isBoss(c, seat, hands) && c > h)
                        h = c;
                if (h >= 0) {
                    const int hs = suitOf(h);
                    const int64_t higher = suitMask(hs) & ~((int64_t(1) << (h + 1)) - 1);
                    const int64_t against = m_opt.honest
                        ? (remainingOf(hands) & ~hands[seat])
                        : (hands[(seat + 1) % 4] | hands[(seat + 3) % 4]);
                    const int n_hi = popcount(against & higher);
                    const int guards = popcount(hands[seat] & suitMask(hs)) - 2;
                    if (guards < n_hi)
                        return h;
                }
            }
            return best_secure;
        }
        if (ws % 2 != seat % 2 && best_win >= 0)
            return best_win;
        const int p_seat = (seat + 2) % 4;
        bool p_behind = false;
        for (int i = n + 1; i < 4; ++i)
            if ((trick[0].first + i) % 4 == p_seat)
                p_behind = true;
        if ((m_tr & 8) && !m_opt.honest && suitOf(legal[0]) != led && p_behind
                && canBeat(hands[p_seat], wc, led)) {
            // partner takes this trick: the biggest non-master side card on it
            const int64_t rem = remainingOf(hands);
            int best_v = -1;
            std::pair<int, int> bv;
            for (int c : legal) {
                if (isTrump(c) || isMaster(c, hands[seat], rem))
                    continue;
                auto k = std::make_pair(PTS[rankOf(c)], -rankOf(c));
                if (best_v < 0 || k > bv) {
                    best_v = c;
                    bv = k;
                }
            }
            if (best_v >= 0)
                return best_v;
        }
        if ((m_nt & 16) && suitOf(legal[0]) != led) {
            std::vector<int> free, boss, losers;
            discardClasses(seat, legal, hands, free, boss);
            for (int c : free)
                if (PTS[rankOf(c)] <= 2)
                    losers.push_back(c);
            if (!losers.empty())
                return cheapest(losers);
            const int low = lowestOfLongest(boss, 3);
            if (low >= 0 && PTS[rankOf(low)] <= 2)
                return low;
            if (!free.empty())
                return cheapest(free);
            return cheapest(legal);
        }
        int best = -1;
        std::tuple<int, int, int> bk;
        if ((m_nt & 8) && suitOf(legal[0]) != led) {
            for (int c : legal) {
                if (PTS[rankOf(c)] > 2 || isBoss(c, seat, hands))
                    continue;
                auto k = std::make_tuple(PTS[rankOf(c)], isTrump(c) ? 1 : 0, rankOf(c));
                if (best < 0 || k < bk) {
                    best = c;
                    bk = k;
                }
            }
            if (best >= 0)
                return best;
        }
        for (int c : legal) {
            auto k = std::make_tuple(PTS[rankOf(c)], isTrump(c) ? 1 : 0, rankOf(c));
            if (best < 0 || k < bk) {
                best = c;
                bk = k;
            }
        }
        return best;
    }
};

ManilleGame freshGame(const World& start, int trump, int declarer) {
    return ManilleGame(start, std::array<int32_t, 4>{}, std::array<int32_t, 4>{}, 0,
                       (declarer + 1) % 4, trump, 0, 0, 0, 0,
                       std::array<int32_t, 32>{}, std::array<int32_t, 32>{}, 0,
                       false, declarer);
}

std::vector<int> legalOf(const ManilleGame& g) {
    std::vector<int> out;
    for (const Card& c : g.validMoves())
        out.push_back(c.index());
    return out;
}

World handsOf(const ManilleGame& g) {
    return {g.handMask(0), g.handMask(1), g.handMask(2), g.handMask(3)};
}

struct Replayed {
    ManilleGame game;
    Policy policy;
    std::vector<std::pair<int, int>> trick;
};

void play(Replayed& r, int seat, int card) {
    r.policy.note(seat, card, r.trick);
    r.game.doMove(Card::from_index(card));
    r.trick.emplace_back(seat, card);
    if (r.trick.size() == 4)
        r.trick.clear();
}

World startHands(const World& w, const History& history) {
    World start = w;
    for (auto& sc : history)
        start[sc.first] |= int64_t(1) << sc.second;
    return start;
}

} // namespace

std::vector<std::vector<double>> evaluate_candidates(
    int trump, int declarer, int me,
    const std::vector<int>& candidates,
    const std::vector<World>& worlds,
    const History& history,
    const Options& opt,
    const NetChoose* net, int net_plies, bool net_to_trick_end) {
    std::vector<std::vector<double>> out(candidates.size(),
                                         std::vector<double>(worlds.size(), 0.0));
    for (size_t wi = 0; wi < worlds.size(); ++wi) {
        Replayed base{freshGame(startHands(worlds[wi], history), trump, declarer),
                      Policy(trump, declarer, opt), {}};
        base.game.startPlaying();
        for (auto& sc : history)
            play(base, sc.first, sc.second);
        for (size_t ci = 0; ci < candidates.size(); ++ci) {
            Replayed r = base;
            play(r, me, candidates[ci]);
            int ply = 0;
            // once the net hands over it stays off: the trick test would
            // otherwise switch it back on at the next trick's second card
            bool net_on = net && net_plies > 0;
            while (!r.game.finished()) {
                int seat = static_cast<int>(r.game.currentPlayer());
                net_on = net_on && (ply < net_plies || (net_to_trick_end && !r.trick.empty()));
                int c;
                if (net_on) {
                    const int64_t hm = r.game.handMask(seat);
                    if ((hm & (hm - 1)) == 0)
                        c = manille::ctz64(static_cast<unsigned long long>(hm));
                    else
                        c = (*net)(r.game, seat);
                } else {
                    std::vector<int> legal = legalOf(r.game);
                    c = r.policy.choose(seat, legal, handsOf(r.game), r.trick);
                }
                play(r, seat, c);
                ++ply;
            }
            out[ci][wi] = r.game.teamPoints(me % 2);
        }
    }
    return out;
}

std::vector<double> world_weights(
    int trump, int declarer, int me,
    const std::vector<World>& worlds,
    const History& history,
    double eps,
    const Options& opt) {
    std::vector<double> out(worlds.size(), 1.0);
    for (size_t wi = 0; wi < worlds.size(); ++wi) {
        Replayed r{freshGame(startHands(worlds[wi], history), trump, declarer),
                   Policy(trump, declarer, opt), {}};
        r.game.startPlaying();
        double w = 1.0;
        for (auto& sc : history) {
            if (sc.first != me) {
                std::vector<int> legal = legalOf(r.game);
                if (legal.size() > 1
                        && r.policy.choose(sc.first, legal, handsOf(r.game), r.trick) != sc.second)
                    w *= eps;
            }
            play(r, sc.first, sc.second);
        }
        out[wi] = w;
    }
    return out;
}

std::vector<std::array<int, 4>> world_mismatches(
    int trump, int declarer, int me,
    const std::vector<World>& worlds,
    const History& history,
    const Options& opt) {
    std::vector<std::array<int, 4>> out(worlds.size(), {0, 0, 0, 0});
    for (size_t wi = 0; wi < worlds.size(); ++wi) {
        Replayed r{freshGame(startHands(worlds[wi], history), trump, declarer),
                   Policy(trump, declarer, opt), {}};
        r.game.startPlaying();
        for (auto& sc : history) {
            if (sc.first != me) {
                std::vector<int> legal = legalOf(r.game);
                if (legal.size() > 1
                        && r.policy.choose(sc.first, legal, handsOf(r.game), r.trick) != sc.second)
                    ++out[wi][sc.first];
            }
            play(r, sc.first, sc.second);
        }
    }
    return out;
}

namespace {

inline uint32_t suitMask32(int s) { return 0xFFu << (s * 8); }
inline uint32_t higherInSuit32(int card) {
    return suitMask32(card >> 3) & static_cast<uint32_t>(~((uint64_t(2) << card) - 1ULL));
}
inline bool beatsCard(int a, int b, int trump) {
    const int sa = a >> 3, sb = b >> 3;
    if (sa == sb) return (a & 7) > (b & 7);
    return sa == trump;
}

inline uint32_t fastLegalMask(uint32_t hand, int n_trick, int led,
                              int w_card, int w_player, int cur, int trump) {
    if (n_trick == 0)
        return hand;
    const uint32_t following = hand & suitMask32(led);
    const bool partner_winning = (w_player == ((cur + 2) & 3));
    if (following) {
        if (partner_winning)
            return following;
        if ((w_card >> 3) == led) {
            const uint32_t beating = following & higherInSuit32(w_card);
            return beating ? beating : following;
        }
        return following;
    }
    if (partner_winning || trump < 0)
        return hand;
    const uint32_t tr_mask = suitMask32(trump);
    const uint32_t trumps = hand & tr_mask;
    if (!trumps)
        return hand;
    if ((w_card >> 3) == trump) {
        const uint32_t higher_tr = trumps & higherInSuit32(w_card);
        if (higher_tr)
            return higher_tr;
        const uint32_t non_tr = hand & ~tr_mask;
        return non_tr ? non_tr : trumps;
    }
    return trumps;
}

// Prune strictly equivalent 0-point cards (7, 8, 9 of the same suit) when no
// live card in another hand or current winning card lies between them.
inline uint32_t pruneEquivalentZeroPts(uint32_t legal, uint32_t others, int n_trick, int w_card) {
    uint32_t out = legal;
    for (int s = 0; s < 4; ++s) {
        const uint32_t low3 = (legal >> (s * 8)) & 7u;
        if ((low3 & (low3 - 1u)) == 0)
            continue;  // 0 or 1 of {7,8,9}
        const int base = s * 8;
        if ((low3 & 3u) == 3u)  // both 7 and 8 -> drop 8
            out &= ~(1u << (base + 1));
        if ((low3 & 6u) == 6u)  // both 8 and 9 -> drop 9
            out &= ~(1u << (base + 2));
        else if (low3 == 5u) {  // 7 and 9 without 8 -> drop 9 if 8 is dead
            const int c8 = base + 1;
            const bool c8_live = ((others >> c8) & 1u) != 0 || (n_trick > 0 && w_card == c8);
            if (!c8_live)
                out &= ~(1u << (base + 2));
        }
    }
    return out;
}

struct EndgameTTEntry {
    uint32_t h[4];
    uint16_t gen;
    uint8_t cur;
    int8_t lo;
    int8_t hi;
};

struct EndgameSolver {
    static constexpr size_t TT_SIZE = 1u << 14;  // 16384 entries (~384 KB)
    static constexpr size_t TT_MASK = TT_SIZE - 1;
    std::vector<EndgameTTEntry> tt;
    uint16_t gen = 1;
    int trump = -1;

    EndgameSolver() : tt(TT_SIZE) {
        for (auto& e : tt) e.gen = 0;
    }

    void newWorld(int t) {
        trump = t;
        ++gen;
        if (gen == 0) {
            for (auto& e : tt) e.gen = 0;
            gen = 1;
        }
    }

    static inline size_t hashState(const uint32_t h[4], int cur) {
        uint64_t x = (uint64_t(h[0]) << 32) | h[1];
        uint64_t y = (uint64_t(h[2]) << 32) | h[3];
        x ^= y * 0x9e3779b97f4a7c15ULL;
        x ^= static_cast<uint64_t>(cur) * 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return static_cast<size_t>(x) & TT_MASK;
    }

    // Returns optimal remaining Team-0 points from the start of the current
    // trick (rem_pts = trick_pts + points of all cards remaining in hands).
    int ab(uint32_t h[4], int cur, int n_trick, int led, int w_card, int w_player,
           int trick_pts, int rem_pts, int alpha, int beta) {
        if (rem_pts == 0)
            return 0;
        if (0 >= beta)
            return 0;
        if (rem_pts <= alpha)
            return rem_pts;

        int orig_alpha = alpha, orig_beta = beta;
        size_t tt_idx = 0;
        if (n_trick == 0) {
            tt_idx = hashState(h, cur);
            const auto& e = tt[tt_idx];
            if (e.gen == gen && e.cur == cur && e.h[0] == h[0] && e.h[1] == h[1]
                    && e.h[2] == h[2] && e.h[3] == h[3]) {
                if (e.lo == e.hi) return e.lo;
                if (e.lo >= beta) return e.lo;
                if (e.hi <= alpha) return e.hi;
                if (e.lo > alpha) alpha = e.lo;
                if (e.hi < beta) beta = e.hi;
                if (alpha >= beta) return alpha;
            }
        }

        uint32_t legal = fastLegalMask(h[cur], n_trick, led, w_card, w_player, cur, trump);
        const uint32_t others = (h[0] | h[1] | h[2] | h[3]) & ~h[cur];
        if ((legal & (legal - 1u)) != 0)
            legal = pruneEquivalentZeroPts(legal, others, n_trick, w_card);

        // Extract moves; order high-rank cards first for faster cutoffs
        int moves[8];
        int nm = 0;
        uint32_t m = legal;
        while (m) {
            const int c = manille::ctz64(m);
            m &= m - 1u;
            moves[nm++] = c;
        }
        // Simple insertion sort by descending (is_trump, rank) when nm > 1
        if (nm > 1) {
            for (int i = 1; i < nm; ++i) {
                const int c = moves[i];
                const int kc = ((c >> 3) == trump ? 8 : 0) + (c & 7);
                int j = i - 1;
                while (j >= 0) {
                    const int kj = ((moves[j] >> 3) == trump ? 8 : 0) + (moves[j] & 7);
                    if (kj >= kc) break;
                    moves[j + 1] = moves[j];
                    --j;
                }
                moves[j + 1] = c;
            }
        }

        const bool maxing = (cur & 1) == 0;
        int best = maxing ? -1 : 61;

        for (int i = 0; i < nm; ++i) {
            const int c = moves[i];
            h[cur] ^= (1u << c);
            const int nt = n_trick + 1;
            const int n_led = (n_trick == 0) ? (c >> 3) : led;
            int n_wc = w_card, n_wp = w_player;
            if (n_trick == 0 || beatsCard(c, w_card, trump)) {
                n_wc = c;
                n_wp = cur;
            }
            const int n_tpts = trick_pts + PTS[c & 7];
            int val;
            if (nt == 4) {
                const int gain = (n_wp & 1) == 0 ? n_tpts : 0;
                const int next_rem = rem_pts - n_tpts;
                if (next_rem == 0 || (h[0] | h[1] | h[2] | h[3]) == 0) {
                    val = gain;
                } else {
                    val = gain + ab(h, n_wp, 0, 0, -1, -1, 0, next_rem,
                                    alpha - gain, beta - gain);
                }
            } else {
                val = ab(h, (cur + 1) & 3, nt, n_led, n_wc, n_wp, n_tpts, rem_pts, alpha, beta);
            }
            h[cur] ^= (1u << c);

            if (maxing) {
                if (val > best) best = val;
                if (best > alpha) alpha = best;
                if (alpha >= beta) break;
            } else {
                if (val < best) best = val;
                if (best < beta) beta = best;
                if (alpha >= beta) break;
            }
        }

        if (n_trick == 0) {
            auto& e = tt[tt_idx];
            const bool same = (e.gen == gen && e.cur == cur && e.h[0] == h[0]
                               && e.h[1] == h[1] && e.h[2] == h[2] && e.h[3] == h[3]);
            if (!same) {
                e.h[0] = h[0]; e.h[1] = h[1]; e.h[2] = h[2]; e.h[3] = h[3];
                e.gen = gen;
                e.cur = static_cast<uint8_t>(cur);
                e.lo = 0;
                e.hi = static_cast<int8_t>(rem_pts);
            }
            if (best <= orig_alpha) {
                if (best < e.hi) e.hi = static_cast<int8_t>(best);
            } else if (best >= orig_beta) {
                if (best > e.lo) e.lo = static_cast<int8_t>(best);
            } else {
                e.lo = e.hi = static_cast<int8_t>(best);
            }
        }
        return best;
    }
};

} // namespace

std::vector<std::vector<double>> evaluate_endgame(
    const ManilleGame& game, int me,
    const std::vector<int>& candidates,
    const std::vector<World>& worlds) {
    thread_local EndgameSolver solver;
    std::vector<std::vector<double>> out(candidates.size(),
                                         std::vector<double>(worlds.size(), 0.0));
    const int trump = game.trumpSuit();
    const int base_pts0 = game.teamPoints(0);
    const int n_trick = game.trickSize();
    int led = 0, w_card = -1, w_player = -1, base_tpts = 0;
    if (n_trick > 0) {
        led = suitOf(game.trickCard(0));
        w_card = game.trickCard(0);
        w_player = game.trickPlayer(0);
        base_tpts = PTS[rankOf(w_card)];
        for (int i = 1; i < n_trick; ++i) {
            const int tc = game.trickCard(i);
            const int tp = game.trickPlayer(i);
            base_tpts += PTS[rankOf(tc)];
            if (beatsCard(tc, w_card, trump)) {
                w_card = tc;
                w_player = tp;
            }
        }
    }

    // Deduplicate worlds to avoid re-solving identical endgame deals
    std::vector<size_t> order(worlds.size());
    for (size_t i = 0; i < worlds.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return worlds[a] < worlds[b];
    });

    const bool my_team0 = (me % 2 == 0);
    for (size_t oi = 0; oi < order.size(); ) {
        const size_t first_wi = order[oi];
        const World& w = worlds[first_wi];
        size_t oj = oi + 1;
        while (oj < order.size() && worlds[order[oj]] == w) ++oj;

        uint32_t h[4] = {
            static_cast<uint32_t>(w[0]), static_cast<uint32_t>(w[1]),
            static_cast<uint32_t>(w[2]), static_cast<uint32_t>(w[3])
        };
        int rem_pts = base_tpts;
        uint32_t all_cards = h[0] | h[1] | h[2] | h[3];
        while (all_cards) {
            const int c = manille::ctz64(all_cards);
            all_cards &= all_cards - 1u;
            rem_pts += PTS[c & 7];
        }

        solver.newWorld(trump);
        for (size_t ci = 0; ci < candidates.size(); ++ci) {
            const int c = candidates[ci];
            h[me] ^= (1u << c);
            const int nt = n_trick + 1;
            const int n_led = (n_trick == 0) ? (c >> 3) : led;
            int n_wc = w_card, n_wp = w_player;
            if (n_trick == 0 || beatsCard(c, w_card, trump)) {
                n_wc = c;
                n_wp = me;
            }
            const int n_tpts = base_tpts + PTS[c & 7];
            int pts0_rem;
            if (nt == 4) {
                const int gain = (n_wp & 1) == 0 ? n_tpts : 0;
                const int next_rem = rem_pts - n_tpts;
                if (next_rem == 0 || (h[0] | h[1] | h[2] | h[3]) == 0)
                    pts0_rem = gain;
                else
                    pts0_rem = gain + solver.ab(h, n_wp, 0, 0, -1, -1, 0, next_rem, -1, next_rem + 1);
            } else {
                pts0_rem = solver.ab(h, (me + 1) & 3, nt, n_led, n_wc, n_wp, n_tpts, rem_pts,
                                     -1, rem_pts + 1);
            }
            h[me] ^= (1u << c);
            const int final_pts0 = base_pts0 + pts0_rem;
            const double team_pts = my_team0 ? double(final_pts0) : double(60 - final_pts0);
            for (size_t k = oi; k < oj; ++k)
                out[ci][order[k]] = team_pts;
        }
        oi = oj;
    }
    return out;
}

} // namespace rollout
