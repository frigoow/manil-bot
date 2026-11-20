#include "fast_policy.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include "bitops.h"
#include "parallel.h"

namespace fastpolicy {
namespace {

inline int suitOf(int c) { return c >> 3; }
inline int64_t bit(int c) { return int64_t(1) << c; }
inline int64_t suitMask(int s) { return int64_t(0xFF) << (8 * s); }
inline bool beatsCard(int a, int b, int trump) {
    const int sa = a >> 3, sb = b >> 3;
    return (sa == sb) ? ((a & 7) > (b & 7)) : (sa == trump);
}

// Fast zero-allocation legal move mask matching ManilleGame::legalMoveMask().
inline int64_t fastLegalMask(int64_t hand, int seat, int nt, int led,
                             int ws, int wc, int trump) {
    if (nt == 0)
        return hand;
    const int64_t flw = hand & suitMask(led);
    const bool par_w = (ws == ((seat + 2) & 3));
    if (flw) {
        if (!par_w && (wc >> 3) == led) {
            const int64_t ab = flw & ~((int64_t(1) << (wc + 1)) - 1);
            return ab ? ab : flw;
        }
        return flw;
    }
    if (par_w || trump < 0)
        return hand;
    const int64_t tr = hand & suitMask(trump);
    if (!tr)
        return hand;
    if ((wc >> 3) == trump) {
        const int64_t ab = tr & ~((int64_t(1) << (wc + 1)) - 1);
        if (ab)
            return ab;
        const int64_t ntr = hand & ~tr;
        return ntr ? ntr : tr;
    }
    return tr;
}

inline void forwardLayers23(const Net& net, const float* a1, float* a2, float out[32]) {
    const int h1 = net.h1, h2 = net.h2;
    std::memcpy(a2, net.b2.data(), static_cast<size_t>(h2) * sizeof(float));
    const float* w2t = net.w2t.data();
    for (int i = 0; i < h1; ++i) {
        const float v = a1[i];
        if (v > 0.0f) {
            const float* col = w2t + static_cast<size_t>(i) * h2;
            for (int o = 0; o < h2; ++o)
                a2[o] += v * col[o];
        }
    }
    std::memcpy(out, net.b3.data(), 32 * sizeof(float));
    const float* w3t = net.w3t.data();
    for (int i = 0; i < h2; ++i) {
        const float v = a2[i];
        if (v > 0.0f) {
            const float* col = w3t + static_cast<size_t>(i) * 32;
            for (int c = 0; c < 32; ++c)
                out[c] += v * col[c];
        }
    }
}

} // namespace

int64_t features(const ManilleGame& game, int seat, std::vector<int>& active,
                 float dense[F_DENSE]) {
    active.clear();
    const int trump = game.trumpSuit();
    const int64_t hand = game.handMask(seat);
    int64_t legal = 0;
    for (const Card& c : game.validMoves())
        legal |= bit(c.index());

    // public record: completed tricks, then the cards on the table
    int64_t played[4] = {0, 0, 0, 0};
    int voids[4] = {0, 0, 0, 0};
    const int n_done = game.playedCardCount();
    for (int i = 0; i < n_done; ++i) {
        const int p = game.playedPlayer(i), c = game.playedCard(i);
        played[p] |= bit(c);
        const int led = suitOf(game.playedCard(i - i % 4));
        if (suitOf(c) != led)
            voids[p] |= 1 << led;
    }
    const int n_trick = game.trickSize();
    std::array<int32_t, 4> tc{-1, -1, -1, -1};
    for (int i = 0; i < n_trick; ++i) {
        const int p = game.trickPlayer(i), c = game.trickCard(i);
        tc[i] = c;
        played[p] |= bit(c);
        if (suitOf(c) != suitOf(game.trickCard(0)))
            voids[p] |= 1 << suitOf(game.trickCard(0));
    }
    const int64_t seen = hand | played[0] | played[1] | played[2] | played[3];

    for (int c = 0; c < 32; ++c)
        if (hand & bit(c))
            active.push_back(F_HAND + c);
    for (int c = 0; c < 32; ++c)
        if (legal & bit(c))
            active.push_back(F_LEGAL + c);
    for (int r = 0; r < 4; ++r) {
        const int64_t m = played[(seat + r) % 4];
        for (int c = 0; c < 32; ++c)
            if (m & bit(c))
                active.push_back(F_PLAYED + r * 32 + c);
    }
    {
        // ascending index order: sort the (at most 3) trick entries by slot
        int idx[3], n = 0;
        for (int i = 0; i < n_trick; ++i) {
            const int r = ((game.trickPlayer(i) - seat) % 4 + 4) % 4;
            if (r >= 1)
                idx[n++] = F_TRICK + (r - 1) * 32 + tc[i];
        }
        std::sort(idx, idx + n);
        for (int i = 0; i < n; ++i)
            active.push_back(idx[i]);
    }
    active.push_back(trump >= 0 ? F_TRUMP + trump : F_TRUMP + 4);
    if (game.declarer() >= 0)
        active.push_back(F_DECLARER + ((game.declarer() - seat) % 4 + 4) % 4);
    for (int r = 1; r < 4; ++r)
        for (int s = 0; s < 4; ++s)
            if ((voids[(seat + r) % 4] >> s) & 1)
                active.push_back(F_VOID + (r - 1) * 4 + s);
    active.push_back(F_TRICK_INDEX + std::min(n_done / 4, 7));
    active.push_back(F_POS + n_trick);
    // own cards nothing unseen outranks in their suit
    for (int c = 0; c < 32; ++c) {
        if (!(hand & bit(c)))
            continue;
        const int64_t above = suitMask(suitOf(c)) & ~((bit(c) << 1) - 1);
        if ((above & ~seen) == 0)
            active.push_back(F_MASTER + c);
    }
    if (n_trick > 0) {
        const int w = ManilleGame::trickWinner(tc, n_trick, trump);
        const Card wc = Card::from_index(tc[w]);
        for (int c = 0; c < 32; ++c)
            if ((legal & bit(c)) && Card::from_index(c).beats(wc, trump))
                active.push_back(F_WINS + c);
        if (game.trickPlayer(w) == (seat + 2) % 4)
            active.push_back(F_PARTNER_WINS);
    }
    dense[0] = game.teamPoints(seat % 2) / 60.0f;
    dense[1] = game.teamPoints(1 - seat % 2) / 60.0f;
    return legal;
}

void Net::finalize() {
    if (h1 <= 0 || h2 <= 0
            || w1.size() != static_cast<size_t>(h1) * F_DIM || b1.size() != static_cast<size_t>(h1)
            || w2.size() != static_cast<size_t>(h2) * h1 || b2.size() != static_cast<size_t>(h2)
            || w3.size() != static_cast<size_t>(32) * h2 || b3.size() != 32)
        throw std::invalid_argument("fast policy net: weight shapes do not match the layout");
    w1t.assign(static_cast<size_t>(F_DIM) * h1, 0.0f);
    for (int o = 0; o < h1; ++o)
        for (int i = 0; i < F_DIM; ++i)
            w1t[static_cast<size_t>(i) * h1 + o] = w1[static_cast<size_t>(o) * F_DIM + i];
    w2t.assign(static_cast<size_t>(h1) * h2, 0.0f);
    for (int o = 0; o < h2; ++o)
        for (int i = 0; i < h1; ++i)
            w2t[static_cast<size_t>(i) * h2 + o] = w2[static_cast<size_t>(o) * h1 + i];
    w3t.assign(static_cast<size_t>(h2) * 32, 0.0f);
    for (int c = 0; c < 32; ++c)
        for (int i = 0; i < h2; ++i)
            w3t[static_cast<size_t>(i) * 32 + c] = w3[static_cast<size_t>(c) * h2 + i];
}

Net load_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::invalid_argument("fast policy net: cannot open " + path);
    char magic[4];
    int32_t hdr[4];
    f.read(magic, 4);
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (!f || std::memcmp(magic, "MFPN", 4) != 0 || hdr[0] != 1)
        throw std::invalid_argument("fast policy net: not a version-1 MFPN file");
    if (hdr[1] != F_DIM)
        throw std::invalid_argument("fast policy net: feature layout differs from this build");
    Net n;
    n.h1 = hdr[2];
    n.h2 = hdr[3];
    if (n.h1 <= 0 || n.h2 <= 0 || n.h1 > 8192 || n.h2 > 8192)
        throw std::invalid_argument("fast policy net: bad layer sizes");
    auto read = [&](std::vector<float>& v, size_t count) {
        v.resize(count);
        f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(count * sizeof(float)));
    };
    read(n.w1, static_cast<size_t>(n.h1) * F_DIM);
    read(n.b1, static_cast<size_t>(n.h1));
    read(n.w2, static_cast<size_t>(n.h2) * n.h1);
    read(n.b2, static_cast<size_t>(n.h2));
    read(n.w3, static_cast<size_t>(32) * n.h2);
    read(n.b3, 32);
    if (!f)
        throw std::invalid_argument("fast policy net: file is truncated");
    n.finalize();
    return n;
}

int choose(const Net& net, const ManilleGame& game, int seat) {
    // Early exit on forced moves before touching features or network weights.
    const int trump = game.trumpSuit();
    const int64_t hand = game.handMask(seat);
    const int nt = game.trickSize();
    int tc[4], tp[4];
    int ws = -1, wc = -1, led = 0;
    if (nt > 0) {
        for (int i = 0; i < nt; ++i) {
            tc[i] = game.trickCard(i);
            tp[i] = game.trickPlayer(i);
        }
        led = suitOf(tc[0]);
        ws = tp[0];
        wc = tc[0];
        for (int i = 1; i < nt; ++i)
            if (beatsCard(tc[i], wc, trump)) {
                ws = tp[i];
                wc = tc[i];
            }
    }
    const int64_t legal = fastLegalMask(hand, seat, nt, led, ws, wc, trump);
    if ((legal & (legal - 1)) == 0)
        return manille::ctz64(static_cast<unsigned long long>(legal));

    // thread_local scratch: avoids heap allocations across tens of thousands of calls.
    thread_local std::vector<float> a1, a2;
    const int h1 = net.h1, h2 = net.h2;
    a1.resize(static_cast<size_t>(h1));
    a2.resize(static_cast<size_t>(h2));
    std::memcpy(a1.data(), net.b1.data(), static_cast<size_t>(h1) * sizeof(float));

    const float* w1t = net.w1t.data();
    auto add1 = [&](int idx) {
        const float* col = w1t + static_cast<size_t>(idx) * h1;
        for (int o = 0; o < h1; ++o)
            a1[o] += col[o];
    };

    int64_t played[4] = {0, 0, 0, 0};
    int voids[4] = {0, 0, 0, 0};
    const int n_done = game.playedCardCount();
    for (int i = 0; i < n_done; i += 4) {
        const int t_led = suitOf(game.playedCard(i));
        for (int j = 0; j < 4; ++j) {
            const int p = game.playedPlayer(i + j), c = game.playedCard(i + j);
            played[p] |= bit(c);
            if (suitOf(c) != t_led)
                voids[p] |= 1 << t_led;
        }
    }
    for (int i = 0; i < nt; ++i) {
        played[tp[i]] |= bit(tc[i]);
        if (suitOf(tc[i]) != led)
            voids[tp[i]] |= 1 << led;
    }
    const int64_t seen = hand | played[0] | played[1] | played[2] | played[3];

    int64_t m = hand;
    while (m) {
        const int c = manille::ctz64(static_cast<unsigned long long>(m));
        m &= m - 1;
        add1(F_HAND + c);
        const int64_t above = suitMask(suitOf(c)) & ~((bit(c) << 1) - 1);
        if ((above & ~seen) == 0)
            add1(F_MASTER + c);
    }
    m = legal;
    while (m) {
        const int c = manille::ctz64(static_cast<unsigned long long>(m));
        m &= m - 1;
        add1(F_LEGAL + c);
        if (nt > 0 && beatsCard(c, wc, trump))
            add1(F_WINS + c);
    }
    for (int r = 0; r < 4; ++r) {
        int64_t pm = played[(seat + r) % 4];
        while (pm) {
            const int c = manille::ctz64(static_cast<unsigned long long>(pm));
            pm &= pm - 1;
            add1(F_PLAYED + r * 32 + c);
        }
    }
    for (int i = 0; i < nt; ++i) {
        const int r = ((tp[i] - seat) % 4 + 4) % 4;
        if (r >= 1)
            add1(F_TRICK + (r - 1) * 32 + tc[i]);
    }
    add1(trump >= 0 ? F_TRUMP + trump : F_TRUMP + 4);
    if (game.declarer() >= 0)
        add1(F_DECLARER + ((game.declarer() - seat) % 4 + 4) % 4);
    for (int r = 1; r < 4; ++r)
        for (int s = 0; s < 4; ++s)
            if ((voids[(seat + r) % 4] >> s) & 1)
                add1(F_VOID + (r - 1) * 4 + s);
    add1(F_TRICK_INDEX + std::min(n_done / 4, 7));
    add1(F_POS + nt);
    if (nt > 0 && ws == ((seat + 2) & 3))
        add1(F_PARTNER_WINS);

    const float d0 = game.teamPoints(seat % 2) / 60.0f;
    const float d1 = game.teamPoints(1 - seat % 2) / 60.0f;
    const float* cd0 = w1t + static_cast<size_t>(F_BINARY) * h1;
    const float* cd1 = w1t + static_cast<size_t>(F_BINARY + 1) * h1;
    for (int o = 0; o < h1; ++o)
        a1[o] += d0 * cd0[o] + d1 * cd1[o];

    alignas(32) float logits[32];
    forwardLayers23(net, a1.data(), a2.data(), logits);

    int best = -1;
    float best_v = 0.0f;
    m = legal;
    while (m) {
        const int c = manille::ctz64(static_cast<unsigned long long>(m));
        m &= m - 1;
        if (best < 0 || logits[c] > best_v) {
            best = c;
            best_v = logits[c];
        }
    }
    return best;
}

std::vector<double> history_log_likelihood(
    const Net& net, int trump, int declarer, int me,
    const std::vector<rollout::World>& worlds,
    const rollout::History& history,
    double tau, double partner_w, double opp_w) {
    const size_t nw = worlds.size();
    std::vector<double> loglik(nw, 0.0);
    if (net.empty() || history.empty() || nw == 0 || (partner_w <= 0.0 && opp_w <= 0.0))
        return loglik;

    const int h1 = net.h1, h2 = net.h2;
    const double inv_tau = 1.0 / std::max(tau, 1e-6);
    std::vector<rollout::World> hands(nw);
    int64_t played_all[4] = {0, 0, 0, 0};
    for (const auto& sc : history)
        played_all[sc.first] |= bit(sc.second);
    for (size_t wi = 0; wi < nw; ++wi)
        for (int s = 0; s < 4; ++s)
            hands[wi][s] = worlds[wi][s] | played_all[s];

    int64_t played[4] = {0, 0, 0, 0};
    int voids[4] = {0, 0, 0, 0};
    int pts[2] = {0, 0};
    int tc[4], tp[4], nt = 0, n_done = 0;
    constexpr int PTS[8] = {0, 0, 0, 1, 2, 3, 4, 5};
    const int partner = (me + 2) % 4;
    const float* w1t = net.w1t.data();

    thread_local std::vector<float> pub_a1, a1, a2;
    pub_a1.resize(static_cast<size_t>(h1));
    a1.resize(static_cast<size_t>(h1));
    a2.resize(static_cast<size_t>(h2));

    for (size_t ply = 0; ply < history.size(); ++ply) {
        const int seat = history[ply].first;
        const int actual = history[ply].second;

        if (seat != me) {
            const double seat_w = (seat == partner) ? partner_w : opp_w;
            if (seat_w > 0.0) {
                std::memcpy(pub_a1.data(), net.b1.data(), static_cast<size_t>(h1) * sizeof(float));
                auto add_pub = [&](int idx, float scale = 1.0f) {
                    const float* col = w1t + static_cast<size_t>(idx) * h1;
                    if (scale == 1.0f) {
                        for (int o = 0; o < h1; ++o)
                            pub_a1[o] += col[o];
                    } else {
                        for (int o = 0; o < h1; ++o)
                            pub_a1[o] += scale * col[o];
                    }
                };
                int64_t pl_cur[4] = {played[0], played[1], played[2], played[3]};
                int v_cur[4] = {voids[0], voids[1], voids[2], voids[3]};
                const int led = nt > 0 ? suitOf(tc[0]) : 0;
                for (int i = 0; i < nt; ++i) {
                    pl_cur[tp[i]] |= bit(tc[i]);
                    if (suitOf(tc[i]) != led)
                        v_cur[tp[i]] |= 1 << led;
                }
                const int64_t pub_seen = pl_cur[0] | pl_cur[1] | pl_cur[2] | pl_cur[3];

                for (int r = 0; r < 4; ++r) {
                    int64_t m = pl_cur[(seat + r) % 4];
                    while (m) {
                        const int c = manille::ctz64(static_cast<unsigned long long>(m));
                        m &= m - 1;
                        add_pub(F_PLAYED + r * 32 + c);
                    }
                }
                for (int i = 0; i < nt; ++i) {
                    const int r = ((tp[i] - seat) % 4 + 4) % 4;
                    if (r >= 1)
                        add_pub(F_TRICK + (r - 1) * 32 + tc[i]);
                }
                add_pub(trump >= 0 ? F_TRUMP + trump : F_TRUMP + 4);
                if (declarer >= 0)
                    add_pub(F_DECLARER + ((declarer - seat) % 4 + 4) % 4);
                for (int r = 1; r < 4; ++r)
                    for (int s = 0; s < 4; ++s)
                        if ((v_cur[(seat + r) % 4] >> s) & 1)
                            add_pub(F_VOID + (r - 1) * 4 + s);
                add_pub(F_TRICK_INDEX + std::min(n_done / 4, 7));
                add_pub(F_POS + nt);

                int ws = -1, wc = -1;
                if (nt > 0) {
                    ws = tp[0];
                    wc = tc[0];
                    for (int i = 1; i < nt; ++i)
                        if (beatsCard(tc[i], wc, trump)) {
                            ws = tp[i];
                            wc = tc[i];
                        }
                    if (ws == ((seat + 2) & 3))
                        add_pub(F_PARTNER_WINS);
                }
                add_pub(F_BINARY + 0, pts[seat % 2] / 60.0f);
                add_pub(F_BINARY + 1, pts[1 - seat % 2] / 60.0f);

                std::unordered_map<int64_t, size_t> hand_to_id;
                hand_to_id.reserve(nw);
                std::vector<int64_t> uniq_hands;
                uniq_hands.reserve(nw);
                std::vector<size_t> world_to_id(nw);
                for (size_t wi = 0; wi < nw; ++wi) {
                    const int64_t hand = hands[wi][seat];
                    auto it = hand_to_id.find(hand);
                    if (it == hand_to_id.end()) {
                        const size_t id = uniq_hands.size();
                        hand_to_id.emplace(hand, id);
                        uniq_hands.push_back(hand);
                        world_to_id[wi] = id;
                    } else {
                        world_to_id[wi] = it->second;
                    }
                }
                std::vector<double> uniq_lp(uniq_hands.size(), 0.0);
                const float* pub_ptr = pub_a1.data();
                manille_parallel::parallel_for(uniq_hands.size(), 32, [&](size_t b, size_t e) {
                    thread_local std::vector<float> tl_a1, tl_a2;
                    tl_a1.resize(static_cast<size_t>(h1));
                    tl_a2.resize(static_cast<size_t>(h2));
                    auto add_a1 = [&](int idx) {
                        const float* col = w1t + static_cast<size_t>(idx) * h1;
                        for (int o = 0; o < h1; ++o)
                            tl_a1[o] += col[o];
                    };
                    for (size_t ui = b; ui < e; ++ui) {
                        const int64_t hand = uniq_hands[ui];
                        const int64_t leg = fastLegalMask(hand, seat, nt, led, ws, wc, trump);
                        if ((leg & (leg - 1)) == 0) {
                            uniq_lp[ui] = 0.0;
                            continue;
                        }
                        if (!((leg >> actual) & 1)) {
                            uniq_lp[ui] = -20.0;
                            continue;
                        }
                        std::memcpy(tl_a1.data(), pub_ptr, static_cast<size_t>(h1) * sizeof(float));
                        const int64_t seen = pub_seen | hand;
                        int64_t m = hand;
                        while (m) {
                            const int c = manille::ctz64(static_cast<unsigned long long>(m));
                            m &= m - 1;
                            add_a1(F_HAND + c);
                            const int64_t ab = suitMask(suitOf(c)) & ~((bit(c) << 1) - 1);
                            if ((ab & ~seen) == 0)
                                add_a1(F_MASTER + c);
                        }
                        m = leg;
                        while (m) {
                            const int c = manille::ctz64(static_cast<unsigned long long>(m));
                            m &= m - 1;
                            add_a1(F_LEGAL + c);
                            if (nt > 0 && beatsCard(c, wc, trump))
                                add_a1(F_WINS + c);
                        }
                        alignas(32) float logits[32];
                        forwardLayers23(net, tl_a1.data(), tl_a2.data(), logits);

                        double max_l = -1e18, actual_l = -1e18, lgs[8];
                        int nl = 0;
                        m = leg;
                        while (m) {
                            const int c = manille::ctz64(static_cast<unsigned long long>(m));
                            m &= m - 1;
                            const double sc = double(logits[c]) * inv_tau;
                            lgs[nl++] = sc;
                            if (sc > max_l) max_l = sc;
                            if (c == actual) actual_l = sc;
                        }
                        double sum_e = 0.0;
                        for (int i = 0; i < nl; ++i)
                            sum_e += std::exp(lgs[i] - max_l);
                        uniq_lp[ui] = (actual_l - max_l) - std::log(sum_e);
                    }
                });
                for (size_t wi = 0; wi < nw; ++wi)
                    loglik[wi] += seat_w * uniq_lp[world_to_id[wi]];
            }
        }

        for (size_t wi = 0; wi < nw; ++wi)
            hands[wi][seat] &= ~bit(actual);
        tc[nt] = actual;
        tp[nt] = seat;
        ++nt;
        if (nt == 4) {
            int ws = tp[0], wc = tc[0], tpts = PTS[tc[0] & 7];
            for (int j = 1; j < 4; ++j) {
                tpts += PTS[tc[j] & 7];
                if (beatsCard(tc[j], wc, trump)) {
                    ws = tp[j];
                    wc = tc[j];
                }
            }
            pts[ws % 2] += tpts;
            const int led = suitOf(tc[0]);
            for (int j = 0; j < 4; ++j) {
                played[tp[j]] |= bit(tc[j]);
                if (suitOf(tc[j]) != led)
                    voids[tp[j]] |= 1 << led;
            }
            n_done += 4;
            nt = 0;
        }
    }
    return loglik;
}

std::vector<std::vector<double>> evaluate_candidates(
    const Net& net, int trump, int declarer, int me,
    const std::vector<int>& candidates,
    const std::vector<rollout::World>& worlds,
    const rollout::History& history) {
    std::vector<std::vector<double>> out(candidates.size(),
                                         std::vector<double>(worlds.size(), 0.0));
    manille_parallel::parallel_for(worlds.size(), 16, [&](size_t b, size_t e) {
        for (size_t wi = b; wi < e; ++wi) {
            rollout::World start = worlds[wi];
            for (auto& sc : history)
                start[sc.first] |= bit(sc.second);
            ManilleGame base(start, std::array<int32_t, 4>{}, std::array<int32_t, 4>{}, 0,
                             (declarer + 1) % 4, trump, 0, 0, 0, 0,
                             std::array<int32_t, 32>{}, std::array<int32_t, 32>{}, 0,
                             false, declarer);
            base.startPlaying();
            for (auto& sc : history)
                base.doMove(Card::from_index(sc.second));
            for (size_t ci = 0; ci < candidates.size(); ++ci) {
                ManilleGame g = base;
                g.doMove(Card::from_index(candidates[ci]));
                while (!g.finished()) {
                    const int seat = static_cast<int>(g.currentPlayer());
                    const int64_t hand = g.handMask(seat);
                    int c;
                    if ((hand & (hand - 1)) == 0) {
                        c = manille::ctz64(static_cast<unsigned long long>(hand));  // last card
                    } else {
                        c = choose(net, g, seat);
                    }
                    g.doMove(Card::from_index(c));
                }
                out[ci][wi] = g.teamPoints(me % 2);
            }
        }
    });
    return out;
}

} // namespace fastpolicy
