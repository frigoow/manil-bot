#include "fast_policy.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include "bitops.h"

namespace fastpolicy {
namespace {

inline int suitOf(int c) { return c >> 3; }
inline int64_t bit(int c) { return int64_t(1) << c; }
inline int64_t suitMask(int s) { return int64_t(0xFF) << (8 * s); }

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
    // thread_local scratch: the play-out calls this tens of thousands of
    // times per move
    thread_local std::vector<int> active;
    thread_local std::vector<float> a1, a2;
    float dense[F_DENSE];
    const int64_t legal = features(game, seat, active, dense);
    if ((legal & (legal - 1)) == 0)  // forced
        return manille::ctz64(static_cast<unsigned long long>(legal));
    const int h1 = net.h1, h2 = net.h2;
    a1.assign(net.b1.begin(), net.b1.end());
    for (int i : active) {
        const float* col = net.w1t.data() + static_cast<size_t>(i) * h1;
        for (int o = 0; o < h1; ++o)
            a1[o] += col[o];
    }
    for (int d = 0; d < F_DENSE; ++d) {
        const float* col = net.w1t.data() + static_cast<size_t>(F_BINARY + d) * h1;
        for (int o = 0; o < h1; ++o)
            a1[o] += dense[d] * col[o];
    }
    for (int o = 0; o < h1; ++o)
        a1[o] = a1[o] > 0 ? a1[o] : 0;
    a2.resize(h2);
    for (int o = 0; o < h2; ++o) {
        const float* row = net.w2.data() + static_cast<size_t>(o) * h1;
        float s = net.b2[o];
        for (int i = 0; i < h1; ++i)
            s += row[i] * a1[i];
        a2[o] = s > 0 ? s : 0;
    }
    int best = -1;
    float best_v = 0;
    for (int c = 0; c < 32; ++c) {
        if (!(legal & bit(c)))
            continue;
        const float* row = net.w3.data() + static_cast<size_t>(c) * h2;
        float s = net.b3[c];
        for (int i = 0; i < h2; ++i)
            s += row[i] * a2[i];
        if (best < 0 || s > best_v) {
            best = c;
            best_v = s;
        }
    }
    return best;
}

std::vector<std::vector<double>> evaluate_candidates(
    const Net& net, int trump, int declarer, int me,
    const std::vector<int>& candidates,
    const std::vector<rollout::World>& worlds,
    const rollout::History& history) {
    std::vector<std::vector<double>> out(candidates.size(),
                                         std::vector<double>(worlds.size(), 0.0));
    for (size_t wi = 0; wi < worlds.size(); ++wi) {
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
    return out;
}

} // namespace fastpolicy
