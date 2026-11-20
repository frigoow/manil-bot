// Release builds define NDEBUG, which would compile every check below out.
#undef NDEBUG
#include "mobile/manille_mobile.h"
#include "fast_policy.h"
#include "sampler/uniform.h"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <initializer_list>
#include <map>
#include <vector>
#include <cstdint>
#include <cstring>
#include "bitops.h"

int main() {
    ManilleMobileGame* game = nullptr;
    assert(manille_mobile_new_random_deal(3, 42, &game) == MANILLE_MOBILE_OK);
    assert(game != nullptr);

    ManilleMobileState state{};
    assert(manille_mobile_get_state(game, &state) == MANILLE_MOBILE_OK);
    assert(state.current_player == 0);
    assert(state.declarer == 3);
    assert(state.trump_suit >= 0 && state.trump_suit < 4);

    int64_t hand = 0;
    int64_t legal = 0;
    assert(manille_mobile_hand_mask(game, 0, &hand) == MANILLE_MOBILE_OK);
    assert(manille_mobile_legal_mask(game, &legal) == MANILLE_MOBILE_OK);
    assert(hand == legal);

    float obs[MANILLE_MOBILE_OBS_DIM]{};
    float action[MANILLE_MOBILE_MAX_LEGAL * MANILLE_MOBILE_ACT_DIM]{};
    int32_t cards[MANILLE_MOBILE_MAX_LEGAL]{};
    int32_t n_legal = 0;
    assert(manille_mobile_encode_q(game, 0, obs, action, cards, &n_legal)
           == MANILLE_MOBILE_OK);
    assert(n_legal == 8);
    assert(cards[0] >= 0 && cards[0] < 32);
    assert(manille_mobile_apply_card(game, cards[0]) == MANILLE_MOBILE_OK);

    float raw[MANILLE_MOBILE_MAX_PLAYS * MANILLE_MOBILE_RAW_TOKEN_DIM]{};
    float glob[MANILLE_MOBILE_GLOBAL_DIM]{};
    int64_t n_raw = 0;
    assert(manille_mobile_encode_gru(game, 1, raw, &n_raw, glob, action, cards, &n_legal)
           == MANILLE_MOBILE_OK);
    assert(n_raw == 1);
    assert(n_legal > 0);

    assert(manille_mobile_get_state(game, &state) == MANILLE_MOBILE_OK);
    assert(state.n_played_cards == 0);  // completed tricks only
    assert(state.n_trick == 1);
    assert(state.current_player == 1);
    manille_mobile_destroy(game);

    // A human declaration must recreate the exact same deal with the chosen
    // suit (including mull) and preserve the announcement window.
    game = nullptr;
    assert(manille_mobile_new_random_deal_with_trump(0, 99, -1, &game)
           == MANILLE_MOBILE_OK);
    assert(manille_mobile_get_state(game, &state) == MANILLE_MOBILE_OK);
    assert(state.declarer == 0);
    assert(state.trump_suit == -1);
    assert(state.multiplier == 2);
    assert(state.doubling_phase == 1);
    assert(manille_mobile_announce_double(game, 1) == MANILLE_MOBILE_OK);
    assert(manille_mobile_announce_redouble(game, 0) == MANILLE_MOBILE_OK);
    assert(manille_mobile_get_state(game, &state) == MANILLE_MOBILE_OK);
    assert(state.multiplier == 8);
    manille_mobile_destroy(game);

    // The rule agent plays whole deals (trump and mull, uniform belief so
    // the test needs no nets): every card it picks is legal.
    for (int32_t trump : {2, -1}) {
        game = nullptr;
        assert(manille_mobile_new_random_deal_with_trump(1, 7, trump, &game)
               == MANILLE_MOBILE_OK);
        assert(manille_mobile_start_playing(game) == MANILLE_MOBILE_OK);
        float belief[96];
        for (float& b : belief)
            b = 1.0f / 3.0f;
        for (int ply = 0; ply < 32; ++ply) {
            int64_t legal_mask = 0;
            assert(manille_mobile_legal_mask(game, &legal_mask) == MANILLE_MOBILE_OK);
            int32_t card = -1;
            assert(manille_mobile_rule_choose(game, belief, nullptr, 0, 64, &card)
                   == MANILLE_MOBILE_OK);
            assert(card >= 0 && ((legal_mask >> card) & 1));
            assert(manille_mobile_apply_card(game, card) == MANILLE_MOBILE_OK);
        }
        assert(manille_mobile_get_state(game, &state) == MANILLE_MOBILE_OK);
        assert(state.finished == 1);
        assert(state.pts0 + state.pts1 == 60);
        // A finished deal still reports its last trick as the table; encoding
        // it once overflowed the 32-play history (app crash at hand end).
        {
            std::vector<float> o(MANILLE_MOBILE_OBS_DIM), a(MANILLE_MOBILE_MAX_LEGAL * MANILLE_MOBILE_ACT_DIM);
            int32_t cards[MANILLE_MOBILE_MAX_LEGAL];
            int32_t n_cards = -1;
            assert(manille_mobile_encode_q(game, state.current_player, o.data(), a.data(), cards, &n_cards)
                   == MANILLE_MOBILE_OK);
            assert(n_cards == 0);
        }
        manille_mobile_destroy(game);
    }

    // Save/restore round trip: every field of the state comes back, at the
    // announcement window, mid-deal with a meegaan + terug, and finished.
    {
        auto roundtrip = [](ManilleMobileGame* g) {
            ManilleMobileState a{}, b{};
            assert(manille_mobile_get_state(g, &a) == MANILLE_MOBILE_OK);
            int64_t hands[4];
            for (int p = 0; p < 4; ++p)
                assert(manille_mobile_hand_mask(g, p, &hands[p]) == MANILLE_MOBILE_OK);
            ManilleMobileGame* r = nullptr;
            assert(manille_mobile_create(hands, &a, &r) == MANILLE_MOBILE_OK);
            assert(manille_mobile_get_state(r, &b) == MANILLE_MOBILE_OK);
            assert(std::memcmp(&a, &b, sizeof a) == 0);
            manille_mobile_destroy(r);
        };
        game = nullptr;
        assert(manille_mobile_new_random_deal_with_trump(0, 21, -1, &game) == MANILLE_MOBILE_OK);
        roundtrip(game);
        assert(manille_mobile_announce_double(game, 1) == MANILLE_MOBILE_OK);
        roundtrip(game);
        assert(manille_mobile_announce_redouble(game, 0) == MANILLE_MOBILE_OK);
        roundtrip(game);
        assert(manille_mobile_start_playing(game) == MANILLE_MOBILE_OK);
        for (int ply = 0; ply < 32; ++ply) {
            int64_t lm = 0;
            assert(manille_mobile_legal_mask(game, &lm) == MANILLE_MOBILE_OK);
            assert(manille_mobile_apply_card(game, manille::ctz64(lm)) == MANILLE_MOBILE_OK);
            roundtrip(game);
        }
        assert(manille_mobile_get_state(game, &state) == MANILLE_MOBILE_OK);
        assert(state.finished == 1 && state.multiplier == 8);
        manille_mobile_destroy(game);
    }

    // trick winner from the engine: trump over the led suit, else highest led
    {
        int32_t idx = -1;
        const int32_t t1[4] = {7, 15, 2, 6};      // 10D led, 10H, 9D, AD; trump H
        assert(manille_mobile_trick_winner(t1, 4, 1, &idx) == MANILLE_MOBILE_OK && idx == 1);
        assert(manille_mobile_trick_winner(t1, 4, -1, &idx) == MANILLE_MOBILE_OK && idx == 0);
        const int32_t t2[4] = {2, 6, 7, 31};      // 9D, AD, 10D, 10S; no trump
        assert(manille_mobile_trick_winner(t2, 4, -1, &idx) == MANILLE_MOBILE_OK && idx == 2);
    }

    // Two-step API on a clone: sampled worlds agree with the observer's own
    // hand and the public record; dealt hands = current | played; the
    // choice with per-world weights is legal.
    {
        game = nullptr;
        assert(manille_mobile_new_random_deal_with_trump(2, 11, 1, &game) == MANILLE_MOBILE_OK);
        assert(manille_mobile_start_playing(game) == MANILLE_MOBILE_OK);
        for (int ply = 0; ply < 6; ++ply) {
            int64_t legal_mask = 0;
            assert(manille_mobile_legal_mask(game, &legal_mask) == MANILLE_MOBILE_OK);
            assert(manille_mobile_apply_card(game, manille::ctz64(legal_mask)) == MANILLE_MOBILE_OK);
        }
        ManilleMobileGame* copy = nullptr;
        assert(manille_mobile_clone(game, &copy) == MANILLE_MOBILE_OK);
        assert(manille_mobile_get_state(copy, &state) == MANILLE_MOBILE_OK);
        const int me = state.current_player;
        int64_t mine = 0;
        assert(manille_mobile_hand_mask(copy, me, &mine) == MANILLE_MOBILE_OK);
        int64_t cur[4 * 50], init[4 * 50];
        int32_t n = 0;
        assert(manille_mobile_sample_worlds(copy, 0, 50, cur, init, &n) == MANILLE_MOBILE_OK);
        assert(n == 50);
        int64_t played = 0;
        for (int i = 0; i < state.n_played_cards; ++i)
            played |= int64_t(1) << state.played_cards[i];
        for (int i = 0; i < state.n_trick; ++i)
            played |= int64_t(1) << state.trick_cards[i];
        std::vector<double> extra(n);
        for (int i = 0; i < n; ++i) {
            assert(cur[4 * i + me] == mine);
            int64_t all = 0, dealt = 0;
            for (int p = 0; p < 4; ++p) {
                assert((cur[4 * i + p] & played) == 0);
                assert((init[4 * i + p] & cur[4 * i + p]) == cur[4 * i + p]);
                assert(manille::popcount64(init[4 * i + p]) == 8);
                all |= cur[4 * i + p];
                dealt |= init[4 * i + p];
            }
            assert((all | played) == (int64_t(1) << 32) - 1);
            assert(dealt == (int64_t(1) << 32) - 1);
            extra[i] = -0.1 * i;
        }
        int64_t legal_mask = 0;
        assert(manille_mobile_legal_mask(copy, &legal_mask) == MANILLE_MOBILE_OK);
        int32_t card = -1;
        double values[32];
        assert(manille_mobile_rule_choose_worlds(copy, cur, n, nullptr, nullptr, extra.data(), &card,
                                                 values) == MANILLE_MOBILE_OK);
        assert(card >= 0 && ((legal_mask >> card) & 1));
        // play-out values for exactly the legal cards, or none at all (the
        // play-out needs >= 20 worlds that survive the rank caps)
        int n_values = 0;
        for (int c = 0; c < 32; ++c)
            n_values += !std::isnan(values[c]);
        for (int c = 0; c < 32; ++c)
            if (n_values)
                assert(((legal_mask >> c) & 1) == !std::isnan(values[c]));
        for (int c = 0; c < 32; ++c)
            if (n_values && ((legal_mask >> c) & 1))
                assert(values[c] >= 0 && values[c] <= 60);
        // with enough worlds the play-out runs and every legal card is valued
        std::vector<int64_t> cur2(4 * 400), init2(4 * 400);
        int32_t n2 = 0;
        assert(manille_mobile_sample_worlds(copy, 0, 400, cur2.data(), init2.data(), &n2) == MANILLE_MOBILE_OK);
        assert(manille_mobile_rule_choose_worlds(copy, cur2.data(), n2, nullptr, nullptr, nullptr, &card,
                                                 values) == MANILLE_MOBILE_OK);
        if (manille::popcount64(legal_mask) > 1)
            for (int c = 0; c < 32; ++c)
                assert(((legal_mask >> c) & 1) == !std::isnan(values[c]));
        // the fast play-out net: a tiny synthetic file in the MFPN layout,
        // hybrid (4 / 8 plies, cut or to the trick's end) and net-throughout
        // all pick legal cards
        {
            const char* path = "test_mobile_fastnet.bin";
            FILE* f = std::fopen(path, "wb");
            assert(f != nullptr);
            const int32_t hdr[4] = {1, fastpolicy::F_DIM, 4, 3};
            std::fwrite("MFPN", 1, 4, f);
            std::fwrite(hdr, 4, 4, f);
            const int n_w = 4 * fastpolicy::F_DIM + 4 + 3 * 4 + 3 + 32 * 3 + 32;
            for (int i = 0; i < n_w; ++i) {
                float w = 0.01f * float((i * 37) % 23 - 11);
                std::fwrite(&w, 4, 1, f);
            }
            std::fclose(f);
            ManilleFastNet* net = nullptr;
            assert(manille_mobile_fastnet_load("no/such/file.bin", &net) != MANILLE_MOBILE_OK);
            assert(manille_mobile_fastnet_load(path, &net) == MANILLE_MOBILE_OK && net != nullptr);
            assert(manille_mobile_fastnet_set_plies(net, -1) != MANILLE_MOBILE_OK);
            assert(manille_mobile_fastnet_set_plies(nullptr, 4) != MANILLE_MOBILE_OK);
            assert(manille_mobile_fastnet_set_trick_end(nullptr, 1) != MANILLE_MOBILE_OK);
            for (int variant : {4, 8, 104, 108, 32, 132, 0}) {
                // 100+: the net finishes the trick its plies end in
                const int plies = variant % 100;
                assert(manille_mobile_fastnet_set_plies(net, plies) == MANILLE_MOBILE_OK);
                assert(manille_mobile_fastnet_set_trick_end(net, variant >= 100) == MANILLE_MOBILE_OK);
                card = -1;
                assert(manille_mobile_rule_choose_worlds_net(copy, cur2.data(), n2, nullptr, nullptr,
                                                             nullptr, net, &card, values)
                       == MANILLE_MOBILE_OK);
                assert(card >= 0 && ((legal_mask >> card) & 1));
                if (manille::popcount64(legal_mask) > 1)
                    for (int c = 0; c < 32; ++c)
                        assert(((legal_mask >> c) & 1) == !std::isnan(values[c]));
            }
            // Also verify full-deal play through the mobile API (including late-hand
            // exact enumeration, mull exact alpha-beta endgame, and suit fast-net
            // history likelihood).
            for (int32_t trump : {1, -1}) {
                ManilleMobileGame* fg = nullptr;
                assert(manille_mobile_new_random_deal_with_trump(0, 77 + trump, trump, &fg)
                       == MANILLE_MOBILE_OK);
                assert(manille_mobile_start_playing(fg) == MANILLE_MOBILE_OK);
                assert(manille_mobile_fastnet_set_plies(net, 32) == MANILLE_MOBILE_OK);
                assert(manille_mobile_fastnet_set_trick_end(net, 1) == MANILLE_MOBILE_OK);
                float belief[96];
                for (float& b : belief) b = 1.0f / 3.0f;
                float q[32]{};
                std::vector<int64_t> cw(4 * 64), iw(4 * 64);
                for (int ply = 0; ply < 32; ++ply) {
                    int64_t lm = 0;
                    assert(manille_mobile_legal_mask(fg, &lm) == MANILLE_MOBILE_OK);
                    int32_t nw = 0, pick = -1, bs = -1;
                    assert(manille_mobile_sample_worlds_belief(
                               fg, 0, 64, belief, 0.75, cw.data(), iw.data(), &nw, &bs)
                           == MANILLE_MOBILE_OK && nw > 0 && nw <= 64);
                    // On trick 1 (ply == 0), 24 hidden cards cannot enumerate into 64 -> belief sampled.
                    if (ply == 0) assert(bs == 1);
                    // On trick 8 (ply >= 28), <= 3 hidden cards always enumerate -> not belief sampled.
                    if (ply >= 28) assert(bs == 0);
                    assert(manille_mobile_rule_choose_worlds_net_ex(
                               fg, cw.data(), nw, bs, belief, q, nullptr, net, &pick, values)
                           == MANILLE_MOBILE_OK);
                    assert(pick >= 0 && ((lm >> pick) & 1));
                    assert(manille_mobile_apply_card(fg, pick) == MANILLE_MOBILE_OK);
                }
                assert(manille_mobile_get_state(fg, &state) == MANILLE_MOBILE_OK);
                assert(state.finished == 1 && state.pts0 + state.pts1 == 60);
                manille_mobile_destroy(fg);
            }
            manille_mobile_fastnet_free(net);
            std::remove(path);
        }
        manille_mobile_destroy(copy);
        manille_mobile_destroy(game);
    }

    // Verify exact belief-weighted DP sampling distribution against exact enumeration
    // on a small public state (6 hidden cards across 3 seats: 2 cards each -> 90 worlds).
    {
        sampler::PublicState ps{};
        ps.observer = 1;
        ps.current_player = 1;
        ps.observer_hand = (int64_t(1) << 0) | (int64_t(1) << 1);
        ps.partner_hand = 0;
        // Cards 2..7 are unseen (6 cards), 8..31 already played across 6 tricks
        // (no void constraints: every player followed suit in each played trick).
        ps.trick_cards.fill(-1);
        ps.trick_players.fill(-1);
        ps.played_cards.fill(-1);
        ps.played_players.fill(-1);
        ps.n_played_cards = 24;
        for (int i = 0; i < 24; ++i) {
            ps.played_cards[i] = 8 + i;
            ps.played_players[i] = i % 4;
        }
        ps.hand_sizes = {2, 2, 2, 2};
        ps.declarer = -1;
        ps.trump_suit = -1;

        float belief[96]{};
        // Non-uniform beliefs for cards 2..7 across rel seats 0, 1, 2
        const float w_table[6][3] = {
            {0.80f, 0.15f, 0.05f},
            {0.10f, 0.70f, 0.20f},
            {0.25f, 0.25f, 0.50f},
            {0.60f, 0.30f, 0.10f},
            {0.05f, 0.85f, 0.10f},
            {0.30f, 0.20f, 0.50f},
        };
        for (int i = 0; i < 6; ++i)
            for (int r = 0; r < 3; ++r)
                belief[r * 32 + (2 + i)] = w_table[i][r];

        sampler::UniformDealSampler us;
        auto exact = us.enumerateDeals(ps, 1000);
        assert(exact.size() == 90);

        const double bw = 0.75;
        std::map<std::array<int64_t, 4>, double> target_p;
        double z_sum = 0.0;
        for (const auto& d : exact) {
            double wt = 1.0;
            for (int r = 0; r < 3; ++r) {
                const int seat = (ps.observer + 1 + r) % 4;
                int64_t m = d.hands[seat];
                while (m) {
                    int c = manille::ctz64(static_cast<unsigned long long>(m));
                    m &= m - 1;
                    wt *= std::pow(double(belief[r * 32 + c]) + 1e-9, bw);
                }
            }
            target_p[d.hands] = wt;
            z_sum += wt;
        }
        for (auto& kv : target_p) kv.second /= z_sum;

        sampler::UniformDealSampler::seedPrng(1234567);
        bool enumerated = true;
        const int N = 20000;
        auto draws = us.sampleDealsBelief(ps, N, belief, bw, /*enumerate_if_small=*/false, &enumerated);
        assert(!enumerated && static_cast<int>(draws.size()) == N);

        std::map<std::array<int64_t, 4>, int> counts;
        for (const auto& d : draws) counts[d.hands]++;

        double max_err = 0.0;
        for (const auto& kv : target_p) {
            const double emp = double(counts[kv.first]) / double(N);
            max_err = std::max(max_err, std::abs(emp - kv.second));
        }
        assert(max_err < 0.015);
    }
    return 0;
}
