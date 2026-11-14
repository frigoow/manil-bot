#include <jni.h>

#include "mobile/manille_mobile.h"

#include <array>
#include <cstdint>
#include <vector>

namespace {

ManilleMobileGame* game_from(jlong handle) {
    return reinterpret_cast<ManilleMobileGame*>(handle);
}

void throw_engine_error(JNIEnv* env) {
    jclass error_class = env->FindClass("java/lang/IllegalStateException");
    if (error_class != nullptr)
        env->ThrowNew(error_class, manille_mobile_last_error());
}

bool check(int status, JNIEnv* env) {
    if (status == MANILLE_MOBILE_OK)
        return true;
    throw_engine_error(env);
    return false;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_manille_androidapp_NativeGame_nativeNewDeal(
    JNIEnv* env,
    jclass,
    jint dealer,
    jlong seed
) {
    ManilleMobileGame* game = nullptr;
    if (!check(manille_mobile_new_random_deal(dealer, static_cast<uint64_t>(seed), &game), env))
        return 0;
    return reinterpret_cast<jlong>(game);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_manille_androidapp_NativeGame_nativeNewDealWithTrump(
    JNIEnv* env, jclass, jint dealer, jlong seed, jint trump
) {
    ManilleMobileGame* game = nullptr;
    if (!check(manille_mobile_new_random_deal_with_trump(
            dealer, static_cast<uint64_t>(seed), trump, &game), env))
        return 0;
    return reinterpret_cast<jlong>(game);
}

extern "C" JNIEXPORT void JNICALL
Java_com_manille_androidapp_NativeGame_nativeClose(JNIEnv*, jclass, jlong handle) {
    manille_mobile_destroy(game_from(handle));
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_manille_androidapp_NativeGame_nativeState(JNIEnv* env, jclass, jlong handle) {
    ManilleMobileState state{};
    if (!check(manille_mobile_get_state(game_from(handle), &state), env))
        return nullptr;
    std::array<jint, 22> values{
        state.current_player, state.trump_suit, state.declarer,
        state.pts0, state.pts1, state.tricks_won0, state.tricks_won1,
        state.multiplier, state.doubled, state.redoubled, state.doubling_phase,
        state.finished, state.n_trick, state.n_played_cards,
        state.trick_cards[0], state.trick_cards[1], state.trick_cards[2], state.trick_cards[3],
        state.trick_players[0], state.trick_players[1], state.trick_players[2], state.trick_players[3],
    };
    jintArray output = env->NewIntArray(static_cast<jsize>(values.size()));
    if (output != nullptr)
        env->SetIntArrayRegion(output, 0, static_cast<jsize>(values.size()), values.data());
    return output;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_manille_androidapp_NativeGame_nativeHandMask(
    JNIEnv* env, jclass, jlong handle, jint player
) {
    int64_t mask = 0;
    if (!check(manille_mobile_hand_mask(game_from(handle), player, &mask), env))
        return 0;
    return static_cast<jlong>(mask);
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_manille_androidapp_NativeGame_nativeLegalCards(JNIEnv* env, jclass, jlong handle) {
    float obs[MANILLE_MOBILE_OBS_DIM]{};
    float action[MANILLE_MOBILE_MAX_LEGAL * MANILLE_MOBILE_ACT_DIM]{};
    int32_t cards[MANILLE_MOBILE_MAX_LEGAL]{};
    int32_t count = 0;
    ManilleMobileState state{};
    if (!check(manille_mobile_get_state(game_from(handle), &state), env)
        || !check(manille_mobile_encode_q(game_from(handle), state.current_player, obs, action, cards, &count), env))
        return nullptr;
    jintArray output = env->NewIntArray(count);
    if (output != nullptr)
        env->SetIntArrayRegion(output, 0, count, cards);
    return output;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_manille_androidapp_NativeGame_nativeObservation(
    JNIEnv* env, jclass, jlong handle, jint observer
) {
    std::array<float, MANILLE_MOBILE_OBS_DIM> obs{};
    std::array<float, MANILLE_MOBILE_MAX_LEGAL * MANILLE_MOBILE_ACT_DIM> action{};
    std::array<int32_t, MANILLE_MOBILE_MAX_LEGAL> cards{};
    int32_t count = 0;
    if (!check(manille_mobile_encode_q(game_from(handle), observer, obs.data(), action.data(), cards.data(), &count), env))
        return nullptr;
    jfloatArray output = env->NewFloatArray(static_cast<jsize>(obs.size()));
    if (output != nullptr)
        env->SetFloatArrayRegion(output, 0, static_cast<jsize>(obs.size()), obs.data());
    return output;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_manille_androidapp_NativeGame_nativeActionFeatures(
    JNIEnv* env, jclass, jlong handle, jint observer
) {
    std::array<float, MANILLE_MOBILE_OBS_DIM> obs{};
    std::array<float, MANILLE_MOBILE_MAX_LEGAL * MANILLE_MOBILE_ACT_DIM> action{};
    std::array<int32_t, MANILLE_MOBILE_MAX_LEGAL> cards{};
    int32_t count = 0;
    if (!check(manille_mobile_encode_q(game_from(handle), observer, obs.data(), action.data(), cards.data(), &count), env))
        return nullptr;
    jfloatArray output = env->NewFloatArray(static_cast<jsize>(action.size()));
    if (output != nullptr)
        env->SetFloatArrayRegion(output, 0, static_cast<jsize>(action.size()), action.data());
    return output;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_manille_androidapp_NativeGame_nativeGruRaw(
    JNIEnv* env, jclass, jlong handle, jint observer
) {
    std::array<float, MANILLE_MOBILE_MAX_PLAYS * MANILLE_MOBILE_RAW_TOKEN_DIM> raw{};
    std::array<float, MANILLE_MOBILE_GLOBAL_DIM> glob{};
    std::array<float, MANILLE_MOBILE_MAX_LEGAL * MANILLE_MOBILE_ACT_DIM> action{};
    std::array<int32_t, MANILLE_MOBILE_MAX_LEGAL> cards{};
    int64_t n_raw = 0;
    int32_t n_legal = 0;
    if (!check(manille_mobile_encode_gru(
            game_from(handle), observer, raw.data(), &n_raw, glob.data(), action.data(),
            cards.data(), &n_legal), env))
        return nullptr;
    jfloatArray output = env->NewFloatArray(static_cast<jsize>(raw.size()));
    if (output != nullptr)
        env->SetFloatArrayRegion(output, 0, static_cast<jsize>(raw.size()), raw.data());
    return output;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_manille_androidapp_NativeGame_nativeGruLength(
    JNIEnv* env, jclass, jlong handle, jint observer
) {
    std::array<float, MANILLE_MOBILE_MAX_PLAYS * MANILLE_MOBILE_RAW_TOKEN_DIM> raw{};
    std::array<float, MANILLE_MOBILE_GLOBAL_DIM> glob{};
    std::array<float, MANILLE_MOBILE_MAX_LEGAL * MANILLE_MOBILE_ACT_DIM> action{};
    std::array<int32_t, MANILLE_MOBILE_MAX_LEGAL> cards{};
    int64_t n_raw = 0;
    int32_t n_legal = 0;
    if (!check(manille_mobile_encode_gru(
            game_from(handle), observer, raw.data(), &n_raw, glob.data(), action.data(),
            cards.data(), &n_legal), env))
        return 0;
    return static_cast<jlong>(n_raw);
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_manille_androidapp_NativeGame_nativeGruGlobal(
    JNIEnv* env, jclass, jlong handle, jint observer
) {
    std::array<float, MANILLE_MOBILE_MAX_PLAYS * MANILLE_MOBILE_RAW_TOKEN_DIM> raw{};
    std::array<float, MANILLE_MOBILE_GLOBAL_DIM> glob{};
    std::array<float, MANILLE_MOBILE_MAX_LEGAL * MANILLE_MOBILE_ACT_DIM> action{};
    std::array<int32_t, MANILLE_MOBILE_MAX_LEGAL> cards{};
    int64_t n_raw = 0;
    int32_t n_legal = 0;
    if (!check(manille_mobile_encode_gru(
            game_from(handle), observer, raw.data(), &n_raw, glob.data(), action.data(),
            cards.data(), &n_legal), env))
        return nullptr;
    jfloatArray output = env->NewFloatArray(static_cast<jsize>(glob.size()));
    if (output != nullptr)
        env->SetFloatArrayRegion(output, 0, static_cast<jsize>(glob.size()), glob.data());
    return output;
}

extern "C" JNIEXPORT void JNICALL
Java_com_manille_androidapp_NativeGame_nativeApplyCard(
    JNIEnv* env, jclass, jlong handle, jint card
) {
    check(manille_mobile_apply_card(game_from(handle), card), env);
}

extern "C" JNIEXPORT void JNICALL
Java_com_manille_androidapp_NativeGame_nativeAnnounceDouble(
    JNIEnv* env, jclass, jlong handle, jint player
) {
    check(manille_mobile_announce_double(game_from(handle), player), env);
}

extern "C" JNIEXPORT void JNICALL
Java_com_manille_androidapp_NativeGame_nativeAnnounceRedouble(
    JNIEnv* env, jclass, jlong handle, jint player
) {
    check(manille_mobile_announce_redouble(game_from(handle), player), env);
}

extern "C" JNIEXPORT void JNICALL
Java_com_manille_androidapp_NativeGame_nativeStartPlaying(JNIEnv* env, jclass, jlong handle) {
    check(manille_mobile_start_playing(game_from(handle)), env);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_manille_androidapp_NativeGame_nativeHandScore(
    JNIEnv* env, jclass, jlong handle, jint team
) {
    int32_t score = 0;
    if (!check(manille_mobile_hand_score(game_from(handle), team, &score), env))
        return 0;
    return score;
}

// Rule agent move (cpp/rule_agent.h) for the player to move. belief: 96
// floats (BeliefNet on nativeObservation) or null; q: 32 floats by card
// (QNet values of the legal cards) or null.
extern "C" JNIEXPORT jint JNICALL
Java_com_manille_androidapp_NativeGame_nativeRuleChoose(
    JNIEnv* env, jclass, jlong handle, jfloatArray belief, jfloatArray q,
    jboolean trump_rule_known, jint n_worlds
) {
    std::array<float, 96> b{};
    std::array<float, 32> qv{};
    const float* bp = nullptr;
    const float* qp = nullptr;
    if (belief != nullptr) {
        if (env->GetArrayLength(belief) != 96) {
            jclass error_class = env->FindClass("java/lang/IllegalArgumentException");
            if (error_class != nullptr)
                env->ThrowNew(error_class, "belief must hold 3 x 32 floats");
            return -1;
        }
        env->GetFloatArrayRegion(belief, 0, 96, b.data());
        bp = b.data();
    }
    if (q != nullptr) {
        if (env->GetArrayLength(q) != 32) {
            jclass error_class = env->FindClass("java/lang/IllegalArgumentException");
            if (error_class != nullptr)
                env->ThrowNew(error_class, "q must hold 32 floats indexed by card");
            return -1;
        }
        env->GetFloatArrayRegion(q, 0, 32, qv.data());
        qp = qv.data();
    }
    int32_t card = -1;
    if (!check(manille_mobile_rule_choose(game_from(handle), bp, qp,
                                          trump_rule_known ? 1 : 0, n_worlds, &card), env))
        return -1;
    return card;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_manille_androidapp_NativeGame_nativeClone(JNIEnv* env, jclass, jlong handle) {
    ManilleMobileGame* copy = nullptr;
    if (!check(manille_mobile_clone(game_from(handle), &copy), env))
        return 0;
    return reinterpret_cast<jlong>(copy);
}

// [4 x n current hands, then 4 x n dealt hands], n = worlds actually drawn.
extern "C" JNIEXPORT jlongArray JNICALL
Java_com_manille_androidapp_NativeGame_nativeSampleWorlds(
    JNIEnv* env, jclass, jlong handle, jboolean trump_rule_known, jint n_worlds
) {
    if (n_worlds < 0)
        n_worlds = 0;
    std::vector<int64_t> cur(4 * static_cast<size_t>(n_worlds)), init(cur.size());
    int32_t n = 0;
    if (!check(manille_mobile_sample_worlds(game_from(handle), trump_rule_known ? 1 : 0, n_worlds,
                                            cur.data(), init.data(), &n), env))
        return nullptr;
    jlongArray output = env->NewLongArray(8 * n);
    if (output != nullptr) {
        env->SetLongArrayRegion(output, 0, 4 * n, reinterpret_cast<const jlong*>(cur.data()));
        env->SetLongArrayRegion(output, 4 * n, 4 * n, reinterpret_cast<const jlong*>(init.data()));
    }
    return output;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_manille_androidapp_NativeGame_nativeRuleChooseWorlds(
    JNIEnv* env, jclass, jlong handle, jlongArray worlds, jint n_worlds,
    jfloatArray belief, jfloatArray q, jdoubleArray extra, jdoubleArray out_values, jlong fast_net
) {
    auto fail_arg = [&](const char* message) {
        jclass error_class = env->FindClass("java/lang/IllegalArgumentException");
        if (error_class != nullptr)
            env->ThrowNew(error_class, message);
        return -1;
    };
    if (n_worlds < 0 || (n_worlds > 0 && (worlds == nullptr || env->GetArrayLength(worlds) < 4 * n_worlds)))
        return fail_arg("worlds must hold 4 x n_worlds hand masks");
    std::vector<int64_t> w(4 * static_cast<size_t>(n_worlds));
    if (n_worlds > 0)
        env->GetLongArrayRegion(worlds, 0, 4 * n_worlds, reinterpret_cast<jlong*>(w.data()));
    std::array<float, 96> b{};
    std::array<float, 32> qv{};
    std::vector<double> ex;
    if (belief != nullptr) {
        if (env->GetArrayLength(belief) != 96)
            return fail_arg("belief must hold 3 x 32 floats");
        env->GetFloatArrayRegion(belief, 0, 96, b.data());
    }
    if (q != nullptr) {
        if (env->GetArrayLength(q) != 32)
            return fail_arg("q must hold 32 floats indexed by card");
        env->GetFloatArrayRegion(q, 0, 32, qv.data());
    }
    if (extra != nullptr) {
        if (env->GetArrayLength(extra) != n_worlds)
            return fail_arg("extra must hold one log-weight per world");
        ex.resize(static_cast<size_t>(n_worlds));
        env->GetDoubleArrayRegion(extra, 0, n_worlds, ex.data());
    }
    if (out_values != nullptr && env->GetArrayLength(out_values) != 32)
        return fail_arg("out_values must hold 32 doubles");
    int32_t card = -1;
    std::array<double, 32> values{};
    if (!check(manille_mobile_rule_choose_worlds_net(
            game_from(handle), w.data(), n_worlds, belief ? b.data() : nullptr,
            q ? qv.data() : nullptr, extra ? ex.data() : nullptr,
            reinterpret_cast<const ManilleFastNet*>(fast_net), &card,
            out_values ? values.data() : nullptr), env))
        return -1;
    if (out_values != nullptr)
        env->SetDoubleArrayRegion(out_values, 0, 32, values.data());
    return card;
}

// The fast learned play-out net (manille_fast_policy_v1.bin): a native
// handle shared by every rule-agent call, 0 when it failed to load.
extern "C" JNIEXPORT jlong JNICALL
Java_com_manille_androidapp_NativeGame_nativeFastNetLoad(JNIEnv* env, jclass, jstring path) {
    const char* p = env->GetStringUTFChars(path, nullptr);
    ManilleFastNet* net = nullptr;
    const int status = manille_mobile_fastnet_load(p, &net);
    env->ReleaseStringUTFChars(path, p);
    if (!check(status, env))
        return 0;
    return reinterpret_cast<jlong>(net);
}

extern "C" JNIEXPORT void JNICALL
Java_com_manille_androidapp_NativeGame_nativeFastNetFree(JNIEnv*, jclass, jlong net) {
    manille_mobile_fastnet_free(reinterpret_cast<ManilleFastNet*>(net));
}

// Net-guided plies before the rule policy takes over (default 4); set it
// before the handle is shared with worker threads.
extern "C" JNIEXPORT void JNICALL
Java_com_manille_androidapp_NativeGame_nativeFastNetSetPlies(JNIEnv* env, jclass, jlong net,
                                                             jint plies) {
    check(manille_mobile_fastnet_set_plies(reinterpret_cast<ManilleFastNet*>(net), plies), env);
}

// Full engine snapshot for saved games: the 22 nativeState values, then the
// 32 played cards and their 32 players, then the 4 current hands (as 8 ints:
// low, high 32 bits per seat). nativeRestore takes the same array back.
extern "C" JNIEXPORT jintArray JNICALL
Java_com_manille_androidapp_NativeGame_nativeSnapshot(JNIEnv* env, jclass, jlong handle) {
    ManilleMobileState s{};
    if (!check(manille_mobile_get_state(game_from(handle), &s), env))
        return nullptr;
    std::vector<jint> v{
        s.current_player, s.trump_suit, s.declarer, s.pts0, s.pts1, s.tricks_won0, s.tricks_won1,
        s.multiplier, s.doubled, s.redoubled, s.doubling_phase, s.finished, s.n_trick,
        s.n_played_cards,
    };
    for (int i = 0; i < 4; ++i) v.push_back(s.trick_cards[i]);
    for (int i = 0; i < 4; ++i) v.push_back(s.trick_players[i]);
    for (int i = 0; i < 32; ++i) v.push_back(s.played_cards[i]);
    for (int i = 0; i < 32; ++i) v.push_back(s.played_players[i]);
    for (int p = 0; p < 4; ++p) {
        int64_t mask = 0;
        if (!check(manille_mobile_hand_mask(game_from(handle), p, &mask), env))
            return nullptr;
        v.push_back(static_cast<jint>(mask & 0xFFFFFFFFLL));
        v.push_back(static_cast<jint>((mask >> 32) & 0xFFFFFFFFLL));
    }
    jintArray output = env->NewIntArray(static_cast<jsize>(v.size()));
    if (output != nullptr)
        env->SetIntArrayRegion(output, 0, static_cast<jsize>(v.size()), v.data());
    return output;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_manille_androidapp_NativeGame_nativeRestore(JNIEnv* env, jclass, jintArray snapshot) {
    const jsize n = snapshot ? env->GetArrayLength(snapshot) : 0;
    if (n != 14 + 8 + 64 + 8) {
        jclass error_class = env->FindClass("java/lang/IllegalArgumentException");
        if (error_class != nullptr)
            env->ThrowNew(error_class, "snapshot must hold 94 ints");
        return 0;
    }
    std::vector<jint> v(static_cast<size_t>(n));
    env->GetIntArrayRegion(snapshot, 0, n, v.data());
    ManilleMobileState s{};
    s.current_player = v[0]; s.trump_suit = v[1]; s.declarer = v[2]; s.pts0 = v[3]; s.pts1 = v[4];
    s.tricks_won0 = v[5]; s.tricks_won1 = v[6]; s.multiplier = v[7]; s.doubled = v[8];
    s.redoubled = v[9]; s.doubling_phase = v[10]; s.finished = v[11]; s.n_trick = v[12];
    s.n_played_cards = v[13];
    for (int i = 0; i < 4; ++i) { s.trick_cards[i] = v[14 + i]; s.trick_players[i] = v[18 + i]; }
    for (int i = 0; i < 32; ++i) { s.played_cards[i] = v[22 + i]; s.played_players[i] = v[54 + i]; }
    int64_t hands[4];
    for (int p = 0; p < 4; ++p)
        hands[p] = (static_cast<int64_t>(static_cast<uint32_t>(v[86 + 2 * p])))
                 | (static_cast<int64_t>(static_cast<uint32_t>(v[87 + 2 * p])) << 32);
    ManilleMobileGame* game = nullptr;
    if (!check(manille_mobile_create(hands, &s, &game), env))
        return 0;
    return reinterpret_cast<jlong>(game);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_manille_androidapp_NativeGame_nativeTrickWinner(JNIEnv* env, jclass, jintArray cards, jint trump) {
    const jsize n = cards ? env->GetArrayLength(cards) : 0;
    int32_t c[4] = {-1, -1, -1, -1};
    if (n < 1 || n > 4) {
        jclass error_class = env->FindClass("java/lang/IllegalArgumentException");
        if (error_class != nullptr)
            env->ThrowNew(error_class, "a trick holds 1..4 cards");
        return -1;
    }
    env->GetIntArrayRegion(cards, 0, n, reinterpret_cast<jint*>(c));
    int32_t idx = -1;
    if (!check(manille_mobile_trick_winner(c, n, trump, &idx), env))
        return -1;
    return idx;
}
