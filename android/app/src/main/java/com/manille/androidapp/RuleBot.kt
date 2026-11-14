package com.manille.androidapp

import android.content.Context
import java.io.Closeable
import kotlin.math.exp
import kotlin.math.ln1p

/**
 * The rule agent (cpp/rule_agent.cpp, the desktop GUI's "Rule-based, phone
 * config" bot) for the app. The C++ side samples the worlds and plays them
 * out; the nets run here on the same observation:
 *  - belief net (trump or no-trump file): who holds which unseen card
 *  - Q GRU (trump or no-trump): value of each legal card, blended in
 *  - announce net: on a doubled deal, how likely each sampled world's hands
 *    are to have made the meegaan / terug calls that were made (and the
 *    passes before them), as the bots' own announce policy decides them.
 *    Same formula as learn/rule_agent.py::decl_log_weights with
 *    double_tilt='called', checked against it on the served policy.
 *
 * choose() blocks for the play-out: call it off the UI thread, on a
 * [NativeGame.copy].
 */
class RuleBot(context: Context, private val trumpQ: QNet, private val announce: AnnounceNet?) : Closeable {
    private val beliefTrump = BeliefNet(context, BeliefNet.TRUMP_FILE)
    private val beliefMull = BeliefNet(context, BeliefNet.MULL_FILE)
    private val mullQ = QNet(context, QNet.MULL_MODEL_FILE)
    /**
     * The fast learned play-out (cpp/fast_policy.h): every simulated seat
     * follows a small net for the first plies of the play-out, then the rule
     * policy. Per declared-deal A/B vs the rule play-out (pts/deal):
     * suit deals, 4 plies ~+0.2 (seeds 99/100); mull deals, 4 plies ~+0.35
     * (3 seeds, 9000 deals), 8 plies +0.68 and +0.81 (seeds 102/103, 3000
     * deals each, both CIs clear of 0). Cost on
     * one desktop core ~94 ms/move at 4 plies, ~174 ms at 8 (rule play-out
     * 8 ms). Two handles of the same 0.5 MB net, one per ply count. 0 when
     * the asset is missing or bad: the rule play-out then runs alone.
     */
    private fun loadFastNet(context: Context, plies: Int): Long = if (!FAST_PLAYOUT) 0L else runCatching {
        val net = NativeGame.nativeFastNetLoad(
            ModelFiles.install(context, FAST_NET_FILE, "Unzip manille-app-models.zip (release models-v1) into android/app/src/main/assets/models, then rebuild.").path)
        if (net != 0L) NativeGame.nativeFastNetSetPlies(net, plies)
        net
    }.getOrDefault(0L)
    private val fastNetSuit: Long = loadFastNet(context, FAST_PLIES_SUIT)
    private val fastNetMull: Long = loadFastNet(context, FAST_PLIES_MULL)

    /**
     * The deal's calls. [declUs]/[declThem] = the declaring team's and the
     * other team's match score when the calls were made. [humanSeats]: calls
     * from these seats carry no likelihood (a human's calls are not the
     * policy's).
     */
    data class Calls(
        val dealer: Int,
        val meegaanSeat: Int?,
        val terugSeat: Int?,
        val declUs: Int,
        val declThem: Int,
        val trumpRuleKnown: Boolean,
        val humanSeats: Set<Int>,
    )

    /** Milliseconds of the last choose(): sample, belief, Q, doubles, play-out. */
    val lastTimings = LongArray(5)

    fun choose(game: NativeGame, calls: Calls): Int = analyse(game, calls, null)

    /**
     * The card, and into [values] (32, by card) each legal card's play-out
     * value -- expected final team points, what the desktop hint shows; NaN
     * for the other cards or when the play-out did not run.
     */
    fun analyse(game: NativeGame, calls: Calls, values: DoubleArray?): Int {
        values?.fill(Double.NaN)
        lastTimings.fill(0)
        val state = game.state()
        val me = state.currentPlayer
        val legal = game.legalCards()
        if (legal.size == 1) return legal[0]
        val mull = state.trumpSuit < 0
        var t = System.nanoTime()
        fun lap(i: Int) { val now = System.nanoTime(); lastTimings[i] = (now - t) / 1_000_000; t = now }
        val worlds = game.sampleWorlds(calls.trumpRuleKnown, N_WORLDS); lap(0)
        val belief = (if (mull) beliefMull else beliefTrump).probabilities(game.observation(me)); lap(1)
        val scores = (if (mull) mullQ else trumpQ).scoreCards(
            game.gruRaw(me), game.gruLength(me), game.gruGlobal(me), game.actionFeatures(me), legal)
        val q = FloatArray(32)
        for (s in scores) q[s.card] = s.value
        lap(2)
        val extra = if (state.doubled) doubleWeights(worlds, state, calls) else null; lap(3)
        val card = game.ruleChooseWorlds(
            worlds, belief, q, extra, values, if (mull) fastNetMull else fastNetSuit); lap(4)
        return card
    }

    /** log P(the calls made | each world's dealt hands), or null if none apply. */
    private fun doubleWeights(worlds: NativeGame.Worlds, state: BoardState, calls: Calls): DoubleArray? {
        val net = announce ?: return null
        val d = calls.dealer
        val n = worlds.n
        if (n == 0) return null
        val decl = if (state.trumpSuit < 0) 4 else state.trumpSuit
        val defenders = intArrayOf((d + 1) % 4, (d + 3) % 4)
        if (defenders.any { it in calls.humanSeats }) return null
        val mee = calls.meegaanSeat ?: return null
        val logw = DoubleArray(n)

        fun dealt(seat: Int) = LongArray(n) { worlds.dealt[4 * it + seat] }

        // meegaan: defenders asked from the dealer's left, first yes takes it;
        // scores from the defenders' own view
        val pMee = Array(2) { DoubleArray(n) }
        val qMee = Array(2) { DoubleArray(n) }
        for ((i, off) in intArrayOf(1, 3).withIndex()) {
            val ev = net.equitiesBatch(dealt((d + off) % 4), decl, off, calls.declThem, calls.declUs)
            for (w in 0 until n) {
                val delta = (ev[w][1] - ev[w][0]).toDouble() / TEMP
                pMee[i][w] = logSigmoid(delta)
                qMee[i][w] = logSigmoid(-delta)
            }
        }
        for (w in 0 until n)
            logw[w] += if (mee == (d + 1) % 4) pMee[0][w] else qMee[0][w] + pMee[1][w]

        // terug: the declaring side answers, dealer first; its own view
        if (intArrayOf(d, (d + 2) % 4).any { it in calls.humanSeats }) return logw
        val pTer = Array(2) { DoubleArray(n) }
        val qTer = Array(2) { DoubleArray(n) }
        for ((i, off) in intArrayOf(0, 2).withIndex()) {
            val ev = net.equitiesBatch(dealt((d + off) % 4), decl, off, calls.declUs, calls.declThem)
            for (w in 0 until n) {
                val delta = (ev[w][3] - ev[w][2]).toDouble() / TEMP
                pTer[i][w] = logSigmoid(delta)
                qTer[i][w] = logSigmoid(-delta)
            }
        }
        for (w in 0 until n) {
            logw[w] += when {
                !state.redoubled -> qTer[0][w] + qTer[1][w]
                calls.terugSeat == d -> pTer[0][w]
                else -> qTer[0][w] + pTer[1][w]
            }
        }
        return logw
    }

    override fun close() {
        beliefTrump.close(); beliefMull.close(); mullQ.close()
        if (fastNetSuit != 0L) NativeGame.nativeFastNetFree(fastNetSuit)
        if (fastNetMull != 0L) NativeGame.nativeFastNetFree(fastNetMull)
    }

    companion object {
        const val N_WORLDS = 400
        // off until the hybrid play-out is timed on a phone (budget < 0.5 s/move)
        const val FAST_PLAYOUT = false
        const val FAST_NET_FILE = "manille_fast_policy_v1.bin"
        const val FAST_PLIES_SUIT = 4
        const val FAST_PLIES_MULL = 8
        // softmax temperature on the P(match win) scale (desktop decl_temp)
        private const val TEMP = 0.01

        fun logSigmoid(x: Double): Double =
            if (x >= 0) -ln1p(exp(-x)) else x - ln1p(exp(x))
    }
}
