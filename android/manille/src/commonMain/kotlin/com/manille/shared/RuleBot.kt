package com.manille.shared

import com.cardtable.platform.*

import kotlin.concurrent.Volatile
import kotlin.math.exp
import kotlin.math.ln1p
import kotlin.time.TimeSource

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
 * choose() blocks for the play-out: call it off the UI thread, on an
 * [Engine.copy].
 */
class RuleBot(private val platform: Platform, private val trumpQ: QModel, private val announce: AnnounceModel?) : AutoCloseable {
    private val beliefTrump = platform.beliefModel(ModelNames.BELIEF_TRUMP)

    /**
     * The no-trump (mull) belief and Q nets: ~200 MB that only a mull deal
     * uses, so they load when one needs them ([prepareMull] as soon as a
     * mull is declared) and go again at the next suit deal ([releaseMull]).
     * Touched only on the worker thread, like everything that runs the nets.
     */
    private var mullNets: Pair<BeliefModel, QModel>? = null

    private fun mull(): Pair<BeliefModel, QModel> = mullNets ?: run {
        val belief = platform.beliefModel(ModelNames.BELIEF_MULL)
        val q = try { platform.qModel(ModelNames.Q_MULL) } catch (e: Throwable) { belief.close(); throw e }
        (belief to q).also { mullNets = it }
    }

    /** Load the mull nets now (a mull was declared). On the worker. */
    fun prepareMull() { runCatching { mull() } }

    /** Free the mull nets (a suit deal). On the worker. */
    fun releaseMull() {
        mullNets?.let { (belief, q) -> belief.close(); q.close() }
        mullNets = null
    }
    /**
     * The fast learned play-out (cpp/fast_policy.h): every simulated seat
     * follows a small net for the first [netPlies] plies of the play-out, then
     * the rule policy. Per declared-deal A/B vs the rule play-out (pts/deal):
     * mull deals, 8 plies +0.68 and +0.81 (2 x 3000 deals), 4 plies ~+0.35;
     * suit deals, 4 plies ~+0.2. Cost on one desktop core ~94 ms/move at 4
     * plies, ~174 ms at 8 (rule play-out alone 8 ms).
     *
     * With [netTrickEnd] the net also finishes the trick its plies end in,
     * so it hands over on a trick boundary (4 plies -> 4..7 cards).
     *
     * One native handle per (plies, trick end) -- the handle carries both --
     * so a setting change never touches a handle a move is using. Empty when
     * the asset is missing or bad: the rule play-out then runs alone.
     */
    private fun loadFastNets(file: String): Map<Pair<Int, Boolean>, FastNet> =
        PLY_CHOICES.filter { it > 0 }
            .map { plies -> plies to TRICK_END }
            .mapNotNull { key ->
                runCatching { platform.fastNet(file, key.first, key.second) }
                    .getOrNull()?.let { key to it }
            }.toMap()

    private val fastNets: Map<Pair<Int, Boolean>, FastNet> = loadFastNets(ModelNames.FAST_NET)
    /** Optional dedicated trump-deal fast play-out net; falls back to [fastNets] when absent. */
    private val fastNetsSuit: Map<Pair<Int, Boolean>, FastNet> = loadFastNets(ModelNames.FAST_NET_SUIT)

    /** Net plies for the play-out, one of [PLY_CHOICES]; 0 = the rule play-out alone. */
    @Volatile var netPlies: Int = WHOLE_DEAL
    /** The net finishes the trick its plies end in ([TRICK_END]). */
    @Volatile var netTrickEnd: Boolean = TRICK_END

    /** True when the play-out net loaded, so [netPlies] has an effect. */
    val hasFastNet: Boolean get() = fastNets.isNotEmpty() || fastNetsSuit.isNotEmpty()

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

    fun choose(game: Engine, calls: Calls): Int = analyse(game, calls, null)

    /**
     * The card, and into [values] (32, by card) each legal card's play-out
     * value -- expected final team points, what the desktop hint shows; NaN
     * for the other cards or when the play-out did not run.
     */
    fun analyse(game: Engine, calls: Calls, values: DoubleArray?): Int {
        values?.fill(Double.NaN)
        lastTimings.fill(0)
        val state = game.state()
        val me = state.currentPlayer
        val legal = game.legalCards()
        if (legal.size == 1) return legal[0]
        val mull = state.trumpSuit < 0
        var t = TimeSource.Monotonic.markNow()
        fun lap(i: Int) { lastTimings[i] = t.elapsedNow().inWholeMilliseconds; t = TimeSource.Monotonic.markNow() }
        val key = netPlies to netTrickEnd
        val fastNet = if (!mull) (fastNetsSuit[key] ?: fastNets[key]) else fastNets[key]
        val nWorlds = if (fastNet != null) N_WORLDS_NET else N_WORLDS
        val nets = if (mull) mull() else null
        val belief = (nets?.first ?: beliefTrump).probabilities(game.observation(me)); lap(1)
        val worlds = game.sampleWorldsBelief(calls.trumpRuleKnown, nWorlds, belief); lap(0)
        val scores = (nets?.second ?: trumpQ).scoreCards(
            game.gruRaw(me), game.gruLength(me), game.gruGlobal(me), game.actionFeatures(me), legal)
        val q = FloatArray(32)
        for (s in scores) q[s.card] = s.value
        lap(2)
        val extra = if (state.doubled) doubleWeights(worlds, state, calls) else null; lap(3)
        val card = game.ruleChooseWorlds(worlds, belief, q, extra, values, fastNet); lap(4)
        return card
    }

    /** log P(the calls made | each world's dealt hands), or null if none apply. */
    private fun doubleWeights(worlds: Worlds, state: BoardState, calls: Calls): DoubleArray? {
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

        // Deduplicate identical dealt hands per seat before calling the
        // announce ONNX net (late in a deal or after enumeration, many worlds
        // share the same seat hand; mathematically identical, ~2-3x faster).
        fun equitiesDedup(hands: LongArray, rel: Int, us: Int, them: Int): Array<FloatArray> {
            val indexByHand = HashMap<Long, Int>(hands.size)
            val unique = LongArray(hands.size)
            val map = IntArray(hands.size)
            var u = 0
            for (i in hands.indices) {
                val h = hands[i]
                val existing = indexByHand[h]
                if (existing != null) {
                    map[i] = existing
                } else {
                    indexByHand[h] = u
                    unique[u] = h
                    map[i] = u
                    u++
                }
            }
            val uniqEv = net.equitiesBatch(unique.copyOf(u), decl, rel, us, them)
            return Array(hands.size) { uniqEv[map[it]] }
        }

        // meegaan: defenders asked from the dealer's left, first yes takes it;
        // scores from the defenders' own view
        val pMee = Array(2) { DoubleArray(n) }
        val qMee = Array(2) { DoubleArray(n) }
        for ((i, off) in intArrayOf(1, 3).withIndex()) {
            val ev = equitiesDedup(dealt((d + off) % 4), off, calls.declThem, calls.declUs)
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
            val ev = equitiesDedup(dealt((d + off) % 4), off, calls.declUs, calls.declThem)
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
        beliefTrump.close(); releaseMull()
        fastNets.values.forEach { it.close() }
        fastNetsSuit.values.forEach { it.close() }
    }

    companion object {
        const val N_WORLDS = 400
        /** Sampled worlds when the fast play-out net is enabled (exact belief-weighted DP + SIR + racing). */
        const val N_WORLDS_NET = 1000
        /**
         * Hand the play-out over to the rule policy on a trick boundary: the
         * net finishes the trick its plies end in, so "1 trick" = 4 plies
         * rounded up to the end of a trick. Chosen by the trump A/B (scripts/
         * declared_ab.py, tag trump_plies: k4e/k8e vs k4/k8).
         */
        const val TRICK_END = true
        /** The selectable play-out depths in plies (4 per trick; the "Bot thinking" setting). */
        val PLY_CHOICES = listOf(0, 4, 8, 12, WHOLE_DEAL)
        /** More plies than any deal has left: the net plays the whole simulated deal. */
        const val WHOLE_DEAL = 32
        // softmax temperature on the P(match win) scale (desktop decl_temp)
        private const val TEMP = 0.01

        fun logSigmoid(x: Double): Double =
            if (x >= 0) -ln1p(exp(-x)) else x - ln1p(exp(x))
    }
}
