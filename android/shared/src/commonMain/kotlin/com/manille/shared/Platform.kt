package com.manille.shared

/**
 * What the shared game needs from the platform. The rules, the encoders and
 * the play-out live in C++ (cpp/mobile, the same code on every platform);
 * the nets run on the platform's ONNX Runtime; files sit in the app's
 * private storage. Android and the desktop: com.manille.jvm.JvmPlatform
 * (shared/src/jvmShared), set up by app/AndroidPlatform.kt and
 * desktop/Main.kt. iOS: to come.
 */
interface Platform {
    /** A fresh engine; start it with [Engine.newDeal]. */
    fun newEngine(): Engine
    /** Index of the winning card of a (partial) trick, by the engine's rule. */
    fun trickWinner(cards: List<Int>, trump: Int): Int

    /** The nets by file name ([ModelNames]); each throws when its file is missing or bad. */
    fun qModel(file: String): QModel
    fun beliefModel(file: String): BeliefModel
    fun announceModel(): AnnounceModel
    fun declarationModel(): DeclarationModel
    /** The fast play-out net set to [plies] (and [trickEnd]); null when it cannot load. */
    fun fastNet(file: String, plies: Int, trickEnd: Boolean): FastNet?

    val storage: Storage
    /** Diagnostics (Android: logcat). */
    fun log(message: String)
    /** Wall-clock time, milliseconds since 1970 (the statistics' result dates). */
    fun epochMillis(): Long
}

/** The model files shipped with the app (release models-v1). */
object ModelNames {
    /** Trump-deal QGRU blueprint; the no-trump one is [Q_MULL]. */
    const val Q_TRUMP = "manille_qgru_v1.onnx"
    const val Q_MULL = "manille_qgru_mull_v1.onnx"
    const val BELIEF_TRUMP = "manille_belief_v1.onnx"
    const val BELIEF_MULL = "manille_belief_mull_v1.onnx"
    const val ANNOUNCE = "manille_announce_v1.onnx"
    const val DECLARATION = "manille_decl_v1.onnx"
    const val FAST_NET = "manille_fast_policy_v1.bin"
    /** Optional dedicated trump-deal fast play-out net (falls back to [FAST_NET] when absent). */
    const val FAST_NET_SUIT = "manille_fast_policy_suit_v1.bin"
}

/** Small named text files in the app's private storage. */
interface Storage {
    fun read(name: String): String?
    /** Replaces the file atomically (write aside, then rename). */
    fun write(name: String, text: String)
    fun delete(name: String)
    fun exists(name: String): Boolean
    fun rename(from: String, to: String): Boolean
}

data class BoardState(
    val currentPlayer: Int,
    val trumpSuit: Int,
    val declarer: Int,
    val team0Points: Int,
    val team1Points: Int,
    val team0Tricks: Int,
    val team1Tricks: Int,
    val multiplier: Int,
    val doubled: Boolean,
    val redoubled: Boolean,
    val announcementOpen: Boolean,
    val finished: Boolean,
    val trickCards: List<Int>,
    val trickPlayers: List<Int>,
)

/** Sampled determinizations for the player to move: 4 hand masks per world. */
class Worlds(
    val n: Int,
    val current: LongArray,
    val dealt: LongArray,
    val beliefSampled: Boolean = false,
)

/** An opaque loaded fast play-out net (cpp/fast_policy.h). */
interface FastNet : AutoCloseable

/** The C++ engine (cpp/mobile/manille_mobile.h): rules, encoders, rule agent. */
interface Engine : AutoCloseable {
    /** Full engine state for a saved game (see [restore]). */
    fun snapshot(): IntArray
    /** Replace the current deal with a saved [snapshot]; throws if it cannot. */
    fun restore(snapshot: IntArray)
    /** An independent copy (for a worker thread); close it when done. */
    fun copy(): Engine
    fun newDeal(dealer: Int, seed: Long, trump: Int? = null)
    fun state(): BoardState
    fun handMask(player: Int): Long
    fun legalCards(): IntArray
    fun applyCard(card: Int)
    fun announceDouble(player: Int)
    fun announceRedouble(player: Int)
    fun startPlaying()
    fun handScore(team: Int): Int

    fun actionFeatures(observer: Int): FloatArray
    fun gruRaw(observer: Int): FloatArray
    fun gruLength(observer: Int): Long
    fun gruGlobal(observer: Int): FloatArray
    /** Encoder observation of [observer] (the belief nets' input). */
    fun observation(observer: Int): FloatArray

    fun sampleWorlds(trumpRuleKnown: Boolean, nWorlds: Int): Worlds
    /**
     * Exact belief-weighted DP world sampling (UniformDealSampler::sampleDealsBelief):
     * draws worlds directly from the belief distribution when possible, falling back
     * to exact enumeration or uniform sampling otherwise.
     */
    fun sampleWorldsBelief(
        trumpRuleKnown: Boolean, nWorlds: Int, belief: FloatArray, beliefWeight: Double = 0.75,
    ): Worlds = sampleWorlds(trumpRuleKnown, nWorlds)
    /**
     * Rule agent move (cpp/rule_agent.h) on [worlds]; [extra] = optional
     * log-weight per world. [outValues] (32, by card) receives each legal
     * card's play-out value; NaN elsewhere or when it did not run. Blocks for
     * the play-out: never on the UI thread.
     */
    fun ruleChooseWorlds(
        worlds: Worlds, belief: FloatArray?, q: FloatArray?, extra: DoubleArray?,
        outValues: DoubleArray?, fastNet: FastNet?,
    ): Int
}

data class CardScore(val card: Int, val value: Float)

/** The recurrent QGRU(s, a) blueprint. */
interface QModel : AutoCloseable {
    /** Raw Q values of the legal cards (hints, the rule agent's blend, the fallback bot). */
    fun scoreCards(raw: FloatArray, nRaw: Long, glob: FloatArray, allActions: FloatArray, legalCards: IntArray): List<CardScore>
}

fun QModel.chooseCard(raw: FloatArray, nRaw: Long, glob: FloatArray, allActions: FloatArray, legalCards: IntArray): Int {
    require(legalCards.isNotEmpty()) { "Cannot choose from no legal cards." }
    return scoreCards(raw, nRaw, glob, allActions, legalCards).maxBy { it.value }.card
}

/** Encoder observation -> P(seat observer+1+r holds card c), 3 x 32 floats. */
interface BeliefModel : AutoCloseable {
    fun probabilities(observation: FloatArray): FloatArray
}

/**
 * The meegaan / terug equity policy (learn.decision_layer.EquityPolicy).
 * Scores from the deciding team's own view; rel = (seat - dealer) % 4.
 */
interface AnnounceModel : AutoCloseable {
    /** `[meegaan_pass, meegaan_call, terug_pass, terug_call]` match equities. */
    fun equities(handMask: Long, declaration: Int, rel: Int, us: Int, them: Int): FloatArray
    /** [equities] for many hands at once: row i = hands[i]. */
    fun equitiesBatch(hands: LongArray, declaration: Int, rel: Int, us: Int, them: Int): Array<FloatArray>
}

fun AnnounceModel.meegaan(handMask: Long, declaration: Int, rel: Int, us: Int, them: Int): Boolean {
    val ev = equities(handMask, declaration, rel, us, them)
    return ev[1] > ev[0]
}

fun AnnounceModel.terug(handMask: Long, declaration: Int, rel: Int, us: Int, them: Int): Boolean {
    val ev = equities(handMask, declaration, rel, us, them)
    return ev[3] > ev[2]
}

/** The score-aware declaration policy: equity of trump 0..3 and mull (4). */
interface DeclarationModel : AutoCloseable {
    fun equities(handMask: Long, us: Int, them: Int): FloatArray
}
