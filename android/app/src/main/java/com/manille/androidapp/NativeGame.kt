package com.manille.androidapp

import com.manille.shared.*
import java.io.Closeable

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

/** Thin Kotlin ownership wrapper; rules and encoders remain in C++. */
class NativeGame() : Closeable {
    private var handle: Long = 0

    private constructor(handle: Long) : this() { this.handle = handle }

    /** Full engine state for a saved game (see [restore]). */
    fun snapshot(): IntArray = nativeSnapshot(requireHandle())

    /** Replace the current deal with a saved [snapshot]. */
    fun restore(snapshot: IntArray) {
        val restored = nativeRestore(snapshot)
        check(restored != 0L) { "Could not restore the saved game." }
        close()
        handle = restored
    }

    /** An independent copy (for a worker thread); close it when done. */
    fun copy(): NativeGame = NativeGame(nativeClone(requireHandle()))

    /** Sampled determinizations for the player to move. */
    class Worlds(val n: Int, val current: LongArray, val dealt: LongArray)

    fun sampleWorlds(trumpRuleKnown: Boolean, nWorlds: Int = 400): Worlds {
        val raw = nativeSampleWorlds(requireHandle(), trumpRuleKnown, nWorlds)
        val n = raw.size / 8
        return Worlds(n, raw.copyOfRange(0, 4 * n), raw.copyOfRange(4 * n, 8 * n))
    }

    /**
     * Rule agent on given worlds; [extra] = optional log-weight per world.
     * [outValues] (32, by card) receives each legal card's play-out value
     * (expected final team points); NaN elsewhere or when it did not run.
     */
    fun ruleChooseWorlds(
        worlds: Worlds, belief: FloatArray?, q: FloatArray?, extra: DoubleArray?,
        outValues: DoubleArray? = null, fastNet: Long = 0L,
    ): Int = nativeRuleChooseWorlds(requireHandle(), worlds.current, worlds.n, belief, q, extra, outValues, fastNet)

    fun newDeal(dealer: Int = 3, seed: Long = System.nanoTime(), trump: Int? = null) {
        close()
        handle = if (trump == null) nativeNewDeal(dealer, seed) else nativeNewDealWithTrump(dealer, seed, trump)
    }

    fun state(): BoardState {
        val raw = nativeState(requireHandle())
        return BoardState(
            currentPlayer = raw[0],
            trumpSuit = raw[1],
            declarer = raw[2],
            team0Points = raw[3],
            team1Points = raw[4],
            team0Tricks = raw[5],
            team1Tricks = raw[6],
            multiplier = raw[7],
            doubled = raw[8] != 0,
            redoubled = raw[9] != 0,
            announcementOpen = raw[10] != 0,
            finished = raw[11] != 0,
            trickCards = raw.slice(14..17),
            trickPlayers = raw.slice(18..21),
        )
    }

    fun handMask(player: Int): Long = nativeHandMask(requireHandle(), player)

    fun legalCards(): IntArray = nativeLegalCards(requireHandle())

    fun actionFeatures(observer: Int): FloatArray = nativeActionFeatures(requireHandle(), observer)

    fun gruRaw(observer: Int): FloatArray = nativeGruRaw(requireHandle(), observer)

    fun gruLength(observer: Int): Long = nativeGruLength(requireHandle(), observer)

    fun gruGlobal(observer: Int): FloatArray = nativeGruGlobal(requireHandle(), observer)

    /** Encoder observation of the player to move (the belief nets' input). */
    fun observation(observer: Int): FloatArray = nativeObservation(requireHandle(), observer)

    /**
     * Rule agent move (cpp/rule_agent.h) for the player to move. [belief] is
     * BeliefNet's 96 outputs on [observation], [q] the QNet value of each
     * legal card indexed by card (32 floats); either may be null.
     * [trumpRuleKnown]: trump came from the C++ trump rule, not a chooser or
     * a human. Blocks for the play-out: call it off the UI thread.
     */
    fun ruleChoose(belief: FloatArray?, q: FloatArray?, trumpRuleKnown: Boolean, nWorlds: Int = 400): Int =
        nativeRuleChoose(requireHandle(), belief, q, trumpRuleKnown, nWorlds)

    fun applyCard(card: Int) {
        nativeApplyCard(requireHandle(), card)
    }

    fun announceDouble(player: Int) = nativeAnnounceDouble(requireHandle(), player)

    fun announceRedouble(player: Int) = nativeAnnounceRedouble(requireHandle(), player)

    fun startPlaying() = nativeStartPlaying(requireHandle())

    fun handScore(team: Int): Int = nativeHandScore(requireHandle(), team)

    private fun requireHandle(): Long {
        check(handle != 0L) { "Start a deal before using the game engine." }
        return handle
    }

    override fun close() {
        if (handle != 0L) {
            nativeClose(handle)
            handle = 0
        }
    }

    companion object {
        init {
            System.loadLibrary("manille_mobile")
        }

        @JvmStatic private external fun nativeNewDeal(dealer: Int, seed: Long): Long
        @JvmStatic private external fun nativeNewDealWithTrump(dealer: Int, seed: Long, trump: Int): Long
        @JvmStatic private external fun nativeClose(handle: Long)
        @JvmStatic private external fun nativeState(handle: Long): IntArray
        @JvmStatic private external fun nativeHandMask(handle: Long, player: Int): Long
        @JvmStatic private external fun nativeLegalCards(handle: Long): IntArray
        @JvmStatic private external fun nativeActionFeatures(handle: Long, observer: Int): FloatArray
        @JvmStatic private external fun nativeGruRaw(handle: Long, observer: Int): FloatArray
        @JvmStatic private external fun nativeGruLength(handle: Long, observer: Int): Long
        @JvmStatic private external fun nativeGruGlobal(handle: Long, observer: Int): FloatArray
        @JvmStatic private external fun nativeObservation(handle: Long, observer: Int): FloatArray
        /** Index of the winning card of a (partial) trick, by the engine's rule. */
        fun trickWinner(cards: List<Int>, trump: Int): Int = nativeTrickWinner(cards.toIntArray(), trump)

        @JvmStatic private external fun nativeTrickWinner(cards: IntArray, trump: Int): Int
        @JvmStatic private external fun nativeClone(handle: Long): Long
        @JvmStatic private external fun nativeSnapshot(handle: Long): IntArray
        @JvmStatic private external fun nativeRestore(snapshot: IntArray): Long
        @JvmStatic private external fun nativeSampleWorlds(handle: Long, trumpRuleKnown: Boolean, nWorlds: Int): LongArray
        @JvmStatic private external fun nativeRuleChooseWorlds(
            handle: Long, worlds: LongArray, nWorlds: Int, belief: FloatArray?, q: FloatArray?, extra: DoubleArray?,
            outValues: DoubleArray?, fastNet: Long,
        ): Int
        /** The fast play-out net from its file (cpp/fast_policy.h); throws on a bad file. */
        @JvmStatic external fun nativeFastNetLoad(path: String): Long
        @JvmStatic external fun nativeFastNetFree(net: Long)
        @JvmStatic external fun nativeFastNetSetPlies(net: Long, plies: Int)
        @JvmStatic external fun nativeFastNetSetTrickEnd(net: Long, on: Boolean)
        @JvmStatic private external fun nativeRuleChoose(
            handle: Long, belief: FloatArray?, q: FloatArray?, trumpRuleKnown: Boolean, nWorlds: Int,
        ): Int
        @JvmStatic private external fun nativeApplyCard(handle: Long, card: Int)
        @JvmStatic private external fun nativeAnnounceDouble(handle: Long, player: Int)
        @JvmStatic private external fun nativeAnnounceRedouble(handle: Long, player: Int)
        @JvmStatic private external fun nativeStartPlaying(handle: Long)
        @JvmStatic private external fun nativeHandScore(handle: Long, team: Int): Int
    }
}
