package com.manille.jvm

import com.cardtable.platform.*

import com.manille.shared.*

/**
 * The C++ engine through JNI (cpp/jni/jni_bridge.cpp): a thin Kotlin
 * ownership wrapper; rules and encoders remain in C++.
 */
class NativeGame() : Engine {
    private var handle: Long = 0

    private constructor(handle: Long) : this() { this.handle = handle }

    /** Full engine state for a saved game (see [restore]). */
    override fun snapshot(): IntArray = nativeSnapshot(requireHandle())

    /** Replace the current deal with a saved [snapshot]. */
    override fun restore(snapshot: IntArray) {
        val restored = nativeRestore(snapshot)
        check(restored != 0L) { "Could not restore the saved game." }
        close()
        handle = restored
    }

    /** An independent copy (for a worker thread); close it when done. */
    override fun copy(): NativeGame = NativeGame(nativeClone(requireHandle()))

    override fun sampleWorlds(trumpRuleKnown: Boolean, nWorlds: Int): Worlds {
        val raw = nativeSampleWorlds(requireHandle(), trumpRuleKnown, nWorlds)
        val n = raw.size / 8
        return Worlds(n, raw.copyOfRange(0, 4 * n), raw.copyOfRange(4 * n, 8 * n))
    }

    override fun sampleWorldsBelief(
        trumpRuleKnown: Boolean, nWorlds: Int, belief: FloatArray, beliefWeight: Double,
    ): Worlds {
        val raw = nativeSampleWorldsBelief(requireHandle(), trumpRuleKnown, nWorlds, belief, beliefWeight)
        val n = (raw.size - 1) / 8
        val beliefSampled = raw.isNotEmpty() && raw[8 * n] != 0L
        return Worlds(n, raw.copyOfRange(0, 4 * n), raw.copyOfRange(4 * n, 8 * n), beliefSampled)
    }

    /**
     * Rule agent on given worlds; [extra] = optional log-weight per world.
     * [outValues] (32, by card) receives each legal card's play-out value
     * (expected final team points); NaN elsewhere or when it did not run.
     */
    override fun ruleChooseWorlds(
        worlds: Worlds, belief: FloatArray?, q: FloatArray?, extra: DoubleArray?,
        outValues: DoubleArray?, fastNet: FastNet?,
    ): Int = nativeRuleChooseWorlds(requireHandle(), worlds.current, worlds.n, worlds.beliefSampled,
        belief, q, extra, outValues, (fastNet as? NativeFastNet)?.handle ?: 0L)

    override fun newDeal(dealer: Int, seed: Long, trump: Int?) {
        close()
        handle = if (trump == null) nativeNewDeal(dealer, seed) else nativeNewDealWithTrump(dealer, seed, trump)
    }

    override fun state(): BoardState {
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

    override fun handMask(player: Int): Long = nativeHandMask(requireHandle(), player)

    override fun legalCards(): IntArray = nativeLegalCards(requireHandle())

    override fun actionFeatures(observer: Int): FloatArray = nativeActionFeatures(requireHandle(), observer)

    override fun gruRaw(observer: Int): FloatArray = nativeGruRaw(requireHandle(), observer)

    override fun gruLength(observer: Int): Long = nativeGruLength(requireHandle(), observer)

    override fun gruGlobal(observer: Int): FloatArray = nativeGruGlobal(requireHandle(), observer)

    /** Encoder observation of the player to move (the belief nets' input). */
    override fun observation(observer: Int): FloatArray = nativeObservation(requireHandle(), observer)

    /**
     * Rule agent move (cpp/rule_agent.h) for the player to move. [belief] is
     * BeliefNet's 96 outputs on [observation], [q] the QNet value of each
     * legal card indexed by card (32 floats); either may be null.
     * [trumpRuleKnown]: trump came from the C++ trump rule, not a chooser or
     * a human. Blocks for the play-out: call it off the UI thread.
     */
    fun ruleChoose(belief: FloatArray?, q: FloatArray?, trumpRuleKnown: Boolean, nWorlds: Int = 400): Int =
        nativeRuleChoose(requireHandle(), belief, q, trumpRuleKnown, nWorlds)

    override fun applyCard(card: Int) {
        nativeApplyCard(requireHandle(), card)
    }

    override fun announceDouble(player: Int) = nativeAnnounceDouble(requireHandle(), player)

    override fun announceRedouble(player: Int) = nativeAnnounceRedouble(requireHandle(), player)

    override fun startPlaying() = nativeStartPlaying(requireHandle())

    override fun handScore(team: Int): Int = nativeHandScore(requireHandle(), team)

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
            NativeLibrary.load()
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
        @JvmStatic private external fun nativeSampleWorldsBelief(
            handle: Long, trumpRuleKnown: Boolean, nWorlds: Int, belief: FloatArray, beliefWeight: Double,
        ): LongArray
        @JvmStatic private external fun nativeRuleChooseWorlds(
            handle: Long, worlds: LongArray, nWorlds: Int, beliefSampled: Boolean, belief: FloatArray?,
            q: FloatArray?, extra: DoubleArray?, outValues: DoubleArray?, fastNet: Long,
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

/**
 * How the JNI library gets loaded. Android finds libmanille_mobile.so in the
 * APK by name; the desktop shell points [load] at the file it ships before
 * the first engine is made.
 */
object NativeLibrary {
    @Volatile var load: () -> Unit = { System.loadLibrary("manille_mobile") }
}

/** A loaded fast play-out net (cpp/fast_policy.h), freed on close. */
class NativeFastNet(val handle: Long) : FastNet {
    override fun close() = NativeGame.nativeFastNetFree(handle)
}
