package com.manille.shared

import com.cardtable.platform.*

import kotlinx.serialization.json.JsonObject

enum class TablePhase { LOBBY, DECLARATION, MEEGAAN, TERUG, PLAYING, HAND_OVER, MATCH_OVER }

/*
 * A table as one player sees it, and the moves that player can make there.
 *
 * Seats are numbered from the viewer's point of view: 0 is the local player,
 * 1 on their left (West), 2 the partner (North), 3 East; teams are seat % 2,
 * so "us" is team 0. A local game is played from seat 0; a networked one
 * will rotate the server's seats so the viewer is always 0.
 *
 * The session is the referee (it deals, checks every move, runs the bots
 * and keeps the score); GameController turns what the session reports into
 * the table on screen, the messages, the bots' reactions and the stats.
 */

internal const val HUMAN = 0
internal const val PARTNER = 2
internal const val HUMAN_TEAM = 0
internal const val WINNING_SCORE = 101
internal fun teamOf(seat: Int) = seat % 2

/** Everything the viewer may know of the table -- never another player's cards. */
data class TableView(
    val phase: TablePhase,
    val board: BoardState?,
    /** The viewer's own cards (bit c = card c). */
    val hand: Long,
    /** What the viewer may play now: empty unless it is their turn to play a card. */
    val legalCards: List<Int>,
    val handCounts: List<Int>,
    val dealer: Int,
    /** Deals started in this game, 1 for the first. */
    val handNumber: Int,
    val match: Boolean,
    val usScore: Int,
    val themScore: Int,
    /** Who knocked (meegaan) and who knocked back (terug) this deal. */
    val meegaanSeat: Int?,
    val terugSeat: Int?,
) {
    companion object {
        val LOBBY = TableView(TablePhase.LOBBY, null, 0L, emptyList(), List(4) { 0 }, 0, 0, true, 0, 0, null, null)
    }
}

/** A trick the last card completed. [before] is the board as that card was played. */
data class CompletedTrick(val cards: List<Int>, val players: List<Int>, val taker: Int, val before: BoardState)

/** What happened at the table, in order; each comes with the [TableView] right after it. */
sealed interface TableEvent {
    /** New cards are dealt; [TableView.dealer] declares next. */
    data object DealStarted : TableEvent
    /** Trump suit 0..3, or -1 for mull (no trump). */
    data class Declared(val seat: Int, val trump: Int) : TableEvent
    /** Meegaan: the defenders double. */
    data class Knocked(val seat: Int) : TableEvent
    /** Terug: the declarers double again. */
    data class KnockedBack(val seat: Int) : TableEvent
    /** The viewer must answer: [TableView.phase] says what (declare, knock, knock back). */
    data object YourCall : TableEvent
    /** The calls are done; the player left of the dealer leads. */
    data object PlayStarted : TableEvent
    /**
     * [seat] played [card]. [legal]: what that seat could have played instead
     * (a local game knows; over the network it would give away a hand, so
     * null). [trick]: set when this card completed a trick.
     */
    data class CardPlayed(val seat: Int, val card: Int, val legal: Set<Int>?, val trick: CompletedTrick?) : TableEvent
    /** The deal is over: match points for each team ([TableView] has the new score). */
    data class DealScored(val usDelta: Int, val themDelta: Int) : TableEvent
}

/** A private copy of the viewer's position for a hint; close [engine] when done. */
class HintPosition(val engine: Engine, val calls: RuleBot.Calls, val key: Int)

/**
 * One table seen from seat 0. Commands that do not fit the moment (not your
 * turn, an illegal card) are ignored and return false; the session never
 * trusts the screen.
 */
interface TableSession : AutoCloseable {
    /** The table now. */
    val view: TableView

    /** Gets every event, on the main thread, in order. */
    var listener: ((TableEvent, TableView) -> Unit)?

    fun declare(trump: Int): Boolean
    fun answerMeegaan(accept: Boolean): Boolean
    fun answerTerug(accept: Boolean): Boolean
    fun play(card: Int): Boolean

    /** The viewer's position for a hint, or null when this session cannot give one. */
    fun hintPosition(): HintPosition? = null
    /** Another seat's cards (the "open cards" view), or null when the viewer may not see them. */
    fun openHand(seat: Int): Long? = null
}

/** A local game's own state for the saved game (see [LocalSession.save]). */
typealias SessionSave = JsonObject
