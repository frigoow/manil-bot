package com.manille.shared

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.serialization.json.JsonObject
import kotlin.random.Random

/**
 * The offline table: seat 0 is the local player, the other three are bots.
 * A [Referee] with one player; since that player sits in seat 0, its
 * absolute seats are already the player's own.
 */
class LocalSession(
    platform: Platform,
    bots: Bots,
    worker: Worker,
    pacing: Pacing = Pacing(),
    random: Random = Random.Default,
    main: CoroutineDispatcher = Dispatchers.Main,
) : TableSession {
    private val referee = Referee(platform, bots, worker, setOf(HUMAN), pacing, random, main).also { r ->
        r.listener = { event -> listener?.invoke(Seats.event(event, HUMAN), view) }
    }

    override var listener: ((TableEvent, TableView) -> Unit)? = null
    /** A bot's thinking time on a real decision (more than one legal card), in ms. */
    var onBotThought: ((Long) -> Unit)?
        get() = referee.onBotThought
        set(value) { referee.onBotThought = value }

    override val view: TableView get() = referee.view(HUMAN)

    /** A new quick game (one deal) or match to [WINNING_SCORE]. */
    fun start(match: Boolean) = referee.start(match)
    /** Leave the table: nothing more happens until the next [start] or [restore]. */
    fun stop() = referee.stop()
    /** The next deal (after HAND_OVER), dealt by the next seat. */
    fun newDeal() = referee.newDeal()

    override fun declare(trump: Int): Boolean = referee.declare(HUMAN, trump)
    override fun answerMeegaan(accept: Boolean): Boolean = referee.answer(HUMAN, TablePhase.MEEGAAN, accept)
    override fun answerTerug(accept: Boolean): Boolean = referee.answer(HUMAN, TablePhase.TERUG, accept)
    override fun play(card: Int): Boolean = referee.play(HUMAN, card)

    override fun hintPosition(): HintPosition? = referee.hintPosition(HUMAN)
    /** Changes whenever the position does (a hint for an old position is dropped). */
    fun positionKey(): Int = referee.positionKey()
    override fun openHand(seat: Int): Long? = referee.openHand(seat)

    /** The session's part of the saved game (GameController adds its own), or null in the lobby. */
    fun save(): SessionSave? = referee.save()
    /** Back to a saved table; throws when it cannot be read. Call [resume] once the screen shows it. */
    fun restore(save: JsonObject) = referee.restore(save)
    /** Carry on after [restore]: a finished deal is scored, a bot to move moves. */
    fun resume() = referee.resume()

    override fun close() = referee.close()
}
