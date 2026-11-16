package com.manille.server

import com.manille.shared.*
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * One online table: four chairs, the players in them, and once the game is
 * on, a [Referee] that plays the empty chairs with bots. Runs on the hub's
 * game thread only.
 *
 * Every player gets the table from their own chair ([Seats]); nothing in an
 * update shows another player's cards, and the cards a player could have
 * played instead ([TableEvent.CardPlayed.legal]) are not sent at all.
 */
class ServerTable(
    val code: String,
    private val match: Boolean,
    private val hub: Hub,
    private val platform: Platform,
    private val bots: Bots,
    private val worker: Worker,
) {
    private val chairs = arrayOfNulls<Player>(4)
    private val botNames = hub.config.botNames.shuffled(hub.random).let { names -> List(4) { names[it % names.size] } }
    private var host: Player? = null
    private var referee: Referee? = null
    /** After a deal: the chairs whose players asked for the next one. */
    private val ready = mutableSetOf<Int>()
    private val awayTimers = mutableMapOf<Int, Job>()
    private var idleTimer: Job? = null
    private var closed = false

    private fun seatOf(player: Player): Int = chairs.indexOfFirst { it === player }
    private fun connected(seat: Int) = chairs[seat]?.connection != null

    /** A chair a newcomer can take: a free one before the game, one a bot plays once it is on. */
    fun hasRoom(): Boolean = chairs.any { it == null }

    fun join(player: Player) {
        // partners first: a second player joins the first one's team
        val seat = JOIN_ORDER.first { chairs[it] == null }
        chairs[seat] = player
        player.table = this
        if (host == null) host = player
        idleTimer?.cancel(); idleTimer = null
        referee?.let { r ->
            // into a running game: the player takes over from the bot
            r.setHuman(seat, true)
            ready -= seat
        }
        broadcastInfo()
        referee?.let { r -> player.connection?.send(ServerMessage.Update(TableEvent.Resynced, Seats.view(r.view(seat), seat))) }
    }

    fun handle(player: Player, message: ClientMessage) {
        val seat = seatOf(player)
        if (seat < 0) return
        val r = referee
        val ok = when (message) {
            is ClientMessage.Sit -> sit(player, seat, message.seat)
            ClientMessage.Start -> start(player)
            is ClientMessage.Declare -> r?.declare(seat, message.trump) == true
            is ClientMessage.Knock -> (message.call == TablePhase.MEEGAAN || message.call == TablePhase.TERUG) &&
                r?.answer(seat, message.call, message.accept) == true
            is ClientMessage.Play -> r?.play(seat, message.card) == true
            ClientMessage.NextDeal -> nextDeal(seat)
            is ClientMessage.React -> {
                val text = message.text.filter { !it.isISOControl() }.trim().take(Hub.MAX_REACTION)
                if (text.isNotEmpty()) for (other in 0..3) if (other != seat) chairs[other]?.connection?.send(ServerMessage.Reaction(Seats.rel(seat, other), text))
                true
            }
            ClientMessage.Leave -> { leave(player); true }
            is ClientMessage.Hello, is ClientMessage.Create, is ClientMessage.Join -> true
        }
        if (!ok) player.connection?.send(ServerMessage.Problem(Problems.NOT_NOW, "That is not possible now."))
    }

    private fun sit(player: Player, from: Int, to: Int): Boolean {
        if (referee != null || to !in 0..3) return false
        if (chairs[to] != null && chairs[to] !== player) {
            player.connection?.send(ServerMessage.Problem(Problems.SEAT_TAKEN, "That chair is taken."))
            return true
        }
        chairs[from] = null; chairs[to] = player
        broadcastInfo()
        return true
    }

    /** The host starts the game, or a new one once the last is over. */
    private fun start(player: Player): Boolean {
        if (player !== host) {
            player.connection?.send(ServerMessage.Problem(Problems.NOT_HOST, "Only the host can start."))
            return true
        }
        val current = referee
        if (current != null && current.view(0).phase != TablePhase.MATCH_OVER && !(current.handOver && !match)) return false
        val r = current ?: Referee(platform, bots, worker, humanSeats(), hub.config.pacing, hub.random, hub.game).also { r ->
            referee = r
            r.listener = ::onEvent
        }
        ready.clear()
        broadcastInfo()
        r.start(match)
        return true
    }

    /** Chairs played by a connected player; the bots play the rest. */
    private fun humanSeats(): Set<Int> = (0..3).filter { connected(it) }.toSet()

    private fun nextDeal(seat: Int): Boolean {
        val r = referee ?: return false
        if (!r.handOver) return false
        ready += seat
        maybeDeal()
        return true
    }

    /** Deals once every connected player is ready. */
    private fun maybeDeal() {
        val r = referee ?: return
        if (!r.handOver) return
        if ((0..3).filter { connected(it) }.all { it in ready }) { ready.clear(); broadcastInfo(); r.newDeal() }
        else broadcastInfo()
    }

    private fun onEvent(event: TableEvent) {
        val r = referee ?: return
        if (event is TableEvent.DealScored) ready.clear()
        for (seat in 0..3) {
            val connection = chairs[seat]?.connection ?: continue
            val mine = Seats.event(event, seat).let { if (it is TableEvent.CardPlayed) it.copy(legal = null) else it }
            connection.send(ServerMessage.Update(mine, Seats.view(r.view(seat), seat)))
        }
    }

    fun leave(player: Player) {
        val seat = seatOf(player)
        if (seat < 0) return
        chairs[seat] = null
        player.table = null
        awayTimers.remove(seat)?.cancel()
        ready -= seat
        player.connection?.send(ServerMessage.Left("left"))
        if (host === player) host = chairs.firstOrNull { it != null }
        if (chairs.all { it == null }) { close("closed"); return }
        referee?.setHuman(seat, false)
        broadcastInfo()
        maybeDeal()
    }

    fun disconnected(player: Player) {
        val seat = seatOf(player)
        if (seat < 0) return
        broadcastInfo()
        awayTimers.remove(seat)?.cancel()
        awayTimers[seat] = hub.scope.launch {
            delay(hub.config.graceMs)
            awayTimers.remove(seat)
            // still gone: a bot plays the chair until they are back
            referee?.setHuman(seat, false)
            maybeDeal()
        }
        if ((0..3).none { connected(it) }) {
            idleTimer?.cancel()
            idleTimer = hub.scope.launch { delay(hub.config.idleTableMs); if ((0..3).none { connected(it) }) close("closed") }
        }
    }

    fun reconnected(player: Player) {
        val seat = seatOf(player)
        if (seat < 0) return
        awayTimers.remove(seat)?.cancel()
        idleTimer?.cancel(); idleTimer = null
        referee?.setHuman(seat, true)
        resend(player)
    }

    /** The table afresh for [player]: its chairs, and the game from their seat. */
    fun resend(player: Player) {
        val seat = seatOf(player)
        broadcastInfo()
        referee?.let { r -> player.connection?.send(ServerMessage.Update(TableEvent.Resynced, Seats.view(r.view(seat), seat))) }
    }

    private fun broadcastInfo() {
        val started = referee != null
        val seats = (0..3).map { seat ->
            val p = chairs[seat]
            when {
                p != null -> SeatInfo(p.name, bot = false, away = p.connection == null)
                started -> SeatInfo(botNames[seat], bot = true)
                else -> SeatInfo(null, bot = true)
            }
        }
        for (seat in 0..3) {
            val p = chairs[seat] ?: continue
            p.connection?.send(ServerMessage.TableInfo(code, match, seats, seat, p === host, started, ready.sorted()))
        }
    }

    fun close(reason: String) {
        if (closed) return
        closed = true
        awayTimers.values.forEach { it.cancel() }; awayTimers.clear()
        idleTimer?.cancel()
        for (p in chairs) if (p != null) { p.table = null; p.connection?.send(ServerMessage.Left(reason)) }
        chairs.fill(null)
        referee?.close(); referee = null
        hub.closed(this)
    }

    private companion object {
        val JOIN_ORDER = listOf(0, 2, 1, 3)
    }
}
