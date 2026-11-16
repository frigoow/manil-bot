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
    private val store: Store,
) {
    val createdAt: Long = System.currentTimeMillis()
    private val chairs = arrayOfNulls<Player>(4)
    private val botNames = hub.config.botNames.shuffled(hub.random).let { names -> List(4) { names[it % names.size] } }
    private var host: Player? = null
    private var referee: Referee? = null
    /** After a deal: the chairs whose players asked for the next one. */
    private val ready = mutableSetOf<Int>()
    private val awayTimers = mutableMapOf<Int, Job>()
    private var idleTimer: Job? = null
    private var closed = false

    // ---- the record: this table's current game and deal, kept for the database and the API ----
    /** The game being played (or just ended), or null before the first. */
    var gameId: String? = null; private set
    private var gameEnded = false
    /** This game's finished deals. */
    val deals = mutableListOf<DealRecord>()
    private val deal = DealLog()

    /** What happens in one deal, from the referee's events (absolute seats). */
    private class DealLog {
        var dealer = 0; var trump = -1; var meegaan: Int? = null; var terug: Int? = null
        var hands = List(4) { 0L }
        val plays = mutableListOf<List<Int>>()
        val tricksBy = IntArray(4)
        val manilles = IntArray(2)
        var lastTrick: List<List<Int>> = emptyList()
        var lastTaker = -1
        fun reset(dealer: Int) {
            this.dealer = dealer; trump = -1; meegaan = null; terug = null; hands = List(4) { 0L }
            plays.clear(); tricksBy.fill(0); manilles.fill(0); lastTrick = emptyList(); lastTaker = -1
        }
    }

    fun seatOf(player: Player): Int = chairs.indexOfFirst { it === player }
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
            gameId?.takeIf { !gameEnded }?.let { id ->
                val names = seatNames()
                store.post { addGamePlayer(id, player.id, seat); updateSeats(id, names) }
            }
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
            // handled by the hub
            is ClientMessage.Hello, is ClientMessage.Create, is ClientMessage.Join,
            ClientMessage.GetStats, ClientMessage.GetTransferCode -> true
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
        newGame()
        broadcastInfo()
        r.start(match)
        return true
    }

    /** Who sits where, by name (bots by theirs). */
    private fun seatNames(): List<String> = (0..3).map { chairs[it]?.name ?: botNames[it] }

    private fun newGame() {
        val id = hub.newGameId()
        gameId = id; gameEnded = false; deals.clear()
        val names = seatNames()
        val seated = (0..3).mapNotNull { seat -> chairs[seat]?.let { seat to it.id } }
        store.post {
            startGame(id, code, match, names)
            seated.forEach { (seat, player) -> addGamePlayer(id, player, seat) }
        }
    }

    private fun endGame(outcome: String) {
        val id = gameId ?: return
        if (gameEnded) return
        gameEnded = true
        store.post { endGame(id, outcome) }
    }

    /** Books [event]: the deal's record, and at its end the deal, the players' statistics and maybe the game. */
    private fun record(event: TableEvent, r: Referee) {
        val id = gameId ?: return
        when (event) {
            TableEvent.DealStarted -> deal.reset(r.view(0).dealer)
            is TableEvent.Declared -> deal.trump = event.trump
            is TableEvent.Knocked -> deal.meegaan = event.seat
            is TableEvent.KnockedBack -> deal.terug = event.seat
            TableEvent.PlayStarted -> deal.hands = (0..3).map { r.openHand(it) ?: 0L }
            is TableEvent.CardPlayed -> {
                deal.plays += listOf(event.seat, event.card)
                event.trick?.let { t ->
                    deal.tricksBy[t.taker] += 1
                    deal.manilles[t.taker % 2] += t.cards.count { it % 8 == 7 }
                    deal.lastTrick = t.players.zip(t.cards) { seat, card -> listOf(seat, card) }
                    deal.lastTaker = t.taker
                }
            }
            is TableEvent.DealScored -> {
                val view = r.view(0)
                val b = view.board ?: return
                val record = DealRecord(
                    number = deals.size + 1, dealer = deal.dealer, trump = deal.trump, multiplier = b.multiplier,
                    meegaan = deal.meegaan, terug = deal.terug,
                    points = listOf(b.team0Points, b.team1Points), tricks = listOf(b.team0Tricks, b.team1Tricks),
                    score = listOf(event.usDelta, event.themDelta),
                    hands = deal.hands, plays = deal.plays.toList(), ended = System.currentTimeMillis(),
                )
                deals += record
                // a deal counts for the players who played it themselves (not while a bot stood in)
                val credited = (0..3).mapNotNull { seat -> chairs[seat]?.takeIf { seat in r.humans }?.let { seat to it.id } }
                val hands = credited.map { (seat, player) -> player to handRecord(record, seat) }
                val gameOver = view.phase == TablePhase.MATCH_OVER || !match
                val results = if (!gameOver) emptyList() else credited.map { (seat, player) ->
                    val t = seat % 2
                    // a match counts its match points; a quick game its one deal's card points (as the app does)
                    if (match) Triple(player, view.scoreOf(t), view.scoreOf(1 - t)) else Triple(player, record.points[t], record.points[1 - t])
                }
                store.post {
                    addDeal(id, record)
                    hands.forEach { (player, h) -> recordHand(player, h) }
                    results.forEach { (player, us, them) -> recordResult(player, match, us, them) }
                }
                if (gameOver) {
                    val (s0, s1) = if (match) view.usScore to view.themScore else record.points[0] to record.points[1]
                    endGame(when { s0 > s1 -> "team0"; s1 > s0 -> "team1"; else -> "tie" })
                }
            }
            TableEvent.YourCall, TableEvent.Resynced, is TableEvent.Waiting -> {}
        }
    }

    private fun TableView.scoreOf(team: Int) = if (team == 0) usScore else themScore

    /** The deal from [seat]'s side, as the app's statistics count it. */
    private fun handRecord(d: DealRecord, seat: Int): HandRecord {
        val t = seat % 2; val o = 1 - t
        fun ours(s: Int?) = s != null && s % 2 == t
        return HandRecord(
            pointsUs = d.points[t], pointsThem = d.points[o], scoreUs = d.score[t], scoreThem = d.score[o],
            tricksUs = d.tricks[t], tricksMine = deal.tricksBy[seat], manillesUs = deal.manilles[t],
            weDeclared = d.dealer % 2 == t, iDeclared = d.dealer == seat, mull = d.trump < 0, multiplier = d.multiplier,
            meegaanUs = ours(d.meegaan), meegaanThem = d.meegaan != null && !ours(d.meegaan), meegaanMine = d.meegaan == seat,
            terugUs = ours(d.terug), terugThem = d.terug != null && !ours(d.terug), terugMine = d.terug == seat,
        )
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
        record(event, r)
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
        if (gameId != null) endGame("abandoned")
        awayTimers.values.forEach { it.cancel() }; awayTimers.clear()
        idleTimer?.cancel()
        for (p in chairs) if (p != null) { p.table = null; p.connection?.send(ServerMessage.Left(reason)) }
        chairs.fill(null)
        referee?.close(); referee = null
        hub.closed(this)
    }

    // ---- the API's view (on the game thread) -------------------------------------------

    fun summary(): LiveTable {
        val v = referee?.view(0)
        return LiveTable(
            code = code, match = match, started = referee != null, created = createdAt,
            phase = v?.phase?.name, hand = v?.handNumber ?: 0,
            score = listOf(v?.usScore ?: 0, v?.themScore ?: 0),
            seats = (0..3).map { seat ->
                val p = chairs[seat]
                when {
                    p != null -> LiveSeat(p.name, bot = false, away = p.connection == null, player = p.id)
                    referee != null -> LiveSeat(botNames[seat], bot = true)
                    else -> LiveSeat(null, bot = true)
                }
            },
            game = gameId,
        )
    }

    /** The table now; [admin] adds every hand, the connections and who is ready. Hidden cards only for [admin]. */
    fun detail(admin: Boolean): LiveTableDetail {
        val r = referee
        val v = r?.view(0)
        val b = v?.board
        val live = if (r == null || v == null || b == null || v.phase == TablePhase.LOBBY) null else LiveDeal(
            number = v.handNumber, dealer = v.dealer, trump = if (v.phase == TablePhase.DECLARATION) null else b.trumpSuit,
            multiplier = b.multiplier, meegaan = v.meegaanSeat, terug = v.terugSeat,
            turn = b.currentPlayer.takeIf { v.phase == TablePhase.PLAYING && !b.finished },
            asked = v.asked,
            trick = b.trickPlayers.zip(b.trickCards).filter { it.second >= 0 }.map { listOf(it.first, it.second) },
            points = listOf(b.team0Points, b.team1Points), tricks = listOf(b.team0Tricks, b.team1Tricks),
            cardsLeft = v.handCounts, lastTrick = deal.lastTrick, lastTaker = deal.lastTaker.takeIf { it >= 0 },
        )
        return LiveTableDetail(
            table = summary(), deal = live, deals = deals.toList(),
            admin = if (!admin) null else AdminTable(
                hands = if (r == null || live == null) null else (0..3).map { r.openHand(it) ?: 0L },
                host = host?.name, humans = r?.humans?.sorted() ?: emptyList(), ready = ready.sorted(),
                connected = (0..3).map { connected(it) }, awayTimers = awayTimers.keys.sorted(),
            ),
        )
    }

    private companion object {
        val JOIN_ORDER = listOf(0, 2, 1, 3)
    }
}
