package com.manille.shared

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlin.random.Random
import kotlin.time.TimeSource

/**
 * One thread for the bots' thinking, shared by everything that uses the nets
 * (the session's bots, hints, the benchmark). [lock] is held for the whole of
 * a job, so whoever frees the nets can wait for the job in flight.
 */
class Worker {
    val dispatcher: CoroutineDispatcher = Dispatchers.Default.limitedParallelism(1)
    val lock = Mutex()
}

/** The nets the bots play with; any may be missing (a weaker fallback plays instead). */
class Bots(
    val ruleBot: RuleBot?,
    val qnet: QModel?,
    val declarationNet: DeclarationModel?,
    val announceNet: AnnounceModel?,
)

/** How long the table waits: a bot's least thinking time, a finished trick on the table. */
data class Pacing(val botThinkMs: Long = 900L, val trickHoldMs: Long = 1400L)

/**
 * The referee of one table, in absolute seats (team 0 = seats 0 and 2): it
 * deals, asks for the calls, checks every move, runs the bots in the seats
 * that have no player and keeps the score, with the C++ engine as the
 * rulebook (mirrors gui/game_session.py).
 *
 * Any seat may be a player ([humans]); a seat can change hands mid-game
 * ([setHuman]: a player who drops out is replaced by a bot, and takes the
 * seat back on return). The offline game is a referee with one player in
 * seat 0 ([LocalSession]); the server runs one per online table.
 *
 * Runs on [main]; a bot thinks on the [worker] with a copy of the game, and
 * its card comes back to [main]. Events are reported in absolute seats, with
 * the team-0 side as "us" ([Seats] turns them into one player's view).
 * Whoever must answer a call is announced with [TableEvent.Waiting].
 */
class Referee(
    private val platform: Platform,
    private val bots: Bots,
    private val worker: Worker,
    humans: Set<Int>,
    private val pacing: Pacing = Pacing(),
    private val random: Random = Random.Default,
    main: CoroutineDispatcher = Dispatchers.Main,
) : AutoCloseable {
    private val game = platform.newEngine()
    private val scope = CoroutineScope(SupervisorJob() + main)
    private val mainDispatcher = main

    /** Every event, on [main], in order, in absolute seats. Ask [view] for each seat's table. */
    var listener: ((TableEvent) -> Unit)? = null
    /** A bot's thinking time on a real decision (more than one legal card), in ms. */
    var onBotThought: ((Long) -> Unit)? = null

    /** The seats played by people; the bots play the others. */
    var humans: Set<Int> = humans.toSet(); private set
    /** Seats a person held at some point of this deal: their calls are not the bots' policy. */
    private var dealHumans: Set<Int> = this.humans

    private var phase = TablePhase.LOBBY
    private var matchMode = true
    private var team0Score = 0
    private var team1Score = 0
    private var handNumber = 0
    private var dealer = 0
    private var dealSeed = 0L
    /** The player whose call the table waits for (declaration, meegaan or terug), if any. */
    private var asked: Int? = null
    // This deal's calls, for the bots' world weights.
    private var meegaanSeat: Int? = null
    private var terugSeat: Int? = null
    private var declUs = 0
    private var declThem = 0
    private var trumpRuleKnown = false
    /** The calls are done and the cards are being played (phase alone says PLAYING while the calls are still open). */
    private var cardsOpen = false
    /** A finished trick still on the table: the bots wait for it. */
    private var holding = false
    private var turnToken = 0

    // At most one delayed bot/trick step may ever be outstanding. Every timed
    // step is posted through schedule(), which cancels the previous one first,
    // so a stray extra call can never fork a second self-perpetuating loop that
    // races the first and plays the hand out in a blink.
    private var pending: Job? = null

    private fun schedule(delayMs: Long, action: () -> Unit) {
        pending?.cancel()
        lateinit var job: Job
        job = scope.launch(start = CoroutineStart.LAZY) {
            delay(delayMs)
            if (pending === job) { pending = null; action() }
        }
        pending = job
        job.start()
    }

    private fun cancelTurn() { turnToken += 1; pending?.cancel(); pending = null; holding = false }

    /** Nobody is at the table: no deal, nothing happens until [start] or [restore]. */
    val inLobby: Boolean get() = phase == TablePhase.LOBBY
    /** The deal is over and the table waits for [newDeal] (never after the match is over). */
    val handOver: Boolean get() = phase == TablePhase.HAND_OVER

    /**
     * The table as [viewer] may see it, in absolute seats: their own cards,
     * what they may play, and the call they must answer. While someone else
     * is asked for a call, the phase reads PLAYING (with nothing to play), as
     * it does while the bots make theirs.
     */
    fun view(viewer: Int): TableView {
        if (phase == TablePhase.LOBBY) return TableView.LOBBY.copy(match = matchMode)
        val board = game.state()
        val calling = phase == TablePhase.DECLARATION || phase == TablePhase.MEEGAAN || phase == TablePhase.TERUG
        return TableView(
            phase = if (calling && asked != viewer) TablePhase.PLAYING else phase,
            board = board,
            hand = game.handMask(viewer),
            legalCards = if (cardsOpen && phase == TablePhase.PLAYING && board.currentPlayer == viewer && !board.finished)
                game.legalCards().toList() else emptyList(),
            handCounts = List(4) { game.handMask(it).countOneBits() },
            dealer = dealer, handNumber = handNumber, match = matchMode,
            usScore = team0Score, themScore = team1Score,
            meegaanSeat = meegaanSeat, terugSeat = terugSeat,
            asked = asked?.takeIf { calling },
            askedCall = phase.takeIf { calling && asked != null },
        )
    }

    private fun emit(event: TableEvent) { listener?.invoke(event) }

    // ---- the game's course --------------------------------------------------

    /** A new quick game (one deal) or match to [WINNING_SCORE]. */
    fun start(match: Boolean) {
        matchMode = match; team0Score = 0; team1Score = 0; handNumber = 0
        newDeal()
    }

    /** Leave the table: nothing more happens until the next [start] or [restore]. */
    fun stop() { cancelTurn(); phase = TablePhase.LOBBY; cardsOpen = false; asked = null }

    /** The next deal (after HAND_OVER), dealt by the next seat. */
    fun newDeal() {
        cancelTurn()
        dealer = if (handNumber == 0) random.nextInt(4) else (dealer + 1) % 4
        if (!matchMode) { team0Score = 0; team1Score = 0 }
        dealSeed = random.nextLong()
        game.newDeal(dealer, dealSeed); handNumber += 1
        meegaanSeat = null; terugSeat = null; cardsOpen = false; asked = null
        dealHumans = humans
        // match score from the declaring team's side, as the calls see it
        if (teamOf(dealer) == 0) { declUs = team0Score; declThem = team1Score } else { declUs = team1Score; declThem = team0Score }
        trumpRuleKnown = false
        if (dealer in humans) {
            phase = TablePhase.DECLARATION
            asked = dealer
            emit(TableEvent.DealStarted)
            emit(TableEvent.Waiting(dealer, TablePhase.DECLARATION))
            return
        }
        phase = TablePhase.PLAYING
        emit(TableEvent.DealStarted)
        botDeclares()
    }

    /** The dealer's seat is a bot's: it declares now. */
    private fun botDeclares() {
        asked = null; phase = TablePhase.PLAYING
        val model = bots.declarationNet
        if (model != null) {
            val (us, them) = scoresFor(dealer)
            val equities = model.equities(game.handMask(dealer), us, them)
            val choice = equities.indices.maxBy { equities[it] }
            game.newDeal(dealer, dealSeed, if (choice == 4) -1 else choice)
        }
        // without the declaration net the C++ trump rule declared (newDeal's default)
        trumpRuleKnown = model == null
        emit(TableEvent.Declared(dealer, game.state().trumpSuit))
        afterDeclaration()
    }

    fun declare(seat: Int, trump: Int): Boolean {
        if (phase != TablePhase.DECLARATION || asked != seat || seat !in humans || trump !in -1..3) return false
        game.newDeal(dealer, dealSeed, trump)
        asked = null; phase = TablePhase.PLAYING
        emit(TableEvent.Declared(seat, trump))
        afterDeclaration()
        return true
    }

    /** [seat]'s answer to the meegaan (knock) or terug (knock back) it was asked for. */
    fun answer(seat: Int, call: TablePhase, accept: Boolean): Boolean {
        if (phase != call || asked != seat || seat !in humans) return false
        asked = null
        if (accept) accepted(seat) else askNext()
        return true
    }

    fun play(seat: Int, card: Int): Boolean {
        val board = game.state()
        if (!cardsOpen || phase != TablePhase.PLAYING || seat !in humans || board.finished ||
            board.currentPlayer != seat || card !in game.legalCards()) return false
        applyCard(card)
        return true
    }

    /**
     * A person takes [seat] ([human]) or leaves it to a bot. A bot that takes
     * over answers whatever the seat was asked at once; a person who takes a
     * seat back is asked from the next decision on.
     */
    fun setHuman(seat: Int, human: Boolean) {
        if (human == (seat in humans)) return
        humans = if (human) humans + seat else humans - seat
        if (human) { dealHumans = dealHumans + seat; return }
        if (phase == TablePhase.LOBBY) return
        when {
            asked == seat && phase == TablePhase.DECLARATION -> botDeclares()
            asked == seat -> { asked = null; if (botCalls(seat)) accepted(seat) else askNext() }
            cardsOpen && phase == TablePhase.PLAYING -> afterMove()
        }
    }

    // ---- the calls: meegaan, then terug ------------------------------------------

    /** Seats to be asked after [asked] in this round of calls. */
    private var queue: List<Int> = emptyList()

    /** Who is asked in a round, in order; the first yes takes it. */
    private fun order(call: TablePhase): List<Int> =
        if (call == TablePhase.MEEGAAN) listOf((dealer + 1) % 4, (dealer + 3) % 4) // defenders, from the dealer's left
        else listOf(dealer, (dealer + 2) % 4) // the declaring side, dealer first

    private fun afterDeclaration() = round(TablePhase.MEEGAAN)

    private fun round(call: TablePhase) { phase = call; queue = order(call); askNext() }

    /** The next seat in the round: a bot answers at once, a player is asked. No one left: play. */
    private fun askNext() {
        while (true) {
            val seat = queue.firstOrNull() ?: run { asked = null; beginPlaying(); return }
            queue = queue.drop(1)
            if (seat in humans) { asked = seat; emit(TableEvent.Waiting(seat, phase)); return }
            if (botCalls(seat)) { accepted(seat); return }
        }
    }

    private fun accepted(seat: Int) {
        asked = null
        if (phase == TablePhase.MEEGAAN) { knock(seat); round(TablePhase.TERUG) }
        else { knockBack(seat); beginPlaying() }
    }

    private fun knock(seat: Int) { game.announceDouble(seat); meegaanSeat = seat; emit(TableEvent.Knocked(seat)) }
    private fun knockBack(seat: Int) { game.announceRedouble(seat); terugSeat = seat; emit(TableEvent.KnockedBack(seat)) }

    private fun botCalls(seat: Int): Boolean = if (phase == TablePhase.MEEGAAN) botMeegaan(seat) else botTerug(seat)

    // ---- the play -----------------------------------------------------------------

    private fun beginPlaying() {
        game.startPlaying(); phase = TablePhase.PLAYING; cardsOpen = true; asked = null; queue = emptyList()
        emit(TableEvent.PlayStarted)
        afterMove()
    }

    private fun applyCard(card: Int) {
        val before = game.state()
        val seat = before.currentPlayer
        // what the player could have played instead (the engine's answer), so
        // the reactions can tell a forced loss from a gift
        val legal = game.legalCards().toSet()
        val completes = before.trickCards.count { it >= 0 } == 3
        val trick = if (!completes) null else {
            val cards = before.trickCards.filter { it >= 0 } + card
            val players = before.trickPlayers.filter { it >= 0 } + seat
            // the winner from the cards: on the last trick the engine
            // finishes the deal without making the winner the player to move
            CompletedTrick(cards, players, players[platform.trickWinner(cards, before.trumpSuit)], before)
        }
        game.applyCard(card)
        if (trick != null) holding = true
        emit(TableEvent.CardPlayed(seat, card, legal, trick))
        if (trick != null) schedule(pacing.trickHoldMs) { holding = false; afterMove() }
        else afterMove()
    }

    private fun afterMove() {
        if (holding || !cardsOpen || phase != TablePhase.PLAYING) return
        val board = game.state()
        when {
            board.finished -> completeHand()
            board.currentPlayer !in humans -> startBot(board.currentPlayer)
        }
    }

    private fun completeHand() {
        val delta0 = game.handScore(0)
        val delta1 = game.handScore(1)
        team0Score += delta0; team1Score += delta1
        phase = if (matchMode && (team0Score >= WINNING_SCORE || team1Score >= WINNING_SCORE)) TablePhase.MATCH_OVER else TablePhase.HAND_OVER
        emit(TableEvent.DealScored(delta0, delta1))
    }

    // ---- the bots -------------------------------------------------------------

    private fun calls() = RuleBot.Calls(dealer, meegaanSeat, terugSeat, declUs, declThem, trumpRuleKnown, dealHumans)

    /**
     * The rule agent thinks on the worker with a copy of the game; the card
     * is played on the main thread after at least [Pacing.botThinkMs], and
     * only if nothing moved on meanwhile (turnToken: a new deal or leaving
     * the table cancels it). Any failure falls back to the QGRU's card.
     */
    private fun startBot(player: Int) {
        val model = bots.ruleBot
        if (model == null) {
            schedule(pacing.botThinkMs) { if (stillTurnOf(player)) applyCard(qnetCard(player)) }
            return
        }
        val token = turnToken
        val started = TimeSource.Monotonic.markNow()
        val copy = game.copy()
        val calls = calls()
        onWorker {
            val decision = copy.legalCards().size > 1
            val t0 = TimeSource.Monotonic.markNow()
            val card = try { model.choose(copy, calls) } catch (error: Exception) { -1 } finally { copy.close() }
            val ms = t0.elapsedNow().inWholeMilliseconds
            onMain {
                if (decision && card >= 0) onBotThought?.invoke(ms)
                if (token != turnToken) return@onMain
                val wait = (pacing.botThinkMs - started.elapsedNow().inWholeMilliseconds).coerceAtLeast(0L)
                schedule(wait) {
                    if (token != turnToken || !stillTurnOf(player)) return@schedule
                    applyCard(if (card in game.legalCards()) card else qnetCard(player))
                }
            }
        }
    }

    /** The bot may still move for [player]: its turn, and no person has taken the seat meanwhile. */
    private fun stillTurnOf(player: Int): Boolean {
        val now = game.state()
        return cardsOpen && phase == TablePhase.PLAYING && !holding && !now.finished && now.currentPlayer == player && player !in humans
    }

    private fun qnetCard(player: Int): Int =
        bots.qnet?.chooseCard(game.gruRaw(player), game.gruLength(player), game.gruGlobal(player),
            game.actionFeatures(player), game.legalCards()) ?: game.legalCards().first()

    /** Match score (us, them) from [seat]'s team's side. */
    private fun scoresFor(seat: Int) = if (teamOf(seat) == 0) team0Score to team1Score else team1Score to team0Score

    private fun declarationIndex(): Int = game.state().trumpSuit.let { if (it < 0) 4 else it }

    private fun botMeegaan(seat: Int): Boolean {
        val (us, them) = scoresFor(seat)
        return bots.announceNet?.meegaan(game.handMask(seat), declarationIndex(), (seat - dealer + 4) % 4, us, them)
            ?: botShouldCall(seat)
    }

    private fun botTerug(seat: Int): Boolean {
        val (us, them) = scoresFor(seat)
        return bots.announceNet?.terug(game.handMask(seat), declarationIndex(), (seat - dealer + 4) % 4, us, them)
            ?: botShouldCall(seat)
    }

    // Offline fallback only, for when the announcement net fails to load: a
    // crude hand-strength threshold with no notion of trump or the score.
    private fun botShouldCall(player: Int): Boolean {
        val ranks = intArrayOf(0, 0, 1, 3, 4, 5, 6, 8)
        return (0 until 32).filter { game.handMask(player) and (1L shl it) != 0L }.sumOf { ranks[it % 8] } >= 29
    }

    private fun onWorker(block: () -> Unit) { scope.launch(worker.dispatcher) { worker.lock.withLock { block() } } }
    private fun onMain(block: () -> Unit) { scope.launch(mainDispatcher) { block() } }

    // ---- hints and the open-cards view ------------------------------------------

    /** [seat]'s position for a hint, while it is that seat's move; the copy holds every hand (offline only). */
    fun hintPosition(seat: Int): HintPosition? {
        val board = game.state()
        if (!cardsOpen || phase != TablePhase.PLAYING || board.finished || board.currentPlayer != seat) return null
        return HintPosition(game.copy(), calls(), positionKey())
    }

    /** Changes whenever the position does (a hint for an old position is dropped). */
    fun positionKey(): Int = if (phase == TablePhase.LOBBY) 0 else game.snapshot().contentHashCode()

    fun openHand(seat: Int): Long? = if (phase == TablePhase.LOBBY) null else game.handMask(seat)

    // ---- the saved game -------------------------------------------------------------

    /** The table's state for a saved game, or null in the lobby. Seats are absolute; "us" is team 0. */
    fun save(): SessionSave? {
        if (phase == TablePhase.LOBBY) return null
        return buildJsonObject {
            put("match", matchMode); put("us", team0Score); put("them", team1Score)
            put("hand", handNumber); put("dealer", dealer); put("seed", dealSeed)
            put("phase", phase.name)
            put("asked", asked ?: -1)
            put("meegaan", meegaanSeat ?: -1); put("terug", terugSeat ?: -1)
            put("declUs", declUs); put("declThem", declThem); put("ruleTrump", trumpRuleKnown)
            put("engine", JsonArray(game.snapshot().map { JsonPrimitive(it) }))
        }
    }

    /**
     * Back to the table [save] described; throws (and changes nothing a new
     * [start] would not reset) when it cannot be read. Call [resume] once the
     * screen shows it. A save from before multiplayer has no "asked": its
     * calls were always seat 0's.
     */
    fun restore(save: JsonObject) {
        cancelTurn()
        val engine = save.optArray("engine") ?: error("no engine state")
        val restoredPhase = TablePhase.valueOf(save.getString("phase"))
        game.restore(IntArray(engine.size) { engine[it].jsonPrimitive.int })
        matchMode = save.getBoolean("match"); team0Score = save.getInt("us"); team1Score = save.getInt("them")
        handNumber = save.getInt("hand"); dealer = save.getInt("dealer"); dealSeed = save.getLong("seed")
        meegaanSeat = save.getInt("meegaan").takeIf { it >= 0 }
        terugSeat = save.getInt("terug").takeIf { it >= 0 }
        declUs = save.getInt("declUs"); declThem = save.getInt("declThem")
        trumpRuleKnown = save.getBoolean("ruleTrump")
        phase = restoredPhase
        val calling = phase == TablePhase.DECLARATION || phase == TablePhase.MEEGAAN || phase == TablePhase.TERUG
        asked = if (!calling) null else (save["asked"]?.jsonPrimitive?.int ?: HUMAN).takeIf { it >= 0 }
        // the rest of a round of calls follows from who is asked now
        queue = if (phase == TablePhase.MEEGAAN || phase == TablePhase.TERUG) order(phase).dropWhile { it != asked }.drop(1) else emptyList()
        dealHumans = humans
        // a game is saved in PLAYING only once the calls are done
        cardsOpen = phase == TablePhase.PLAYING
    }

    /** Carry on after [restore]: a finished deal is scored, a bot to move moves. */
    fun resume() { afterMove() }

    override fun close() {
        cancelTurn(); scope.cancel()
        // a bot's move in flight works on its own copy; the game itself is
        // only used here, so it can go once that job is done
        CoroutineScope(Dispatchers.Default).launch { worker.lock.withLock { game.close() } }
    }
}

/**
 * Absolute seats to one player's: the player becomes seat 0 and their team
 * "us" (team 0). The offline game is already seen from seat 0.
 */
object Seats {
    fun rel(seat: Int, viewer: Int): Int = (seat - viewer + 4) % 4
    private fun relOrNone(seat: Int, viewer: Int) = if (seat < 0) seat else rel(seat, viewer)

    fun board(b: BoardState, viewer: Int): BoardState {
        if (viewer == 0) return b
        val swap = viewer % 2 == 1
        return b.copy(
            currentPlayer = relOrNone(b.currentPlayer, viewer),
            declarer = relOrNone(b.declarer, viewer),
            team0Points = if (swap) b.team1Points else b.team0Points,
            team1Points = if (swap) b.team0Points else b.team1Points,
            team0Tricks = if (swap) b.team1Tricks else b.team0Tricks,
            team1Tricks = if (swap) b.team0Tricks else b.team1Tricks,
            trickPlayers = b.trickPlayers.map { relOrNone(it, viewer) },
        )
    }

    fun view(v: TableView, viewer: Int): TableView {
        if (viewer == 0) return v
        val swap = viewer % 2 == 1
        return v.copy(
            board = v.board?.let { board(it, viewer) },
            handCounts = List(4) { v.handCounts[(it + viewer) % 4] },
            dealer = rel(v.dealer, viewer),
            usScore = if (swap) v.themScore else v.usScore,
            themScore = if (swap) v.usScore else v.themScore,
            meegaanSeat = v.meegaanSeat?.let { rel(it, viewer) },
            terugSeat = v.terugSeat?.let { rel(it, viewer) },
            asked = v.asked?.let { rel(it, viewer) },
        )
    }

    /** [event] as [viewer] sees it: their own call becomes [TableEvent.YourCall]. */
    fun event(event: TableEvent, viewer: Int): TableEvent {
        if (event is TableEvent.Waiting && event.seat == viewer) return TableEvent.YourCall
        if (viewer == 0) return event
        val swap = viewer % 2 == 1
        return when (event) {
            is TableEvent.Declared -> event.copy(seat = rel(event.seat, viewer))
            is TableEvent.Knocked -> event.copy(seat = rel(event.seat, viewer))
            is TableEvent.KnockedBack -> event.copy(seat = rel(event.seat, viewer))
            is TableEvent.Waiting -> event.copy(seat = rel(event.seat, viewer))
            is TableEvent.CardPlayed -> event.copy(seat = rel(event.seat, viewer), trick = event.trick?.let { t ->
                t.copy(players = t.players.map { rel(it, viewer) }, taker = rel(t.taker, viewer), before = board(t.before, viewer))
            })
            is TableEvent.DealScored -> if (swap) TableEvent.DealScored(event.themDelta, event.usDelta) else event
            TableEvent.DealStarted, TableEvent.YourCall, TableEvent.PlayStarted, TableEvent.Resynced -> event
        }
    }
}
