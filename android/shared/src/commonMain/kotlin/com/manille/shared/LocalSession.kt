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
 * The offline table: seat 0 is the local player, the other three are bots.
 * Deals, the calls, the play and the score, with the C++ engine as the
 * rulebook (mirrors gui/game_session.py). Runs on [main]; a bot thinks on
 * the [worker] with a copy of the game, and its card comes back to [main].
 */
class LocalSession(
    private val platform: Platform,
    private val bots: Bots,
    private val worker: Worker,
    private val pacing: Pacing = Pacing(),
    private val random: Random = Random.Default,
    main: CoroutineDispatcher = Dispatchers.Main,
) : TableSession {
    private val game = platform.newEngine()
    private val scope = CoroutineScope(SupervisorJob() + main)
    private val mainDispatcher = main

    override var listener: ((TableEvent, TableView) -> Unit)? = null
    /** A bot's thinking time on a real decision (more than one legal card), in ms. */
    var onBotThought: ((Long) -> Unit)? = null

    private var phase = TablePhase.LOBBY
    private var matchMode = true
    private var usScore = 0
    private var themScore = 0
    private var handNumber = 0
    private var dealer = 0
    private var dealSeed = 0L
    // This deal's calls, for the bots' world weights.
    private var meegaanSeat: Int? = null
    private var terugSeat: Int? = null
    private var declUs = 0
    private var declThem = 0
    private var trumpRuleKnown = false
    /** The calls are done and the cards are being played (phase alone says PLAYING while a bot dealer's calls are still open). */
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

    override val view: TableView get() {
        if (phase == TablePhase.LOBBY) return TableView.LOBBY.copy(match = matchMode)
        val board = game.state()
        return TableView(
            phase = phase,
            board = board,
            hand = game.handMask(HUMAN),
            legalCards = if (cardsOpen && phase == TablePhase.PLAYING && board.currentPlayer == HUMAN && !board.finished)
                game.legalCards().toList() else emptyList(),
            handCounts = List(4) { game.handMask(it).countOneBits() },
            dealer = dealer, handNumber = handNumber, match = matchMode,
            usScore = usScore, themScore = themScore,
            meegaanSeat = meegaanSeat, terugSeat = terugSeat,
        )
    }

    private fun emit(event: TableEvent) { listener?.invoke(event, view) }

    // ---- the game's course --------------------------------------------------

    /** A new quick game (one deal) or match to [WINNING_SCORE]. */
    fun start(match: Boolean) {
        matchMode = match; usScore = 0; themScore = 0; handNumber = 0
        newDeal()
    }

    /** Leave the table: nothing more happens until the next [start] or [restore]. */
    fun stop() { cancelTurn(); phase = TablePhase.LOBBY; cardsOpen = false }

    /** The next deal (after HAND_OVER), dealt by the next seat. */
    fun newDeal() {
        cancelTurn()
        dealer = if (handNumber == 0) random.nextInt(4) else (dealer + 1) % 4
        if (!matchMode) { usScore = 0; themScore = 0 }
        dealSeed = random.nextLong()
        game.newDeal(dealer, dealSeed); handNumber += 1
        meegaanSeat = null; terugSeat = null; cardsOpen = false
        // match score from the declaring team's side, as the calls see it
        if (teamOf(dealer) == HUMAN_TEAM) { declUs = usScore; declThem = themScore } else { declUs = themScore; declThem = usScore }
        // a bot dealer without the declaration net keeps the C++ trump rule
        trumpRuleKnown = dealer != HUMAN && bots.declarationNet == null
        if (dealer == HUMAN) {
            phase = TablePhase.DECLARATION
            emit(TableEvent.DealStarted)
            emit(TableEvent.YourCall)
            return
        }
        phase = TablePhase.PLAYING
        emit(TableEvent.DealStarted)
        bots.declarationNet?.let { model ->
            val (us, them) = scoresFor(dealer)
            val equities = model.equities(game.handMask(dealer), us, them)
            val choice = equities.indices.maxBy { equities[it] }
            game.newDeal(dealer, dealSeed, if (choice == 4) -1 else choice)
        }
        emit(TableEvent.Declared(dealer, game.state().trumpSuit))
        afterDeclaration()
    }

    override fun declare(trump: Int): Boolean {
        if (phase != TablePhase.DECLARATION || trump !in -1..3) return false
        game.newDeal(dealer, dealSeed, trump)
        emit(TableEvent.Declared(HUMAN, trump))
        afterDeclaration()
        return true
    }

    override fun answerMeegaan(accept: Boolean): Boolean {
        if (phase != TablePhase.MEEGAAN) return false
        if (accept) {
            knock(HUMAN)
            botsKnockBack()
        } else if ((dealer + 3) % 4 == PARTNER) {
            // the human passed first; the partner bot still gets its say
            partnerKnocks()
        }
        beginPlaying()
        return true
    }

    override fun answerTerug(accept: Boolean): Boolean {
        if (phase != TablePhase.TERUG) return false
        if (accept) knockBack(HUMAN)
        // the human dealt and passed; the partner bot still gets its say
        else if (dealer == HUMAN) partnerKnocksBack()
        beginPlaying()
        return true
    }

    override fun play(card: Int): Boolean {
        val board = game.state()
        if (!cardsOpen || phase != TablePhase.PLAYING || board.finished || board.currentPlayer != HUMAN || card !in game.legalCards()) return false
        applyCard(card)
        return true
    }

    private fun knock(seat: Int) { game.announceDouble(seat); meegaanSeat = seat; emit(TableEvent.Knocked(seat)) }
    private fun knockBack(seat: Int) { game.announceRedouble(seat); terugSeat = seat; emit(TableEvent.KnockedBack(seat)) }

    private fun afterDeclaration() {
        if (teamOf(dealer) == HUMAN_TEAM) {
            // Both defenders are bots; seat order from the dealer's left, first
            // yes takes the meegaan (mirrors gui/game_session._bot_meegaan).
            val defender = listOf((dealer + 1) % 4, (dealer + 3) % 4).firstOrNull { botMeegaan(it) }
            if (defender == null) { beginPlaying(); return }
            knock(defender)
            // Our answer: dealer first, then partner, first yes takes it.
            // The partner bot answers before the human when it dealt.
            if (dealer == PARTNER && partnerKnocksBack()) { beginPlaying(); return }
            phase = TablePhase.TERUG
            emit(TableEvent.YourCall)
        } else {
            // Defenders from the dealer's left, first yes takes the meegaan:
            // the partner bot answers before the human when it sits there.
            if ((dealer + 1) % 4 == PARTNER && partnerKnocks()) { beginPlaying(); return }
            phase = TablePhase.MEEGAAN
            emit(TableEvent.YourCall)
        }
    }

    /** The partner bot's meegaan against the opponents' declaration; the
     *  opponents (both bots) then answer it. False if it passes. */
    private fun partnerKnocks(): Boolean {
        if (!botMeegaan(PARTNER)) return false
        knock(PARTNER)
        botsKnockBack()
        return true
    }

    /** The partner bot's terug against the opponents' meegaan. False if it passes. */
    private fun partnerKnocksBack(): Boolean {
        if (!botTerug(PARTNER)) return false
        knockBack(PARTNER)
        return true
    }

    /** The declaring side is both bots (our side knocked against them):
     *  dealer first, then partner, first yes takes the terug. */
    private fun botsKnockBack() {
        listOf(dealer, (dealer + 2) % 4).firstOrNull { botTerug(it) }?.let { knockBack(it) }
    }

    private fun beginPlaying() {
        game.startPlaying(); phase = TablePhase.PLAYING; cardsOpen = true
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
            board.currentPlayer != HUMAN -> startBot(board.currentPlayer)
        }
    }

    private fun completeHand() {
        val usDelta = game.handScore(HUMAN_TEAM)
        val themDelta = game.handScore(1)
        usScore += usDelta; themScore += themDelta
        phase = if (matchMode && (usScore >= WINNING_SCORE || themScore >= WINNING_SCORE)) TablePhase.MATCH_OVER else TablePhase.HAND_OVER
        emit(TableEvent.DealScored(usDelta, themDelta))
    }

    // ---- the bots -------------------------------------------------------------

    private fun calls() = RuleBot.Calls(dealer, meegaanSeat, terugSeat, declUs, declThem, trumpRuleKnown, setOf(HUMAN))

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

    private fun stillTurnOf(player: Int): Boolean {
        val now = game.state()
        return cardsOpen && phase == TablePhase.PLAYING && !holding && !now.finished && now.currentPlayer == player
    }

    private fun qnetCard(player: Int): Int =
        bots.qnet?.chooseCard(game.gruRaw(player), game.gruLength(player), game.gruGlobal(player),
            game.actionFeatures(player), game.legalCards()) ?: game.legalCards().first()

    /** Match score (us, them) from [seat]'s team's side. */
    private fun scoresFor(seat: Int) = if (teamOf(seat) == HUMAN_TEAM) usScore to themScore else themScore to usScore

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

    override fun hintPosition(): HintPosition? {
        val board = game.state()
        if (!cardsOpen || phase != TablePhase.PLAYING || board.finished || board.currentPlayer != HUMAN) return null
        return HintPosition(game.copy(), calls(), positionKey())
    }

    /** Changes whenever the position does (a hint for an old position is dropped). */
    fun positionKey(): Int = if (phase == TablePhase.LOBBY) 0 else game.snapshot().contentHashCode()

    override fun openHand(seat: Int): Long? = if (phase == TablePhase.LOBBY) null else game.handMask(seat)

    // ---- the saved game -------------------------------------------------------------

    /** The session's part of the saved game (GameController adds its own), or null in the lobby. */
    fun save(): SessionSave? {
        if (phase == TablePhase.LOBBY) return null
        return buildJsonObject {
            put("match", matchMode); put("us", usScore); put("them", themScore)
            put("hand", handNumber); put("dealer", dealer); put("seed", dealSeed)
            put("phase", phase.name)
            put("meegaan", meegaanSeat ?: -1); put("terug", terugSeat ?: -1)
            put("declUs", declUs); put("declThem", declThem); put("ruleTrump", trumpRuleKnown)
            put("engine", JsonArray(game.snapshot().map { JsonPrimitive(it) }))
        }
    }

    /**
     * Back to the table [save] described; throws (and changes nothing a new
     * [start] would not reset) when it cannot be read. Call [resume] once the
     * screen shows it.
     */
    fun restore(save: JsonObject) {
        cancelTurn()
        val engine = save.optArray("engine") ?: error("no engine state")
        val restoredPhase = TablePhase.valueOf(save.getString("phase"))
        game.restore(IntArray(engine.size) { engine[it].jsonPrimitive.int })
        matchMode = save.getBoolean("match"); usScore = save.getInt("us"); themScore = save.getInt("them")
        handNumber = save.getInt("hand"); dealer = save.getInt("dealer"); dealSeed = save.getLong("seed")
        meegaanSeat = save.getInt("meegaan").takeIf { it >= 0 }
        terugSeat = save.getInt("terug").takeIf { it >= 0 }
        declUs = save.getInt("declUs"); declThem = save.getInt("declThem")
        trumpRuleKnown = save.getBoolean("ruleTrump")
        phase = restoredPhase
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
