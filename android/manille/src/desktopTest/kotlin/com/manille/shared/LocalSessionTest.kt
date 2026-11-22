package com.manille.shared

import com.cardtable.platform.*

import com.manille.jvm.JvmPlatform
import com.manille.jvm.NativeLibrary
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import java.io.File
import java.nio.file.Files
import java.util.concurrent.Executors
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * Whole games through [LocalSession] with the real C++ engine and nets (the
 * desktop build's; see shared/build.gradle.kts), seat 0 played by a random
 * but legal "human". No waiting: [Pacing] is zero.
 */
class LocalSessionTest {
    private val resources = File(System.getProperty("manille.resources") ?: error("run through Gradle: manille.resources"))
    private val platform: Platform by lazy {
        val library = File(resources, "native/" + System.mapLibraryName("manille_mobile"))
        NativeLibrary.load = { System.load(library.absolutePath) }
        JvmPlatform(
            modelFile = { File(resources, "models/$it").also { f -> check(f.isFile) { "missing $f" } } },
            dataDir = Files.createTempDirectory("manille-test").toFile(),
            logger = { println("manille: $it") },
        )
    }
    private val executor = Executors.newSingleThreadExecutor()
    private val main = executor.asCoroutineDispatcher()
    private val closeables = mutableListOf<AutoCloseable>()
    private val workers = mutableListOf<Worker>()

    // Nothing may touch the nets once they are closed: first let the table
    // thread and every bot job finish, then close.
    @AfterTest fun tearDown() {
        executor.shutdown()
        executor.awaitTermination(30, java.util.concurrent.TimeUnit.SECONDS)
        runBlocking { workers.forEach { w -> w.lock.lock(); w.lock.unlock() } }
        closeables.reversed().forEach { runCatching { it.close() } }
    }

    private fun fastBots(): Bots = Bots(
        ruleBot = null,
        qnet = platform.qModel(ModelNames.Q_TRUMP).also { closeables += it },
        declarationNet = platform.declarationModel().also { closeables += it },
        announceNet = platform.announceModel().also { closeables += it },
    )

    private fun ruleBots(): Bots {
        val q = platform.qModel(ModelNames.Q_TRUMP).also { closeables += it }
        val announce = platform.announceModel().also { closeables += it }
        return Bots(RuleBot(platform, q, announce).also { closeables += it }, q,
            platform.declarationModel().also { closeables += it }, announce)
    }

    /**
     * Seat 0 plays [session] at random until [done] says stop; every event
     * is checked against the rules on the way. Returns what was seen.
     */
    private class Table(val session: LocalSession, val rng: Random, val done: (TableEvent, TableView) -> Boolean) {
        val events = mutableListOf<Pair<TableEvent, TableView>>()
        val finished = CompletableDeferred<Unit>()
        var deals = 0
        var cardsThisDeal = 0
        var problem: String? = null
        /** The session's save at the moment [done] said stop (the session is stopped then). */
        var savedAtDone: SessionSave? = null

        fun check(ok: Boolean, what: () -> String) { if (!ok && problem == null) problem = what() }

        fun onEvent(event: TableEvent, view: TableView) {
            if (finished.isCompleted) return
            events += event to view
            check(view.hand == session.openHand(HUMAN)) { "the view shows cards that are not seat 0's" }
            when (event) {
                TableEvent.DealStarted -> { deals += 1; cardsThisDeal = 0 }
                is TableEvent.CardPlayed -> {
                    cardsThisDeal += 1
                    check(event.legal != null && event.card in event.legal) { "illegal card ${event.card} by ${event.seat}" }
                    event.trick?.let { t ->
                        check(t.cards.size == 4 && t.players.toSet().size == 4) { "a trick of ${t.cards}" }
                        check(t.taker in t.players) { "taker ${t.taker} not in ${t.players}" }
                    }
                }
                is TableEvent.DealScored -> {
                    val b = view.board!!
                    check(cardsThisDeal == 32) { "a deal of $cardsThisDeal cards" }
                    check(b.team0Points + b.team1Points == 60) { "points ${b.team0Points}+${b.team1Points}" }
                    check(b.team0Tricks + b.team1Tricks == 8) { "tricks ${b.team0Tricks}+${b.team1Tricks}" }
                    val expected = (maxOf(b.team0Points, b.team1Points) - 30) * b.multiplier
                    val (winner, loser) = if (b.team0Points >= b.team1Points) event.usDelta to event.themDelta else event.themDelta to event.usDelta
                    check(winner == expected && loser == 0) { "scored ${event.usDelta}/${event.themDelta} for ${b.team0Points}-${b.team1Points} x${b.multiplier}" }
                    check(view.phase == TablePhase.HAND_OVER || view.phase == TablePhase.MATCH_OVER) { "phase ${view.phase} after scoring" }
                }
                else -> {}
            }
            if (problem != null || done(event, view)) {
                savedAtDone = session.save()
                session.stop()
                finished.complete(Unit)
                return
            }
            // answer later, not from inside the session's own call
            when {
                event == TableEvent.YourCall && view.phase == TablePhase.DECLARATION ->
                    later { check(session.declare(rng.nextInt(-1, 4))) { "declare refused" } }
                event == TableEvent.YourCall && view.phase == TablePhase.MEEGAAN ->
                    later { check(session.answerMeegaan(rng.nextInt(3) == 0)) { "meegaan refused" } }
                event == TableEvent.YourCall && view.phase == TablePhase.TERUG ->
                    later { check(session.answerTerug(rng.nextInt(3) == 0)) { "terug refused" } }
                event is TableEvent.DealScored && view.phase == TablePhase.HAND_OVER ->
                    later { session.newDeal() }
                view.phase == TablePhase.PLAYING && view.legalCards.isNotEmpty() -> {
                    val card = view.legalCards[rng.nextInt(view.legalCards.size)]
                    later { check(session.play(card)) { "card $card refused" } }
                }
            }
        }

        lateinit var later: (() -> Unit) -> Unit
    }

    private fun play(bots: Bots, seed: Int, timeoutMs: Long = 120_000, cardsBefore: Int = 0, setup: (LocalSession) -> Unit,
                     done: (TableEvent, TableView) -> Boolean): Table {
        val worker = Worker().also { workers += it }
        val session = LocalSession(platform, bots, worker, Pacing(0, 0), Random(seed), main).also { closeables += it }
        val table = Table(session, Random(seed + 1), done)
        table.cardsThisDeal = cardsBefore
        table.later = { block -> executor.execute { if (!table.finished.isCompleted) block() } }
        session.listener = table::onEvent
        executor.execute { setup(session) }
        runBlocking { withTimeout(timeoutMs) { table.finished.await() } }
        table.problem?.let { fail(it) }
        return table
    }

    @Test fun aMatchToTheEnd() {
        val table = play(fastBots(), seed = 1, setup = { it.start(match = true) }) { _, view -> view.phase == TablePhase.MATCH_OVER }
        val last = table.events.last().second
        assertTrue(last.usScore >= WINNING_SCORE || last.themScore >= WINNING_SCORE)
        val scored = table.events.mapNotNull { it.first as? TableEvent.DealScored }
        assertEquals(last.usScore, scored.sumOf { it.usDelta })
        assertEquals(last.themScore, scored.sumOf { it.themDelta })
        assertEquals(table.deals, scored.size)
        println("match: ${table.deals} deals, ${last.usScore}-${last.themScore}")
    }

    @Test fun manyQuickGamesCoverEveryCall() {
        val kinds = mutableSetOf<String>()
        for (seed in 10 until 40) {
            val table = play(fastBots(), seed, setup = { it.start(match = false) }) { event, _ -> event is TableEvent.DealScored }
            table.events.forEach { (e, v) ->
                when (e) {
                    is TableEvent.Declared -> kinds += if (e.seat == HUMAN) "human declares" else "bot declares"
                    is TableEvent.Knocked -> kinds += if (e.seat == HUMAN) "human knocks" else "bot knocks"
                    is TableEvent.KnockedBack -> kinds += if (e.seat == HUMAN) "human knocks back" else "bot knocks back"
                    TableEvent.YourCall -> kinds += "asked: ${v.phase}"
                    else -> {}
                }
            }
            assertEquals(TablePhase.HAND_OVER, table.events.last().second.phase)
        }
        println("seen: ${kinds.sorted()}")
        for (k in listOf("human declares", "bot declares", "asked: DECLARATION", "asked: MEEGAAN")) assertTrue(k in kinds, "never saw: $k")
    }

    @Test fun theRuleAgentPlaysADeal() {
        val table = play(ruleBots(), seed = 5, timeoutMs = 300_000, setup = { it.start(match = false) }) { event, _ -> event is TableEvent.DealScored }
        assertEquals(32, table.events.count { it.first is TableEvent.CardPlayed })
    }

    /** The mull nets load on demand, can be released and come back (RuleBot). */
    @Test fun theRuleAgentLoadsTheMullNetsWhenNeeded() {
        val q = platform.qModel(ModelNames.Q_TRUMP).also { closeables += it }
        val announce = platform.announceModel().also { closeables += it }
        val bot = RuleBot(platform, q, announce).also { closeables += it }
        fun deal(trump: Int, seed: Long) {
            val g = platform.newEngine()
            try {
                g.newDeal(0, seed, trump); g.startPlaying()
                val calls = RuleBot.Calls(0, null, null, 0, 0, false, emptySet())
                while (!g.state().finished) {
                    val card = bot.choose(g, calls)
                    assertTrue(card in g.legalCards(), "illegal card $card")
                    g.applyCard(card)
                }
                assertEquals(60, g.state().team0Points + g.state().team1Points)
            } finally { g.close() }
        }
        deal(trump = -1, seed = 1)   // loads them
        bot.releaseMull()
        deal(trump = 2, seed = 2)    // a suit deal without them
        bot.prepareMull()
        deal(trump = -1, seed = 3)   // back again
    }

    @Test fun aSavedGameCarriesOn() {
        // play the first trick of a match, keep the save, leave the table
        val first = play(fastBots(), seed = 77, setup = { it.start(match = true) }) { event, _ ->
            event is TableEvent.CardPlayed && event.trick != null
        }
        val save = assertNotNull(first.savedAtDone)
        val played = first.events.count { it.first is TableEvent.CardPlayed }
        assertEquals(4, played)

        // a new session from the save finishes that deal: 32 cards in all
        val second = play(fastBots(), seed = 78, cardsBefore = played, setup = { s ->
            s.restore(save)
            s.resume()
            // when it is seat 0's move nothing happens until it plays: start it off
            if (s.view.legalCards.isNotEmpty()) s.listener!!(TableEvent.PlayStarted, s.view)
        }) { event, _ -> event is TableEvent.DealScored }
        assertEquals(32 - played, second.events.count { it.first is TableEvent.CardPlayed })
        val scored = second.events.last().second.board!!
        assertEquals(60, scored.team0Points + scored.team1Points)
    }
}
