package com.manille.shared

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
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * A table with several players ([Referee], as the server runs it): every
 * player sees only their own cards, from their own seat, and is asked
 * exactly their own calls; a player who drops out is replaced by a bot.
 */
class RefereeTest {
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
    private val worker = Worker()

    @AfterTest fun tearDown() {
        executor.shutdown()
        executor.awaitTermination(30, java.util.concurrent.TimeUnit.SECONDS)
        runBlocking { worker.lock.lock(); worker.lock.unlock() }
        closeables.reversed().forEach { runCatching { it.close() } }
    }

    private fun bots(): Bots = Bots(
        ruleBot = null,
        qnet = platform.qModel(ModelNames.Q_TRUMP).also { closeables += it },
        declarationNet = platform.declarationModel().also { closeables += it },
        announceNet = platform.announceModel().also { closeables += it },
    )

    /** What one run saw: per seat, the events as that player got them. */
    private class Run {
        val seen = mutableMapOf<Int, MutableList<Pair<TableEvent, TableView>>>()
        val asked = mutableListOf<Pair<Int, TablePhase>>()
        val dealers = mutableListOf<Int>()
        var problem: String? = null
        val finished = CompletableDeferred<Unit>()
        fun check(ok: Boolean, what: () -> String) { if (!ok && problem == null) problem = what() }
    }

    /**
     * Plays a game at a table where [players] are people choosing at random;
     * [onAsked] may act instead of the player (return true when it did).
     */
    private fun play(players: Set<Int>, seed: Int, match: Boolean, until: (TableEvent, Referee) -> Boolean,
                     onAsked: (Referee, Int, TableEvent) -> Boolean = { _, _, _ -> false }): Run {
        val referee = Referee(platform, bots(), worker, players, Pacing(0, 0), Random(seed), main).also { closeables += it }
        val rng = Random(seed + 1)
        val run = Run()
        fun later(block: () -> Unit) = executor.execute { if (!run.finished.isCompleted) block() }
        referee.listener = listener@{ event ->
            if (run.finished.isCompleted) return@listener
            if (event == TableEvent.DealStarted) run.dealers += referee.view(0).dealer
            for (seat in referee.humans) {
                val view = Seats.view(referee.view(seat), seat)
                val mine = Seats.event(event, seat)
                run.seen.getOrPut(seat) { mutableListOf() } += mine to view
                run.check(view.hand == referee.openHand(seat)) { "seat $seat sees cards that are not its own" }
                run.check(view.legalCards.isEmpty() || view.board!!.currentPlayer == HUMAN) { "seat $seat may play out of turn" }
                if (mine is TableEvent.CardPlayed) {
                    val abs = event as TableEvent.CardPlayed
                    run.check(mine.seat == Seats.rel(abs.seat, seat)) { "card seat not rotated for $seat" }
                    mine.trick?.let { t -> run.check(t.players.last() == mine.seat) { "trick players not rotated" } }
                }
                if (mine is TableEvent.DealScored) {
                    val abs = event as TableEvent.DealScored
                    run.check(mine.usDelta == if (seat % 2 == 0) abs.usDelta else abs.themDelta) { "score not from $seat's side" }
                    run.check(view.usScore == if (seat % 2 == 0) referee.view(0).usScore else referee.view(0).themScore) { "match score not from $seat's side" }
                }
            }
            if (run.problem != null || until(event, referee)) { referee.stop(); run.finished.complete(Unit); return@listener }
            when {
                event is TableEvent.Waiting -> {
                    val seat = event.seat
                    run.asked += seat to event.call
                    run.check(seat in referee.humans) { "a bot seat $seat was asked" }
                    // every other player sees the wait, only the asked one gets the panel
                    for (other in referee.humans) {
                        val v = referee.view(other)
                        if (other == seat) run.check(v.phase == event.call) { "asked seat sees ${v.phase}" }
                        else run.check(v.phase == TablePhase.PLAYING && v.asked == seat && v.askedCall == event.call) { "seat $other sees ${v.phase}" }
                    }
                    if (onAsked(referee, seat, event)) return@listener
                    later {
                        val ok = when (event.call) {
                            TablePhase.DECLARATION -> referee.declare(seat, rng.nextInt(-1, 4))
                            else -> {
                                run.check(!referee.answer((seat + 1) % 4, event.call, true)) { "another seat answered for $seat" }
                                referee.answer(seat, event.call, rng.nextInt(3) == 0)
                            }
                        }
                        run.check(ok) { "${event.call} by $seat refused" }
                    }
                }
                event is TableEvent.DealScored && referee.handOver -> later { referee.newDeal() }
                else -> {
                    val b = referee.view(0).board
                    if (b != null && !b.finished && b.currentPlayer in referee.humans) {
                        val seat = b.currentPlayer
                        val legal = referee.view(seat).legalCards
                        if (legal.isNotEmpty()) later {
                            val card = legal[rng.nextInt(legal.size)]
                            run.check(!referee.play((seat + 1) % 4, card)) { "a card was played out of turn" }
                            referee.play(seat, card)
                        }
                    }
                }
            }
        }
        executor.execute { referee.start(match) }
        runBlocking { withTimeout(120_000) { run.finished.await() } }
        run.problem?.let { fail(it) }
        return run
    }

    private fun Referee.over() = view(0).phase == TablePhase.MATCH_OVER

    @Test fun fourPlayersPlayAMatch() {
        var final = TableView.LOBBY
        val run = play(setOf(0, 1, 2, 3), seed = 3, match = true, until = { e, r -> final = r.view(0); e is TableEvent.DealScored && r.over() })
        assertEquals(TablePhase.MATCH_OVER, final.phase)
        assertTrue(maxOf(final.usScore, final.themScore) >= WINNING_SCORE)
        // every dealer was asked to declare (all are players), the deal passing to the left
        assertEquals(run.dealers, run.asked.filter { it.second == TablePhase.DECLARATION }.map { it.first })
        assertTrue(run.dealers.zipWithNext().all { (a, b) -> b == (a + 1) % 4 })
        for (seat in 0..3) {
            val mine = run.seen.getValue(seat)
            val asked = run.asked.filter { it.first == seat }.size
            assertEquals(asked, mine.count { it.first == TableEvent.YourCall }, "seat $seat's calls")
            assertTrue(mine.none { (e, _) -> e is TableEvent.Waiting && e.seat == HUMAN }, "seat $seat waited for itself")
            val scored = mine.mapNotNull { it.first as? TableEvent.DealScored }
            assertEquals(mine.last().second.usScore, scored.sumOf { it.usDelta }, "seat $seat's own score")
        }
        println("4 players: ${run.asked.size} calls asked, final ${final.usScore}-${final.themScore}")
    }

    @Test fun twoPlayersAgainstBotsInEverySeating() {
        for ((i, players) in listOf(setOf(0, 2), setOf(1, 3), setOf(0, 1), setOf(2, 3)).withIndex()) {
            for (seed in 0 until 8) {
                val run = play(players, seed = 100 + 10 * i + seed, match = false, until = { e, _ -> e is TableEvent.DealScored })
                assertTrue(run.asked.all { it.first in players })
                for (seat in players) assertEquals(32, run.seen.getValue(seat).count { it.first is TableEvent.CardPlayed })
            }
        }
    }

    @Test fun aBotTakesOverWhoDropsOutAndThePlayerComesBack() {
        var droppedAt: TableEvent? = null
        var back = false
        lateinit var table: Referee
        val run = play(setOf(0, 2), seed = 41, match = true, until = { e, r -> table = r; e is TableEvent.DealScored && r.view(0).handNumber >= 3 },
            onAsked = { r, seat, event ->
                // seat 2 walks away the first time it is asked anything; the bot answers for it
                if (seat == 2 && droppedAt == null) {
                    droppedAt = event
                    executor.execute { r.setHuman(2, false) }
                    true
                } else false
            }) 
        assertTrue(droppedAt != null, "seat 2 was never asked")
        assertTrue(2 !in table.humans)
        // the deal went on without seat 2: the bot answered, the cards were played
        assertTrue(run.seen.getValue(0).count { it.first is TableEvent.DealScored } >= 3)
        // and the seat can be taken back
        executor.submit { table.setHuman(2, true); back = 2 in table.humans }.get()
        assertTrue(back)
    }

    @Test fun aSavedCallRoundCarriesOnWithTheRightSeat() {
        // stop when a player is asked to knock, keep the save
        var save: SessionSave? = null
        var askedSeat = -1
        play(setOf(1, 3), seed = 7, match = true, until = { e, r ->
            if (e is TableEvent.Waiting && e.call == TablePhase.MEEGAAN) { save = r.save(); askedSeat = e.seat; true } else false
        })
        val saved = save ?: fail("no meegaan asked")
        assertEquals(askedSeat, saved["asked"].toString().toInt())
        // a fresh table from that save asks the same seat
        val referee = Referee(platform, bots(), worker, setOf(1, 3), Pacing(0, 0), Random(1), main).also { closeables += it }
        executor.submit {
            referee.restore(saved)
            val v = referee.view(askedSeat)
            check(v.phase == TablePhase.MEEGAAN) { "restored as ${v.phase}" }
            check(referee.view((askedSeat + 2) % 4).phase == TablePhase.PLAYING)
            check(referee.answer(askedSeat, TablePhase.MEEGAAN, false))
        }.get()
    }
}
