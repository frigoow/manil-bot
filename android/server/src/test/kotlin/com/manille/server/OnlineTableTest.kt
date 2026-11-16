package com.manille.server

import com.manille.shared.*
import io.ktor.client.HttpClient
import io.ktor.client.plugins.websocket.WebSockets
import io.ktor.client.plugins.websocket.webSocket
import io.ktor.server.cio.CIO
import io.ktor.server.engine.EmbeddedServer
import io.ktor.server.engine.embeddedServer
import io.ktor.websocket.Frame
import io.ktor.websocket.readText
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import java.io.File
import java.net.ServerSocket
import java.util.concurrent.Executors
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * A real server on a local port and real clients ([RemoteSession]) playing
 * at it: who sits where, what each player gets to see, a player dropping
 * out and coming back, and what the server refuses.
 */
class OnlineTableTest {
    private val engines = Engines(File(System.getProperty("manille.resources") ?: error("run through Gradle")))
    private val worker = Worker()
    private val bots = engines.bots(ruleAgent = false)
    private var hub = Hub(engines.platform, bots, worker, ServerConfig(graceMs = 300, pacing = Pacing(0, 0)))
    private val port = ServerSocket(0).use { it.localPort }
    private val server: EmbeddedServer<*, *> = embeddedServer(CIO, port = port, host = "127.0.0.1") { manille(hub) }.start(wait = false)
    private val url = "ws://127.0.0.1:$port/play"
    private val executor = Executors.newSingleThreadExecutor()
    private val main = executor.asCoroutineDispatcher()
    private val sessions = mutableListOf<RemoteSession>()

    @AfterTest fun tearDown() {
        sessions.forEach { s -> executor.submit { s.close() }.get() }
        server.stop(100, 1000)
        hub.close()
        executor.shutdown()
        runBlocking { worker.lock.lock(); worker.lock.unlock() }
    }

    /** A client that plays at random, and remembers what it saw. */
    private inner class Client(val name: String, seed: Int, token: String? = null) {
        val rng = Random(seed)
        val session = RemoteSession(url, name, token, main).also { sessions += it }
        val events = mutableListOf<Pair<TableEvent, TableView>>()
        val problems = mutableListOf<String>()
        var info: ServerMessage.TableInfo? = null
        var playedAt = -1
        var autoplay = true
        val welcomed = CompletableDeferred<Unit>()
        var onEvent: (TableEvent, TableView) -> Unit = { _, _ -> }

        init {
            session.onToken = { welcomed.complete(Unit) }
            session.onInfo = { info = it }
            session.onProblem = { code, _ -> problems += code }
            session.listener = { event, view ->
                events += event to view
                onEvent(event, view)
                if (autoplay) move(event, view)
            }
        }

        fun move(event: TableEvent, view: TableView) {
            when {
                event == TableEvent.YourCall && view.phase == TablePhase.DECLARATION -> session.declare(rng.nextInt(-1, 4))
                event == TableEvent.YourCall && view.phase == TablePhase.MEEGAAN -> session.answerMeegaan(rng.nextInt(3) == 0)
                event == TableEvent.YourCall && view.phase == TablePhase.TERUG -> session.answerTerug(rng.nextInt(3) == 0)
                view.phase == TablePhase.PLAYING && view.legalCards.isNotEmpty() && view.hand.countOneBits() != playedAt -> {
                    playedAt = view.hand.countOneBits()
                    session.play(view.legalCards[rng.nextInt(view.legalCards.size)])
                }
            }
        }

        fun on(block: () -> Unit) = executor.submit(block).get()
        fun waitFor(timeoutMs: Long = 30_000, what: String, ok: () -> Boolean) {
            val end = System.currentTimeMillis() + timeoutMs
            while (!executor.submit<Boolean> { ok() }.get()) {
                if (System.currentTimeMillis() > end) fail("timed out waiting for $what (${name})")
                Thread.sleep(10)
            }
        }
        fun cardsPlayed() = executor.submit<Int> { events.count { it.first is TableEvent.CardPlayed } }.get()
    }

    private fun Client.awaitWelcome() = runBlocking { withTimeout(10_000) { welcomed.await() } }

    /** Host and guest at one table, guest in [guestSeat] (2 = partners), the game started. */
    private fun table(guestSeat: Int, match: Boolean = false): Pair<Client, Client> {
        val host = Client("Vivi", 1).also { it.awaitWelcome() }
        host.on { host.session.create(match) }
        host.waitFor(what = "a table") { host.info != null }
        val code = host.info!!.code
        assertEquals(5, code.length)
        val guest = Client("Jonas", 2).also { it.awaitWelcome() }
        guest.on { guest.session.join(code.lowercase()) }
        guest.waitFor(what = "joining") { guest.info?.code == code }
        assertEquals(2, guest.info!!.you, "the second player joins as partner")
        if (guestSeat != 2) {
            guest.on { guest.session.sit(guestSeat) }
            host.waitFor(what = "the guest's new chair") { host.info?.seats?.get(guestSeat)?.name == "Jonas" }
        }
        assertTrue(host.info!!.host); assertTrue(!guest.info!!.host)
        // only the host starts the game
        guest.on { guest.session.startGame() }
        guest.waitFor(what = "the guest's start refused") { guest.problems == listOf(Problems.NOT_HOST) }
        guest.on { guest.problems.clear() }
        host.on { host.session.startGame() }
        host.waitFor(what = "the deal") { host.events.isNotEmpty() }
        return host to guest
    }

    @Test fun twoPlayersFinishADealSeeingOnlyTheirOwnCards() {
        for (guestSeat in listOf(2, 1)) {
            val (host, guest) = table(guestSeat)
            host.waitFor(60_000, "the end of the deal") { host.events.any { it.first is TableEvent.DealScored } }
            guest.waitFor(10_000, "the guest's end of the deal") { guest.events.any { it.first is TableEvent.DealScored } }
            for (c in listOf(host, guest)) c.on {
                assertEquals(32, c.events.count { it.first is TableEvent.CardPlayed }, "${c.name}'s cards")
                // never the cards another seat could have played
                assertTrue(c.events.all { (e, _) -> e !is TableEvent.CardPlayed || e.legal == null })
                // every card a player saw played came from that seat's hand earlier (seen from their chair)
                assertTrue(c.problems.isEmpty(), "${c.name}: ${c.problems}")
            }
            // the two chairs' views of one deal agree, turned by their seats
            val hostScore = host.events.firstNotNullOf { it.first as? TableEvent.DealScored }
            val guestScore = guest.events.firstNotNullOf { it.first as? TableEvent.DealScored }
            if (guestSeat % 2 == 0) assertEquals(hostScore, guestScore)
            else assertEquals(TableEvent.DealScored(hostScore.themDelta, hostScore.usDelta), guestScore)
            val hostHand = host.events.first { it.first == TableEvent.PlayStarted || it.first is TableEvent.CardPlayed }.second.hand
            val guestHand = guest.events.first { it.first == TableEvent.PlayStarted || it.first is TableEvent.CardPlayed }.second.hand
            assertEquals(0L, hostHand and guestHand, "two players hold the same card")
            // the guest's first card, as the host saw it, came from the guest's seat
            val guestCard = guest.events.firstNotNullOf { (e, _) -> (e as? TableEvent.CardPlayed)?.takeIf { it.seat == 0 } }
            val asHost = host.events.firstNotNullOf { (e, _) -> (e as? TableEvent.CardPlayed)?.takeIf { it.card == guestCard.card } }
            assertEquals(guestSeat, asHost.seat)
            assertEquals(listOf("Vivi", "Jonas"), listOf(0, guestSeat).map { host.info!!.seats[it].name })
            assertTrue(host.info!!.seats.filterIndexed { i, _ -> i != 0 && i != guestSeat }.all { it.bot && it.name != null })
            // both ready: the next deal is dealt
            host.on { host.session.nextDeal() }
            host.waitFor(what = "the host's ready") { host.info?.ready == listOf(0) }
            guest.on { guest.session.nextDeal() }
            host.waitFor(what = "the second deal") { host.events.count { it.first == TableEvent.DealStarted } == 2 }
            host.on { host.autoplay = false; host.session.leave() }
            guest.on { guest.autoplay = false; guest.session.leave() }
            host.waitFor(what = "the table closing") { hub.tableCount == 0 }
        }
    }

    @Test fun aPlayerWhoDropsOutIsReplacedByABotAndGetsTheChairBack() {
        val (host, guest) = table(guestSeat = 2, match = true)
        // the guest's app goes away after the first card it plays
        val token = assertNotNull(guest.session.token)
        guest.waitFor(60_000, "the guest's first card") { guest.playedAt >= 0 }
        guest.on { guest.autoplay = false; guest.session.close() }
        host.waitFor(what = "the guest shown as away") { host.info?.seats?.get(2)?.away == true }
        // the deal goes on without them: the bot plays chair 2 after the grace time
        host.waitFor(60_000, "the deal without the guest") { host.events.any { it.first is TableEvent.DealScored } }
        val seenByHost = host.cardsPlayed()

        // back, with the same token: the same chair, and the table as it is now
        val back = Client("Jonas", 3, token).also { it.awaitWelcome() }
        back.waitFor(what = "the chair back") { back.info?.you == 2 }
        back.waitFor(what = "the table afresh") { back.events.any { it.first == TableEvent.Resynced } }
        back.on {
            val (_, view) = back.events.first { it.first == TableEvent.Resynced }
            assertEquals(TablePhase.HAND_OVER, view.phase)
            assertEquals(host.events.last { it.first is TableEvent.DealScored }.second.usScore, view.usScore)
            assertTrue(back.info!!.seats[2].let { !it.away && !it.bot })
        }
        assertTrue(seenByHost >= 32)
        // and the next deal waits for them as a player again
        host.on { host.session.nextDeal() }
        Thread.sleep(200)
        assertEquals(1, host.events.count { it.first == TableEvent.DealStarted })
        back.on { back.session.nextDeal() }
        host.waitFor(what = "the second deal") { host.events.count { it.first == TableEvent.DealStarted } == 2 }
        back.waitFor(60_000, "the returning player's cards") { back.events.count { it.first is TableEvent.CardPlayed } > 0 }
    }

    /** One raw connection: send [lines], collect what comes back. */
    private fun raw(vararg lines: String, expect: Int): List<ServerMessage> = runBlocking {
        val got = mutableListOf<ServerMessage>()
        HttpClient(io.ktor.client.engine.cio.CIO) { install(WebSockets) }.use { client ->
            withTimeout(10_000) {
                client.webSocket(url) {
                    for (line in lines) send(Frame.Text(line))
                    for (frame in incoming) {
                        if (frame is Frame.Text) got += ProtocolJson.decodeFromString(ServerMessage.serializer(), frame.readText())
                        if (got.size >= expect) break
                    }
                }
            }
        }
        got
    }

    @Test fun theServerRefusesWhatIsNotAllowed() {
        val hello = """{"type":"hello","protocol":$PROTOCOL_VERSION,"token":null,"name":"  \u0007Eve with a very long name indeed  "}"""
        // unreadable, out of order, before hello, other version
        assertEquals(Problems.HELLO_FIRST, (raw("""{"type":"create","match":true}""", expect = 1).single() as ServerMessage.Problem).code)
        assertEquals(Problems.VERSION, (raw("""{"type":"hello","protocol":999,"token":null,"name":"x"}""", expect = 1).single() as ServerMessage.Problem).code)
        val got = raw(hello, "not json", """{"type":"join","code":"ZZZZZ"}""", """{"type":"play","card":3}""", expect = 4)
        val welcome = got[0] as ServerMessage.Welcome
        assertEquals("Eve with a very long", welcome.name)
        assertEquals(listOf(Problems.BAD_MESSAGE, Problems.NO_TABLE, Problems.NO_TABLE), got.drop(1).map { (it as ServerMessage.Problem).code })
        // at a table: a card out of turn, a guest's start, a chair that is taken
        val host = Client("Vivi", 9).also { it.awaitWelcome() }
        host.autoplay = false
        host.on { host.session.create(true) }
        host.waitFor(what = "a table") { host.info != null }
        val code = host.info!!.code
        val atTable = raw(hello, """{"type":"join","code":"$code"}""", """{"type":"sit","seat":0}""", """{"type":"start"}""",
            """{"type":"play","card":3}""", expect = 5)
        val problems = atTable.filterIsInstance<ServerMessage.Problem>().map { it.code }
        assertEquals(listOf(Problems.SEAT_TAKEN, Problems.NOT_HOST, Problems.NOT_NOW), problems)
        assertNull(atTable.firstOrNull { it is ServerMessage.Update })
    }
}
