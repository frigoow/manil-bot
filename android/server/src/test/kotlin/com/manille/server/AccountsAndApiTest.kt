package com.manille.server

import com.manille.shared.*
import io.ktor.server.cio.CIO
import io.ktor.server.engine.EmbeddedServer
import io.ktor.server.engine.embeddedServer
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import java.io.File
import java.net.ServerSocket
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.util.concurrent.Executors
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * The server as the home of players and games: accounts that survive a
 * restart and move to another device, statistics counted by the server,
 * games stored with every card, and the JSON API (public and admin) on a
 * real server with real clients.
 */
class AccountsAndApiTest {
    private val engines = Engines(File(System.getProperty("manille.resources") ?: error("run through Gradle")))
    private val worker = Worker()
    private val bots = engines.bots(ruleAgent = false)
    private val dbFile = Files.createTempFile("manille-test", ".db").toFile().also { it.deleteOnExit() }
    private val executor = Executors.newSingleThreadExecutor()
    private val main = executor.asCoroutineDispatcher()
    private val http = HttpClient.newHttpClient()
    private val sessions = mutableListOf<RemoteSession>()
    private var instance = Instance()

    /** One server process: a hub on the test database, on a free port. */
    private inner class Instance {
        val store = Store(dbFile.path)
        val hub = Hub(engines.platform, bots, worker, ServerConfig(graceMs = 300, pacing = Pacing(0, 0)), store)
        val port = ServerSocket(0).use { it.localPort }
        val server: EmbeddedServer<*, *> = embeddedServer(CIO, port = port, host = "127.0.0.1") { manille(hub, ADMIN) }.start(wait = false)
        val url = "ws://127.0.0.1:$port/play"
        fun stop() {
            server.stop(100, 1000)
            runBlocking { withContext(hub.game) { hub.close() } }
            store.close()
        }
    }

    @AfterTest fun tearDown() {
        sessions.forEach { s -> executor.submit { s.close() }.get() }
        instance.stop()
        executor.shutdown()
        runBlocking { worker.lock.lock(); worker.lock.unlock() }
    }

    private inner class Client(val name: String, seed: Int, token: String? = null, transfer: String? = null) {
        val rng = Random(seed)
        val session = RemoteSession(instance.url, name, token, main, transfer = transfer).also { sessions += it }
        val events = mutableListOf<Pair<TableEvent, TableView>>()
        val problems = mutableListOf<String>()
        var info: ServerMessage.TableInfo? = null
        var stats: Stats? = null
        var transferCode: String? = null
        var welcomed = false
        private var playedAt = -1

        init {
            session.onToken = { welcomed = true }
            session.onInfo = { info = it }
            session.onProblem = { code, _ -> problems += code }
            session.onStats = { stats = it }
            session.onTransferCode = { code, _ -> transferCode = code }
            session.listener = { event, view -> events += event to view; move(event, view) }
        }

        private fun move(event: TableEvent, view: TableView) {
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
        fun <T> get(block: () -> T): T = executor.submit<T> { block() }.get()
        fun waitFor(timeoutMs: Long = 30_000, what: String, ok: () -> Boolean) {
            val end = System.currentTimeMillis() + timeoutMs
            while (!get(ok)) {
                if (System.currentTimeMillis() > end) fail("timed out waiting for $what ($name)")
                Thread.sleep(10)
            }
        }
        fun welcome() = apply { waitFor(10_000, "welcome") { welcomed } }
        val id get() = get { session.playerId }
        val token get() = get { session.token }
    }

    private fun api(path: String, admin: String? = null): Pair<Int, JsonElement?> {
        val request = HttpRequest.newBuilder(URI("http://127.0.0.1:${instance.port}$path")).GET()
            .apply { if (admin != null) header("Authorization", "Bearer $admin") }.build()
        val response = http.send(request, HttpResponse.BodyHandlers.ofString())
        return response.statusCode() to runCatching { Json.parseToJsonElement(response.body()) }.getOrNull()
    }

    private fun post(path: String, admin: String?): Int = http.send(
        HttpRequest.newBuilder(URI("http://127.0.0.1:${instance.port}$path")).POST(HttpRequest.BodyPublishers.noBody())
            .apply { if (admin != null) header("Authorization", "Bearer $admin") }.build(),
        HttpResponse.BodyHandlers.ofString()).statusCode()

    private val JsonElement.obj get() = jsonObject
    private operator fun JsonElement.get(key: String): JsonElement = jsonObject[key] ?: JsonNull

    /** Host and partner at a quick-game table, started. */
    private fun quickTable(): Triple<Client, Client, String> {
        val host = Client("Vivi", 1).welcome()
        host.on { host.session.create(match = false) }
        host.waitFor(what = "a table") { host.info != null }
        val code = host.get { host.info!!.code }
        val guest = Client("Jonas", 2).welcome()
        guest.on { guest.session.join(code) }
        guest.waitFor(what = "joining") { guest.info?.code == code }
        host.on { host.session.startGame() }
        return Triple(host, guest, code)
    }

    @Test fun aPlayerIsKnownAgainAfterTheServerRestarts() {
        val first = Client("Vivi", 1).welcome()
        val id = assertNotNull(first.id)
        val token = assertNotNull(first.token)
        assertEquals(10, id.length)
        assertNotEquals(id, token, "the public id is not the secret")
        first.on { first.session.close() }
        instance.stop()
        instance = Instance()   // a new process on the same database file
        val again = Client("Vivi B", 3, token = token).welcome()
        assertEquals(id, again.id)
        assertEquals(token, again.token)
        val (code, page) = api("/api/players/$id")
        assertEquals(200, code)
        assertEquals("Vivi B", page!!["profile"]["name"].jsonPrimitive.content, "the name follows the latest hello")
    }

    @Test fun aTransferCodeMovesThePlayerToAnotherDevice() {
        val phone = Client("Vivi", 1).welcome()
        phone.on { phone.session.requestTransferCode() }
        phone.waitFor(what = "a transfer code") { phone.transferCode != null }
        val code = phone.get { phone.transferCode!! }
        assertEquals(8, code.length)
        val laptop = Client("Vivi", 5, transfer = code.lowercase()).welcome()
        assertEquals(phone.id, laptop.id, "the same player")
        assertNotEquals(phone.token, laptop.token, "with its own token")
        // a code works once
        val other = Client("Mallory", 6, transfer = code)
        other.waitFor(what = "the refusal") { other.problems.contains(Problems.BAD_TRANSFER) }
        // the old device still plays as the player (a new connection replaces the other device's)
        phone.on { phone.session.close() }
        val phoneAgain = Client("Vivi", 7, token = phone.token).welcome()
        assertEquals(phone.id, phoneAgain.id)
    }

    @Test fun aGameIsStoredAndCountsForThePlayersWhoPlayedIt() {
        val (host, guest, code) = quickTable()
        host.waitFor(60_000, "the end of the deal") { host.events.any { it.first is TableEvent.DealScored } }
        guest.waitFor(10_000, "the guest's end") { guest.events.any { it.first is TableEvent.DealScored } }
        val scored = host.get { host.events.firstNotNullOf { it.first as? TableEvent.DealScored } }
        val board = host.get { host.events.last { it.first is TableEvent.DealScored }.second.board!! }

        // the players' statistics, as the app counts them
        for (c in listOf(host, guest)) {
            c.on { c.session.requestStats() }
            c.waitFor(what = "stats") { c.stats != null }
            val s = c.get { c.stats!! }
            assertEquals(1, s["hands"]); assertEquals(1, s["games"])
            assertEquals(1, s["hands_won"] + s["hands_lost"] + s["hands_tied"])
            assertEquals(board.team0Points, s["points_for"], "${c.name} and the host are partners: the same side")
            assertEquals(scored.usDelta, s["score_for"])
            assertEquals(1, s.history.size)
        }

        // the game, with every card
        val (_, games) = api("/api/games")
        val game = games!!.jsonArray.single()
        val gameId = game["id"].jsonPrimitive.content
        assertEquals(code, game["table"].jsonPrimitive.content)
        assertEquals(if (board.team0Points > board.team1Points) "team0" else if (board.team1Points > board.team0Points) "team1" else "tie",
            game["outcome"].jsonPrimitive.content)
        val (_, detail) = api("/api/games/$gameId")
        val deal = detail!!["deals"].jsonArray.single()
        assertEquals(32, deal["plays"].jsonArray.size)
        val hands = deal["hands"].jsonArray.map { it.jsonPrimitive.long }
        assertEquals(List(4) { 8 }, hands.map { it.countOneBits() })
        assertEquals(-1L ushr 32, hands.fold(0L) { m, h -> m or h }, "the four hands are the deck")
        assertEquals(60, deal["points"].jsonArray.sumOf { it.jsonPrimitive.int })
        assertEquals(setOf("Vivi", "Jonas"), detail["players"].jsonArray.map { it["name"].jsonPrimitive.content }.toSet())
        // the player's page shows the game
        val (_, page) = api("/api/players/${host.id}")
        assertEquals(gameId, page!!["recent"].jsonArray.single()["id"].jsonPrimitive.content)
        assertEquals(1, page["stats"]["hands"].jsonPrimitive.int)
    }

    @Test fun theApiShowsATableButNeverAHandInPlay() {
        val (host, guest, code) = quickTable()
        host.waitFor(30_000, "cards on the table") { host.events.count { it.first is TableEvent.CardPlayed } >= 2 }
        val (status, tables) = api("/api/tables")
        assertEquals(200, status)
        val table = tables!!.jsonArray.single { it["code"].jsonPrimitive.content == code }
        assertEquals(listOf("Vivi", "Jonas"), listOf(0, 2).map { table["seats"].jsonArray[it]["name"].jsonPrimitive.content })
        assertNotNull(table["seats"].jsonArray[0]["player"].jsonPrimitive.content)
        val (_, detail) = api("/api/tables/$code")
        assertTrue(detail!!["admin"] is JsonNull, "no admin part in public")
        assertTrue("hands" !in detail["deal"].obj, "no hands in a public live deal")
        assertEquals(404, api("/api/tables/ZZZZZ").first)
        // admin: refused without (or with a wrong) token, everything with it
        assertEquals(401, api("/admin/tables").first)
        assertEquals(401, api("/admin/tables", admin = "wrong").first)
        val (adminStatus, full) = api("/admin/tables/$code", admin = ADMIN)
        assertEquals(200, adminStatus)
        val adminHands = full!!["admin"]["hands"]
        assertTrue(adminHands is JsonArray && adminHands.size == 4)
        val (_, st) = api("/admin/status", admin = ADMIN)
        assertEquals(1, st!!["tables"].jsonPrimitive.int)
        assertEquals(2, st["connected"].jsonPrimitive.int)
        val (_, online) = api("/admin/players", admin = ADMIN)
        assertEquals(setOf(code), online!!.jsonArray.map { it["table"].jsonPrimitive.content }.toSet())
        // closing it ends the game as abandoned
        assertEquals(200, post("/admin/tables/$code/close", ADMIN))
        host.waitFor(what = "the table closed") { host.info == null }
        val (_, games) = api("/api/games")
        assertEquals("abandoned", games!!.jsonArray.single()["outcome"].jsonPrimitive.content)
    }

    @Test fun theAdminApiIsOffWithoutAToken() {
        // the test instance has a token; served without one, admin answers 401 to everything
        val bare = embeddedServer(CIO, port = 0, host = "127.0.0.1") { manille(instance.hub, null) }.start(wait = false)
        try {
            val port = runBlocking { bare.engine.resolvedConnectors().first().port }
            val status = http.send(HttpRequest.newBuilder(URI("http://127.0.0.1:$port/admin/status"))
                .header("Authorization", "Bearer ").GET().build(), HttpResponse.BodyHandlers.ofString()).statusCode()
            assertEquals(401, status)
        } finally { bare.stop(100, 1000) }
    }

    private companion object { const val ADMIN = "test-admin-token" }
}
