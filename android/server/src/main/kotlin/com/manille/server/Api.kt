package com.manille.server

import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.ApplicationCall
import io.ktor.server.response.respondText
import io.ktor.server.routing.Route
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.route
import kotlinx.coroutines.withContext
import kotlinx.serialization.KSerializer
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import java.security.MessageDigest

/*
 * The server's JSON API, read-only:
 *
 *   GET /api/tables                    open tables: chairs, score, phase
 *   GET /api/tables/{code}             one table now: the deal being played (the
 *                                      cards on the table, never a hand still in
 *                                      play) and this game's finished deals
 *   GET /api/games?limit=              finished games, newest first
 *   GET /api/games/{id}                one game: its players and every deal,
 *                                      with each deal's hands and cards in order
 *   GET /api/players?name=             players by name
 *   GET /api/players/{id}              a profile and its online statistics
 *   GET /api/players/{id}/games        that player's games
 *
 * and, with "Authorization: Bearer <MANILLE_ADMIN_TOKEN>" (off without it):
 *
 *   GET  /admin/status                 uptime, memory, connections, database
 *   GET  /admin/tables                 every table in full: hands, connections
 *   GET  /admin/tables/{code}          one table in full
 *   GET  /admin/players                who is connected, and where
 *   POST /admin/tables/{code}/close    close a table (its game: abandoned)
 *
 * Seats and teams are the table's own: seat 0..3, team = seat % 2. Cards are
 * suit * 8 + rank (ranks 7 8 9 J Q K A 10); a hand is a bit mask of cards.
 * Times are milliseconds since 1970.
 */

private val ApiJson = Json { encodeDefaults = true; explicitNulls = true }

@Serializable
data class LiveSeat(val name: String?, val bot: Boolean, val away: Boolean = false, val player: String? = null)

@Serializable
data class LiveTable(
    val code: String, val match: Boolean, val started: Boolean, val created: Long,
    /** The game's phase (LOBBY, DECLARATION, MEEGAAN, TERUG, PLAYING, HAND_OVER, MATCH_OVER), null before it starts. */
    val phase: String?,
    /** Deals dealt in this game. */
    val hand: Int,
    /** Match points per team. */
    val score: List<Int>,
    val seats: List<LiveSeat>,
    /** The game's id (see /api/games/{id}), null before the first. */
    val game: String?,
)

@Serializable
data class LiveDeal(
    val number: Int, val dealer: Int,
    /** 0..3, -1 for mull; null while the dealer still chooses. */
    val trump: Int?,
    val multiplier: Int, val meegaan: Int?, val terug: Int?,
    /** Whose card it is, or null. */
    val turn: Int?,
    /** Who must answer a call now, or null. */
    val asked: Int?,
    /** The cards on the table: [seat, card]. */
    val trick: List<List<Int>>,
    val points: List<Int>, val tricks: List<Int>,
    val cardsLeft: List<Int>,
    val lastTrick: List<List<Int>>, val lastTaker: Int?,
)

@Serializable
data class AdminTable(
    /** Every seat's cards now (admin only), or null outside a deal. */
    val hands: List<Long>?,
    val host: String?,
    /** Seats a person plays now (the rest: bots). */
    val humans: List<Int>,
    /** After a deal: seats ready for the next one. */
    val ready: List<Int>,
    val connected: List<Boolean>,
    /** Seats whose player dropped and is being waited for. */
    val awayTimers: List<Int>,
)

@Serializable
data class LiveTableDetail(val table: LiveTable, val deal: LiveDeal?, val deals: List<DealRecord>, val admin: AdminTable? = null)

@Serializable
data class PlayerPage(val profile: Store.Profile, val stats: Map<String, Int>, val recent: List<Store.GameSummary>)

@Serializable
data class OnlinePlayer(val id: String, val name: String, val table: String?, val seat: Int?)

@Serializable
data class Status(
    val uptimeSeconds: Long, val tables: Int, val connected: Int, val playing: Int,
    val heapUsedMb: Long, val heapMaxMb: Long, val database: Map<String, Long>,
)

private suspend fun <T> ApplicationCall.json(serializer: KSerializer<T>, value: T) =
    respondText(ApiJson.encodeToString(serializer, value), ContentType.Application.Json)

private suspend fun ApplicationCall.notFound(what: String) =
    respondText("""{"error":"$what not found"}""", ContentType.Application.Json, HttpStatusCode.NotFound)

private fun ApplicationCall.limit(default: Int = 50): Int =
    request.queryParameters["limit"]?.toIntOrNull()?.coerceIn(1, 500) ?: default

/** The public API (see the top of this file). */
fun Route.publicApi(hub: Hub) = route("/api") {
    val store = hub.store
    get("/tables") {
        val tables = withContext(hub.game) { hub.liveTables.map { it.summary() } }
        call.json(ListSerializer(LiveTable.serializer()), tables)
    }
    get("/tables/{code}") {
        val detail = withContext(hub.game) { hub.table(call.parameters["code"]!!)?.detail(admin = false) }
            ?: return@get call.notFound("table")
        call.json(LiveTableDetail.serializer(), detail)
    }
    get("/games") {
        val limit = call.limit()
        call.json(ListSerializer(Store.GameSummary.serializer()), store.read { games(limit) })
    }
    get("/games/{id}") {
        val id = call.parameters["id"]!!
        val game = store.read { game(id) } ?: return@get call.notFound("game")
        call.json(Store.GameDetail.serializer(), game)
    }
    get("/players") {
        val name = call.request.queryParameters["name"].orEmpty()
        val limit = call.limit(20)
        call.json(ListSerializer(Store.Profile.serializer()), store.read { findPlayers(name, limit) })
    }
    get("/players/{id}") {
        val id = call.parameters["id"]!!
        val page = store.read { profile(id)?.let { PlayerPage(it, stats(id).c, games(10, playerId = id, all = true)) } }
            ?: return@get call.notFound("player")
        call.json(PlayerPage.serializer(), page)
    }
    get("/players/{id}/games") {
        val id = call.parameters["id"]!!
        val limit = call.limit()
        val games = store.read { if (profile(id) == null) null else games(limit, playerId = id, all = true) }
            ?: return@get call.notFound("player")
        call.json(ListSerializer(Store.GameSummary.serializer()), games)
    }
}

/** The admin API, only with the right bearer token; off when [token] is null or blank. */
fun Route.adminApi(hub: Hub, token: String?) = route("/admin") {
    suspend fun ApplicationCall.allowed(): Boolean {
        val expected = token?.takeIf { it.isNotBlank() }
        val given = request.headers["Authorization"]?.removePrefix("Bearer ")?.trim()
        // constant-time comparison: the time taken says nothing about the token
        val ok = expected != null && given != null && MessageDigest.isEqual(expected.toByteArray(), given.toByteArray())
        if (!ok) respondText("""{"error":"unauthorized"}""", ContentType.Application.Json, HttpStatusCode.Unauthorized)
        return ok
    }
    get("/status") {
        if (!call.allowed()) return@get
        val (tables, connected, playing) = withContext(hub.game) {
            Triple(hub.tableCount, hub.connectedCount, hub.liveTables.count { it.summary().started })
        }
        val rt = Runtime.getRuntime()
        val status = Status(
            uptimeSeconds = (System.currentTimeMillis() - hub.startedAt) / 1000, tables = tables, connected = connected, playing = playing,
            heapUsedMb = (rt.totalMemory() - rt.freeMemory()) / 1_048_576, heapMaxMb = rt.maxMemory() / 1_048_576,
            database = hub.store.read { counts() },
        )
        call.json(Status.serializer(), status)
    }
    get("/tables") {
        if (!call.allowed()) return@get
        val tables = withContext(hub.game) { hub.liveTables.map { it.detail(admin = true) } }
        call.json(ListSerializer(LiveTableDetail.serializer()), tables)
    }
    get("/tables/{code}") {
        if (!call.allowed()) return@get
        val detail = withContext(hub.game) { hub.table(call.parameters["code"]!!)?.detail(admin = true) }
            ?: return@get call.notFound("table")
        call.json(LiveTableDetail.serializer(), detail)
    }
    post("/tables/{code}/close") {
        if (!call.allowed()) return@post
        val closed = withContext(hub.game) { hub.closeTable(call.parameters["code"]!!) }
        if (closed) call.respondText("""{"closed":true}""", ContentType.Application.Json) else call.notFound("table")
    }
    get("/players") {
        if (!call.allowed()) return@get
        val online = withContext(hub.game) {
            hub.onlinePlayers.map { p -> OnlinePlayer(p.id, p.name, p.table?.code, p.table?.seatOf(p)?.takeIf { it >= 0 }) }
        }
        call.json(ListSerializer(OnlinePlayer.serializer()), online)
    }
}
