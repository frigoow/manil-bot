package com.manille.server

import com.manille.shared.HandRecord
import com.manille.shared.Stats
import com.manille.shared.StatsCounting
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import java.security.MessageDigest
import java.security.SecureRandom
import java.sql.Connection
import java.sql.DriverManager
import java.sql.ResultSet
import java.util.Base64
import java.util.concurrent.Executors

/**
 * What the server keeps: players (a public id, a name, their online
 * statistics), the devices that may play as them (tokens, kept only as
 * hashes), transfer codes, and every game with its deals and the cards of
 * each deal. One SQLite file; one thread ([db]) does all the work, so the
 * game thread never waits for the disk: it [post]s its writes.
 */
class Store(path: String, private val clock: () -> Long = System::currentTimeMillis) : AutoCloseable {
    private val executor = Executors.newSingleThreadExecutor { r -> Thread(r, "manille-db").apply { isDaemon = true } }
    val db: CoroutineDispatcher = executor.asCoroutineDispatcher()
    private val writes = CoroutineScope(SupervisorJob() + db)
    private val secure = SecureRandom()
    private val conn: Connection = DriverManager.getConnection("jdbc:sqlite:$path")

    init {
        // created on this thread, used on [db] from now on
        conn.createStatement().use { st ->
            if (path != ":memory:") st.execute("PRAGMA journal_mode = WAL")
            st.execute("PRAGMA foreign_keys = ON")
            st.execute("PRAGMA synchronous = NORMAL")
        }
        migrate()
    }

    /** Runs [block] on the database thread, after every write posted before it. */
    suspend fun <T> read(block: Store.() -> T): T = withContext(db) { block() }

    /** Queues a write; the caller goes on. Writes run in the order posted. */
    fun post(block: Store.() -> Unit) {
        writes.launch { runCatching { block() }.onFailure { System.err.println("manille: database: $it") } }
    }

    private fun migrate() {
        val version = conn.createStatement().use { st -> st.executeQuery("PRAGMA user_version").use { it.getInt(1) } }
        if (version < 1) conn.createStatement().use { st ->
            st.executeUpdate("""CREATE TABLE players(
                id TEXT PRIMARY KEY, name TEXT NOT NULL, created INTEGER NOT NULL, last_seen INTEGER NOT NULL,
                stats TEXT NOT NULL DEFAULT '{}')""")
            st.executeUpdate("""CREATE TABLE tokens(
                hash TEXT PRIMARY KEY, player_id TEXT NOT NULL REFERENCES players(id), created INTEGER NOT NULL)""")
            st.executeUpdate("""CREATE TABLE transfers(
                code TEXT PRIMARY KEY, player_id TEXT NOT NULL REFERENCES players(id), expires INTEGER NOT NULL)""")
            st.executeUpdate("""CREATE TABLE games(
                id TEXT PRIMARY KEY, table_code TEXT NOT NULL, match INTEGER NOT NULL,
                started INTEGER NOT NULL, ended INTEGER, team0 INTEGER NOT NULL DEFAULT 0, team1 INTEGER NOT NULL DEFAULT 0,
                outcome TEXT, seats TEXT NOT NULL)""")
            st.executeUpdate("""CREATE TABLE game_players(
                game_id TEXT NOT NULL REFERENCES games(id), player_id TEXT NOT NULL REFERENCES players(id), seat INTEGER NOT NULL,
                PRIMARY KEY (game_id, player_id, seat))""")
            st.executeUpdate("CREATE INDEX game_players_player ON game_players(player_id)")
            st.executeUpdate("""CREATE TABLE deals(
                game_id TEXT NOT NULL REFERENCES games(id), number INTEGER NOT NULL, dealer INTEGER NOT NULL,
                trump INTEGER NOT NULL, multiplier INTEGER NOT NULL, meegaan INTEGER, terug INTEGER,
                team0_points INTEGER NOT NULL, team1_points INTEGER NOT NULL, team0_tricks INTEGER NOT NULL, team1_tricks INTEGER NOT NULL,
                team0_score INTEGER NOT NULL, team1_score INTEGER NOT NULL, hands TEXT NOT NULL, plays TEXT NOT NULL, ended INTEGER NOT NULL,
                PRIMARY KEY (game_id, number))""")
            st.executeUpdate("PRAGMA user_version = 1")
        }
    }

    // ---- players ------------------------------------------------------------------

    /** Who a connection is: [token] is the secret the device keeps; [id] is public. */
    data class Account(val id: String, val token: String, val name: String)

    /**
     * The player behind [token] (renamed to [name]); with a valid [transfer]
     * code, that code's player, with a new token for this device; else a new
     * player. Null when [transfer] is given but unknown or expired.
     */
    fun signIn(token: String?, name: String, transfer: String?): Account? {
        val now = clock()
        if (transfer != null) {
            val code = transfer.trim().uppercase()
            val id = query("SELECT player_id FROM transfers WHERE code = ? AND expires > ?", code, now) { it.getString(1) }.firstOrNull()
                ?: return null
            update("DELETE FROM transfers WHERE code = ?", code)
            val fresh = newSecret(24)
            update("INSERT INTO tokens(hash, player_id, created) VALUES (?, ?, ?)", hash(fresh), id, now)
            update("UPDATE players SET name = ?, last_seen = ? WHERE id = ?", name, now, id)
            return Account(id, fresh, name)
        }
        if (token != null) {
            val id = query("SELECT player_id FROM tokens WHERE hash = ?", hash(token)) { it.getString(1) }.firstOrNull()
            if (id != null) {
                update("UPDATE players SET name = ?, last_seen = ? WHERE id = ?", name, now, id)
                return Account(id, token, name)
            }
        }
        // a new player (or a token this server does not know, e.g. after a reset)
        val id = newId()
        val fresh = newSecret(24)
        update("INSERT INTO players(id, name, created, last_seen) VALUES (?, ?, ?, ?)", id, name, now, now)
        update("INSERT INTO tokens(hash, player_id, created) VALUES (?, ?, ?)", hash(fresh), id, now)
        return Account(id, fresh, name)
    }

    fun touch(id: String) { update("UPDATE players SET last_seen = ? WHERE id = ?", clock(), id) }

    /** A code for [playerId] to sign in on another device, valid for [minutes]. */
    fun transferCode(playerId: String, minutes: Int): String {
        val now = clock()
        update("DELETE FROM transfers WHERE expires <= ? OR player_id = ?", now, playerId)
        while (true) {
            val code = String(CharArray(8) { CODE_ALPHABET[secure.nextInt(CODE_ALPHABET.length)] })
            if (query("SELECT 1 FROM transfers WHERE code = ?", code) { 1 }.isEmpty()) {
                update("INSERT INTO transfers(code, player_id, expires) VALUES (?, ?, ?)", code, playerId, now + minutes * 60_000L)
                return code
            }
        }
    }

    @Serializable
    data class Profile(val id: String, val name: String, val created: Long, val lastSeen: Long)

    fun profile(id: String): Profile? =
        query("SELECT id, name, created, last_seen FROM players WHERE id = ?", id) { Profile(it.getString(1), it.getString(2), it.getLong(3), it.getLong(4)) }.firstOrNull()

    /** Players by name (part of it), most recently seen first. */
    fun findPlayers(name: String, limit: Int): List<Profile> =
        query("SELECT id, name, created, last_seen FROM players WHERE name LIKE ? ESCAPE '\\' ORDER BY last_seen DESC LIMIT ?",
            "%" + name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%", limit) {
            Profile(it.getString(1), it.getString(2), it.getLong(3), it.getLong(4))
        }

    fun stats(id: String): Stats =
        query("SELECT stats FROM players WHERE id = ?", id) { StatsCounting.read(Json.parseToJsonElement(it.getString(1)) as JsonObject) }
            .firstOrNull() ?: Stats(emptyMap(), emptyList())

    private fun changeStats(id: String, change: (MutableMap<String, Int>, MutableList<JsonObject>) -> Unit) {
        val s = stats(id)
        val c = s.c.toMutableMap(); val h = s.history.toMutableList()
        change(c, h)
        update("UPDATE players SET stats = ? WHERE id = ?", StatsCounting.write(c, h).toString(), id)
    }

    fun recordHand(id: String, record: HandRecord) = changeStats(id) { c, _ -> StatsCounting.addHand(c, record) }

    fun recordResult(id: String, match: Boolean, us: Int, them: Int) = changeStats(id) { c, h ->
        StatsCounting.addResult(c, match, us, them)
        h += StatsCounting.historyEntry(clock() / 1000, match, us, them)
    }

    // ---- games --------------------------------------------------------------------

    fun startGame(id: String, tableCode: String, match: Boolean, seats: List<String>) {
        update("INSERT INTO games(id, table_code, match, started, seats) VALUES (?, ?, ?, ?, ?)",
            id, tableCode, if (match) 1 else 0, clock(), Json.encodeToString(seats))
    }

    /** [playerId] sat in [seat] during game [gameId] (from its start, or joining later). */
    fun addGamePlayer(gameId: String, playerId: String, seat: Int) {
        update("INSERT OR IGNORE INTO game_players(game_id, player_id, seat) VALUES (?, ?, ?)", gameId, playerId, seat)
    }

    fun updateSeats(gameId: String, seats: List<String>) {
        update("UPDATE games SET seats = ? WHERE id = ?", Json.encodeToString(seats), gameId)
    }

    fun addDeal(gameId: String, deal: DealRecord) {
        update("""INSERT INTO deals(game_id, number, dealer, trump, multiplier, meegaan, terug, team0_points, team1_points,
                  team0_tricks, team1_tricks, team0_score, team1_score, hands, plays, ended)
                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            gameId, deal.number, deal.dealer, deal.trump, deal.multiplier, deal.meegaan, deal.terug,
            deal.points[0], deal.points[1], deal.tricks[0], deal.tricks[1], deal.score[0], deal.score[1],
            Json.encodeToString(deal.hands), Json.encodeToString(deal.plays), deal.ended)
        update("UPDATE games SET team0 = team0 + ?, team1 = team1 + ? WHERE id = ?", deal.score[0], deal.score[1], gameId)
    }

    /** [outcome]: "team0", "team1", "tie" or "abandoned". */
    fun endGame(gameId: String, outcome: String) {
        update("UPDATE games SET ended = ?, outcome = ? WHERE id = ? AND ended IS NULL", clock(), outcome, gameId)
    }

    @Serializable
    data class GameSummary(
        val id: String, val table: String, val match: Boolean, val started: Long, val ended: Long?,
        /** Match points per team (a quick game: the one deal's score). */
        val score: List<Int>,
        /** "team0", "team1", "tie", "abandoned", or null while it is played. */
        val outcome: String?,
        /** Who sat where when the game started or ended (names; bots included). */
        val seats: List<String>,
    )

    private fun gameSummary(rs: ResultSet) = GameSummary(
        rs.getString("id"), rs.getString("table_code"), rs.getInt("match") == 1, rs.getLong("started"),
        rs.getLong("ended").takeIf { !rs.wasNull() }, listOf(rs.getInt("team0"), rs.getInt("team1")), rs.getString("outcome"),
        Json.decodeFromString(rs.getString("seats")),
    )

    /** Recent games, newest first; only finished ones unless [all]; only [playerId]'s if given. */
    fun games(limit: Int, playerId: String? = null, all: Boolean = false): List<GameSummary> {
        val where = buildList {
            if (!all) add("g.ended IS NOT NULL")
            if (playerId != null) add("g.id IN (SELECT game_id FROM game_players WHERE player_id = ?)")
        }.joinToString(" AND ").let { if (it.isEmpty()) "" else "WHERE $it" }
        val args: List<Any> = listOfNotNull<Any>(playerId) + limit
        return query("SELECT g.* FROM games g $where ORDER BY g.started DESC LIMIT ?", *args.toTypedArray()) { gameSummary(it) }
    }

    @Serializable
    data class GamePlayer(val id: String, val name: String, val seat: Int)

    @Serializable
    data class GameDetail(val game: GameSummary, val players: List<GamePlayer>, val deals: List<DealRecord>)

    fun game(id: String): GameDetail? {
        val summary = query("SELECT * FROM games g WHERE id = ?", id) { gameSummary(it) }.firstOrNull() ?: return null
        val players = query("""SELECT p.id, p.name, gp.seat FROM game_players gp JOIN players p ON p.id = gp.player_id
                                WHERE gp.game_id = ? ORDER BY gp.seat""", id) { GamePlayer(it.getString(1), it.getString(2), it.getInt(3)) }
        val deals = query("SELECT * FROM deals WHERE game_id = ? ORDER BY number", id) { rs ->
            DealRecord(
                number = rs.getInt("number"), dealer = rs.getInt("dealer"), trump = rs.getInt("trump"), multiplier = rs.getInt("multiplier"),
                meegaan = rs.getInt("meegaan").takeIf { !rs.wasNull() }, terug = rs.getInt("terug").takeIf { !rs.wasNull() },
                points = listOf(rs.getInt("team0_points"), rs.getInt("team1_points")),
                tricks = listOf(rs.getInt("team0_tricks"), rs.getInt("team1_tricks")),
                score = listOf(rs.getInt("team0_score"), rs.getInt("team1_score")),
                hands = Json.decodeFromString(rs.getString("hands")), plays = Json.decodeFromString(rs.getString("plays")),
                ended = rs.getLong("ended"),
            )
        }
        return GameDetail(summary, players, deals)
    }

    /** Games and players stored, and the database file's size (admin). */
    fun counts(): Map<String, Long> = mapOf(
        "players" to query("SELECT COUNT(*) FROM players") { it.getLong(1) }.first(),
        "games" to query("SELECT COUNT(*) FROM games") { it.getLong(1) }.first(),
        "deals" to query("SELECT COUNT(*) FROM deals") { it.getLong(1) }.first(),
        "bytes" to query("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()") { it.getLong(1) }.first(),
    )

    // ---- plumbing -------------------------------------------------------------------

    private fun bind(sql: String, args: Array<out Any?>) = conn.prepareStatement(sql).also { st ->
        args.forEachIndexed { i, a ->
            when (a) {
                null -> st.setObject(i + 1, null)
                is Int -> st.setInt(i + 1, a)
                is Long -> st.setLong(i + 1, a)
                is String -> st.setString(i + 1, a)
                else -> error("unsupported parameter ${a::class}")
            }
        }
    }

    private fun update(sql: String, vararg args: Any?): Int = bind(sql, args).use { it.executeUpdate() }

    private fun <T> query(sql: String, vararg args: Any?, row: (ResultSet) -> T): List<T> =
        bind(sql, args).use { st -> st.executeQuery().use { rs -> buildList { while (rs.next()) add(row(rs)) } } }

    private fun newSecret(bytes: Int): String =
        ByteArray(bytes).also(secure::nextBytes).let { Base64.getUrlEncoder().withoutPadding().encodeToString(it) }

    private fun newId(): String {
        while (true) {
            val id = String(CharArray(10) { ID_ALPHABET[secure.nextInt(ID_ALPHABET.length)] })
            if (query("SELECT 1 FROM players WHERE id = ?", id) { 1 }.isEmpty()) return id
        }
    }

    override fun close() {
        writes.cancel()
        executor.submit { runCatching { conn.close() } }.get()
        executor.shutdown()
    }

    companion object {
        /** No 0/O, 1/I: easy to read out and type. */
        private const val CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        private const val ID_ALPHABET = "abcdefghijkmnpqrstuvwxyz23456789"

        fun hash(token: String): String =
            MessageDigest.getInstance("SHA-256").digest(token.toByteArray()).joinToString("") { "%02x".format(it) }
    }
}

/** One finished deal, as stored and as the API shows it. Seats and teams are the table's own (absolute). */
@Serializable
data class DealRecord(
    val number: Int,
    val dealer: Int,
    /** 0..3 or -1 for mull. */
    val trump: Int,
    val multiplier: Int,
    val meegaan: Int? = null,
    val terug: Int? = null,
    /** Card points, tricks and match points per team. */
    val points: List<Int>,
    val tricks: List<Int>,
    val score: List<Int>,
    /** The cards each seat held when play began (bit c = card c). */
    val hands: List<Long>,
    /** Every card in order: [seat, card]. */
    val plays: List<List<Int>>,
    val ended: Long,
)
