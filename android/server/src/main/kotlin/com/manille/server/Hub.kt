package com.manille.server

import com.manille.shared.*
import io.ktor.websocket.Frame
import io.ktor.websocket.WebSocketSession
import io.ktor.websocket.close
import io.ktor.websocket.readText
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.security.SecureRandom
import java.util.Base64
import kotlin.random.Random
import kotlin.random.asKotlinRandom

/** Server settings; the defaults are the production ones. */
data class ServerConfig(
    /** How long a dropped player's chair waits for them before a bot plays it. */
    val graceMs: Long = 20_000,
    /** A table nobody is connected to is closed after this long. */
    val idleTableMs: Long = 10 * 60_000,
    val pacing: Pacing = Pacing(),
    /** Names the bots get at a table (the app's public defaults). */
    val botNames: List<String> = AppConfig.botNames,
)

/** Someone who said hello: a stored player ([id], public), on this device by [token]. */
class Player(val id: String, var token: String, var name: String) {
    var connection: Connection? = null
    var table: ServerTable? = null
}

/**
 * One WebSocket. Messages go out through a bounded outbox so the game thread
 * never waits for the network; a client that cannot keep up is dropped.
 */
class Connection(private val socket: WebSocketSession) {
    var player: Player? = null
    private val outbox = Channel<ServerMessage>(OUTBOX)
    private val sender = socket.launch {
        for (message in outbox) socket.send(Frame.Text(ProtocolJson.encodeToString(ServerMessage.serializer(), message)))
        // closed on purpose: what was queued (a refusal, say) has gone out first
        runCatching { socket.close() }
    }

    fun send(message: ServerMessage) {
        if (outbox.trySend(message).isFailure) drop()
    }

    /** Closes once the messages already queued are sent. */
    fun close() { outbox.close() }

    /** Closes now, unsent messages and all (a client that cannot keep up). */
    private fun drop() {
        outbox.cancel()
        socket.launch { runCatching { socket.close() } }
    }

    private companion object { const val OUTBOX = 512 }
}

/**
 * Everyone connected and every table. All game state is touched on one
 * thread ([game]): the referees' own work is light, the bots think on the
 * shared [worker].
 */
class Hub(
    private val platform: Platform,
    private val bots: Bots,
    private val worker: Worker,
    val config: ServerConfig = ServerConfig(),
    /** Players, statistics and games (an in-memory database unless one is given). */
    val store: Store = Store(":memory:"),
) : AutoCloseable {
    val game: CoroutineDispatcher = Dispatchers.Default.limitedParallelism(1)
    val scope = CoroutineScope(SupervisorJob() + game)
    private val secure = SecureRandom()
    val random: Random = secure.asKotlinRandom()
    /** Connected or seated players, by id. */
    private val players = mutableMapOf<String, Player>()
    private val tables = mutableMapOf<String, ServerTable>()
    val startedAt: Long = System.currentTimeMillis()

    val tableCount: Int get() = tables.size
    /** The tables now (on the game thread). */
    val liveTables: List<ServerTable> get() = tables.values.sortedBy { it.code }
    fun table(code: String): ServerTable? = tables[code.trim().uppercase()]
    val connectedCount: Int get() = players.values.count { it.connection != null }
    val onlinePlayers: List<Player> get() = players.values.filter { it.connection != null }.sortedBy { it.name }

    /** Serves one WebSocket until it closes. */
    suspend fun serve(socket: WebSocketSession) {
        val connection = Connection(socket)
        try {
            for (frame in socket.incoming) {
                if (frame !is Frame.Text) continue
                val message = runCatching { ProtocolJson.decodeFromString(ClientMessage.serializer(), frame.readText()) }.getOrNull()
                if (message is ClientMessage.Hello) {
                    // who this is comes from the database, before the game thread sees them
                    val account = if (message.protocol != PROTOCOL_VERSION) null
                        else store.read { signIn(message.token, cleanName(message.name), message.transfer) }
                    withContext(game) { hello(connection, message, account) }
                    continue
                }
                withContext(game) {
                    if (message == null) connection.send(ServerMessage.Problem(Problems.BAD_MESSAGE, "Unreadable message."))
                    else handle(connection, message)
                }
            }
        } finally {
            withContext(NonCancellable + game) { disconnected(connection) }
        }
    }

    private fun handle(connection: Connection, message: ClientMessage) {
        val player = connection.player ?: run {
            connection.send(ServerMessage.Problem(Problems.HELLO_FIRST, "Say hello first.")); return
        }
        val table = player.table
        when (message) {
            is ClientMessage.Create -> {
                table?.leave(player)
                val code = newCode()
                ServerTable(code, message.match, this, platform, bots, worker, store).also { tables[code] = it }.join(player)
            }
            is ClientMessage.Join -> {
                val target = tables[message.code.trim().uppercase()]
                when {
                    target == null -> connection.send(ServerMessage.Problem(Problems.NO_TABLE, "No table with that code."))
                    target === table -> table.resend(player)
                    !target.hasRoom() -> connection.send(ServerMessage.Problem(Problems.TABLE_FULL, "That table is full."))
                    else -> { table?.leave(player); target.join(player) }
                }
            }
            ClientMessage.GetStats -> scope.launch {
                val s = store.read { stats(player.id) }
                connection.send(ServerMessage.PlayerStats(s.c, s.history))
            }
            ClientMessage.GetTransferCode -> scope.launch {
                val code = store.read { transferCode(player.id, TRANSFER_MINUTES) }
                connection.send(ServerMessage.TransferCode(code, TRANSFER_MINUTES))
            }
            else -> table?.handle(player, message)
                ?: connection.send(ServerMessage.Problem(Problems.NO_TABLE, "You are not at a table."))
        }
    }

    private fun hello(connection: Connection, hello: ClientMessage.Hello, account: Store.Account?) {
        if (hello.protocol != PROTOCOL_VERSION) {
            connection.send(ServerMessage.Problem(Problems.VERSION, "This app is too old or too new for the server: please update."))
            connection.close(); return
        }
        if (account == null) {
            connection.send(ServerMessage.Problem(Problems.BAD_TRANSFER, "That code is unknown or has expired."))
            connection.close(); return
        }
        // a connection that already said hello as someone else lets go of them first
        connection.player?.takeIf { it.id != account.id }?.let { disconnected(connection) }
        val player = players[account.id] ?: Player(account.id, account.token, account.name).also { players[it.id] = it }
        player.name = account.name
        player.token = account.token
        // a newer connection replaces the old one (the app reconnected, or the same player on another device)
        player.connection?.takeIf { it !== connection }?.let { old -> old.player = null; old.close() }
        player.connection = connection
        connection.player = player
        connection.send(ServerMessage.Welcome(account.token, player.name, player.id))
        player.table?.reconnected(player)
    }

    private fun disconnected(connection: Connection) {
        val player = connection.player ?: return
        connection.player = null
        if (player.connection !== connection) return
        player.connection = null
        store.post { touch(player.id) }
        val table = player.table
        if (table == null) players.remove(player.id) else table.disconnected(player)
    }

    internal fun closed(table: ServerTable) {
        tables.remove(table.code)
        // players who never came back are forgotten with their table
        players.values.removeAll { it.table == null && it.connection == null }
    }

    private fun newCode(): String {
        while (true) {
            val code = String(CharArray(5) { CODE_ALPHABET[secure.nextInt(CODE_ALPHABET.length)] })
            if (code !in tables) return code
        }
    }

    /** A new game's id: public, unguessable enough that a game's page is not found by counting. */
    fun newGameId(): String = ByteArray(9).also(secure::nextBytes).let { Base64.getUrlEncoder().withoutPadding().encodeToString(it) }

    /** Closes a table (admin): its players are told, its game ends as abandoned. */
    fun closeTable(code: String): Boolean = tables[code.trim().uppercase()]?.let { it.close("closed"); true } ?: false

    override fun close() {
        scope.cancel()
        tables.values.toList().forEach { it.close("closed") }
    }

    companion object {
        /** No 0/O, 1/I: easy to read out loud. */
        private const val CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        const val MAX_NAME = 20
        const val MAX_REACTION = 40
        const val TRANSFER_MINUTES = 10

        fun cleanName(raw: String): String =
            raw.filter { !it.isISOControl() }.trim().take(MAX_NAME).ifEmpty { "Player" }
    }
}
