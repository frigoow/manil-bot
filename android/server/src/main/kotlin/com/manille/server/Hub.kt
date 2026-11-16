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

/** Someone who said hello: known by [token] across connections. */
class Player(val token: String, var name: String) {
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
    }

    fun send(message: ServerMessage) {
        if (outbox.trySend(message).isFailure) close()
    }

    fun close() {
        outbox.close()
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
) : AutoCloseable {
    val game: CoroutineDispatcher = Dispatchers.Default.limitedParallelism(1)
    val scope = CoroutineScope(SupervisorJob() + game)
    private val secure = SecureRandom()
    val random: Random = secure.asKotlinRandom()
    private val players = mutableMapOf<String, Player>()
    private val tables = mutableMapOf<String, ServerTable>()

    val tableCount: Int get() = tables.size

    /** Serves one WebSocket until it closes. */
    suspend fun serve(socket: WebSocketSession) {
        val connection = Connection(socket)
        try {
            for (frame in socket.incoming) {
                if (frame !is Frame.Text) continue
                val message = runCatching { ProtocolJson.decodeFromString(ClientMessage.serializer(), frame.readText()) }.getOrNull()
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
        if (message is ClientMessage.Hello) { hello(connection, message); return }
        val player = connection.player ?: run {
            connection.send(ServerMessage.Problem(Problems.HELLO_FIRST, "Say hello first.")); return
        }
        val table = player.table
        when (message) {
            is ClientMessage.Create -> {
                table?.leave(player)
                val code = newCode()
                ServerTable(code, message.match, this, platform, bots, worker).also { tables[code] = it }.join(player)
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
            else -> table?.handle(player, message)
                ?: connection.send(ServerMessage.Problem(Problems.NO_TABLE, "You are not at a table."))
        }
    }

    private fun hello(connection: Connection, hello: ClientMessage.Hello) {
        if (hello.protocol != PROTOCOL_VERSION) {
            connection.send(ServerMessage.Problem(Problems.VERSION, "This app is too old or too new for the server: please update."))
            connection.close(); return
        }
        val name = cleanName(hello.name)
        val player = hello.token?.let { players[it] } ?: Player(newToken(), name).also { players[it.token] = it }
        player.name = name
        // a newer connection replaces the old one (the app reconnected)
        player.connection?.takeIf { it !== connection }?.let { old -> old.player = null; old.close() }
        player.connection = connection
        connection.player = player
        connection.send(ServerMessage.Welcome(player.token, player.name))
        player.table?.reconnected(player)
    }

    private fun disconnected(connection: Connection) {
        val player = connection.player ?: return
        connection.player = null
        if (player.connection !== connection) return
        player.connection = null
        val table = player.table
        if (table == null) players.remove(player.token) else table.disconnected(player)
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

    private fun newToken(): String = ByteArray(24).also(secure::nextBytes).let { Base64.getUrlEncoder().withoutPadding().encodeToString(it) }

    override fun close() {
        scope.cancel()
        tables.values.toList().forEach { it.close("closed") }
    }

    companion object {
        /** No 0/O, 1/I: easy to read out loud. */
        private const val CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        const val MAX_NAME = 20
        const val MAX_REACTION = 40

        fun cleanName(raw: String): String =
            raw.filter { !it.isISOControl() }.trim().take(MAX_NAME).ifEmpty { "Player" }
    }
}
