package com.manille.shared

import io.ktor.client.HttpClient
import io.ktor.client.plugins.websocket.WebSockets
import io.ktor.client.plugins.websocket.webSocket
import io.ktor.websocket.Frame
import io.ktor.websocket.readText
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * A table on the server, seen from this player's chair. The server is the
 * referee: moves go out as requests and come back as events, already turned
 * to this player's seat (Protocol.kt). The connection is kept up by itself:
 * after a drop it reconnects with the same token and the server sends the
 * table afresh ([TableEvent.Resynced]).
 *
 * Every callback runs on [main].
 */
class RemoteSession(
    /** ws:// or wss:// address of the server's /play endpoint. */
    private val url: String,
    private val name: String,
    /** The token from an earlier connection, so the server knows this player again. */
    token: String? = null,
    private val main: CoroutineDispatcher = Dispatchers.Main,
    private val client: HttpClient = HttpClient { install(WebSockets) },
) : TableSession {
    enum class Status { CONNECTING, ONLINE, OFFLINE }

    var token: String? = token; private set
    var status: Status = Status.CONNECTING; private set
    /** The table this player sits at, or null. */
    var info: ServerMessage.TableInfo? = null; private set
    override var view: TableView = TableView.LOBBY; private set

    override var listener: ((TableEvent, TableView) -> Unit)? = null
    var onStatus: ((Status) -> Unit)? = null
    var onInfo: ((ServerMessage.TableInfo?) -> Unit)? = null
    /** The server's token for this player: keep it for the next session. */
    var onToken: ((String) -> Unit)? = null
    /** Another player's smiley or line: seat (0 = this player) and text. */
    var onReaction: ((Int, String) -> Unit)? = null
    /** A request was refused ([Problems] code, message). */
    var onProblem: ((String, String) -> Unit)? = null
    /** Left the table: "left" (asked to), "closed" (the table was closed). */
    var onLeft: ((String) -> Unit)? = null

    private val scope = CoroutineScope(SupervisorJob() + main)
    private var outbox = Channel<ClientMessage>(Channel.UNLIMITED)
    private var closed = false

    init { scope.launch { connectLoop() } }

    private suspend fun connectLoop() {
        var backoff = 500L
        while (!closed) {
            setStatus(Status.CONNECTING)
            try {
                client.webSocket(url) {
                    val out = Channel<ClientMessage>(Channel.UNLIMITED).also { outbox = it }
                    send(Frame.Text(ProtocolJson.encodeToString(ClientMessage.serializer(), ClientMessage.Hello(PROTOCOL_VERSION, token, name))))
                    val sender = launch {
                        for (message in out) send(Frame.Text(ProtocolJson.encodeToString(ClientMessage.serializer(), message)))
                    }
                    for (frame in incoming) {
                        if (frame !is Frame.Text) continue
                        val message = runCatching { ProtocolJson.decodeFromString(ServerMessage.serializer(), frame.readText()) }.getOrNull() ?: continue
                        withContext(main) { receive(message) }
                        backoff = 500L
                    }
                    sender.cancel(); out.close()
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                // no server, or the connection dropped: try again
            }
            outbox.close()
            if (closed) break
            setStatus(Status.OFFLINE)
            delay(backoff)
            backoff = (backoff * 2).coerceAtMost(8_000L)
        }
    }

    private fun setStatus(s: Status) { if (status != s) { status = s; onStatus?.invoke(s) } }

    private fun receive(message: ServerMessage) {
        when (message) {
            is ServerMessage.Welcome -> { token = message.token; onToken?.invoke(message.token); setStatus(Status.ONLINE) }
            is ServerMessage.TableInfo -> { info = message; onInfo?.invoke(message) }
            is ServerMessage.Update -> { view = message.view; listener?.invoke(message.event, message.view) }
            is ServerMessage.Reaction -> onReaction?.invoke(message.seat, message.text)
            is ServerMessage.Problem -> onProblem?.invoke(message.code, message.message)
            is ServerMessage.Left -> { info = null; view = TableView.LOBBY; onInfo?.invoke(null); onLeft?.invoke(message.reason) }
        }
    }

    /** Sends [message] if the connection is up; false when it is not (the move is dropped, not queued). */
    private fun send(message: ClientMessage): Boolean =
        status == Status.ONLINE && outbox.trySend(message).isSuccess

    // ---- the lobby ----------------------------------------------------------------
    fun create(match: Boolean) = send(ClientMessage.Create(match))
    fun join(code: String) = send(ClientMessage.Join(code))
    fun sit(seat: Int) = send(ClientMessage.Sit(seat))
    /** The host starts the game (or the next one). */
    fun startGame() = send(ClientMessage.Start)
    fun nextDeal() = send(ClientMessage.NextDeal)
    fun react(text: String) = send(ClientMessage.React(text))
    fun leave() = send(ClientMessage.Leave)

    // ---- the moves: the server checks them -------------------------------------------
    override fun declare(trump: Int): Boolean = view.phase == TablePhase.DECLARATION && send(ClientMessage.Declare(trump))
    override fun answerMeegaan(accept: Boolean): Boolean = view.phase == TablePhase.MEEGAAN && send(ClientMessage.Knock(TablePhase.MEEGAAN, accept))
    override fun answerTerug(accept: Boolean): Boolean = view.phase == TablePhase.TERUG && send(ClientMessage.Knock(TablePhase.TERUG, accept))
    override fun play(card: Int): Boolean = card in view.legalCards && send(ClientMessage.Play(card))

    /** Disconnects (the chair stays this player's for a while; [leave] first to give it up). */
    override fun close() {
        closed = true
        outbox.close()
        scope.cancel()
        client.close()
    }
}
