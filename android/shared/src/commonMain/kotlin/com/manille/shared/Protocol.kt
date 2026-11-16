package com.manille.shared

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/*
 * The online game's messages, one JSON object per WebSocket text frame,
 * tagged by "type". The server runs every table (deals, checks each move,
 * plays the empty seats) and sends each player only what that player may
 * see: [ServerMessage.Update] is already turned to the player's own seat
 * (seat 0 = you, "us" = your team; see Seats), and never carries another
 * player's cards.
 *
 * Seats in [ServerMessage.TableInfo] are the table's own (absolute) seats,
 * so everyone can pick a chair; [ServerMessage.TableInfo.you] says which is
 * yours. Everything else is in your seats.
 */

/** Bump on any change a client of the previous version would misread. */
const val PROTOCOL_VERSION = 1

val ProtocolJson = Json {
    classDiscriminator = "type"
    ignoreUnknownKeys = true
    encodeDefaults = true
}

/** One chair at an online table. */
@Serializable
data class SeatInfo(
    /** The player's name, a bot's name, or null for a free chair (a bot plays it once the game starts). */
    val name: String?,
    val bot: Boolean,
    /** A player whose connection dropped: a bot plays for them until they are back. */
    val away: Boolean = false,
)

@Serializable
sealed interface ClientMessage {
    /** First message on every connection; [token] is the one a [ServerMessage.Welcome] gave earlier (null the first time). */
    @Serializable @SerialName("hello")
    data class Hello(val protocol: Int, val token: String?, val name: String) : ClientMessage
    /** Open a new table and sit at it; it gets a code to share. */
    @Serializable @SerialName("create")
    data class Create(val match: Boolean) : ClientMessage
    @Serializable @SerialName("join")
    data class Join(val code: String) : ClientMessage
    /** Move to a free chair (absolute seat 0..3). */
    @Serializable @SerialName("sit")
    data class Sit(val seat: Int) : ClientMessage
    /** The host starts the game; bots take the free chairs. Also: a new game after the last one ended. */
    @Serializable @SerialName("start")
    data object Start : ClientMessage
    @Serializable @SerialName("declare")
    data class Declare(val trump: Int) : ClientMessage
    /** Answer a meegaan ([TablePhase.MEEGAAN]) or terug ([TablePhase.TERUG]) call. */
    @Serializable @SerialName("knock")
    data class Knock(val call: TablePhase, val accept: Boolean) : ClientMessage
    @Serializable @SerialName("play")
    data class Play(val card: Int) : ClientMessage
    /** Ready for the next deal; it is dealt once every player at the table is. */
    @Serializable @SerialName("next")
    data object NextDeal : ClientMessage
    /** A smiley or short line for the table. */
    @Serializable @SerialName("react")
    data class React(val text: String) : ClientMessage
    /** Leave the table for good (a bot takes the chair). */
    @Serializable @SerialName("leave")
    data object Leave : ClientMessage
}

@Serializable
sealed interface ServerMessage {
    /** Keep [token] to be recognised on the next connection (and so get your chair back). */
    @Serializable @SerialName("welcome")
    data class Welcome(val token: String, val name: String) : ServerMessage
    /** The table you sit at: its code, its chairs, whether the game is on. Sent on every change. */
    @Serializable @SerialName("table")
    data class TableInfo(
        val code: String,
        val match: Boolean,
        val seats: List<SeatInfo>,
        /** Your chair (absolute seat). */
        val you: Int,
        val host: Boolean,
        val started: Boolean,
        /** After a deal: the players who are ready for the next one (absolute seats). */
        val ready: List<Int> = emptyList(),
    ) : ServerMessage
    /** Something happened at the table, and the table now, both from your seat. */
    @Serializable @SerialName("update")
    data class Update(val event: TableEvent, val view: TableView) : ServerMessage
    /** [seat] (yours: 0 = you) said [text]. */
    @Serializable @SerialName("reaction")
    data class Reaction(val seat: Int, val text: String) : ServerMessage
    /** A request was refused; [code] is one of [Problems]. */
    @Serializable @SerialName("problem")
    data class Problem(val code: String, val message: String) : ServerMessage
    /** You are no longer at a table ([reason]: "left", "closed"). */
    @Serializable @SerialName("left")
    data class Left(val reason: String) : ServerMessage
}

/** [ServerMessage.Problem] codes. */
object Problems {
    const val VERSION = "version"
    const val HELLO_FIRST = "hello-first"
    const val NO_TABLE = "no-table"
    const val TABLE_FULL = "table-full"
    const val SEAT_TAKEN = "seat-taken"
    const val NOT_HOST = "not-host"
    const val NOT_NOW = "not-now"
    const val BAD_MESSAGE = "bad-message"
}
