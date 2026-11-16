package com.manille.shared

/** What the online screens show: the connection (null before connecting), the table, the last refusal. */
data class OnlineState(
    val status: RemoteSession.Status?,
    val info: ServerMessage.TableInfo? = null,
    val problem: String? = null,
    /** This player's public id on the server, once connected. */
    val playerId: String? = null,
    /** The online statistics screen is open; [stats] once the server answered. */
    val showStats: Boolean = false,
    val stats: Stats? = null,
    /** A code for another device, and how long it stays valid. */
    val transferCode: String? = null,
    val transferMinutes: Int = 0,
)
