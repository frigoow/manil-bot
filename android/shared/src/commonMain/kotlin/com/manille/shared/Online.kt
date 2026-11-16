package com.manille.shared

/** What the online screens show: the connection (null before connecting), the table, the last refusal. */
data class OnlineState(
    val status: RemoteSession.Status?,
    val info: ServerMessage.TableInfo? = null,
    val problem: String? = null,
)
