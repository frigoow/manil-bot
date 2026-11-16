package com.manille.server

import com.manille.jvm.JvmPlatform
import com.manille.jvm.NativeLibrary
import com.manille.shared.*
import io.ktor.server.application.Application
import io.ktor.server.application.install
import io.ktor.server.cio.CIO
import io.ktor.server.engine.embeddedServer
import io.ktor.server.response.respondText
import io.ktor.server.routing.get
import io.ktor.server.routing.routing
import io.ktor.server.websocket.WebSockets
import io.ktor.server.websocket.pingPeriod
import io.ktor.server.websocket.timeout
import io.ktor.server.websocket.webSocket
import java.io.File
import java.nio.file.Files
import kotlin.time.Duration.Companion.seconds

fun main() {
    val resources = File(System.getenv("MANILLE_RESOURCES") ?: System.getProperty("manille.resources")
        ?: error("Set MANILLE_RESOURCES to a folder with native/ and models/ (./gradlew :server:run does)."))
    val port = System.getenv("PORT")?.toInt() ?: 8080
    val grace = System.getenv("MANILLE_GRACE_SECONDS")?.toLong()?.times(1000) ?: ServerConfig().graceMs
    val engine = Engines(resources)
    val hub = Hub(engine.platform, engine.bots(), Worker(), ServerConfig(graceMs = grace))
    engine.platform.log("listening on port $port")
    embeddedServer(CIO, port = port, host = "0.0.0.0") { manille(hub) }.start(wait = true)
}

/** The routes: a health check, and the game's WebSocket. */
fun Application.manille(hub: Hub) {
    install(WebSockets) {
        pingPeriod = 15.seconds
        timeout = 30.seconds
        maxFrameSize = 16_384
    }
    routing {
        get("/health") { call.respondText("ok ${hub.tableCount} tables") }
        webSocket("/play") { hub.serve(this) }
    }
}

/** The C++ engine and the nets from [resources] (native/, models/), as the desktop app ships them. */
class Engines(private val resources: File) {
    val platform: Platform by lazy {
        val library = File(resources, "native/" + System.mapLibraryName("manille_mobile"))
        NativeLibrary.load = { System.load(library.absolutePath) }
        JvmPlatform(
            modelFile = { File(resources, "models/$it").also { f -> check(f.isFile) { "missing $f" } } },
            dataDir = Files.createTempDirectory("manille-server").toFile(),
            logger = { println("manille: $it") },
        )
    }

    /** The app's bots: the rule agent at its standard depth, or the plain QGRU when [ruleAgent] is false (tests). */
    fun bots(ruleAgent: Boolean = true): Bots {
        val q = platform.qModel(ModelNames.Q_TRUMP)
        val announce = platform.announceModel()
        val rule = if (ruleAgent) RuleBot(platform, q, announce) else null
        return Bots(rule, q, platform.declarationModel(), announce)
    }
}
