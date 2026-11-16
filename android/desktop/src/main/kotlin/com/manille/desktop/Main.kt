package com.manille.desktop

import androidx.compose.ui.graphics.painter.BitmapPainter
import androidx.compose.ui.graphics.toComposeImageBitmap
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState
import com.manille.jvm.JvmPlatform
import com.manille.jvm.NativeLibrary
import com.manille.jvm.OrtSessions
import com.manille.shared.AppConfig
import com.manille.shared.ManilleRoot
import org.jetbrains.skia.Image
import java.io.File

/**
 * The desktop shell: per-build names, the platform, then the shared UI in a
 * phone-shaped window (the table is laid out for portrait).
 */
fun main() {
    val resources = appResources()
    val library = File(resources, "native/" + System.mapLibraryName("manille_mobile"))
    NativeLibrary.load = { System.load(library.absolutePath) }
    // memory over a few ms per bot move (see OrtSessions)
    OrtSessions.prepacking = false
    System.getProperty("manille.botNames")?.split(",")?.map { it.trim() }?.filter { it.isNotEmpty() }
        ?.takeIf { it.size >= 3 }?.let { AppConfig.botNames = it }
    System.getProperty("manille.cafeThemeName")?.takeIf { it.isNotBlank() }?.let { AppConfig.cafeThemeName = it }

    val models = File(resources, "models")
    val platform = JvmPlatform(
        modelFile = { name ->
            File(models, name).takeIf { it.isFile } ?: throw IllegalStateException(
                "Missing $name. Run scripts/fetch_models.sh, then rebuild the desktop app.")
        },
        dataDir = dataDir(),
        logger = { System.err.println("manille: $it") },
    )
    val icon = Thread.currentThread().contextClassLoader.getResourceAsStream("icon.png")
        ?.use { BitmapPainter(Image.makeFromEncoded(it.readBytes()).toComposeImageBitmap()) }

    application {
        Window(
            onCloseRequest = ::exitApplication,
            title = "Manille",
            icon = icon,
            state = rememberWindowState(size = DpSize(460.dp, 900.dp)),
        ) {
            ManilleRoot(platform)
        }
    }
}

/** Where Compose put build/appResources/common (models/, native/). */
private fun appResources(): File =
    System.getProperty("compose.application.resources.dir")?.let(::File)
        ?: error("compose.application.resources.dir is not set: start the app with ./gradlew :desktop:run")

/** The player's profiles, statistics and saved game, in the usual place per OS. */
private fun dataDir(): File {
    val os = System.getProperty("os.name").lowercase()
    val home = System.getProperty("user.home")
    return when {
        "win" in os -> File(System.getenv("APPDATA") ?: home, "Manille")
        "mac" in os -> File(home, "Library/Application Support/Manille")
        else -> File(System.getenv("XDG_DATA_HOME")?.takeIf { it.isNotBlank() } ?: "$home/.local/share", "manille")
    }
}
