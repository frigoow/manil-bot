package com.manille.desktop

import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.ImageComposeScene
import androidx.compose.ui.unit.Density
import com.manille.jvm.JvmPlatform
import com.manille.jvm.NativeLibrary
import com.manille.shared.GameController
import com.manille.shared.ManilleRoot
import com.manille.shared.TablePhase
import org.jetbrains.skia.EncodedImageFormat
import java.io.File
import java.nio.file.Files
import javax.swing.SwingUtilities
import kotlin.test.Test

/**
 * The Manille app offscreen, saved as PNGs at the lobby, the declaration and
 * the play. Runs only with -Pmanille.screenshots=<dir>; otherwise it skips.
 */
@OptIn(ExperimentalComposeUiApi::class)
class ScreenshotTest {
    private fun <T> edt(block: () -> T): T { var r: T? = null; SwingUtilities.invokeAndWait { r = block() }; @Suppress("UNCHECKED_CAST") return r as T }

    @Test fun screens() {
        val out = System.getProperty("manille.screenshots")?.takeIf { it.isNotBlank() }?.let(::File) ?: return
        out.mkdirs()
        val resources = File(System.getProperty("compose.application.resources.dir"))
        NativeLibrary.load = { System.load(File(resources, "native/" + System.mapLibraryName("manille_mobile")).absolutePath) }
        val platform = JvmPlatform({ File(resources, "models/$it") }, Files.createTempDirectory("manille").toFile(), {})
        var c: GameController? = null
        val scene = edt { ImageComposeScene(920, 1800, Density(2f)) { ManilleRoot(platform) { c = it } } }
        fun frame() = edt { scene.render(System.nanoTime()) }
        fun shot(name: String) = File(out, "$name.png").writeBytes(frame().encodeToData(EncodedImageFormat.PNG)!!.bytes)
        fun runUntil(limitMs: Long = 30_000, ms: Long = 0, cond: (GameController) -> Boolean = { true }) {
            val start = System.currentTimeMillis()
            while (System.currentTimeMillis() - start < ms) { frame(); Thread.sleep(40) }
            while (System.currentTimeMillis() - start < limitMs) { frame(); if (edt { cond(c!!) }) return; Thread.sleep(40) }
        }
        runUntil(ms = 600); shot("m1-lobby")
        edt { c!!.start(true) }
        runUntil { it.phase == TablePhase.DECLARATION || (it.phase == TablePhase.PLAYING && it.legalCards.isNotEmpty()) }
        runUntil(ms = 600); shot("m2-first-decision")
        edt { if (c!!.phase == TablePhase.DECLARATION) c!!.declare(0) }
        runUntil(60_000) { if (it.phase == TablePhase.MEEGAAN) it.answerMeegaan(false); if (it.phase == TablePhase.TERUG) it.answerTerug(false)
            it.phase == TablePhase.PLAYING && it.legalCards.isNotEmpty() && (it.board?.trickCards?.count { x -> x >= 0 } ?: 0) >= 1 }
        runUntil(ms = 800); shot("m3-play")
        edt { scene.close() }
    }
}
