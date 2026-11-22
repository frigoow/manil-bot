package com.manille.jvm

import com.cardtable.platform.*
import com.cardtable.platform.jvm.FileStorage

import com.manille.shared.*
import java.io.File

/**
 * The shared game's JVM side, the same on Android and the desktop: the C++
 * engine through JNI ([NativeGame]), the nets on ONNX Runtime, files in one
 * private directory. The shells say where the model files and the data
 * directory are, and where diagnostics go.
 *
 * @param modelFile the file for a model name ([ModelNames]); throws with a
 *   hint for the developer when it is missing.
 */
open class JvmPlatform(
    private val modelFile: (String) -> File,
    dataDir: File,
    private val logger: (String) -> Unit,
) : Platform {
    override fun newEngine(): Engine = NativeGame()
    override fun trickWinner(cards: List<Int>, trump: Int): Int = NativeGame.trickWinner(cards, trump)

    override fun qModel(file: String): QModel = QNet(modelFile(file))
    override fun beliefModel(file: String): BeliefModel = BeliefNet(modelFile(file))
    override fun announceModel(): AnnounceModel = AnnounceNet(modelFile(ModelNames.ANNOUNCE))
    override fun declarationModel(): DeclarationModel = DeclarationNet(modelFile(ModelNames.DECLARATION))

    override fun fastNet(file: String, plies: Int, trickEnd: Boolean): FastNet? {
        val handle = NativeGame.nativeFastNetLoad(modelFile(file).path)
        if (handle == 0L) return null
        NativeGame.nativeFastNetSetPlies(handle, plies)
        NativeGame.nativeFastNetSetTrickEnd(handle, trickEnd)
        return NativeFastNet(handle)
    }

    override val storage: Storage = FileStorage(dataDir)

    override fun log(message: String) = logger(message)

    override fun epochMillis(): Long = System.currentTimeMillis()
}
