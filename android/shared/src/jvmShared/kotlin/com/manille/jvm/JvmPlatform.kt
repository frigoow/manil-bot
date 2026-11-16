package com.manille.jvm

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

/** Named files in one directory; writes go to a temp file and are renamed into place. */
class FileStorage(private val dir: File) : Storage {
    init { dir.mkdirs() }

    override fun read(name: String): String? = File(dir, name).takeIf { it.isFile }?.readText()

    override fun write(name: String, text: String) {
        val file = File(dir, name)
        val tmp = File(dir, "$name.tmp")
        tmp.writeText(text)
        if (!tmp.renameTo(file)) { file.delete(); tmp.renameTo(file) }
    }

    override fun delete(name: String) { File(dir, name).delete() }
    override fun exists(name: String): Boolean = File(dir, name).isFile
    override fun rename(from: String, to: String): Boolean = File(dir, from).renameTo(File(dir, to))
}
