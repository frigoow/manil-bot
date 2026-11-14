package com.manille.androidapp

import android.content.Context
import android.util.Log
import com.manille.shared.*
import java.io.File

/**
 * The shared game's Android side: the C++ engine through JNI (NativeGame),
 * the nets on ONNX Runtime, files in the app's private storage.
 */
class AndroidPlatform(context: Context) : Platform {
    private val app = context.applicationContext

    override fun newEngine(): Engine = NativeGame()
    override fun trickWinner(cards: List<Int>, trump: Int): Int = NativeGame.trickWinner(cards, trump)

    override fun qModel(file: String): QModel = QNet(app, file)
    override fun beliefModel(file: String): BeliefModel = BeliefNet(app, file)
    override fun announceModel(): AnnounceModel = AnnounceNet(app)
    override fun declarationModel(): DeclarationModel = DeclarationNet(app)

    override fun fastNet(file: String, plies: Int, trickEnd: Boolean): FastNet? {
        val path = ModelFiles.install(app, file, MODELS_HINT).path
        val handle = NativeGame.nativeFastNetLoad(path)
        if (handle == 0L) return null
        NativeGame.nativeFastNetSetPlies(handle, plies)
        NativeGame.nativeFastNetSetTrickEnd(handle, trickEnd)
        return AndroidFastNet(handle)
    }

    override val storage: Storage = FileStorage(app.filesDir)

    override fun log(message: String) { Log.i("manille", message) }

    override fun epochMillis(): Long = System.currentTimeMillis()

    companion object {
        const val MODELS_HINT =
            "Unzip manille-app-models.zip (release models-v1) into android/app/src/main/assets/models, then rebuild."
    }
}

/** Named files in one directory; writes go to a temp file and are renamed into place. */
private class FileStorage(private val dir: File) : Storage {
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
