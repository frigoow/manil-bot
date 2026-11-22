package com.manille.androidapp

import com.cardtable.platform.*

import com.manille.shared.*
import android.content.Context
import java.io.File
import java.io.FileOutputStream

/**
 * Copies a bundled model out of the APK assets for ONNX Runtime. The copy is
 * refreshed whenever the app was installed or updated after it was made:
 * re-exported models ship under the same file names, and a stale copy from an
 * older install would be fed the current encoder's inputs.
 */
object ModelFiles {
    fun install(context: Context, name: String, hint: String): File {
        val target = File(context.filesDir, name)
        val installed = runCatching {
            context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime
        }.getOrDefault(Long.MAX_VALUE)
        if (target.isFile && target.lastModified() >= installed) return target
        try {
            val tmp = File(context.filesDir, "$name.tmp")
            context.assets.open("models/$name").use { input ->
                FileOutputStream(tmp).use(input::copyTo)
            }
            if (!tmp.renameTo(target)) throw IllegalStateException("could not install $name")
        } catch (error: Exception) {
            throw IllegalStateException("Missing $name. $hint", error)
        }
        return target
    }
}
