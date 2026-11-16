package com.manille.androidapp

import android.content.Context
import android.util.Log
import com.manille.jvm.JvmPlatform

/**
 * The shared game's Android side (com.manille.jvm.JvmPlatform): the nets come
 * out of the APK assets into the app's private storage, files live there too,
 * diagnostics go to logcat.
 */
class AndroidPlatform(context: Context) : JvmPlatform(
    modelFile = { name -> ModelFiles.install(context.applicationContext, name, MODELS_HINT) },
    dataDir = context.applicationContext.filesDir,
    logger = { Log.i("manille", it) },
) {
    companion object {
        const val MODELS_HINT =
            "Unzip manille-app-models.zip (release models-v1) into android/app/src/main/assets/models, then rebuild."
    }
}
