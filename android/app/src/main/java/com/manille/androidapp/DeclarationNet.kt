package com.manille.androidapp

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

/** Exact mobile export of the desktop score-aware declaration policy. */
class DeclarationNet(context: Context) : Closeable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session = environment.createSession(
        installModel(context).absolutePath, OrtSession.SessionOptions())

    fun equities(handMask: Long, us: Int, them: Int): FloatArray {
        val hand = FloatArray(32) { if (handMask and (1L shl it) != 0L) 1f else 0f }
        OnnxTensor.createTensor(
            environment, FloatBuffer.wrap(hand), longArrayOf(1, 32),
        ).use { handTensor ->
            OnnxTensor.createTensor(
                environment, LongBuffer.wrap(longArrayOf(us.toLong(), them.toLong())), longArrayOf(1, 2),
            ).use { scoreTensor ->
                session.run(mapOf("hand" to handTensor, "scores" to scoreTensor)).use { result ->
                    @Suppress("UNCHECKED_CAST")
                    return (result[0].value as Array<FloatArray>)[0].copyOf()
                }
            }
        }
    }

    override fun close() = session.close()

    private fun installModel(context: Context): File =
        ModelFiles.install(context, MODEL_FILE, "Unzip manille-app-models.zip (release models-v1) into android/app/src/main/assets/models, then rebuild.")

    private companion object { const val MODEL_FILE = "manille_decl_v1.onnx" }
}
