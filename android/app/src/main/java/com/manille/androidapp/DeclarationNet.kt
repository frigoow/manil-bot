package com.manille.androidapp

import com.manille.shared.*
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

/** Exact mobile export of the desktop score-aware declaration policy. */
class DeclarationNet(context: Context) : DeclarationModel {
    private val environment = OrtEnvironment.getEnvironment()
    private val session = environment.createSession(
        installModel(context).absolutePath, OrtSession.SessionOptions())

    override fun equities(handMask: Long, us: Int, them: Int): FloatArray {
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
        ModelFiles.install(context, ModelNames.DECLARATION, AndroidPlatform.MODELS_HINT)
}
