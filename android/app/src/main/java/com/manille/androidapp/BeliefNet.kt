package com.manille.androidapp

import com.manille.shared.*
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer

/**
 * The rule agent's belief net (exported from PyTorch; shipped in manille-app-models.zip):
 * encoder observation -> P(seat observer+1+r holds card c), 3 x 32 floats,
 * the prior-residual softmax already applied. One file per declaration
 * kind: trump deals and no-trump (mull) deals.
 */
class BeliefNet(context: Context, private val modelFile: String = TRUMP_FILE) : Closeable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val model = installModel(context)
        session = environment.createSession(model.absolutePath, OrtSession.SessionOptions())
    }

    fun probabilities(observation: FloatArray): FloatArray {
        require(observation.size == OBS_DIM) { "Unexpected observation length ${observation.size}." }
        OnnxTensor.createTensor(
            environment, FloatBuffer.wrap(observation), longArrayOf(1, OBS_DIM.toLong()),
        ).use { obs ->
            val result = session.run(mapOf("obs" to obs))
            try {
                @Suppress("UNCHECKED_CAST")
                val probs = (result[0].value as Array<Array<FloatArray>>)[0]
                return FloatArray(96) { probs[it / 32][it % 32] }
            } finally {
                result.close()
            }
        }
    }

    override fun close() {
        session.close()
    }

    private fun installModel(context: Context): File =
        ModelFiles.install(context, modelFile, "Unzip manille-app-models.zip (release models-v1) into android/app/src/main/assets/models, then rebuild.")

    companion object {
        const val TRUMP_FILE = "manille_belief_v1.onnx"
        const val MULL_FILE = "manille_belief_mull_v1.onnx"
        private const val OBS_DIM = 2492
    }
}
