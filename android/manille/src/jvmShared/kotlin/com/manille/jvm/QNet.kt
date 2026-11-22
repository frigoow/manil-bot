package com.manille.jvm

import com.cardtable.platform.*

import com.manille.shared.*
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

/** ONNX Runtime inference for the exported recurrent QGRU(s, a) model. */
class QNet(model: File) : QModel {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = OrtSessions.open(environment, model)

    /** Raw Q values for the on-device hint overlay; the same model picks bots. */
    override fun scoreCards(
        raw: FloatArray,
        nRaw: Long,
        glob: FloatArray,
        allActions: FloatArray,
        legalCards: IntArray,
    ): List<CardScore> {
        require(raw.size == MAX_PLAYS * RAW_TOKEN_DIM) { "Unexpected history length ${raw.size}." }
        require(glob.size == GLOBAL_DIM) { "Unexpected global token length ${glob.size}." }
        require(legalCards.isNotEmpty()) { "Cannot score no legal cards." }
        val count = legalCards.size
        val actions = allActions.copyOf(count * ACT_DIM)
        val batchedRaw = FloatArray(count * raw.size) { raw[it % raw.size] }
        val batchedGlob = FloatArray(count * glob.size) { glob[it % glob.size] }
        val lengths = LongArray(count) { nRaw }
        OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(batchedRaw),
            longArrayOf(count.toLong(), MAX_PLAYS.toLong(), RAW_TOKEN_DIM.toLong()),
        ).use { rawTensor ->
            OnnxTensor.createTensor(
                environment,
                LongBuffer.wrap(lengths), longArrayOf(count.toLong()),
            ).use { lengthTensor ->
                OnnxTensor.createTensor(
                    environment,
                    FloatBuffer.wrap(batchedGlob), longArrayOf(count.toLong(), GLOBAL_DIM.toLong()),
                ).use { globalTensor ->
                    OnnxTensor.createTensor(
                        environment,
                        FloatBuffer.wrap(actions), longArrayOf(count.toLong(), ACT_DIM.toLong()),
                    ).use { actionTensor ->
                        val result = session.run(mapOf(
                            "raw" to rawTensor,
                            "n_raw" to lengthTensor,
                            "glob" to globalTensor,
                            "act" to actionTensor,
                        ))
                        try {
                            val values = result[0].value as FloatArray
                            return legalCards.indices.map { CardScore(legalCards[it], values[it]) }
                        } finally {
                            result.close()
                        }
                    }
                }
            }
        }
    }

    override fun close() {
        session.close()
    }


    companion object {
        private const val ACT_DIM = 40
        private const val RAW_TOKEN_DIM = 58
        private const val MAX_PLAYS = 31
        private const val GLOBAL_DIM = 130
    }
}
