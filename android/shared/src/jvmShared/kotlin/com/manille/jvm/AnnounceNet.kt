package com.manille.jvm

import com.manille.shared.*
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Exact mobile export of the desktop meegaan / terug equity policy
 * (`learn.decision_layer.EquityPolicy.meegaan` / `.terug`). All scores are
 * from the deciding team's own perspective; `rel = (seat - dealer) % 4`
 * (1 or 3 for a meegaan, 0 or 2 for a terug).
 */
class AnnounceNet(model: File) : AnnounceModel {
    private val environment = OrtEnvironment.getEnvironment()
    private val session = OrtSessions.open(environment, model)

    /** `[meegaan_pass, meegaan_call, terug_pass, terug_call]` match equities. */
    override fun equities(handMask: Long, declaration: Int, rel: Int, us: Int, them: Int): FloatArray {
        val hand = FloatArray(32) { if (handMask and (1L shl it) != 0L) 1f else 0f }
        // The graph's suit-permutation table puts no-trump at index 4; a mull
        // is declaration 4 here, never -1.
        val decl = if (declaration < 0) 4L else declaration.toLong()
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(hand), longArrayOf(1, 32)).use { handTensor ->
            OnnxTensor.createTensor(environment, LongBuffer.wrap(longArrayOf(decl)), longArrayOf(1)).use { declTensor ->
                OnnxTensor.createTensor(environment, LongBuffer.wrap(longArrayOf(rel.toLong())), longArrayOf(1)).use { relTensor ->
                    OnnxTensor.createTensor(
                        environment, LongBuffer.wrap(longArrayOf(us.toLong(), them.toLong())), longArrayOf(1, 2),
                    ).use { scoreTensor ->
                        session.run(mapOf(
                            "hand" to handTensor, "declaration" to declTensor,
                            "rel" to relTensor, "scores" to scoreTensor,
                        )).use { result ->
                            @Suppress("UNCHECKED_CAST")
                            return (result[0].value as Array<FloatArray>)[0].copyOf()
                        }
                    }
                }
            }
        }
    }

    /** [equities] for many hands at once (one batched run): row i = hands[i]. */
    override fun equitiesBatch(hands: LongArray, declaration: Int, rel: Int, us: Int, them: Int): Array<FloatArray> {
        val n = hands.size
        if (n == 0) return emptyArray()
        val flat = FloatArray(n * 32) { if (hands[it / 32] and (1L shl (it % 32)) != 0L) 1f else 0f }
        val decl = if (declaration < 0) 4L else declaration.toLong()
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(flat), longArrayOf(n.toLong(), 32)).use { handTensor ->
            OnnxTensor.createTensor(environment, LongBuffer.wrap(LongArray(n) { decl }), longArrayOf(n.toLong())).use { declTensor ->
                OnnxTensor.createTensor(environment, LongBuffer.wrap(LongArray(n) { rel.toLong() }), longArrayOf(n.toLong())).use { relTensor ->
                    OnnxTensor.createTensor(
                        environment, LongBuffer.wrap(LongArray(2 * n) { if (it % 2 == 0) us.toLong() else them.toLong() }),
                        longArrayOf(n.toLong(), 2),
                    ).use { scoreTensor ->
                        session.run(mapOf(
                            "hand" to handTensor, "declaration" to declTensor,
                            "rel" to relTensor, "scores" to scoreTensor,
                        )).use { result ->
                            @Suppress("UNCHECKED_CAST")
                            return (result[0].value as Array<FloatArray>).map { it.copyOf() }.toTypedArray()
                        }
                    }
                }
            }
        }
    }

    override fun close() = session.close()

}
