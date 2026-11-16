package com.manille.jvm

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File

/** One place that opens ONNX Runtime sessions, so every net runs with the same CPU kernels. */
internal object OrtSessions {
    // ORT 1.23.2 crashed with SIGILL (ILL_ILLOPC) in a thread-pool worker on an
    // arm64 phone running Android 16: its KleidiAI kernels (SME/SME2, i8mm) were
    // chosen on a CPU that cannot run them (onnxruntime#26377). The nets are tiny,
    // so the plain MLAS NEON kernels cost nothing noticeable and run everywhere.
    const val DISABLE_KLEIDIAI = "mlas.disable_kleidiai"

    fun open(environment: OrtEnvironment, model: File): OrtSession =
        OrtSession.SessionOptions().use { options ->
            options.addConfigEntry(DISABLE_KLEIDIAI, "1")
            environment.createSession(model.absolutePath, options)
        }
}
