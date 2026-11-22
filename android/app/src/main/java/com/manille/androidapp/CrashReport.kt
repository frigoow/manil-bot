package com.manille.androidapp

import com.cardtable.platform.*

import android.app.Activity
import android.app.ActivityManager
import android.app.AlertDialog
import android.app.ApplicationExitInfo
import android.content.Context
import android.content.Intent
import android.os.Build
import java.io.File

/**
 * When the app died last time -- a Kotlin/Java crash, a native crash, a
 * freeze ("isn't responding") or the system killing it for memory -- the
 * next launch shows what happened and offers to share the report, so a
 * crash on someone else's phone can be diagnosed. Nothing leaves the phone
 * unless the player shares it.
 *
 * Two sources: our own uncaught-exception handler writes crash.txt (every
 * Android version), and from Android 11 the system's record of why the
 * process ended (ApplicationExitInfo), which also covers native crashes,
 * freezes and memory kills that no handler can catch.
 */
object CrashReport {
    private const val FILE = "crash.txt"
    private const val PREFS = "crash_report"
    private const val SEEN = "seen_exit_ms"

    /** Record uncaught exceptions to crash.txt, then let the app die as usual. */
    fun install(context: Context) {
        val dir = context.filesDir
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, error ->
            runCatching {
                File(dir, FILE).writeText(header() + "Thread: ${thread.name}\n\n" + error.stackTraceToString().take(60_000))
            }
            previous?.uncaughtException(thread, error)
        }
    }

    /** If the last run ended badly, ask whether to share the report. */
    fun showIfAny(activity: Activity) {
        val report = runCatching { collect(activity) }.getOrNull() ?: return
        val dutch = java.util.Locale.getDefault().language == "nl"
        AlertDialog.Builder(activity)
            .setTitle(if (dutch) "Manille is gecrasht" else "Manille crashed")
            .setMessage(
                if (dutch) "De vorige keer stopte de app onverwacht. Wil je het crashrapport delen, zodat we het kunnen oplossen?\n\n${report.lineSequence().take(8).joinToString("\n")}"
                else "The app stopped unexpectedly last time. Share the crash report so it can be fixed?\n\n${report.lineSequence().take(8).joinToString("\n")}")
            .setPositiveButton(if (dutch) "Delen" else "Share") { _, _ ->
                val send = Intent(Intent.ACTION_SEND).apply {
                    type = "text/plain"
                    putExtra(Intent.EXTRA_SUBJECT, "Manille crash report")
                    putExtra(Intent.EXTRA_TEXT, report)
                }
                activity.startActivity(Intent.createChooser(send, null))
            }
            .setNegativeButton(if (dutch) "Sluiten" else "Close", null)
            .show()
    }

    /** The report for the last bad exit not shown yet, or null. Marks it shown. */
    private fun collect(context: Context): String? {
        val parts = mutableListOf<String>()
        val file = File(context.filesDir, FILE)
        if (file.isFile) {
            parts += file.readText()
            file.delete()
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) systemExit(context)?.let { parts += it }
        return if (parts.isEmpty()) null else parts.joinToString("\n\n---\n\n")
    }

    private fun systemExit(context: Context): String? {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) return null
        val am = context.getSystemService(ActivityManager::class.java) ?: return null
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val seen = prefs.getLong(SEEN, 0L)
        val exit = am.getHistoricalProcessExitReasons(context.packageName, 0, 1).firstOrNull() ?: return null
        prefs.edit().putLong(SEEN, exit.timestamp).apply()
        if (exit.timestamp <= seen) return null
        val reason = when (exit.reason) {
            ApplicationExitInfo.REASON_CRASH -> "crash (Kotlin/Java)"
            ApplicationExitInfo.REASON_CRASH_NATIVE -> "native crash"
            ApplicationExitInfo.REASON_ANR -> "not responding (ANR)"
            ApplicationExitInfo.REASON_LOW_MEMORY -> "killed for low memory"
            ApplicationExitInfo.REASON_INITIALIZATION_FAILURE -> "failed to start"
            else -> return null   // a normal exit, or the player / system closed it
        }
        // the first run after an update has no "seen" yet: report only exits
        // from the last day, not old history
        if (seen == 0L && System.currentTimeMillis() - exit.timestamp > 24 * 3600_000L) return null
        val trace = runCatching {
            exit.traceInputStream?.bufferedReader()?.use { it.readText().take(60_000) }
        }.getOrNull()
        return header() + "Exit: $reason\n" +
            "When: ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.ROOT).format(exit.timestamp)}\n" +
            "Memory (PSS/RSS kB): ${exit.pss} / ${exit.rss}\n" +
            "Description: ${exit.description ?: "-"}\n" +
            (if (trace.isNullOrBlank()) "" else "\n$trace")
    }

    private fun header(): String =
        "Manille ${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE})\n" +
            "Device: ${Build.MANUFACTURER} ${Build.MODEL}, Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})\n" +
            "ABI: ${Build.SUPPORTED_ABIS.joinToString()}\n"
}
