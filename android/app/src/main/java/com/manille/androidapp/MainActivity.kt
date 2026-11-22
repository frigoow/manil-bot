package com.manille.androidapp

import com.cardtable.platform.*

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.manille.shared.ManilleRoot

/** The Android shell: per-build names, the platform, then the shared UI. */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        CrashReport.install(this)
        if (savedInstanceState == null) CrashReport.showIfAny(this)
        AppConfig.botNames = BuildConfig.BOT_NAMES.split(",").map { it.trim() }.filter { it.isNotEmpty() }
        AppConfig.cafeThemeName = BuildConfig.CAFE_THEME_NAME
        val platform = AndroidPlatform(this)
        val bench = BuildConfig.DEBUG && intent?.getBooleanExtra("bench", false) == true
        setContent {
            ManilleRoot(platform) { if (bench) it.benchmark() }
        }
    }
}
