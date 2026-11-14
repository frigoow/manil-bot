plugins {
    id("com.android.application") version "9.1.1" apply false
    // The game, its screens and the bots' Kotlin side live in :shared
    // (Kotlin Multiplatform: Android now, iOS later); :app is the Android shell.
    id("com.android.kotlin.multiplatform.library") version "9.1.1" apply false
    id("org.jetbrains.kotlin.multiplatform") version "2.3.21" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.3.21" apply false
}
