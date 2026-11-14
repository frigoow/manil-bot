package com.manille.shared

/**
 * Per-build settings the platform shell sets before the first screen: the
 * bots' names and the café theme's name (Android: from Gradle properties,
 * so personal names stay out of git; see app/build.gradle.kts).
 */
object AppConfig {
    var botNames: List<String> = listOf("Jef", "Mieke", "Staf", "Lieve", "Rik", "Greet", "Fons", "Marie", "Warre", "Leen")
    var cafeThemeName: String = "Café"
}
