package com.manille.shared

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

/**
 * English / Dutch. Every user-visible string is written as tr(english, dutch)
 * at its point of use, so the two stay side by side and a missing translation
 * is visible in review. The language is observable state: switching it
 * recomposes the screen at once (messages already built keep their language
 * until the next one).
 */
object Lang {
    var dutch by mutableStateOf(false)
}

fun tr(en: String, nl: String): String = if (Lang.dutch) nl else en
