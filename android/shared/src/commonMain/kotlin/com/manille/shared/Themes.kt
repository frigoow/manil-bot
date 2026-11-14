package com.manille.shared

import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color

/**
 * A table theme: felt, rim, accents, text, card paper and card back. Every
 * colour of the game screen comes from here (MainActivity reads it through
 * LocalAppTheme), so a theme is just another entry in THEMES.
 */
data class AppTheme(
    val id: String,
    val name: String,
    val feltCenter: Color, val feltMid: Color, val feltEdge: Color,
    val rim: Color,
    val accent: Color, val accentBright: Color,
    val ink: Color, val inkDim: Color,
    val panel: Color,
    val paperTop: Color, val paperBottom: Color,
    val backA: Color, val backB: Color, val backLine: Color,
    val emblem: String,
    /** Night sky behind the table instead of the cloth weave. */
    val stars: Boolean = false,
    /** A soft coloured glow at the top of the screen (transparent = none). */
    val glow: Color = Color.Transparent,
    val nameNl: String = name,
    /** "cafe": the cosy café look -- wooden table with drinks, fairy lights, illustrated court cards. */
    val style: String = "classic",
    /** Screen background outside the table; null = the felt colours. */
    val bgTop: Color? = null, val bgBottom: Color? = null,
)

val THEMES: List<AppTheme> by lazy { listOf(
    AppTheme("cafe", AppConfig.cafeThemeName,
        Color(0xFF8E5C3A), Color(0xFF6E4429), Color(0xFF4A2B17), Color(0xFF2A170C),
        Color(0xFFD4A24C), Color(0xFFF0CF85), Color(0xFFF6EEDD), Color(0xFFCDBFA5), Color(0xC0141C2C),
        Color(0xFFFFFBF2), Color(0xFFF2E8D5), Color(0xFF29497A), Color(0xFF0F1D36), Color(0x66D4A24C), "◉︎",
        stars = true, style = "cafe", bgTop = Color(0xFF0F1A2E), bgBottom = Color(0xFF2B1D12)),
    AppTheme("midnight", "Midnight",
        Color(0xFF2B3F63), Color(0xFF17233C), Color(0xFF090E1C), Color(0xFF070A13),
        Color(0xFF7FD3C8), Color(0xFFB8F4EC), Color(0xFFEAF0F7), Color(0xFFA2AEC0), Color(0xB30A0F1C),
        Color(0xFFFDFDFF), Color(0xFFE5EAF3), Color(0xFF3B2B70), Color(0xFF150F33), Color(0x667FD3C8), "☾︎",
        stars = true, glow = Color(0x40508CC8), nameNl = "Middernacht"),
    AppTheme("classic", "Classic",
        Color(0xFF126043), Color(0xFF0A3826), Color(0xFF041F16), Color(0xFF2C1F14),
        Color(0xFFD8B36A), Color(0xFFF0D192), Color(0xFFF0EAD9), Color(0xFFB9B09A), Color(0xB30A1F17),
        Color(0xFFFFFEF9), Color(0xFFF3ECDC), Color(0xFF8E2F33), Color(0xFF4D1418), Color(0x66F0D192), "♦︎", nameNl = "Klassiek"),
    AppTheme("royal", "Royal",
        Color(0xFF24499A), Color(0xFF15306A), Color(0xFF0A173A), Color(0xFF1B1530),
        Color(0xFFE2C275), Color(0xFFF6DE9C), Color(0xFFEFF1F8), Color(0xFFAFB6CC), Color(0xB30B1430),
        Color(0xFFFFFDF7), Color(0xFFEDE6D3), Color(0xFF203A7C), Color(0xFF0D1B45), Color(0x77E2C275), "♛︎", nameNl = "Koninklijk"),
    AppTheme("bordeaux", "Bordeaux",
        Color(0xFF7A2334), Color(0xFF511523), Color(0xFF2B0A12), Color(0xFF1E120C),
        Color(0xFFE8C07A), Color(0xFFF7DDA6), Color(0xFFF7ECE4), Color(0xFFC8B2A6), Color(0xB32A0A12),
        Color(0xFFFFFBF2), Color(0xFFF1E5D0), Color(0xFF2F4A35), Color(0xFF14231A), Color(0x77E8C07A), "❦︎"),
    AppTheme("deco", "Art Deco",
        Color(0xFF242424), Color(0xFF151515), Color(0xFF070707), Color(0xFF8C6D2F),
        Color(0xFFD4AF37), Color(0xFFF2D675), Color(0xFFF4EEDC), Color(0xFFB5AC94), Color(0xB3111111),
        Color(0xFFFFFCF2), Color(0xFFEFE6CF), Color(0xFF151515), Color(0xFF000000), Color(0xAAD4AF37), "◆︎", nameNl = "Art deco"),
) }

fun themeById(id: String) = THEMES.firstOrNull { it.id == id } ?: THEMES[0]  // Café is the default

val LocalAppTheme = staticCompositionLocalOf { THEMES[0] }
