package com.manille.shared

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.longOrNull
import kotlinx.serialization.json.put

// The files are the ones the Android app wrote with org.json before the
// shared module existed (same names, same keys): an update keeps every
// profile, statistic and saved game. The accessors read leniently, as
// org.json's opt* did.

internal fun JsonObject.prim(key: String): JsonPrimitive? = this[key] as? JsonPrimitive

fun JsonObject.optInt(key: String, default: Int = 0): Int =
    prim(key)?.let { it.intOrNull ?: it.doubleOrNull?.toInt() ?: it.content.toIntOrNull() } ?: default

fun JsonObject.optLong(key: String, default: Long = 0L): Long =
    prim(key)?.let { it.longOrNull ?: it.doubleOrNull?.toLong() ?: it.content.toLongOrNull() } ?: default

fun JsonObject.optBoolean(key: String, default: Boolean = false): Boolean =
    prim(key)?.let { it.booleanOrNull ?: it.content.toBooleanStrictOrNull() } ?: default

fun JsonObject.optString(key: String, default: String = ""): String =
    prim(key)?.takeIf { it !is kotlinx.serialization.json.JsonNull }?.content ?: default

internal fun JsonObject.req(key: String): JsonPrimitive =
    prim(key) ?: throw NoSuchElementException("missing $key")

internal fun JsonObject.getInt(key: String): Int = req(key).let { it.intOrNull ?: error("$key is not an int") }
internal fun JsonObject.getLong(key: String): Long = req(key).let { it.longOrNull ?: error("$key is not a long") }
internal fun JsonObject.getBoolean(key: String): Boolean = req(key).let { it.booleanOrNull ?: error("$key is not a boolean") }
internal fun JsonObject.getString(key: String): String = req(key).content
internal fun JsonObject.optArray(key: String): JsonArray? = this[key] as? JsonArray
internal fun JsonObject.optObject(key: String): JsonObject? = this[key] as? JsonObject

/** Atomic small-JSON files in the app's private storage. */
internal class JsonFiles(private val storage: Storage) {
    fun read(name: String): JsonObject? =
        runCatching { storage.read(name)?.let { Json.parseToJsonElement(it).jsonObject } }.getOrNull()

    fun write(name: String, value: JsonObject) = storage.write(name, value.toString())
}

data class Profile(val id: String, val name: String)

/**
 * Local player profiles (gui/profiles.py's role): each has its own saved game
 * and statistics. The first run creates "You" and moves the single-profile
 * files of earlier versions into it.
 */
class ProfileStore(private val storage: Storage) {
    private val files = JsonFiles(storage)

    var profiles: List<Profile> = emptyList(); private set
    var activeId: String = ""; private set
    var fourColor: Boolean = true; private set
    var smileys: Boolean = true; private set
    var animations: Boolean = true; private set
    var netPlies: Int = RuleBot.WHOLE_DEAL; private set
    /** "nl" (the default) or "en"; only an explicit choice in the lobby or menu overrides Dutch. */
    var language: String = "nl"; private set
    private var languageChosen = false
    var theme: String = "cafe"; private set

    init { load() }

    val active: Profile get() = profiles.first { it.id == activeId }

    private fun load() {
        val j = files.read(FILE)
        if (j == null) {
            profiles = listOf(Profile("p1", "You")); activeId = "p1"
            // migrate the pre-profile files into the first profile
            if (storage.exists("stats.json")) storage.rename("stats.json", "stats_p1.json")
            if (storage.exists("savegame.json")) storage.rename("savegame.json", "savegame_p1.json")
            save(); return
        }
        profiles = (j.optArray("profiles") ?: JsonArray(emptyList())).mapNotNull { e ->
            (e as? JsonObject)?.let { p -> Profile(p.getString("id"), p.getString("name")) }
        }
        if (profiles.isEmpty()) profiles = listOf(Profile("p1", "You"))
        activeId = j.optString("active").takeIf { id -> profiles.any { it.id == id } } ?: profiles[0].id
        fourColor = j.optBoolean("fourColor", true)
        smileys = j.optBoolean("smileys", true)
        animations = j.optBoolean("animations", true)
        netPlies = j.optInt("netPlies", RuleBot.WHOLE_DEAL)
        languageChosen = j.optBoolean("languageChosen", false)
        language = if (languageChosen) j.optString("language", "nl") else "nl"
        theme = j.optString("theme", "cafe")
    }

    private fun save() {
        val arr = JsonArray(profiles.map { buildJsonObject { put("id", it.id); put("name", it.name) } })
        runCatching {
            files.write(FILE, buildJsonObject {
                put("profiles", arr); put("active", activeId); put("fourColor", fourColor)
                put("smileys", smileys); put("animations", animations); put("netPlies", netPlies)
                put("language", language); put("languageChosen", languageChosen); put("theme", theme)
            })
        }
    }

    fun select(id: String) { if (profiles.any { it.id == id }) { activeId = id; save() } }

    fun create(name: String): Profile {
        val next = (profiles.mapNotNull { it.id.removePrefix("p").toIntOrNull() }.maxOrNull() ?: 0) + 1
        val p = Profile("p$next", name.trim().ifEmpty { "Player $next" })
        profiles = profiles + p; activeId = p.id; save()
        return p
    }

    fun rename(id: String, name: String) {
        if (name.isBlank()) return
        profiles = profiles.map { if (it.id == id) it.copy(name = name.trim()) else it }; save()
    }

    /** Deletes a profile with its save and statistics; the last one stays. */
    fun delete(id: String) {
        if (profiles.size <= 1) return
        profiles = profiles.filter { it.id != id }
        storage.delete("stats_$id.json")
        storage.delete("savegame_$id.json")
        if (activeId == id) activeId = profiles[0].id
        save()
    }

    fun setFourColor(on: Boolean) { fourColor = on; save() }
    fun setSmileys(on: Boolean) { smileys = on; save() }
    fun setAnimations(on: Boolean) { animations = on; save() }
    fun setNetPlies(plies: Int) { netPlies = plies; save() }
    fun setLanguage(lang: String) { language = lang; languageChosen = true; save() }
    fun setTheme(id: String) { theme = id; save() }

    private companion object { const val FILE = "profiles.json" }
}

/**
 * The game in progress (gui/saves.py's role) for one profile: written at
 * every settled point of a deal, read back by "Resume" in the lobby, removed
 * when the game or match is over.
 */
class SaveStore(private val storage: Storage, profileId: String) {
    private val files = JsonFiles(storage)
    private val name = "savegame_$profileId.json"

    fun read(): JsonObject? = files.read(name)?.takeIf { it.optInt("version") == VERSION }
    fun write(save: JsonObject) = runCatching {
        files.write(name, JsonObject(save + ("version" to JsonPrimitive(VERSION))))
    }
    fun clear() { storage.delete(name) }

    companion object { const val VERSION = 1 }
}

/** Everything one finished hand contributes to the statistics. */
data class HandRecord(
    val pointsUs: Int, val pointsThem: Int,      // card points
    val scoreUs: Int, val scoreThem: Int,        // match points scored (after the multiplier)
    val tricksUs: Int, val tricksMine: Int, val manillesUs: Int,
    val weDeclared: Boolean, val iDeclared: Boolean, val mull: Boolean,
    val multiplier: Int,
    val meegaanUs: Boolean, val meegaanThem: Boolean, val meegaanMine: Boolean,
    val terugUs: Boolean, val terugThem: Boolean, val terugMine: Boolean,
)

/**
 * Lifetime statistics of one profile: named counters (so new ones can be
 * added without breaking old files) plus the last results. The desktop
 * profile's totals (gui/profiles.py) are a subset.
 */
class Stats(val c: Map<String, Int>, val history: List<JsonObject>) {
    operator fun get(key: String) = c[key] ?: 0
    fun rate(won: String, of: String): String = if (get(of) == 0) "–" else "${formatFixed(get(won) * 100.0 / get(of), 0)}%"
    fun avg(sum: String, of: String): String = if (get(of) == 0) "–" else formatFixed(get(sum).toDouble() / get(of), 1)
}

class StatsStore(storage: Storage, profileId: String, private val clock: () -> Long) {
    private val files = JsonFiles(storage)
    private val name = "stats_$profileId.json"

    fun load(): Stats {
        val root = files.read(name) ?: JsonObject(emptyMap())
        val c = mutableMapOf<String, Int>()
        root.optObject("counters")?.let { j -> j.keys.forEach { c[it] = j.optInt(it) } }
        // the first app version kept only these totals
        root.optObject("totals")?.let { j -> j.keys.forEach { k -> if (k !in c) c[k] = j.optInt(k) } }
        val h = root.optArray("history") ?: JsonArray(emptyList())
        return Stats(c, h.mapNotNull { it as? JsonObject })
    }

    private fun write(c: Map<String, Int>, history: List<JsonObject>) {
        runCatching {
            files.write(name, buildJsonObject {
                put("counters", JsonObject(c.mapValues { JsonPrimitive(it.value) }))
                put("history", JsonArray(history.takeLast(HISTORY_CAP)))
            })
        }
    }

    fun recordHand(r: HandRecord): Stats {
        val s = load(); val c = s.c.toMutableMap()
        fun add(k: String, v: Int = 1) { c[k] = (c[k] ?: 0) + v }
        val won = r.pointsUs > r.pointsThem; val lost = r.pointsUs < r.pointsThem
        add("hands"); if (won) add("hands_won"); if (lost) add("hands_lost"); if (!won && !lost) add("hands_tied")
        add("points_for", r.pointsUs); add("points_against", r.pointsThem)
        add("score_for", r.scoreUs); add("score_against", r.scoreThem)
        add("tricks_team", r.tricksUs); add("tricks_mine", r.tricksMine); add("manilles_team", r.manillesUs)
        if (r.tricksUs == 8) add("slams"); if (r.tricksUs == 0) add("slams_against")
        c["best_hand_points"] = maxOf(c["best_hand_points"] ?: 0, r.pointsUs)
        if (r.weDeclared) { add("decl_hands"); if (won) add("decl_won") } else { add("def_hands"); if (won) add("def_won") }
        if (r.iDeclared) {
            add("my_decl"); if (won) add("my_decl_won")
            if (r.mull) { add("my_mull"); if (won) add("my_mull_won") }
        }
        if (r.mull) { add("mull_hands"); if (won) add("mull_won") }
        if (r.multiplier >= 4) { add("big_hands"); if (won) add("big_won") }
        if (r.meegaanUs) { add("mee_us"); if (won) add("mee_us_won") }
        if (r.meegaanMine) { add("mee_mine"); if (won) add("mee_mine_won") }
        if (r.meegaanThem) { add("mee_them"); if (won) add("mee_them_won") }
        if (r.terugUs) { add("ter_us"); if (won) add("ter_us_won") }
        if (r.terugMine) { add("ter_mine"); if (won) add("ter_mine_won") }
        if (r.terugThem) { add("ter_them"); if (won) add("ter_them_won") }
        val streak = if (won) (c["hand_streak"] ?: 0) + 1 else 0
        c["hand_streak"] = streak; c["best_hand_streak"] = maxOf(c["best_hand_streak"] ?: 0, streak)
        write(c, s.history)
        return load()
    }

    /** A quick game (one hand) or a match to 101 ended. */
    fun recordResult(match: Boolean, us: Int, them: Int): Stats {
        val s = load(); val c = s.c.toMutableMap()
        fun add(k: String, v: Int = 1) { c[k] = (c[k] ?: 0) + v }
        if (match) {
            add("matches"); if (us > them) add("matches_won") else add("matches_lost")
            val streak = if (us > them) (c["match_streak"] ?: 0) + 1 else 0
            c["match_streak"] = streak; c["best_match_streak"] = maxOf(c["best_match_streak"] ?: 0, streak)
            if (us > them) c["best_match_margin"] = maxOf(c["best_match_margin"] ?: 0, us - them)
        } else {
            add("games"); when { us > them -> add("games_won"); us < them -> add("games_lost"); else -> add("games_tied") }
        }
        val h = s.history + buildJsonObject {
            put("t", clock() / 1000); put("mode", if (match) "match" else "game"); put("us", us); put("them", them)
        }
        write(c, h)
        return load()
    }

    fun countHint(): Stats {
        val s = load(); val c = s.c.toMutableMap(); c["hints_used"] = (c["hints_used"] ?: 0) + 1
        write(c, s.history); return load()
    }

    private companion object { const val HISTORY_CAP = 200 }
}

/**
 * [x] with [decimals] digits after the point, rounded half up like
 * String.format; the decimal separator follows the app language (a comma in
 * Dutch).
 */
fun formatFixed(x: Double, decimals: Int): String {
    var scale = 1L
    repeat(decimals) { scale *= 10 }
    val neg = x < 0
    val v = kotlin.math.floor(kotlin.math.abs(x) * scale + 0.5).toLong()
    val whole = v / scale
    val frac = (v % scale).toString().padStart(decimals, '0')
    val sep = if (Lang.dutch) "," else "."
    return (if (neg && v != 0L) "-" else "") + whole + (if (decimals > 0) sep + frac else "")
}
