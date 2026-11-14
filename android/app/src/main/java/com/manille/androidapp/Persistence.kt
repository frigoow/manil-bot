package com.manille.androidapp

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** Atomic small-JSON files in the app's private storage. */
private object JsonFile {
    fun read(file: File): JSONObject? =
        runCatching { if (file.isFile) JSONObject(file.readText()) else null }.getOrNull()

    fun write(file: File, value: JSONObject) {
        val tmp = File(file.parentFile, file.name + ".tmp")
        tmp.writeText(value.toString())
        if (!tmp.renameTo(file)) { file.delete(); tmp.renameTo(file) }
    }
}

data class Profile(val id: String, val name: String)

/**
 * Local player profiles (gui/profiles.py's role): each has its own saved game
 * and statistics. The first run creates "You" and moves the single-profile
 * files of earlier versions into it.
 */
class ProfileStore(private val context: Context) {
    private val file = File(context.filesDir, "profiles.json")

    var profiles: List<Profile> = emptyList(); private set
    var activeId: String = ""; private set
    var fourColor: Boolean = true; private set
    var smileys: Boolean = true; private set
    var animations: Boolean = true; private set
    /** "nl" (the default) or "en"; only an explicit choice in the lobby or menu overrides Dutch. */
    var language: String = "nl"; private set
    private var languageChosen = false
    var theme: String = "cafe"; private set

    init { load() }

    val active: Profile get() = profiles.first { it.id == activeId }

    private fun load() {
        val j = JsonFile.read(file)
        if (j == null) {
            profiles = listOf(Profile("p1", "You")); activeId = "p1"
            // migrate the pre-profile files into the first profile
            File(context.filesDir, "stats.json").takeIf { it.isFile }?.renameTo(File(context.filesDir, "stats_p1.json"))
            File(context.filesDir, "savegame.json").takeIf { it.isFile }?.renameTo(File(context.filesDir, "savegame_p1.json"))
            save(); return
        }
        val arr = j.optJSONArray("profiles") ?: JSONArray()
        profiles = (0 until arr.length()).map { arr.getJSONObject(it).let { p -> Profile(p.getString("id"), p.getString("name")) } }
        if (profiles.isEmpty()) profiles = listOf(Profile("p1", "You"))
        activeId = j.optString("active").takeIf { id -> profiles.any { it.id == id } } ?: profiles[0].id
        fourColor = j.optBoolean("fourColor", true)
        smileys = j.optBoolean("smileys", true)
        animations = j.optBoolean("animations", true)
        languageChosen = j.optBoolean("languageChosen", false)
        language = if (languageChosen) j.optString("language", "nl") else "nl"
        theme = j.optString("theme", "cafe")
    }

    private fun save() {
        val arr = JSONArray()
        profiles.forEach { arr.put(JSONObject().put("id", it.id).put("name", it.name)) }
        runCatching { JsonFile.write(file, JSONObject().put("profiles", arr).put("active", activeId).put("fourColor", fourColor).put("smileys", smileys).put("animations", animations).put("language", language).put("languageChosen", languageChosen).put("theme", theme)) }
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
        File(context.filesDir, "stats_$id.json").delete()
        File(context.filesDir, "savegame_$id.json").delete()
        if (activeId == id) activeId = profiles[0].id
        save()
    }

    fun setFourColor(on: Boolean) { fourColor = on; save() }
    fun setSmileys(on: Boolean) { smileys = on; save() }
    fun setAnimations(on: Boolean) { animations = on; save() }
    fun setLanguage(lang: String) { language = lang; languageChosen = true; save() }
    fun setTheme(id: String) { theme = id; save() }
}

/**
 * The game in progress (gui/saves.py's role) for one profile: written at
 * every settled point of a deal, read back by "Resume" in the lobby, removed
 * when the game or match is over.
 */
class SaveStore(context: Context, profileId: String) {
    private val file = File(context.filesDir, "savegame_$profileId.json")

    fun read(): JSONObject? = JsonFile.read(file)?.takeIf { it.optInt("version") == VERSION }
    fun write(save: JSONObject) = runCatching { JsonFile.write(file, save.put("version", VERSION)) }
    fun clear() { file.delete() }

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
class Stats(val c: Map<String, Int>, val history: List<JSONObject>) {
    operator fun get(key: String) = c[key] ?: 0
    fun rate(won: String, of: String): String = if (get(of) == 0) "–" else "%.0f%%".format(get(won) * 100f / get(of))
    fun avg(sum: String, of: String): String = if (get(of) == 0) "–" else "%.1f".format(get(sum).toFloat() / get(of))
}

class StatsStore(context: Context, profileId: String) {
    private val file = File(context.filesDir, "stats_$profileId.json")

    fun load(): Stats {
        val root = JsonFile.read(file) ?: JSONObject()
        val c = mutableMapOf<String, Int>()
        root.optJSONObject("counters")?.let { j -> j.keys().forEach { c[it] = j.optInt(it) } }
        // the first app version kept only these totals
        root.optJSONObject("totals")?.let { j -> j.keys().forEach { k -> c.putIfAbsent(k, j.optInt(k)) } }
        val h = root.optJSONArray("history") ?: JSONArray()
        return Stats(c, (0 until h.length()).map { h.getJSONObject(it) })
    }

    private fun write(c: Map<String, Int>, history: List<JSONObject>) {
        val counters = JSONObject(); c.forEach { (k, v) -> counters.put(k, v) }
        val arr = JSONArray(); history.takeLast(HISTORY_CAP).forEach { arr.put(it) }
        runCatching { JsonFile.write(file, JSONObject().put("counters", counters).put("history", arr)) }
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
        val h = s.history + JSONObject().put("t", System.currentTimeMillis() / 1000)
            .put("mode", if (match) "match" else "game").put("us", us).put("them", them)
        write(c, h)
        return load()
    }

    fun countHint(): Stats {
        val s = load(); val c = s.c.toMutableMap(); c["hints_used"] = (c["hints_used"] ?: 0) + 1
        write(c, s.history); return load()
    }

    private companion object { const val HISTORY_CAP = 200 }
}
