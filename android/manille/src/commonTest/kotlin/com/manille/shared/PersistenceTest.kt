package com.manille.shared

import com.cardtable.platform.*

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

private class MemoryStorage(val files: MutableMap<String, String> = mutableMapOf()) : Storage {
    override fun read(name: String) = files[name]
    override fun write(name: String, text: String) { files[name] = text }
    override fun delete(name: String) { files.remove(name) }
    override fun exists(name: String) = name in files
    override fun rename(from: String, to: String): Boolean {
        val v = files.remove(from) ?: return false
        files[to] = v; return true
    }
}

/**
 * The files as the Android app's org.json code (v1.0.3 and earlier) wrote
 * them: an update must keep every profile, statistic and saved game.
 */
class PersistenceTest {
    private val oldProfiles = """{"profiles":[{"id":"p1","name":"You"},{"id":"p2","name":"Sam"}],""" +
        """"active":"p2","fourColor":false,"smileys":true,"animations":false,"netPlies":32,""" +
        """"language":"en","languageChosen":true,"theme":"midnight"}"""
    private val oldStats = """{"counters":{"hands":12,"hands_won":7,"points_for":400,"hints_used":3},""" +
        """"history":[{"t":1763112000,"mode":"match","us":101,"them":87},{"t":1763115600,"mode":"game","us":20,"them":40}]}"""
    private val oldSave = """{"match":true,"us":34,"them":21,"hand":4,"dealer":2,"seed":-6178302741095913472,""" +
        """"names":["Jef","Mieke","Staf"],"phase":"PLAYING","message":"Your turn.","meegaan":-1,"terug":-1,""" +
        """"declUs":21,"declThem":34,"ruleTrump":false,"declSummary":"Staf declared ♥","tricksMine":1,""" +
        """"manilles":0,"meeSummary":"","terSummary":"","engine":[1,2,3,-1,40],"version":1}"""

    @Test fun profilesWrittenByOrgJsonLoad() {
        val p = ProfileStore(MemoryStorage(mutableMapOf("profiles.json" to oldProfiles)))
        assertEquals(listOf(Profile("p1", "You"), Profile("p2", "Sam")), p.profiles)
        assertEquals("p2", p.activeId)
        assertFalse(p.fourColor); assertTrue(p.smileys); assertFalse(p.animations)
        assertEquals(32, p.botSetting); assertEquals("en", p.language); assertEquals("midnight", p.theme)
    }

    @Test fun profilesRoundTrip() {
        val storage = MemoryStorage(mutableMapOf("profiles.json" to oldProfiles))
        ProfileStore(storage).apply { setBotSetting(8); create("Robin"); setTheme("deco") }
        val again = ProfileStore(storage)
        assertEquals(3, again.profiles.size); assertEquals("p3", again.activeId)
        assertEquals(8, again.botSetting); assertEquals("deco", again.theme); assertEquals("en", again.language)
    }

    @Test fun firstRunMigratesThePreProfileFiles() {
        val storage = MemoryStorage(mutableMapOf("stats.json" to oldStats, "savegame.json" to oldSave))
        val p = ProfileStore(storage)
        assertEquals("p1", p.activeId)
        assertTrue(storage.exists("stats_p1.json")); assertTrue(storage.exists("savegame_p1.json"))
        assertFalse(storage.exists("stats.json"))
    }

    @Test fun statsWrittenByOrgJsonLoadAndKeepCounting() {
        val storage = MemoryStorage(mutableMapOf("stats_p1.json" to oldStats))
        val store = StatsStore(storage, "p1") { 1_763_200_000_000L }
        val s = store.load()
        assertEquals(12, s["hands"]); assertEquals(3, s["hints_used"]); assertEquals(2, s.history.size)
        assertEquals(101, s.history[0].optInt("us")); assertEquals("game", s.history[1].optString("mode"))
        val after = store.recordResult(true, 101, 99)
        assertEquals(1, after["matches_won"]); assertEquals(3, after.history.size)
        assertEquals(1_763_200_000L, after.history.last().optLong("t"))
        assertEquals(4, store.countHint()["hints_used"])
    }

    @Test fun firstVersionTotalsStillCount() {
        val storage = MemoryStorage(mutableMapOf("stats_p1.json" to """{"totals":{"hands":5,"games":2}}"""))
        val s = StatsStore(storage, "p1") { 0L }.load()
        assertEquals(5, s["hands"]); assertEquals(2, s["games"])
    }

    @Test fun savedGameWrittenByOrgJsonReads() {
        val save = SaveStore(MemoryStorage(mutableMapOf("savegame_p1.json" to oldSave)), "p1").read()
        assertNotNull(save)
        assertEquals(-6178302741095913472L, save.getLong("seed"))
        assertEquals("PLAYING", save.getString("phase")); assertEquals("Staf declared ♥", save.optString("declSummary"))
        assertEquals(listOf(1, 2, 3, -1, 40), save.optArray("engine")!!.map { it.toString().toInt() })
        assertEquals(-1, save.getInt("meegaan")); assertFalse(save.getBoolean("ruleTrump"))
    }

    @Test fun aSaveFromAnotherVersionIsIgnored() {
        val storage = MemoryStorage(mutableMapOf("savegame_p1.json" to oldSave.replace("\"version\":1", "\"version\":2")))
        assertNull(SaveStore(storage, "p1").read())
    }

    @Test fun formatFixedRoundsHalfUpWithTheLanguagesSeparator() {
        val was = Lang.code
        try {
            Lang.code = "en"
            assertEquals("67", formatFixed(66.6666, 0)); assertEquals("2.5", formatFixed(2.46, 1))
            assertEquals("0.0", formatFixed(0.0, 1)); assertEquals("13", formatFixed(12.5, 0))
            Lang.code = "nl"
            assertEquals("2,5", formatFixed(2.46, 1))
        } finally { Lang.code = was }
    }
}
