package com.manille.androidapp

import android.content.Context
import android.os.Handler
import android.os.Looper
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import java.io.Closeable
import kotlin.math.exp
import kotlin.random.Random

enum class TablePhase { LOBBY, DECLARATION, MEEGAAN, TERUG, PLAYING, HAND_OVER, MATCH_OVER }

/** Local session mirroring the declaration and play lifecycle of gui/game_session.py. */
class GameController(context: Context) : Closeable {
    private val game = NativeGame()
    private val appContext = context.applicationContext
    private val profileStore = ProfileStore(appContext)
    init { Lang.dutch = profileStore.language == "nl" }
    private var saves = SaveStore(appContext, profileStore.activeId)
    private var statsStore = StatsStore(appContext, profileStore.activeId)
    // this hand's tricks won by the human and 10s captured by the human's team
    private var tricksMine = 0
    private var manillesUs = 0
    private val botResult = runCatching { QNet(context.applicationContext) }
    private val bot = botResult.getOrNull()
    private val declarationResult = runCatching { DeclarationNet(context.applicationContext) }
    private val declarationNet = declarationResult.getOrNull()
    private val announceResult = runCatching { AnnounceNet(context.applicationContext) }
    private val announceNet = announceResult.getOrNull()
    // The rule agent (desktop GUI "Rule-based, phone config"); the plain QGRU
    // above stays as the fallback when its assets are missing.
    private val ruleResult = runCatching {
        RuleBot(context.applicationContext, bot ?: error("QGRU model missing"), announceNet)
    }
    private val ruleBot = ruleResult.getOrNull().also {
        ruleResult.exceptionOrNull()?.let { e ->
            android.util.Log.w("manille", "rule agent unavailable, bots play the QGRU: ${e.message}")
        }
    }
    /** Which bot plays the other seats (lobby / header label). */
    val botLabel: String get() = if (ruleBot != null) tr("Rule agent", "Regelbot") else "GRU Blueprint"
    val botDetail: String get() = if (ruleBot != null)
        tr("Card counting + play-out, belief & Q nets · offline", "Kaarten tellen + uitspelen, belief- & Q-netten · offline")
    else ruleResult.exceptionOrNull()?.message?.let { tr("Rule agent unavailable: $it", "Regelbot niet beschikbaar: $it") }
        ?: tr("Latest on-device recurrent model · offline", "Recentste recurrente model op het toestel · offline")

    // The rule agent's play-out runs here, on a copy of the game, never on
    // the UI thread; results come back through the main handler.
    private val worker = java.util.concurrent.Executors.newSingleThreadExecutor()
    // This deal's calls, for the rule agent's world weights.
    private var meegaanSeat: Int? = null
    private var terugSeat: Int? = null
    private var declUs = 0
    private var declThem = 0
    private var trumpRuleKnown = false
    private val handler = Handler(Looper.getMainLooper())
    // reactions clear on their own handler: cancelBotTurn wipes `handler`
    private val reactionHandler = Handler(Looper.getMainLooper())
    private var dealer = 0
    private var dealSeed = 0L
    private var matchMode = true
    private var turnToken = 0

    // At most one delayed bot/trick step may ever be outstanding. Every timed
    // step is posted through schedule(), which cancels the previous one first,
    // so a stray extra call can never fork a second self-perpetuating loop that
    // races the first and plays the hand out in a blink.
    private var pending: Runnable? = null

    private fun schedule(delayMs: Long, action: () -> Unit) {
        pending?.let(handler::removeCallbacks)
        val runnable = object : Runnable {
            override fun run() {
                if (pending === this) { pending = null; action() }
            }
        }
        pending = runnable
        handler.postDelayed(runnable, delayMs)
    }

    var board by mutableStateOf<BoardState?>(null); private set
    var handMask by mutableStateOf(0L); private set
    var legalCards by mutableStateOf<List<Int>>(emptyList()); private set
    var handCounts by mutableStateOf(List(4) { 0 }); private set
    var phase by mutableStateOf(TablePhase.LOBBY); private set
    var message by mutableStateOf(botResult.exceptionOrNull()?.message ?: tr("Choose a table to start.", "Kies een tafel om te beginnen.")); private set
    var usScore by mutableStateOf(0); private set
    var themScore by mutableStateOf(0); private set
    var showOpenCards by mutableStateOf(false); private set
    var hints by mutableStateOf<Map<Int, Float>>(emptyMap()); private set
    var hintPick by mutableStateOf<Int?>(null); private set
    var declarationHints by mutableStateOf<Map<Int, Float>>(emptyMap()); private set
    var declarationHintPick by mutableStateOf<Int?>(null); private set
    var declarationSummary by mutableStateOf(""); private set
    var meegaanSummary by mutableStateOf<String?>(null); private set
    var terugSummary by mutableStateOf<String?>(null); private set
    // (pass%, call%) preference share for the human's meegaan/terug call, and
    // whether the policy itself would call. Null when the hint is hidden.
    var callHints by mutableStateOf<Pair<Float, Float>?>(null); private set
    var callHintPick by mutableStateOf(false); private set
    var handNumber by mutableStateOf(0); private set
    var heldTrickCards by mutableStateOf<List<Int>>(emptyList()); private set
    var heldTrickPlayers by mutableStateOf<List<Int>>(emptyList()); private set
    var trickPaused by mutableStateOf(false); private set
    /** The finished trick's cards sweep towards this seat (-1: not now). */
    var trickCollectTo by mutableStateOf(-1); private set
    var stats by mutableStateOf(statsStore.load()); private set
    var profiles by mutableStateOf(profileStore.profiles); private set
    var profileName by mutableStateOf(profileStore.active.name); private set
    var activeProfileId by mutableStateOf(profileStore.activeId); private set
    var fourColor by mutableStateOf(profileStore.fourColor); private set
    var smileys by mutableStateOf(profileStore.smileys); private set
    var animations by mutableStateOf(profileStore.animations); private set
    var theme by mutableStateOf(themeById(profileStore.theme)); private set
    /** The previous completed trick: cards, players, winner (for "Last trick"). */
    var lastTrickCards by mutableStateOf<List<Int>>(emptyList()); private set
    var lastTrickPlayers by mutableStateOf<List<Int>>(emptyList()); private set
    var lastTrickWinner by mutableStateOf(-1); private set
    /** Emoji reaction currently shown per bot seat. */
    var reactions by mutableStateOf<Map<Int, String>>(emptyMap()); private set
    /** One-line summary of the saved game, or null when there is none. */
    var savedGame by mutableStateOf(saveSummary()); private set

    /** Bot names for seats 1..3 (West, North, East), fresh for every game. */
    var botNames by mutableStateOf(NAMES.take(3)); private set

    private fun pickNames() {
        botNames = NAMES.shuffled().take(3)
    }

    fun start(match: Boolean) {
        pickNames()
        matchMode = match; usScore = 0; themScore = 0; handNumber = 0; newDeal()
    }

    val isMatch: Boolean get() = matchMode
    /** Seats that called this deal (for the table's speech bubbles). */
    val meegaanBy: Int? get() = meegaanSeat
    val terugBy: Int? get() = terugSeat
    val dealerSeat: Int get() = dealer

    /** Start the current kind of game over (the save is replaced). */
    fun restart() { cancelBotTurn(); saves.clear(); start(matchMode) }

    // ---- profiles ---------------------------------------------------------
    private fun reloadProfile() {
        saves = SaveStore(appContext, profileStore.activeId)
        statsStore = StatsStore(appContext, profileStore.activeId)
        stats = statsStore.load(); savedGame = saveSummary()
        profiles = profileStore.profiles; profileName = profileStore.active.name
        activeProfileId = profileStore.activeId
    }
    fun selectProfile(id: String) { lobby(); profileStore.select(id); reloadProfile() }
    fun createProfile(name: String) { lobby(); profileStore.create(name); reloadProfile() }
    fun renameProfile(id: String, name: String) { profileStore.rename(id, name); reloadProfile() }
    fun deleteProfile(id: String) { lobby(); profileStore.delete(id); reloadProfile() }
    fun setFourColorDeck(on: Boolean) { profileStore.setFourColor(on); fourColor = on }
    fun chooseTheme(id: String) { profileStore.setTheme(id); theme = themeById(id) }
    var dutch by mutableStateOf(Lang.dutch); private set
    fun setLanguage(nl: Boolean) {
        profileStore.setLanguage(if (nl) "nl" else "en"); Lang.dutch = nl; dutch = nl
        savedGame = saveSummary()
        if (phase == TablePhase.LOBBY) message = tr("Choose a table to start.", "Kies een tafel om te beginnen.")
    }
    fun enableAnimations(on: Boolean) { profileStore.setAnimations(on); animations = on }
    fun enableSmileys(on: Boolean) { profileStore.setSmileys(on); smileys = on; if (!on) reactions = emptyMap() }

    // ---- bot reactions ------------------------------------------------------
    private fun react(seat: Int, vararg choices: String) {
        if (!smileys || seat == HUMAN || seat !in 0..3) return
        val emoji = choices[Random.nextInt(choices.size)]
        reactions = reactions + (seat to emoji)
        reactionHandler.postDelayed({ if (reactions[seat] == emoji) reactions = reactions - seat }, REACTION_MS)
    }

    /** What the human can send: smileys, then short lines (text to mood). */
    val playerSmileys = listOf("😀", "😏", "😎", "😂", "😅", "😩", "🙈", "🗿", "👍", "👏", "🎉", "🤑", "😤", "🙄", "🔥", "😈", "🪤", "🤝")
    val playerLines: List<Pair<String, String>> get() = listOf(
        tr("Well played!", "Goe gespeeld!") to "cheer", "Haha 😏" to "taunt", tr("Come on, partner!", "Komaan partner!") to "cheer",
        tr("Oops...", "Oei...") to "sad", tr("Bad luck!", "Pech!") to "sad", tr("Too strong 💪", "Te sterk 💪") to "taunt", "GG" to "cheer")

    /** The human sends a smiley or a short line; the bots may answer. */
    fun sendReaction(text: String, lineMood: String? = null) {
        reactions = reactions + (HUMAN to text)
        reactionHandler.postDelayed({ if (reactions[HUMAN] == text) reactions = reactions - HUMAN }, REACTION_MS)
        if (!smileys) return
        val mood = lineMood ?: when (text) {
            "😏", "😎", "😂", "🤑" -> "taunt"; "😩", "😅", "🙈" -> "sad"; "👍", "👏", "🎉" -> "cheer"
            "🪤" -> "trap"; "🤝" -> "respect"; else -> null
        }
        val taunt = mood == "taunt"; val sad = mood == "sad"; val cheer = mood == "cheer"
        reactionHandler.postDelayed({
            when {
                mood == "trap" -> react(if (Random.nextBoolean()) 1 else 3, "😤", "🙄", tr("Sneaky! 😤", "Vuile streek! 😤"), tr("Not falling for that", "Daar trap ik niet in"))
                mood == "respect" -> react(Random.nextInt(1, 4), "🤝", tr("Respect 🤝", "Respect 🤝"), "GG 🤝")
                taunt -> react(if (Random.nextBoolean()) 1 else 3, "😒", "🙄", "🗿", tr("Just wait...", "Wacht maar..."), tr("Easy, easy 😤", "Rustig, rustig 😤"), tr("Lucky!", "Geluk!"))
                sad -> react(2, "💪", tr("We'll be fine 💪", "Komt goed 💪"), tr("The next trick is ours!", "Volgende slag is voor ons!"))
                cheer -> if (Random.nextBoolean()) react(2, "😄", "👍", tr("Stronger together! 🙌", "Samen sterk! 🙌")) else react(1, "🙄", tr("Yeah yeah...", "Ja ja..."))
                Random.nextInt(3) == 0 -> react(Random.nextInt(1, 4), "😄", "🙂", "😉")
            }
        }, 900L)
    }

    /** ", Name" when addressing a bot, nothing for the human. */
    private fun at(p: Int) = if (p == HUMAN) "" else ", ${seatName(p)}"

    /** Tricks since the bots last reacted (keeps the table from chattering). */
    private var quietTricks = 1

    /** Each seat's legal cards when it played into the current trick. */
    private val trickLegal = mutableMapOf<Int, Set<Int>>()

    /**
     * What a finished trick makes the bots feel (and say). The point is
     * surprise: losing an ace you had to play is routine (a shrug at most),
     * losing one you could have kept, or seeing a winning 10 bought by a ruff,
     * is news. Events are ranked; each bot reacts at most once and only the
     * two loudest reactions per trick are shown.
     */
    private fun reactToTrick(cards: List<Int>, players: List<Int>, taker: Int, before: BoardState) {
        val trump = before.trumpSuit
        val pts = intArrayOf(0, 0, 0, 1, 2, 3, 4, 5)
        fun ptsOf(c: Int) = pts[c % 8]
        val points = cards.sumOf { ptsOf(it) }
        val led = cards[0] / 8
        val tIdx = players.indexOf(taker)
        val takerCard = cards[tIdx]
        val partnerOf = { p: Int -> (p + 2) % 4 }
        // who was winning the trick right after card k (the engine decides)
        fun leaderAfter(k: Int) = players[NativeGame.trickWinner(cards.subList(0, k + 1), trump)]
        // all cards no longer in any hand, minus this trick: played earlier
        val inHands = (0 until 4).fold(0L) { m, q -> m or game.handMask(q) }
        val trickMask = cards.fold(0L) { m, c -> m or (1L shl c) }
        val earlier = (0 until 32).filter { (inHands shr it) and 1L == 0L && (trickMask shr it) and 1L == 0L }.toSet()

        data class Line(val prio: Int, val seat: Int, val choices: List<String>)
        val lines = mutableListOf<Line>()
        fun add(prio: Int, seat: Int, vararg c: String) { if (seat != HUMAN) lines += Line(prio, seat, c.toList()) }

        // 1. an ace or a 10 lost to the other team
        cards.forEachIndexed { i, c ->
            val p = players[i]
            if (teamOf(p) == teamOf(taker) || ptsOf(c) < 4) return@forEachIndexed
            val ten = c % 8 == 7
            val legal = trickLegal[p]
            // forced: every other legal card was worth as much (unknown after a resume: say nothing loud)
            val forced = legal == null || legal.none { it != c && ptsOf(it) < ptsOf(c) }
            val wasWinning = i == 0 || leaderAfter(i) == p
            val ruffedLater = trump >= 0 && c / 8 != trump && takerCard / 8 == trump && tIdx > i
            when {
                forced -> {
                    // routine: the card had to go
                    if (Random.nextInt(100) < 35) add(1, p, "😩", "🗿", tr("Had to...", "Moest..."), tr("Nothing to be done", "Niks aan te doen"))
                    if (ten && Random.nextInt(100) < 50) add(2, taker, "😏", tr("Thanks${at(p)}!", "Merci${at(p)}!"))
                }
                !wasWinning -> {
                    // thrown onto a trick the other team already had: a present
                    add(if (ten) 9 else 7, taker, "🎁", "😂", tr("A gift${at(p)}? Thanks! 🎁", "Cadeautje${at(p)}? Merci! 🎁"), tr("Don't mind if I do 😋", "Da pak ik graag mee 😋"))
                    add(if (ten) 8 else 7, partnerOf(p), "😡", "🤦", "😤", tr("Why?? 😡", "Waarom?? 😡"), tr("Seriously, partner?! 😤", "Serieus, partner?! 😤"), tr("What are you doing?!", "Wat doe je nu?!"))
                    add(5, p, "🙈", tr("Oops 🙈", "Oeps 🙈"))
                }
                ruffedLater -> {
                    // it was winning and got bought with a trump
                    add(if (ten) 10 else 8, taker, "🔥", "💪", tr("Trumped${at(p)}! 😏", "Getroefd${at(p)}! 😏"), tr("Bought it!", "Hop, gekocht!"), tr("Too late${at(p)} 😎", "Te laat${at(p)} 😎"))
                    add(if (ten) 9 else 7, p, "😤", "😠", tr("Trumping? Seriously? 😤", "Kopen? Serieus? 😤"), "Grr...")
                }
                ten -> {
                    // a winning 10 overtrumped / beaten otherwise
                    add(9, taker, "😎", tr("Thanks for that 10${at(p)}!", "Merci voor die 10${at(p)}!"), tr("Mmm, tasty manille 😋", "Mmm, lekker die manille 😋"))
                    add(8, p, "😱", tr("Ugh, my 10...", "Pff, mijn 10..."))
                }
                else -> {
                    // an ace beaten by its own 10: a gamble if that 10 was still out
                    // the ace was winning and the 10 of its suit was lying in wait
                    // behind it: a mousetrap. Leading the ace into it is news.
                    val tenOut = (c / 8) * 8 + 7 !in earlier
                    val sprungByTen = takerCard == (c / 8) * 8 + 7 && tIdx > i
                    if (tenOut && sprungByTen) {
                        add(if (i == 0) 7 else 6, taker, "🪤", tr("Trapped! 🪤", "In de val! 🪤"), tr("Walked right into it${at(p)} 🪤", "Recht in de val${at(p)} 🪤"), tr("There's the 10! 😎", "Daar is de 10! 😎"))
                        add(if (i == 0) 6 else 4, p, "😬", "🙈", tr("Oops, my ace", "Oei, mijn aas"), tr("Saw that coming...", "Had ik kunnen weten..."))
                        if (i == 0) add(6, partnerOf(p), "😤", "🤦", tr("Why that ace?! 😤", "Waarom die aas?! 😤"), tr("The 10 was still out! 😠", "De 10 was nog niet weg! 😠"))
                    }
                }
            }
        }
        // 2. overtrumped: a winning trump beaten by a higher one
        if (trump >= 0 && takerCard / 8 == trump) {
            for (i in 0 until tIdx) {
                val p = players[i]
                if (cards[i] / 8 == trump && teamOf(p) != teamOf(taker) && leaderAfter(tIdx - 1) == p && led != trump) {
                    add(7, taker, "😎", tr("Overtrumped! 😎", "Overgetroefd! 😎"))
                    add(6, p, "😩", "😤", tr("Not again...", "Niet weer..."), "Grr... 😤")
                }
            }
        }
        // 3. trumping a trick your partner already had (with a choice not to)
        if (trump >= 0 && takerCard / 8 == trump && led != trump && tIdx > 0) {
            val partner = partnerOf(taker)
            val legal = trickLegal[taker]
            val couldDuck = legal != null && legal.any { alt ->
                alt / 8 != trump && players[NativeGame.trickWinner(cards.subList(0, tIdx) + alt, trump)] == partner }
            if (leaderAfter(tIdx - 1) == partner && couldDuck) add(7, partner, "😠", "🤦", "🗿", tr("That one was ours already! 😠", "Die was al van ons! 😠"), tr("Why trump that?! 😤", "Waarom kopen?! 😤"))
        }
        // 4. partner smears points onto a trick that was already ours
        cards.forEachIndexed { i, c ->
            val p = players[i]
            if (p == taker || teamOf(p) != teamOf(taker) || ptsOf(c) < 4 || i == 0) return@forEachIndexed
            val chose = trickLegal[p]?.any { it != c && ptsOf(it) < ptsOf(c) } == true
            if (chose && teamOf(leaderAfter(i - 1)) == teamOf(taker) && leaderAfter(i) != p)
                add(5, taker, "👍", "🙏", "🤝", tr("Nice one, partner 👍", "Goe gevet, partner 👍"), tr("Thanks partner!", "Merci partner!"))
        }
        // 5. the trick that decides the hand: the team passes 30
        val teamBefore = if (teamOf(taker) == 0) before.team0Points else before.team1Points
        if (teamBefore <= 30 && teamBefore + points > 30) {
            add(5, if (taker != HUMAN) taker else partnerOf(taker), "🎉", tr("That's the hand! 🎉", "Binnen! 🎉"), tr("Over 30! 🎉", "Erover! 🎉"))
            add(4, (taker + 1) % 4, "😞", "🗿", tr("Well, that's that", "Dat is gedaan"))
        }
        // 6. small wonders and wasted effort
        if (takerCard % 8 <= 2 && takerCard / 8 != trump && points >= 6)
            add(6, taker, "😂", tr("With a ${arrayOf("7", "8", "9")[takerCard % 8]}! 😂", "Met een ${arrayOf("7", "8", "9")[takerCard % 8]}! 😂"))
        if (points == 0 && trump >= 0 && takerCard / 8 == trump && takerCard % 8 >= 6)
            add(4, (taker + 1) % 4, "🗿", tr("All that for nothing 🗿", "Veel moeite voor niks 🗿"))
        if (points >= 15) add(4, taker, "🤑", tr("Ka-ching! 🤑", "Kassa! 🤑"), tr("What a trick!", "Den dikke slag!"))
        if (taker == HUMAN && points >= 10) {
            add(4, 2, "👏", tr("Well played! 👏", "Goe gespeeld! 👏"), tr("That's how it's done!", "Zo moet da!"))
            add(3, 1, "😒", tr("Lucky...", "Geluk gehad..."))
        }

        // Only real news gets a voice: big events (prio >= 7) always, medium
        // ones sometimes, small ones rarely; after a reaction the table stays
        // quiet for a trick unless something big happens. At most one
        // reaction per trick, two when two big events coincide.
        val seen = mutableSetOf<Int>()
        val ranked = lines.sortedByDescending { it.prio }.filter { seen.add(it.seat) }
        val top = ranked.firstOrNull() ?: run { quietTricks++; return }
        val chance = when { top.prio >= 7 -> 100; top.prio >= 5 -> 45; else -> 15 }
        if (top.prio < 7 && quietTricks < 1) { quietTricks++; return }
        if (Random.nextInt(100) >= chance) { quietTricks++; return }
        quietTricks = 0
        react(top.seat, *top.choices.toTypedArray())
        ranked.getOrNull(1)?.takeIf { it.prio >= 7 && top.prio >= 7 }?.let { react(it.seat, *it.choices.toTypedArray()) }
    }

    fun lobby() {
        cancelBotTurn(); phase = TablePhase.LOBBY; message = tr("Choose a table to start.", "Kies een tafel om te beginnen.")
        savedGame = saveSummary()
    }

    /**
     * Debug timing (adb shell am start -n com.manille.androidapp/.MainActivity
     * --ez bench true): the rule agent plays a few whole deals in all four
     * seats on the worker thread; per-move stage timings go to logcat, tag
     * manille-bench.
     */
    fun benchmark(deals: Int = 4) {
        val model = ruleBot ?: run { android.util.Log.w("manille-bench", "no rule agent"); return }
        worker.execute {
            val total = mutableListOf<Long>()
            val stages = LongArray(5)
            var doubledMoves = 0
            for (d in 0 until deals) {
                val g = NativeGame()
                g.newDeal(d % 4, 9000L + d, if (d % 4 == 3) -1 else d % 4)
                if (d % 2 == 1) g.announceDouble((d % 4 + 1) % 4)
                g.startPlaying()
                val calls = RuleBot.Calls(d % 4, if (d % 2 == 1) (d % 4 + 1) % 4 else null, null,
                    0, 0, false, emptySet())
                while (!g.state().finished) {
                    val t0 = android.os.SystemClock.elapsedRealtime()
                    val card = model.choose(g, calls)
                    val ms = android.os.SystemClock.elapsedRealtime() - t0
                    if (g.legalCards().size > 1) {
                        total += ms
                        for (i in 0 until 5) stages[i] += model.lastTimings[i]
                        if (g.state().doubled) doubledMoves += 1
                    }
                    g.applyCard(card)
                }
                g.close()
            }
            total.sort()
            val n = total.size.coerceAtLeast(1)
            android.util.Log.i("manille-bench",
                "moves=$n mean=${total.sum() / n}ms median=${total[n / 2]}ms p90=${total[(n * 9) / 10]}ms " +
                "max=${total.last()}ms | per move: sample=${stages[0] / n} belief=${stages[1] / n} " +
                "q=${stages[2] / n} doubles=${stages[3] / n} (over all moves; $doubledMoves doubled) " +
                "playout=${stages[4] / n}")
        }
    }

    // ---- saved game -----------------------------------------------------
    private fun persist() {
        if (phase == TablePhase.LOBBY) return
        val save = org.json.JSONObject().apply {
            put("match", matchMode); put("us", usScore); put("them", themScore)
            put("hand", handNumber); put("dealer", dealer); put("seed", dealSeed)
            put("names", org.json.JSONArray(botNames))
            put("phase", phase.name); put("message", message)
            put("meegaan", meegaanSeat ?: -1); put("terug", terugSeat ?: -1)
            put("declUs", declUs); put("declThem", declThem); put("ruleTrump", trumpRuleKnown)
            put("declSummary", declarationSummary)
            put("tricksMine", tricksMine); put("manilles", manillesUs)
            put("meeSummary", meegaanSummary ?: ""); put("terSummary", terugSummary ?: "")
            put("engine", org.json.JSONArray(game.snapshot().toList()))
        }
        saves.write(save)
    }

    private fun saveSummary(): String? {
        val j = saves.read() ?: return null
        val kind = if (j.optBoolean("match")) "Match ${j.optInt("us")}–${j.optInt("them")}" else tr("Quick game", "Snel spel")
        return tr("$kind · hand ${j.optInt("hand")}", "$kind · spel ${j.optInt("hand")}")
    }

    /** Continue the saved game; false if there is none or it cannot be read. */
    fun resume(): Boolean {
        val j = saves.read() ?: return false
        return try {
            cancelBotTurn()
            val engine = j.getJSONArray("engine")
            game.restore(IntArray(engine.length()) { engine.getInt(it) })
            matchMode = j.getBoolean("match"); usScore = j.getInt("us"); themScore = j.getInt("them")
            handNumber = j.getInt("hand"); dealer = j.getInt("dealer"); dealSeed = j.getLong("seed")
            j.optJSONArray("names")?.let { a -> if (a.length() == 3) botNames = List(3) { a.getString(it) } }
            meegaanSeat = j.getInt("meegaan").takeIf { it >= 0 }
            terugSeat = j.getInt("terug").takeIf { it >= 0 }
            declUs = j.getInt("declUs"); declThem = j.getInt("declThem")
            trumpRuleKnown = j.getBoolean("ruleTrump")
            declarationSummary = j.optString("declSummary")
            tricksMine = j.optInt("tricksMine"); manillesUs = j.optInt("manilles")
            meegaanSummary = j.optString("meeSummary").ifEmpty { null }
            terugSummary = j.optString("terSummary").ifEmpty { null }
            hints = emptyMap(); hintPick = null; declarationHints = emptyMap(); declarationHintPick = null
            callHints = null; callHintPick = false
            heldTrickCards = emptyList(); heldTrickPlayers = emptyList(); trickPaused = false; trickCollectTo = -1
            phase = TablePhase.valueOf(j.getString("phase"))
            message = j.optString("message")
            refresh()
            if (phase == TablePhase.PLAYING) {
                if (board?.finished == true) completeHand() else afterMove()
            }
            true
        } catch (error: Exception) {
            saves.clear(); savedGame = null
            message = tr("The saved game could not be restored.", "Het bewaarde spel kon niet hersteld worden.")
            phase = TablePhase.LOBBY
            false
        }
    }


    fun newDeal() {
        cancelBotTurn()
        dealer = if (handNumber == 0) Random.nextInt(4) else (dealer + 1) % 4
        if (!matchMode) { usScore = 0; themScore = 0 }
        dealSeed = Random.nextLong()
        game.newDeal(dealer, dealSeed); handNumber += 1; hints = emptyMap(); hintPick = null
        declarationHints = emptyMap(); declarationHintPick = null
        callHints = null; callHintPick = false
        declarationSummary = ""; meegaanSummary = null; terugSummary = null
        meegaanSeat = null; terugSeat = null; tricksMine = 0; manillesUs = 0
        lastTrickCards = emptyList(); lastTrickPlayers = emptyList(); lastTrickWinner = -1
        // match score from the declaring team's side, as the calls see it
        if (teamOf(dealer) == HUMAN_TEAM) { declUs = usScore; declThem = themScore } else { declUs = themScore; declThem = usScore }
        // a bot dealer without the declaration net keeps the C++ trump rule
        trumpRuleKnown = dealer != HUMAN && declarationNet == null
        heldTrickCards = emptyList(); heldTrickPlayers = emptyList(); trickPaused = false; trickCollectTo = -1; refresh()
        if (dealer == HUMAN) {
            phase = TablePhase.DECLARATION
            message = ""
            persist()
        } else {
            declarationNet?.let { model ->
                val scores = if (teamOf(dealer) == HUMAN_TEAM) usScore to themScore else themScore to usScore
                val equities = model.equities(game.handMask(dealer), scores.first, scores.second)
                val choice = equities.indices.maxBy { equities[it] }
                game.newDeal(dealer, dealSeed, if (choice == 4) -1 else choice)
                if (choice == 4) react(dealer, "😎", tr("Mull! 😎", "Mull! 😎"), tr("Mull, daring!", "Mull, durven!"))
                refresh()
            }
            phase = TablePhase.PLAYING
            declarationSummary = tr("${seatName(dealer)} declared ${trumpSymbol(board?.trumpSuit ?: -1)}", "${seatName(dealer)} koos ${trumpSymbol(board?.trumpSuit ?: -1)}")
            message = "$declarationSummary."
            afterDeclaration()
        }
    }

    fun declare(trump: Int) {
        if (phase != TablePhase.DECLARATION) return
        game.newDeal(dealer, dealSeed, trump); refresh()
        declarationHints = emptyMap(); declarationHintPick = null
        declarationSummary = tr("You declared ${trumpSymbol(trump)}", "Jij koos ${trumpSymbol(trump)}")
        message = "$declarationSummary."; afterDeclaration()
    }

    fun answerMeegaan(accept: Boolean) {
        if (phase != TablePhase.MEEGAAN) return
        callHints = null; callHintPick = false
        if (accept) {
            game.announceDouble(HUMAN); refresh(); meegaanSeat = HUMAN
            react(dealer, "😬", "😮", tr("You dare? 😬", "Durf je? 😬"), tr("Uh oh...", "Oei oei..."))
            meegaanSummary = tr("You knocked", "Jij klopte")
            message = tr("You knock — stakes are now ×${board?.multiplier}.", "Jij klopt — de inzet is nu ×${board?.multiplier}.")
            botsKnockBack()
        } else if ((dealer + 3) % 4 == PARTNER) {
            // the human passed first; the partner bot still gets its say
            partnerKnocks()
        }
        beginPlaying()
    }

    fun answerTerug(accept: Boolean) {
        if (phase != TablePhase.TERUG) return
        callHints = null; callHintPick = false
        if (accept) {
            game.announceRedouble(HUMAN); refresh(); terugSeat = HUMAN; terugSummary = tr("You knocked back", "Jij klopte terug")
            meegaanSeat?.let { react(it, "😱", "😳", tr("Yikes 😱", "Aiai 😱"), tr("Brave!", "Da durft wel!")) }
            message = tr("$terugSummary — stakes are now ×${board?.multiplier}.", "$terugSummary — de inzet is nu ×${board?.multiplier}.")
        } else if (dealer == HUMAN) {
            // the human dealt and passed; the partner bot still gets its say
            partnerKnocksBack()
        }
        beginPlaying()
    }

    fun playHuman(card: Int) {
        val snapshot = board ?: return
        if (phase != TablePhase.PLAYING || snapshot.currentPlayer != HUMAN || card !in legalCards) {
            message = tr("That card is not legal now.", "Die kaart mag je nu niet spelen."); return
        }
        hints = emptyMap(); hintPick = null; applyCardWithTiming(card)
    }

    fun toggleHints() {
        val snapshot = board ?: return
        if (phase != TablePhase.PLAYING || snapshot.currentPlayer != HUMAN) return
        if (hints.isNotEmpty()) { hints = emptyMap(); hintPick = null; return }
        countHint()
        val rule = ruleBot
        if (rule != null) { ruleHints(rule); return }
        val model = bot ?: return
        hints = run {
            val scores = model.scoreCards(game.gruRaw(HUMAN), game.gruLength(HUMAN), game.gruGlobal(HUMAN),
                game.actionFeatures(HUMAN), game.legalCards())
            val top = scores.maxOf { it.value }
            // Desktop gui/hints.py de-normalizes Q to final team points:
            // points = q * 30 + 30, then softmaxes at TAU_PTS = 1.
            val weights = scores.map { exp((it.value - top) * 30.0).toFloat() }
            val total = weights.sum().coerceAtLeast(.0001f)
            hintPick = scores.maxBy { it.value }.card
            scores.indices.associate { scores[it].card to (weights[it] / total * 100f) }
        }
    }

    /**
     * The rule agent's own analysis of the human's position, as the desktop
     * "Rule-based, phone config" hint shows it (gui/hints.py): its pick, and
     * a softmax at 1 point (TAU_PTS) over each card's play-out value. Runs on
     * the worker; dropped if the position changed meanwhile.
     */
    private fun ruleHints(model: RuleBot) {
        val token = turnToken
        val position = game.snapshot().contentHashCode()
        val copy = game.copy()
        val calls = RuleBot.Calls(dealer, meegaanSeat, terugSeat, declUs, declThem, trumpRuleKnown, setOf(HUMAN))
        message = tr("Thinking about a hint…", "Even nadenken over een hint…")
        worker.execute {
            val values = DoubleArray(32)
            val pick = try { model.analyse(copy, calls, values) } catch (error: Exception) { -1 } finally { copy.close() }
            handler.post {
                if (token != turnToken || phase != TablePhase.PLAYING || board?.currentPlayer != HUMAN
                    || game.snapshot().contentHashCode() != position) return@post
                val legal = game.legalCards().toList()
                val valued = legal.filter { !values[it].isNaN() }
                hints = if (valued.size == legal.size && legal.isNotEmpty()) {
                    val top = legal.maxOf { values[it] }
                    val weights = legal.associateWith { exp((values[it] - top) / TAU_PTS) }
                    val total = weights.values.sum()
                    legal.associate { it to (weights.getValue(it) / total * 100.0).toFloat() }
                } else {
                    legal.associateWith { if (it == pick) 100f else 0f }
                }
                hintPick = pick.takeIf { it in legal }
                message = tr("Your turn.", "Jouw beurt.")
            }
        }
    }

    private fun countHint() { stats = statsStore.countHint() }

    fun toggleContextHint() = when (phase) {
        TablePhase.DECLARATION -> toggleDeclarationHint()
        TablePhase.MEEGAAN, TablePhase.TERUG -> toggleCallHint()
        else -> toggleHints()
    }

    fun toggleCallHint() {
        if (phase != TablePhase.MEEGAAN && phase != TablePhase.TERUG) return
        if (callHints != null) { callHints = null; callHintPick = false; return }
        val model = announceNet ?: run {
            message = announceResult.exceptionOrNull()?.message ?: tr("Hint unavailable.", "Hint niet beschikbaar.")
            return
        }
        val rel = (HUMAN - dealer + 4) % 4
        val ev = model.equities(handMask, declarationIndex(), rel, usScore, themScore)
        val (vPass, vCall) = if (phase == TablePhase.MEEGAAN) ev[0] to ev[1] else ev[2] to ev[3]
        // Same convention as gui/hints.py _call_out: softmax of the two match
        // equities at one percentage point (TAU_EQ_PP = 1) — relative
        // preference, not a probability of being right.
        val top = maxOf(vPass, vCall)
        val wPass = exp((vPass - top) * 100.0).toFloat()
        val wCall = exp((vCall - top) * 100.0).toFloat()
        val total = (wPass + wCall).coerceAtLeast(.0001f)
        callHints = (wPass / total * 100f) to (wCall / total * 100f)
        callHintPick = vCall > vPass
    }

    fun toggleDeclarationHint() {
        if (phase != TablePhase.DECLARATION) return
        if (declarationHints.isNotEmpty()) {
            declarationHints = emptyMap(); declarationHintPick = null; return
        }
        val model = declarationNet ?: run {
            message = declarationResult.exceptionOrNull()?.message ?: tr("Declaration hint unavailable.", "Troefhint niet beschikbaar.")
            return
        }
        val equities = model.equities(handMask, usScore, themScore)
        val top = equities.max()
        val weights = equities.map { exp((it - top) * 100.0).toFloat() }
        val total = weights.sum().coerceAtLeast(.0001f)
        declarationHintPick = equities.indices.maxBy { equities[it] }
        declarationHints = equities.indices.associateWith { weights[it] / total * 100f }
    }

    fun toggleOpenCards() { showOpenCards = !showOpenCards }

    private fun afterDeclaration() {
        val declaringTeam = teamOf(dealer)
        if (declaringTeam == HUMAN_TEAM) {
            // Both defenders are bots; seat order from the dealer's left, first
            // yes takes the meegaan (mirrors gui/game_session._bot_meegaan).
            val defender = listOf((dealer + 1) % 4, (dealer + 3) % 4).firstOrNull { botMeegaan(it) }
            if (defender != null) {
                game.announceDouble(defender); refresh(); meegaanSeat = defender; react(defender, "😈", "😏", tr("I knock! 😈", "Ik klop! 😈"), tr("Knock knock! 😈", "Klop klop! 😈"))
                meegaanSummary = tr("${seatName(defender)} knocked", "${seatName(defender)} klopte")
                // Our answer: dealer first, then partner, first yes takes it.
                // The partner bot answers before the human when it dealt.
                if (dealer == PARTNER && partnerKnocksBack()) { beginPlaying(); return }
                phase = TablePhase.TERUG
                message = ""
                persist()
            } else beginPlaying()
        } else {
            // Defenders from the dealer's left, first yes takes the meegaan:
            // the partner bot answers before the human when it sits there.
            if ((dealer + 1) % 4 == PARTNER && partnerKnocks()) { beginPlaying(); return }
            phase = TablePhase.MEEGAAN
            message = ""
            persist()
        }
    }

    /** The partner bot's meegaan against the opponents' declaration; the
     *  opponents (both bots) then answer it. False if it passes. */
    private fun partnerKnocks(): Boolean {
        if (!botMeegaan(PARTNER)) return false
        game.announceDouble(PARTNER); refresh(); meegaanSeat = PARTNER
        react(PARTNER, "😈", "😏", tr("I knock! 😈", "Ik klop! 😈"), tr("Knock knock! 😈", "Klop klop! 😈"))
        meegaanSummary = tr("${seatName(PARTNER)} knocked", "${seatName(PARTNER)} klopte")
        message = tr("$meegaanSummary — stakes are now ×${board?.multiplier}.", "$meegaanSummary — de inzet is nu ×${board?.multiplier}.")
        botsKnockBack()
        return true
    }

    /** The partner bot's terug against the opponents' meegaan. False if it passes. */
    private fun partnerKnocksBack(): Boolean {
        if (!botTerug(PARTNER)) return false
        game.announceRedouble(PARTNER); refresh(); terugSeat = PARTNER
        meegaanSeat?.let { react(it, "😱", "😳", tr("Yikes 😱", "Aiai 😱"), tr("Brave!", "Da durft wel!")) }
        terugSummary = tr("${seatName(PARTNER)} knocked back", "${seatName(PARTNER)} klopte terug")
        message = tr("$terugSummary — stakes are now ×${board?.multiplier}.", "$terugSummary — de inzet is nu ×${board?.multiplier}.")
        return true
    }

    /** The declaring side is both bots (our side knocked against them):
     *  dealer first, then partner, first yes takes the terug. */
    private fun botsKnockBack() {
        val redoubler = listOf(dealer, (dealer + 2) % 4).firstOrNull { botTerug(it) } ?: return
        game.announceRedouble(redoubler); refresh(); terugSeat = redoubler; react(redoubler, "😤", "😈", tr("Knocked back! 😤", "Teruggeklopt! 😤"), tr("Bring it on!", "Kom maar op!"))
        terugSummary = tr("${seatName(redoubler)} knocked back", "${seatName(redoubler)} klopte terug")
        message = tr("$terugSummary — stakes are now ×${board?.multiplier}.", "$terugSummary — de inzet is nu ×${board?.multiplier}.")
    }

    private fun beginPlaying() {
        callHints = null; callHintPick = false
        game.startPlaying(); phase = TablePhase.PLAYING; refresh()
        message = if (board?.currentPlayer == HUMAN) tr("Your turn.", "Jouw beurt.") else tr("${seatName(board?.currentPlayer ?: 0)} is thinking…", "${seatName(board?.currentPlayer ?: 0)} denkt na…")
        persist()
        scheduleBotIfNeeded()
    }

    private fun scheduleBotIfNeeded() {
        val snapshot = board ?: return
        if (phase != TablePhase.PLAYING || snapshot.finished || snapshot.currentPlayer == HUMAN) return
        if (ruleBot != null) { startRuleBot(snapshot.currentPlayer); return }
        schedule(BOT_THINK_MS) { if (phase == TablePhase.PLAYING && !trickPaused) playOneBot() }
    }

    /**
     * The rule agent thinks on a worker thread with a copy of the game; the
     * card is played on the main thread after at least BOT_THINK_MS, and only
     * if nothing moved on meanwhile (turnToken: a new deal or the lobby
     * cancels it). Any failure falls back to the QGRU's card.
     */
    private fun startRuleBot(player: Int) {
        val model = ruleBot ?: return
        val token = turnToken
        val started = android.os.SystemClock.uptimeMillis()
        val copy = game.copy()
        val calls = RuleBot.Calls(dealer, meegaanSeat, terugSeat, declUs, declThem, trumpRuleKnown, setOf(HUMAN))
        worker.execute {
            val card = try { model.choose(copy, calls) } catch (error: Exception) { -1 } finally { copy.close() }
            handler.post {
                if (token != turnToken) return@post
                val wait = (BOT_THINK_MS - (android.os.SystemClock.uptimeMillis() - started)).coerceAtLeast(0L)
                schedule(wait) {
                    val now = board ?: return@schedule
                    if (token != turnToken || phase != TablePhase.PLAYING || trickPaused
                        || now.finished || now.currentPlayer != player) return@schedule
                    val legal = game.legalCards()
                    applyCardWithTiming(if (card in legal) card else qnetCard(player))
                }
            }
        }
    }

    private fun qnetCard(player: Int): Int =
        bot?.chooseCard(game.gruRaw(player), game.gruLength(player), game.gruGlobal(player),
            game.actionFeatures(player), game.legalCards()) ?: game.legalCards().first()

    private fun playOneBot() {
        val snapshot = board ?: return
        if (snapshot.finished || snapshot.currentPlayer == HUMAN || phase != TablePhase.PLAYING) return
        applyCardWithTiming(qnetCard(snapshot.currentPlayer))
    }

    private fun applyCardWithTiming(card: Int) {
        val before = board ?: return
        // what the player could have played instead (the engine's answer), so
        // the reactions can tell a forced loss from a gift
        if (before.trickCards.none { it >= 0 }) trickLegal.clear()
        trickLegal[before.currentPlayer] = game.legalCards().toSet()
        val completesTrick = before.trickCards.count { it >= 0 } == 3
        val completedCards = if (completesTrick) before.trickCards.filter { it >= 0 } + card else emptyList()
        val completedPlayers = if (completesTrick) before.trickPlayers.filter { it >= 0 } + before.currentPlayer else emptyList()
        // the trick's winner from its cards: on the last trick the engine
        // finishes the deal without making the winner the player to move
        val taker = if (completesTrick) completedPlayers[NativeGame.trickWinner(completedCards, before.trumpSuit)] else -1
        if (taker == HUMAN) tricksMine += 1
        if (taker >= 0 && teamOf(taker) == HUMAN_TEAM) manillesUs += completedCards.count { it % 8 == 7 }
        if (taker >= 0) {
            reactToTrick(completedCards, completedPlayers, taker, before)
            lastTrickCards = completedCards; lastTrickPlayers = completedPlayers; lastTrickWinner = taker
        }
        game.applyCard(card); refresh()
        // the card is in: a restart resumes after it (a finished deal is
        // scored on resume)
        persist()
        if (!completesTrick) { afterMove(); return }

        heldTrickCards = completedCards
        heldTrickPlayers = completedPlayers
        trickPaused = true
        legalCards = emptyList()
        val winningPlayer = taker
        message = if (winningPlayer == HUMAN) tr("You take the trick.", "Jij neemt de slag.") else tr("${seatName(winningPlayer)} takes the trick.", "${seatName(winningPlayer)} neemt de slag.")
        // the cards rest, then (with animations) sweep to the winner
        schedule(TRICK_HOLD_MS - COLLECT_MS) {
            if (animations) trickCollectTo = winningPlayer
            schedule(COLLECT_MS) {
                heldTrickCards = emptyList(); heldTrickPlayers = emptyList(); trickPaused = false; trickCollectTo = -1
                if (board?.finished == true) completeHand() else { refresh(); afterMove() }
            }
        }
    }

    private fun afterMove() {
        if (board?.finished == true) { completeHand(); return }
        message = if (board?.currentPlayer == HUMAN) tr("Your turn.", "Jouw beurt.") else tr("${seatName(board?.currentPlayer ?: 0)} is thinking…", "${seatName(board?.currentPlayer ?: 0)} denkt na…")
        scheduleBotIfNeeded()
    }

    private fun completeHand() {
        val usDelta = game.handScore(HUMAN_TEAM)
        val themDelta = game.handScore(1)
        usScore += usDelta; themScore += themDelta; refresh()
        phase = if (matchMode && (usScore >= WINNING_SCORE || themScore >= WINNING_SCORE)) TablePhase.MATCH_OVER else TablePhase.HAND_OVER
        val b = board!!
        val mee = meegaanSeat; val ter = terugSeat
        stats = statsStore.recordHand(HandRecord(
            pointsUs = b.team0Points, pointsThem = b.team1Points, scoreUs = usDelta, scoreThem = themDelta,
            tricksUs = b.team0Tricks, tricksMine = tricksMine, manillesUs = manillesUs,
            weDeclared = teamOf(dealer) == HUMAN_TEAM, iDeclared = dealer == HUMAN, mull = b.trumpSuit < 0,
            multiplier = b.multiplier,
            meegaanUs = mee != null && teamOf(mee) == HUMAN_TEAM, meegaanThem = mee != null && teamOf(mee) != HUMAN_TEAM,
            meegaanMine = mee == HUMAN,
            terugUs = ter != null && teamOf(ter) == HUMAN_TEAM, terugThem = ter != null && teamOf(ter) != HUMAN_TEAM,
            terugMine = ter == HUMAN,
        ))
        when {
            !matchMode -> stats = statsStore.recordResult(false, b.team0Points, b.team1Points)
            phase == TablePhase.MATCH_OVER -> stats = statsStore.recordResult(true, usScore, themScore)
        }
        val usWon = board!!.team0Points > board!!.team1Points
        val themWon = board!!.team1Points > board!!.team0Points
        val b0 = board!!
        for (seat in 1..3) {
            val mine = if (teamOf(seat) == HUMAN_TEAM) usWon else themWon
            val theirs = if (teamOf(seat) == HUMAN_TEAM) themWon else usWon
            val myPts = if (teamOf(seat) == HUMAN_TEAM) b0.team0Points else b0.team1Points
            val myTricks = if (teamOf(seat) == HUMAN_TEAM) b0.team0Tricks else b0.team1Tricks
            when {
                phase == TablePhase.MATCH_OVER && mine -> react(seat, "🏆", tr("🏆 Champions!", "🏆 Kampioen!"), tr("We won! 🎉", "Gewonnen! 🎉"))
                !usWon && !themWon -> react(seat, "🤝", tr("A draw! 🤝", "Gelijk! 🤝"))
                mine && myTricks == 8 -> react(seat, "👑", tr("Every trick! 👑", "Alle slagen! 👑"))
                mine && myPts >= 45 -> react(seat, "🥳", tr("Too strong for you 💪", "Te sterk voor jullie 💪"), tr("Next! 🥳", "Volgende! 🥳"))
                mine -> react(seat, "🎉", "😄", "🙌", tr("We won! 🎉", "Gewonnen! 🎉"))
                theirs && ((mee != null && teamOf(mee) == teamOf(seat)) || (ter != null && teamOf(ter) == teamOf(seat))) ->
                    react(seat, "😡", "😤", tr("So much for knocking 😤", "Dat kloppen was niks 😤"), tr("Why did we knock?! 😡", "Waarom klopten we?! 😡"))
                theirs && myPts <= 15 -> react(seat, "😠", "🤬", tr("That was terrible 😠", "Dat was slecht 😠"), tr("Embarrassing... 😤", "Beschamend... 😤"))
                theirs && seat == PARTNER -> react(seat, "😠", "😤", "🤦", tr("Come on, partner! 😤", "Allez partner! 😤"), tr("Pay attention, partner 😠", "Let een beetje op, partner 😠"), tr("Better luck next time 😅", "Volgende keer beter 😅"))
                theirs -> react(seat, "😞", "🤦", "😅", "😤", tr("Better luck next time 😅", "Volgende keer beter 😅"), tr("The cards were against us", "De kaarten zaten tegen"))
            }
        }
        val winner = when {
            board!!.team0Points > board!!.team1Points -> tr("You & ${seatName(2)} win this hand", "Jij & ${seatName(2)} winnen dit spel")
            board!!.team1Points > board!!.team0Points -> tr("${seatName(1)} & ${seatName(3)} win this hand", "${seatName(1)} & ${seatName(3)} winnen dit spel")
            else -> tr("This hand is a draw", "Dit spel is gelijk")
        }
        message = tr("$winner · hand ${board!!.team0Points}–${board!!.team1Points} · score +$usDelta / +$themDelta · match $usScore–$themScore", "$winner · spel ${board!!.team0Points}–${board!!.team1Points} · score +$usDelta / +$themDelta · stand $usScore–$themScore")
        // a finished quick game or match has nothing left to resume
        if (!matchMode || phase == TablePhase.MATCH_OVER) { saves.clear(); savedGame = null } else persist()
    }

    private fun declarationIndex(): Int = board?.trumpSuit?.let { if (it < 0) 4 else it } ?: 4

    private fun botMeegaan(seat: Int): Boolean {
        val rel = (seat - dealer + 4) % 4
        val (us, them) = if (teamOf(seat) == HUMAN_TEAM) usScore to themScore else themScore to usScore
        return announceNet?.meegaan(game.handMask(seat), declarationIndex(), rel, us, them)
            ?: botShouldCall(seat)
    }

    private fun botTerug(seat: Int): Boolean {
        val rel = (seat - dealer + 4) % 4
        val (us, them) = if (teamOf(seat) == HUMAN_TEAM) usScore to themScore else themScore to usScore
        return announceNet?.terug(game.handMask(seat), declarationIndex(), rel, us, them)
            ?: botShouldCall(seat)
    }

    // Offline fallback only, for when the announcement net fails to load: a
    // crude hand-strength threshold with no notion of trump or the score.
    private fun botShouldCall(player: Int): Boolean {
        val ranks = intArrayOf(0, 0, 1, 3, 4, 5, 6, 8)
        return (0 until 32).filter { game.handMask(player) and (1L shl it) != 0L }.sumOf { ranks[it % 8] } >= 29
    }

    private fun refresh() {
        board = game.state(); handMask = game.handMask(HUMAN)
        legalCards = if (phase == TablePhase.PLAYING && board?.currentPlayer == HUMAN) game.legalCards().toList() else emptyList()
        handCounts = List(4) { java.lang.Long.bitCount(game.handMask(it)) }
    }

    private fun cancelBotTurn() { turnToken += 1; pending = null; handler.removeCallbacksAndMessages(null) }
    override fun close() {
        cancelBotTurn(); worker.shutdown()
        // a move in flight still uses the nets: let it finish before closing them
        runCatching { worker.awaitTermination(2, java.util.concurrent.TimeUnit.SECONDS) }
        ruleBot?.close(); bot?.close(); declarationNet?.close(); announceNet?.close(); game.close()
    }
    /** Match points the finished deal gave each team (us, them). */
    fun handScores(): Pair<Int, Int> = game.handScore(HUMAN_TEAM) to game.handScore(1)
    val meegaanName: String? get() = meegaanSeat?.let { seatName(it) }
    val terugName: String? get() = terugSeat?.let { seatName(it) }

    fun seatName(player: Int) = if (player == HUMAN) tr("You", "Jij") else botNames[player - 1]
    fun trumpLabel(suit: Int) = if (suit < 0) tr("mull", "mull") else (if (Lang.dutch) arrayOf("ruiten", "harten", "klaveren", "schoppen") else arrayOf("diamonds", "hearts", "clubs", "spades"))[suit]
    fun trumpSymbol(suit: Int) = if (suit < 0) tr("∅ (mull)", "∅ (mull)") else arrayOf("♦", "♥", "♣", "♠")[suit]
    fun playerHandMask(player: Int): Long = game.handMask(player)

    private companion object {
        const val HUMAN = 0; const val PARTNER = 2; const val HUMAN_TEAM = 0; const val WINNING_SCORE = 101
        const val BOT_THINK_MS = 900L
        const val TRICK_HOLD_MS = 1400L
        const val COLLECT_MS = 380L
        const val REACTION_MS = 2600L
        // set per build (app/build.gradle.kts, manille.botNames)
        val NAMES = BuildConfig.BOT_NAMES.split(",").map { it.trim() }.filter { it.isNotEmpty() }
        // hint softmax temperature, in final team points (gui/hints.py TAU_PTS)
        const val TAU_PTS = 1.0
        fun teamOf(player: Int) = player % 2
    }
}
