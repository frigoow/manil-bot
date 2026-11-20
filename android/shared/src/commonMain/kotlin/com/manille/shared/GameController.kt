package com.manille.shared

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlin.math.exp
import kotlin.random.Random
import kotlin.time.TimeSource

/**
 * The table on screen. The game itself is played by a session (the
 * referee: [LocalSession] offline); this class turns what it reports into
 * what the screens show -- the table, the held trick, the messages, the
 * bots' reactions -- and keeps the player's profiles, statistics, settings,
 * hints and saved game.
 *
 * Events are handled twice: at once for the books (statistics, the saved
 * game), then in order for the screen, where a finished trick stays on the
 * table for a moment while later events wait their turn.
 */
class GameController(private val platform: Platform) : AutoCloseable {
    private val storage = platform.storage
    private val profileStore = ProfileStore(storage)
    init { Lang.dutch = profileStore.language == "nl" }
    private var saves = SaveStore(storage, profileStore.activeId)
    private var statsStore = newStatsStore(profileStore.activeId)
    private fun newStatsStore(id: String) = StatsStore(storage, id) { platform.epochMillis() }
    // this hand's tricks won by the human and 10s captured by the human's team
    private var tricksMine = 0
    private var manillesUs = 0
    private val botResult = runCatching { platform.qModel(ModelNames.Q_TRUMP) }
    private val bot = botResult.getOrNull()
    private val declarationResult = runCatching { platform.declarationModel() }
    private val declarationNet = declarationResult.getOrNull()
    private val announceResult = runCatching { platform.announceModel() }
    private val announceNet = announceResult.getOrNull()
    // The rule agent (desktop GUI "Rule-based, phone config"); the plain QGRU
    // above stays as the fallback when its assets are missing.
    private val ruleResult = runCatching {
        RuleBot(platform, bot ?: error("QGRU model missing"), announceNet)
    }
    private val ruleBot = ruleResult.getOrNull().also {
        it?.netPlies = profileStore.netPlies.takeIf { p -> p in RuleBot.PLY_CHOICES } ?: RuleBot.WHOLE_DEAL
        ruleResult.exceptionOrNull()?.let { e ->
            platform.log("rule agent unavailable, bots play the QGRU: ${e.message}")
        }
    }
    /** Which bot plays the other seats (lobby / header label). */
    val botLabel: String get() = if (ruleBot != null) tr("Rule agent", "Regelbot") else "GRU Blueprint"
    val botDetail: String get() = if (ruleBot != null)
        tr("Card counting + play-out, belief & Q nets · offline", "Kaarten tellen + uitspelen, belief- & Q-netten · offline")
    else ruleResult.exceptionOrNull()?.message?.let { tr("Rule agent unavailable: $it", "Regelbot niet beschikbaar: $it") }
        ?: tr("Latest on-device recurrent model · offline", "Recentste recurrente model op het toestel · offline")

    // UI-side state changes run on the main thread. Bots and hints think on
    // the worker, on a copy of the game; their results come back to the main
    // thread. The worker's lock is held for the whole of a job, so close()
    // can wait for the job in flight before it frees the nets.
    private val worker = Worker()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private fun onWorker(block: () -> Unit) { scope.launch(worker.dispatcher) { worker.lock.withLock { block() } } }
    private fun onMain(block: () -> Unit) { scope.launch(Dispatchers.Main) { block() } }
    /** Runs [block] on the main thread after [ms]; not cancelled by a new turn (reactions). */
    private fun later(ms: Long, block: () -> Unit) { scope.launch { delay(ms); block() } }

    /** The offline table. A networked one will come in through [TableSession] as well. */
    private val session = LocalSession(platform, Bots(ruleBot, bot, declarationNet, announceNet), worker).also { s ->
        s.listener = ::onEvent
        s.onBotThought = ::recordBotMs
    }
    private val table: TableSession get() = session

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
    private data class CachedHint(val key: Int, val plies: Int, val hints: Map<Int, Float>, val pick: Int?)
    private var cachedHint: CachedHint? = null
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
    /** Play-out net plies for the bots (RuleBot.PLY_CHOICES); 0 = rule play-out alone. */
    var netPlies by mutableStateOf(profileStore.netPlies.takeIf { it in RuleBot.PLY_CHOICES } ?: RuleBot.WHOLE_DEAL); private set
    /** The bots' thinking time over their recent real decisions (ms), or null before any. */
    var botMsMean by mutableStateOf<Long?>(null); private set
    var botMsMax by mutableStateOf<Long?>(null); private set
    private val botMs = ArrayDeque<Long>()
    var theme by mutableStateOf(themeById(profileStore.theme)); private set
    /** The previous completed trick: cards, players, winner (for "Last trick"). */
    var lastTrickCards by mutableStateOf<List<Int>>(emptyList()); private set
    var lastTrickPlayers by mutableStateOf<List<Int>>(emptyList()); private set
    var lastTrickWinner by mutableStateOf(-1); private set
    /** Emoji reaction currently shown per bot seat. */
    var reactions by mutableStateOf<Map<Int, String>>(emptyMap()); private set
    /** Counts the tricks made of a 7, 8, 9 and 10 (any suits, anyone's): each one makes the café pint toast ("Schol!"). */
    var cheers by mutableStateOf(0); private set
    /** One-line summary of the saved game, or null when there is none. */
    var savedGame by mutableStateOf(saveSummary()); private set

    /** Bot names for seats 1..3 (West, North, East), fresh for every game. */
    var botNames by mutableStateOf(AppConfig.botNames.take(3)); private set

    // what is on screen of the table (the session may be a held trick ahead)
    private var shown = TableView.LOBBY
    /** Match points of the last scored deal (us, them). */
    private var lastDeal = 0 to 0

    private fun pickNames() {
        botNames = AppConfig.botNames.shuffled().take(3)
    }

    fun start(match: Boolean) {
        pickNames()
        resetScreen()
        session.start(match)
    }

    val isMatch: Boolean get() = shown.match
    /** Seats that called this deal (for the table's speech bubbles). */
    val meegaanBy: Int? get() = shown.meegaanSeat
    val terugBy: Int? get() = shown.terugSeat
    val dealerSeat: Int get() = shown.dealer

    /** Start the current kind of game over (the save is replaced). */
    fun restart() { saves.clear(); start(shown.match) }

    // ---- profiles ---------------------------------------------------------
    private fun reloadProfile() {
        saves = SaveStore(storage, profileStore.activeId)
        statsStore = newStatsStore(profileStore.activeId)
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
    val playoutNetAvailable: Boolean get() = ruleBot?.hasFastNet == true
    fun setPlayoutPlies(plies: Int) {
        if (plies !in RuleBot.PLY_CHOICES) return
        profileStore.setNetPlies(plies); netPlies = plies; ruleBot?.netPlies = plies
        botMs.clear(); botMsMean = null; botMsMax = null
    }
    private fun recordBotMs(ms: Long) {
        botMs.addLast(ms); while (botMs.size > 24) botMs.removeFirst()
        botMsMean = botMs.sum() / botMs.size; botMsMax = botMs.max()
    }
    fun enableSmileys(on: Boolean) { profileStore.setSmileys(on); smileys = on; if (!on) reactions = emptyMap() }

    // ---- bot reactions ------------------------------------------------------
    private fun react(seat: Int, vararg choices: String) {
        if (!smileys || seat == HUMAN || seat !in 0..3) return
        val emoji = choices[Random.nextInt(choices.size)]
        reactions = reactions + (seat to emoji)
        later(REACTION_MS) { if (reactions[seat] == emoji) reactions = reactions - seat }
    }

    /** What the human can send: smileys, then short lines (text to mood). */
    val playerSmileys = listOf("😀", "😏", "😎", "😂", "😅", "😩", "🙈", "🗿", "👍", "👏", "🎉", "🤑", "😤", "🙄", "🔥", "😈", "🪤", "🤝")
    val playerLines: List<Pair<String, String>> get() = listOf(
        tr("Well played!", "Goe gespeeld!") to "cheer", "Haha 😏" to "taunt", tr("Come on, partner!", "Komaan partner!") to "cheer",
        tr("Oops...", "Oei...") to "sad", tr("Bad luck!", "Pech!") to "sad", tr("Too strong 💪", "Te sterk 💪") to "taunt", "GG" to "cheer")

    /** The human sends a smiley or a short line; the bots may answer. */
    fun sendReaction(text: String, lineMood: String? = null) {
        reactions = reactions + (HUMAN to text)
        later(REACTION_MS) { if (reactions[HUMAN] == text) reactions = reactions - HUMAN }
        if (!smileys) return
        val mood = lineMood ?: when (text) {
            "😏", "😎", "😂", "🤑" -> "taunt"; "😩", "😅", "🙈" -> "sad"; "👍", "👏", "🎉" -> "cheer"
            "🪤" -> "trap"; "🤝" -> "respect"; else -> null
        }
        val taunt = mood == "taunt"; val sad = mood == "sad"; val cheer = mood == "cheer"
        later(900L) {
            when {
                mood == "trap" -> react(if (Random.nextBoolean()) 1 else 3, "😤", "🙄", tr("Sneaky! 😤", "Vuile streek! 😤"), tr("Not falling for that", "Daar trap ik niet in"))
                mood == "respect" -> react(Random.nextInt(1, 4), "🤝", tr("Respect 🤝", "Respect 🤝"), "GG 🤝")
                taunt -> react(if (Random.nextBoolean()) 1 else 3, "😒", "🙄", "🗿", tr("Just wait...", "Wacht maar..."), tr("Easy, easy 😤", "Rustig, rustig 😤"), tr("Lucky!", "Geluk!"))
                sad -> react(2, "💪", tr("We'll be fine 💪", "Komt goed 💪"), tr("The next trick is ours!", "Volgende slag is voor ons!"))
                cheer -> if (Random.nextBoolean()) react(2, "😄", "👍", tr("Stronger together! 🙌", "Samen sterk! 🙌")) else react(1, "🙄", tr("Yeah yeah...", "Ja ja..."))
                Random.nextInt(3) == 0 -> react(Random.nextInt(1, 4), "😄", "🙂", "😉")
            }
        }
    }

    /** ", Name" when addressing a bot, nothing for the human. */
    private fun at(p: Int) = if (p == HUMAN) "" else ", ${seatName(p)}"

    /** Tricks since the bots last reacted (keeps the table from chattering). */
    private var quietTricks = 1

    /** Each seat's legal cards when it played into the current trick (on screen). */
    private val trickLegal = mutableMapOf<Int, Set<Int>>()
    /** Cards of this deal's finished tricks (on screen). */
    private val playedCards = mutableSetOf<Int>()

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
        fun leaderAfter(k: Int) = players[platform.trickWinner(cards.subList(0, k + 1), trump)]
        // the cards of the earlier tricks
        val earlier: Set<Int> = playedCards

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
                alt / 8 != trump && players[platform.trickWinner(cards.subList(0, tIdx) + alt, trump)] == partner }
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
        session.stop(); resetScreen()
        phase = TablePhase.LOBBY; message = tr("Choose a table to start.", "Kies een tafel om te beginnen.")
        savedGame = saveSummary()
    }

    /**
     * Debug timing (Android: adb shell am start -n
     * com.manille.androidapp/.MainActivity --ez bench true): the rule agent
     * plays a few whole deals in all four seats on the worker; per-move stage
     * timings go to the platform log.
     */
    fun benchmark(deals: Int = 4) {
        val model = ruleBot ?: run { platform.log("bench: no rule agent"); return }
        onWorker {
            val total = mutableListOf<Long>()
            val stages = LongArray(5)
            var doubledMoves = 0
            for (d in 0 until deals) {
                val g = platform.newEngine()
                g.newDeal(d % 4, 9000L + d, if (d % 4 == 3) -1 else d % 4)
                if (d % 2 == 1) g.announceDouble((d % 4 + 1) % 4)
                g.startPlaying()
                val calls = RuleBot.Calls(d % 4, if (d % 2 == 1) (d % 4 + 1) % 4 else null, null,
                    0, 0, false, emptySet())
                while (!g.state().finished) {
                    val t0 = TimeSource.Monotonic.markNow()
                    val card = model.choose(g, calls)
                    val ms = t0.elapsedNow().inWholeMilliseconds
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
            platform.log(
                "bench: moves=$n mean=${total.sum() / n}ms median=${total[n / 2]}ms p90=${total[(n * 9) / 10]}ms " +
                "max=${total.last()}ms | per move: sample=${stages[0] / n} belief=${stages[1] / n} " +
                "q=${stages[2] / n} doubles=${stages[3] / n} (over all moves; $doubledMoves doubled) " +
                "playout=${stages[4] / n}")
        }
    }

    // ---- the session's events: the books ---------------------------------------

    /** Every event, the moment the session reports it: what must not wait for the screen. */
    private fun onEvent(event: TableEvent, view: TableView) {
        when (event) {
            TableEvent.DealStarted -> {
                tricksMine = 0; manillesUs = 0
                declarationSummary = ""; meegaanSummary = null; terugSummary = null
            }
            is TableEvent.Declared -> {
                // the mull nets only while a mull deal needs them (RuleBot)
                ruleBot?.let { bot -> onWorker { if (event.trump < 0) bot.prepareMull() else bot.releaseMull() } }
                declarationSummary =
                    if (event.seat == HUMAN) tr("You declared ${trumpSymbol(event.trump)}", "Jij koos ${trumpSymbol(event.trump)}")
                    else tr("${seatName(event.seat)} declared ${trumpSymbol(event.trump)}", "${seatName(event.seat)} koos ${trumpSymbol(event.trump)}")
            }
            is TableEvent.Knocked -> meegaanSummary =
                if (event.seat == HUMAN) tr("You knocked", "Jij klopte")
                else tr("${seatName(event.seat)} knocked", "${seatName(event.seat)} klopte")
            is TableEvent.KnockedBack -> terugSummary =
                if (event.seat == HUMAN) tr("You knocked back", "Jij klopte terug")
                else tr("${seatName(event.seat)} knocked back", "${seatName(event.seat)} klopte terug")
            is TableEvent.CardPlayed -> event.trick?.let { t ->
                if (t.taker == HUMAN) tricksMine += 1
                if (teamOf(t.taker) == HUMAN_TEAM) manillesUs += t.cards.count { it % 8 == 7 }
            }
            is TableEvent.DealScored -> { scoreDeal(event, view); show(event, view); return }
            TableEvent.YourCall, TableEvent.PlayStarted -> {}
        }
        // saved where a restart can carry on: the player is asked, or cards are on the table
        if (event is TableEvent.YourCall || event is TableEvent.PlayStarted || event is TableEvent.CardPlayed) persist()
        show(event, view)
    }

    private fun scoreDeal(event: TableEvent.DealScored, view: TableView) {
        lastDeal = event.usDelta to event.themDelta
        val b = view.board ?: return
        val mee = view.meegaanSeat; val ter = view.terugSeat
        stats = statsStore.recordHand(HandRecord(
            pointsUs = b.team0Points, pointsThem = b.team1Points, scoreUs = event.usDelta, scoreThem = event.themDelta,
            tricksUs = b.team0Tricks, tricksMine = tricksMine, manillesUs = manillesUs,
            weDeclared = teamOf(view.dealer) == HUMAN_TEAM, iDeclared = view.dealer == HUMAN, mull = b.trumpSuit < 0,
            multiplier = b.multiplier,
            meegaanUs = mee != null && teamOf(mee) == HUMAN_TEAM, meegaanThem = mee != null && teamOf(mee) != HUMAN_TEAM,
            meegaanMine = mee == HUMAN,
            terugUs = ter != null && teamOf(ter) == HUMAN_TEAM, terugThem = ter != null && teamOf(ter) != HUMAN_TEAM,
            terugMine = ter == HUMAN,
        ))
        when {
            !view.match -> stats = statsStore.recordResult(false, b.team0Points, b.team1Points)
            view.phase == TablePhase.MATCH_OVER -> stats = statsStore.recordResult(true, view.usScore, view.themScore)
        }
        // a finished quick game or match has nothing left to resume
        if (!view.match || view.phase == TablePhase.MATCH_OVER) { saves.clear(); savedGame = null } else persist()
    }

    // ---- the session's events: the screen ----------------------------------------

    // Events wait here for the screen, in order; a finished trick holds the
    // line while it rests on the table. A new game starts a new line.
    private var screenQueue = Channel<Pair<TableEvent, TableView>>(Channel.UNLIMITED)
    private var screenJob: Job? = null

    private fun show(event: TableEvent, view: TableView) { screenQueue.trySend(event to view) }

    /** Drops whatever was still waiting for the screen and clears the table. */
    private fun resetScreen() {
        screenJob?.cancel(); screenQueue.close()
        val queue = Channel<Pair<TableEvent, TableView>>(Channel.UNLIMITED)
        screenQueue = queue
        screenJob = scope.launch { for ((event, view) in queue) present(event, view) }
        hints = emptyMap(); hintPick = null; cachedHint = null; declarationHints = emptyMap(); declarationHintPick = null
        callHints = null; callHintPick = false
        heldTrickCards = emptyList(); heldTrickPlayers = emptyList(); trickPaused = false; trickCollectTo = -1
        lastTrickCards = emptyList(); lastTrickPlayers = emptyList(); lastTrickWinner = -1
        trickLegal.clear(); playedCards.clear()
        shown = TableView.LOBBY.copy(match = shown.match)
    }

    /** Puts [view] on the table (the player's cards stay unplayable while a trick is held). */
    private fun display(view: TableView) {
        shown = view
        board = view.board; handMask = view.hand; handCounts = view.handCounts
        legalCards = if (trickPaused) emptyList() else view.legalCards
        phase = view.phase; usScore = view.usScore; themScore = view.themScore; handNumber = view.handNumber
    }

    private fun turnMessage(view: TableView): String {
        val seat = view.board?.currentPlayer ?: 0
        return if (seat == HUMAN) tr("Your turn.", "Jouw beurt.") else tr("${seatName(seat)} is thinking…", "${seatName(seat)} denkt na…")
    }

    private fun stakes(view: TableView) = view.board?.multiplier

    private suspend fun present(event: TableEvent, view: TableView) {
        when (event) {
            TableEvent.DealStarted -> {
                hints = emptyMap(); hintPick = null; cachedHint = null; declarationHints = emptyMap(); declarationHintPick = null
                callHints = null; callHintPick = false
                lastTrickCards = emptyList(); lastTrickPlayers = emptyList(); lastTrickWinner = -1
                heldTrickCards = emptyList(); heldTrickPlayers = emptyList(); trickPaused = false; trickCollectTo = -1
                trickLegal.clear(); playedCards.clear()
                display(view)
            }
            is TableEvent.Declared -> {
                display(view)
                declarationHints = emptyMap(); declarationHintPick = null
                if (event.seat != HUMAN && event.trump < 0) react(event.seat, "😎", tr("Mull! 😎", "Mull! 😎"), tr("Mull, daring!", "Mull, durven!"))
                message = "$declarationSummary."
            }
            is TableEvent.Knocked -> {
                display(view)
                if (event.seat == HUMAN) {
                    react(view.dealer, "😬", "😮", tr("You dare? 😬", "Durf je? 😬"), tr("Uh oh...", "Oei oei..."))
                    message = tr("You knock — stakes are now ×${stakes(view)}.", "Jij klopt — de inzet is nu ×${stakes(view)}.")
                } else {
                    react(event.seat, "😈", "😏", tr("I knock! 😈", "Ik klop! 😈"), tr("Knock knock! 😈", "Klop klop! 😈"))
                    message = tr("$meegaanSummary — stakes are now ×${stakes(view)}.", "$meegaanSummary — de inzet is nu ×${stakes(view)}.")
                }
            }
            is TableEvent.KnockedBack -> {
                display(view)
                if (teamOf(event.seat) == HUMAN_TEAM) view.meegaanSeat?.let { react(it, "😱", "😳", tr("Yikes 😱", "Aiai 😱"), tr("Brave!", "Da durft wel!")) }
                else react(event.seat, "😤", "😈", tr("Knocked back! 😤", "Teruggeklopt! 😤"), tr("Bring it on!", "Kom maar op!"))
                message = tr("$terugSummary — stakes are now ×${stakes(view)}.", "$terugSummary — de inzet is nu ×${stakes(view)}.")
            }
            TableEvent.YourCall -> {
                display(view)
                callHints = null; callHintPick = false
                message = ""
            }
            TableEvent.PlayStarted -> {
                display(view)
                callHints = null; callHintPick = false
                // A bot's knock or knock-back is decided in the same step as the
                // human's answer: keep it on screen with the stake it set, or a ×8
                // deal looks like a miscount at the end.
                val calls = listOfNotNull(meegaanSummary, terugSummary)
                val turn = turnMessage(view)
                message = if (calls.isEmpty()) turn
                    else tr("${calls.joinToString(", ")} — stakes ×${stakes(view)}. $turn", "${calls.joinToString(", ")} — inzet ×${stakes(view)}. $turn")
            }
            is TableEvent.CardPlayed -> presentCard(event, view)
            is TableEvent.DealScored -> presentScore(event, view)
        }
    }

    private suspend fun presentCard(event: TableEvent.CardPlayed, view: TableView) {
        val trick = event.trick
        if (trick == null) {
            if (view.board?.trickCards?.count { it >= 0 } == 1) trickLegal.clear()
            event.legal?.let { trickLegal[event.seat] = it }
            display(view)
            if (view.board?.finished != true) message = turnMessage(view)
            return
        }
        event.legal?.let { trickLegal[event.seat] = it }
        if (isScholTrick(trick.cards)) cheers++
        reactToTrick(trick.cards, trick.players, trick.taker, trick.before)
        playedCards += trick.cards
        lastTrickCards = trick.cards; lastTrickPlayers = trick.players; lastTrickWinner = trick.taker
        heldTrickCards = trick.cards; heldTrickPlayers = trick.players; trickPaused = true
        display(view)
        message = if (trick.taker == HUMAN) tr("You take the trick.", "Jij neemt de slag.") else tr("${seatName(trick.taker)} takes the trick.", "${seatName(trick.taker)} neemt de slag.")
        // the cards rest, then (with animations) sweep to the winner
        delay(TRICK_HOLD_MS - COLLECT_MS)
        if (animations) trickCollectTo = trick.taker
        delay(COLLECT_MS)
        heldTrickCards = emptyList(); heldTrickPlayers = emptyList(); trickPaused = false; trickCollectTo = -1
        display(view)
        if (view.board?.finished != true) message = turnMessage(view)
    }

    private fun presentScore(event: TableEvent.DealScored, view: TableView) {
        display(view)
        val b0 = view.board ?: return
        val usWon = b0.team0Points > b0.team1Points
        val themWon = b0.team1Points > b0.team0Points
        val mee = view.meegaanSeat; val ter = view.terugSeat
        for (seat in 1..3) {
            val mine = if (teamOf(seat) == HUMAN_TEAM) usWon else themWon
            val theirs = if (teamOf(seat) == HUMAN_TEAM) themWon else usWon
            val myPts = if (teamOf(seat) == HUMAN_TEAM) b0.team0Points else b0.team1Points
            val myTricks = if (teamOf(seat) == HUMAN_TEAM) b0.team0Tricks else b0.team1Tricks
            when {
                view.phase == TablePhase.MATCH_OVER && mine -> react(seat, "🏆", tr("🏆 Champions!", "🏆 Kampioen!"), tr("We won! 🎉", "Gewonnen! 🎉"))
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
            usWon -> tr("You & ${seatName(2)} win this hand", "Jij & ${seatName(2)} winnen dit spel")
            themWon -> tr("${seatName(1)} & ${seatName(3)} win this hand", "${seatName(1)} & ${seatName(3)} winnen dit spel")
            else -> tr("This hand is a draw", "Dit spel is gelijk")
        }
        message = tr("$winner · hand ${b0.team0Points}–${b0.team1Points} · score +${event.usDelta} / +${event.themDelta} · match ${view.usScore}–${view.themScore}", "$winner · spel ${b0.team0Points}–${b0.team1Points} · score +${event.usDelta} / +${event.themDelta} · stand ${view.usScore}–${view.themScore}")
    }

    // ---- saved game -----------------------------------------------------
    /** The session's state plus what only the screen knows (names, summaries, this hand's books). */
    private fun persist() {
        val own = session.save() ?: return
        val save = JsonObject(own + buildJsonObject {
            put("names", JsonArray(botNames.map { JsonPrimitive(it) }))
            put("message", message)
            put("declSummary", declarationSummary)
            put("tricksMine", tricksMine); put("manilles", manillesUs)
            put("meeSummary", meegaanSummary ?: ""); put("terSummary", terugSummary ?: "")
        })
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
            resetScreen()
            session.restore(j)
            j.optArray("names")?.let { a -> if (a.size == 3) botNames = List(3) { a[it].jsonPrimitive.content } }
            declarationSummary = j.optString("declSummary")
            tricksMine = j.optInt("tricksMine"); manillesUs = j.optInt("manilles")
            meegaanSummary = j.optString("meeSummary").ifEmpty { null }
            terugSummary = j.optString("terSummary").ifEmpty { null }
            // the cards of the tricks already played: in no hand and not on the table
            val view = session.view
            val inHands = (0 until 4).fold(0L) { m, q -> m or (table.openHand(q) ?: 0L) }
            val onTable = view.board?.trickCards.orEmpty().filter { it >= 0 }.toSet()
            if ((0 until 4).all { table.openHand(it) != null })
                playedCards += (0 until 32).filter { (inHands shr it) and 1L == 0L && it !in onTable }
            display(view)
            message = if (view.phase == TablePhase.PLAYING && view.board?.finished != true) turnMessage(view) else j.optString("message")
            session.resume()
            true
        } catch (error: Exception) {
            session.stop(); resetScreen()
            saves.clear(); savedGame = null
            message = tr("The saved game could not be restored.", "Het bewaarde spel kon niet hersteld worden.")
            phase = TablePhase.LOBBY
            false
        }
    }

    // ---- the player's moves ---------------------------------------------

    /** The next deal, after a hand is over. */
    fun newDeal() { session.newDeal() }

    fun declare(trump: Int) {
        if (phase != TablePhase.DECLARATION) return
        table.declare(trump)
    }

    fun answerMeegaan(accept: Boolean) {
        if (phase != TablePhase.MEEGAAN) return
        callHints = null; callHintPick = false
        table.answerMeegaan(accept)
    }

    fun answerTerug(accept: Boolean) {
        if (phase != TablePhase.TERUG) return
        callHints = null; callHintPick = false
        table.answerTerug(accept)
    }

    fun playHuman(card: Int) {
        val snapshot = board ?: return
        if (phase != TablePhase.PLAYING || snapshot.currentPlayer != HUMAN || card !in legalCards) {
            message = tr("That card is not legal now.", "Die kaart mag je nu niet spelen."); return
        }
        hints = emptyMap(); hintPick = null
        // no second card while the session's answer is on its way to the screen
        legalCards = emptyList()
        if (!table.play(card)) { legalCards = shown.legalCards; message = tr("That card is not legal now.", "Die kaart mag je nu niet spelen.") }
    }

    // ---- hints ------------------------------------------------------------

    fun toggleHints() {
        val snapshot = board ?: return
        if (phase != TablePhase.PLAYING || snapshot.currentPlayer != HUMAN || trickPaused) return
        if (hints.isNotEmpty()) { hints = emptyMap(); hintPick = null; return }
        val position = table.hintPosition() ?: return
        val rule = ruleBot
        val plies = rule?.netPlies ?: -1
        val cached = cachedHint
        if (cached != null && cached.key == position.key && cached.plies == plies) {
            position.engine.close()
            hints = cached.hints
            hintPick = cached.pick
            return
        }
        countHint()
        if (rule != null) { ruleHints(rule, position); return }
        val model = bot ?: run { position.engine.close(); return }
        val game = position.engine
        hints = try {
            val scores = model.scoreCards(game.gruRaw(HUMAN), game.gruLength(HUMAN), game.gruGlobal(HUMAN),
                game.actionFeatures(HUMAN), game.legalCards())
            val top = scores.maxOf { it.value }
            // Desktop gui/hints.py de-normalizes Q to final team points:
            // points = q * 30 + 30, then softmaxes at TAU_PTS = 1.
            val weights = scores.map { exp((it.value - top) * 30.0).toFloat() }
            val total = weights.sum().coerceAtLeast(.0001f)
            hintPick = scores.maxBy { it.value }.card
            val computed = scores.indices.associate { scores[it].card to (weights[it] / total * 100f) }
            cachedHint = CachedHint(position.key, plies, computed, hintPick)
            computed
        } finally { game.close() }
    }

    /**
     * The rule agent's own analysis of the human's position, as the desktop
     * "Rule-based, phone config" hint shows it (gui/hints.py): its pick, and
     * a softmax at 1 point (TAU_PTS) over each card's play-out value. Runs on
     * the worker; dropped if the position changed meanwhile.
     */
    private fun ruleHints(model: RuleBot, position: HintPosition) {
        val copy = position.engine
        val plies = model.netPlies
        message = tr("Thinking about a hint…", "Even nadenken over een hint…")
        onWorker {
            val values = DoubleArray(32)
            val pick = try { model.analyse(copy, position.calls, values) } catch (error: Exception) { -1 } finally { copy.close() }
            onMain {
                if (phase != TablePhase.PLAYING || board?.currentPlayer != HUMAN || session.positionKey() != position.key) return@onMain
                val legal = shown.legalCards
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
                cachedHint = CachedHint(position.key, plies, hints, hintPick)
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
        val rel = (HUMAN - shown.dealer + 4) % 4
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

    private fun declarationIndex(): Int = board?.trumpSuit?.let { if (it < 0) 4 else it } ?: 4

    override fun close() {
        session.close(); scope.cancel()
        // a move or hint in flight still uses the nets: free them once it is done
        CoroutineScope(Dispatchers.Default).launch {
            worker.lock.withLock {
                ruleBot?.close(); bot?.close(); declarationNet?.close(); announceNet?.close()
            }
        }
    }
    /** Match points the finished deal gave each team (us, them). */
    fun handScores(): Pair<Int, Int> = lastDeal
    val meegaanName: String? get() = shown.meegaanSeat?.let { seatName(it) }
    val terugName: String? get() = shown.terugSeat?.let { seatName(it) }

    fun seatName(player: Int) = if (player == HUMAN) tr("You", "Jij") else botNames[player - 1]
    fun trumpLabel(suit: Int) = if (suit < 0) tr("mull", "mull") else (if (Lang.dutch) arrayOf("ruiten", "harten", "klaveren", "schoppen") else arrayOf("diamonds", "hearts", "clubs", "spades"))[suit]
    fun trumpSymbol(suit: Int) = if (suit < 0) tr("∅ (mull)", "∅ (mull)") else arrayOf("♦", "♥", "♣", "♠")[suit]
    fun playerHandMask(player: Int): Long = table.openHand(player) ?: 0L

    private companion object {
        // the screen's side of a finished trick (the session waits as long: Pacing)
        const val TRICK_HOLD_MS = 1400L
        // the sweep to the winner (ManilleUi's TrickCard: at most 500 ms on a big window)
        const val COLLECT_MS = 520L
        const val REACTION_MS = 2600L
        // hint softmax temperature, in final team points (gui/hints.py TAU_PTS)
        const val TAU_PTS = 1.0
    }
}

/**
 * A finished trick of exactly a 7, an 8, a 9 and a 10, any suits: it earns a
 * "Schol!" (card ids are suit * 8 + rank, ranks 7 8 9 J Q K A 10).
 */
internal fun isScholTrick(cards: List<Int>): Boolean =
    cards.size == 4 && cards.map { it % 8 }.toSet() == setOf(0, 1, 2, 7)
