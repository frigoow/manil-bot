package com.manille.shared

import com.cardtable.platform.*

import androidx.compose.ui.layout.positionInWindow
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.material3.HorizontalDivider
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.zIndex
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.LinearEasing
import androidx.compose.foundation.Canvas
import androidx.compose.ui.unit.DpOffset
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.key
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.snap
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.animation.core.Animatable
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.TransformOrigin
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Density
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import kotlin.math.cos
import kotlin.math.roundToInt

/**
 * The whole app: the platform shell (Android: MainActivity) calls this with
 * its [Platform]; [onController] sees the controller once it exists.
 */
@Composable
fun ManilleRoot(platform: Platform, onController: (GameController) -> Unit = {}) {
    MaterialTheme {
        val controller = remember { GameController(platform).also(onController) }
        DisposableEffect(controller) { onDispose { controller.close() } }
        // The table is laid out to exact text sizes: a large system
        // font scale pushed the hand off the screen. Cap it.
        val d = LocalDensity.current
        CompositionLocalProvider(
            LocalDensity provides Density(d.density, d.fontScale.coerceAtMost(1.15f)),
            LocalAppTheme provides controller.theme,
            LocalDeck provides ManilleDeck,
        ) {
            ManilleApp(controller)
        }
    }
}

@Composable
private fun ManilleApp(controller: GameController) {
    var screen by remember { mutableStateOf("lobby") }
    Box(
        Modifier.fillMaxSize()
            .tableBackdrop()
            .systemBarsPadding(),
    ) {
        when {
            controller.phase != TablePhase.LOBBY -> GameScene(controller)
            screen == "stats" -> StatsScreen(controller) { screen = "lobby" }
            else -> Lobby(controller) { screen = "stats" }
        }
    }
}

// ---------------------------------------------------------------- lobby ----

@Composable
private fun Lobby(controller: GameController, openStats: () -> Unit) {
    var profileDialog by remember { mutableStateOf(false) }
    BoxWithConstraints(Modifier.fillMaxSize()) {
        val landscape = maxWidth > maxHeight
        val title = @Composable {
            Text("MANILLE", color = GoldBright, fontSize = if (landscape) 32.sp else 42.sp, fontWeight = FontWeight.ExtraBold, letterSpacing = 7.sp)
            Text(tr("Four cards each trick, one manille above all.", "Vier kaarten per slag, één manille boven alles."), color = InkDim, fontSize = 13.sp)
        }
        val who = @Composable {
            LobbyCard("1", tr("WHO IS PLAYING?  (tap to change)", "WIE SPEELT ER?  (tik om te wisselen)"), controller.profileName,
                tr("${controller.profiles.size} profile${if (controller.profiles.size == 1) "" else "s"} on this phone", "${controller.profiles.size} profiel${if (controller.profiles.size == 1) "" else "en"} op deze telefoon")) { profileDialog = true }
            Spacer(Modifier.height(10.dp))
            LobbyCard("2", tr("YOUR OPPONENTS", "JE TEGENSTANDERS"), controller.botLabel, controller.botDetail)
        }
        val play = @Composable {
            controller.savedGame?.let { summary ->
                Row(Modifier.widthIn(max = 480.dp)) {
                    ModeButton(tr("Resume", "Verder spelen"), summary, Modifier.weight(1f)) { controller.resume() }
                }
                Spacer(Modifier.height(10.dp))
            }
            Row(Modifier.widthIn(max = 480.dp), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                ModeButton(tr("Quick game", "Snel spel"), tr("One standalone hand", "Eén los spel"), Modifier.weight(1f)) { controller.start(false) }
                ModeButton(tr("Match to 101", "Match tot 101"), tr("Keep score across hands", "Score over meerdere spellen"), Modifier.weight(1f)) { controller.start(true) }
            }
            Spacer(Modifier.height(10.dp))
            StatsSummary(controller.stats, openStats)
            Spacer(Modifier.height(10.dp))
            Surface(Modifier.widthIn(max = 480.dp).fillMaxWidth(), shape = RoundedCornerShape(16.dp), color = Panel,
                border = BorderStroke(1.dp, Gold.copy(alpha = .25f))) {
                Column(Modifier.padding(horizontal = 14.dp, vertical = 10.dp)) {
                    Row(Modifier.widthIn(max = 480.dp).fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(tr("Bot smileys", "Smileys van de bots"), color = Ink, fontSize = 13.sp)
                            Text(tr("Bots react to the cards: 😎 🙈 👍 🎉", "Bots reageren op het spel: 😎 🙈 👍 🎉"), color = InkDim, fontSize = 11.sp)
                        }
                        Switch(checked = controller.smileys, onCheckedChange = controller::enableSmileys)
                    }
                    Row(Modifier.widthIn(max = 480.dp).fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                        Text(tr("Language", "Taal"), color = Ink, fontSize = 13.sp, modifier = Modifier.weight(1f))
                        ChoiceChip("English", !controller.dutch) { controller.setLanguage(false) }
                        Spacer(Modifier.width(6.dp))
                        ChoiceChip("Nederlands", controller.dutch) { controller.setLanguage(true) }
                    }
                    Spacer(Modifier.height(6.dp))
                    Row(Modifier.widthIn(max = 480.dp).fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(tr("Card animations", "Kaartanimaties"), color = Ink, fontSize = 13.sp)
                            Text(tr("Cards glide onto the table and to the trick's winner", "Kaarten glijden op tafel en naar wie de slag wint"), color = InkDim, fontSize = 11.sp)
                        }
                        Switch(checked = controller.animations, onCheckedChange = controller::enableAnimations)
                    }
                    if (controller.playoutNetAvailable) {
                        Spacer(Modifier.height(6.dp))
                        BotThinkingPicker(controller)
                    }
                    Spacer(Modifier.height(8.dp))
                    ThemePicker(controller.theme.id, controller::chooseTheme)
                }
            }
        }
        if (landscape) {
            Row(Modifier.fillMaxSize().padding(horizontal = 24.dp, vertical = 12.dp), horizontalArrangement = Arrangement.spacedBy(24.dp)) {
                Column(Modifier.weight(1f).fillMaxHeight().verticalScroll(rememberScrollState()),
                    horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
                    title(); Spacer(Modifier.height(18.dp)); who()
                }
                Column(Modifier.weight(1f).fillMaxHeight().verticalScroll(rememberScrollState()),
                    horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) { play() }
            }
        } else {
            Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 24.dp, vertical = 30.dp),
                horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
                title(); Spacer(Modifier.height(28.dp)); who(); Spacer(Modifier.height(12.dp)); play()
                Spacer(Modifier.height(20.dp))
                Text(tr("All game rules and model inference run on your device.", "Alle spelregels en modellen draaien op je toestel."), color = InkDim, fontSize = 11.sp)
            }
        }
    }
    if (profileDialog) ProfileDialog(controller.profiles, controller.activeProfileId, { profileDialog = false },
        controller::selectProfile, controller::createProfile, controller::renameProfile, controller::deleteProfile)
}

/** Label for the play-out net setting, in tricks (4 plies = 1 trick); 0 = off. */
private fun pliesLabel(plies: Int) = when (val t = plies / 4) {
    0 -> tr("Standard", "Standaard")
    RuleBot.WHOLE_DEAL / 4 -> tr("Whole deal", "Hele spel")
    1 -> tr("1 trick", "1 slag")
    else -> tr("$t tricks", "$t slagen")
}

/** The bots' play-out depth with the learned net, and how long they think. */
@Composable
private fun BotThinkingPicker(controller: GameController) = Column(Modifier.widthIn(max = 480.dp).fillMaxWidth()) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(tr("Bot thinking", "Denkwerk van de bots"), color = Ink, fontSize = 13.sp)
            Text(tr("A learned net plays the first tricks of every simulated deal: stronger, slower",
                "Een geleerd netwerk speelt de eerste slagen van elk gesimuleerd spel: sterker, trager"),
                color = InkDim, fontSize = 11.sp)
        }
    }
    Spacer(Modifier.height(4.dp))
    // five choices: scroll sideways on narrow phones rather than wrap
    Row(Modifier.horizontalScroll(rememberScrollState()), verticalAlignment = Alignment.CenterVertically) {
        RuleBot.PLY_CHOICES.forEachIndexed { i, plies ->
            if (i > 0) Spacer(Modifier.width(6.dp))
            ChoiceChip(pliesLabel(plies), controller.netPlies == plies) { controller.setPlayoutPlies(plies) }
        }
    }
    val mean = controller.botMsMean
    Text(if (mean == null) tr("Thinking time: play a few moves to measure it", "Denktijd: speel enkele zetten om te meten")
         else tr("Thinking time: $mean ms per move (slowest ${controller.botMsMax} ms)", "Denktijd: $mean ms per zet (traagste ${controller.botMsMax} ms)"),
        color = InkDim, fontSize = 11.sp, modifier = Modifier.padding(top = 4.dp))
}

@Composable
private fun StatsSummary(stats: Stats, open: () -> Unit) = Surface(
    modifier = Modifier.fillMaxWidth().widthIn(max = 480.dp).clickable(onClick = open), color = Panel,
    shape = RoundedCornerShape(14.dp), border = BorderStroke(1.dp, Color.White.copy(alpha = .12f)),
) {
    Column(Modifier.padding(16.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(tr("YOUR RECORD", "JOUW RESULTATEN"), color = InkDim, fontSize = 10.sp, letterSpacing = 1.sp, modifier = Modifier.weight(1f))
            Text(tr("All statistics ›", "Alle statistieken ›"), color = GoldBright, fontSize = 11.sp)
        }
        if (stats["hands"] == 0) {
            Text(tr("No hands played yet.", "Nog geen spellen gespeeld."), color = Ink, fontSize = 13.sp)
        } else {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                StatCell(tr("Matches", "Matchen"), "${stats["matches_won"]}–${stats["matches_lost"]}")
                StatCell(tr("Quick games", "Snelle spellen"), "${stats["games_won"]}–${stats["games_lost"]}" + if (stats["games_tied"] > 0) "–${stats["games_tied"]}" else "")
                StatCell(tr("Hands won", "Spellen gewonnen"), stats.rate("hands_won", "hands"))
                StatCell(tr("Avg pts", "Gem. ptn"), stats.avg("points_for", "hands"))
            }
        }
    }
}

// ----------------------------------------------------------- statistics ----

@Composable
private fun StatsScreen(controller: GameController, back: () -> Unit) {
    val s = controller.stats
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 16.dp),
        horizontalAlignment = Alignment.CenterHorizontally) {
        Row(Modifier.widthIn(max = 560.dp).fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text(tr("‹ Back", "‹ Terug"), color = GoldBright, fontSize = 15.sp, modifier = Modifier.clickable(onClick = back).padding(8.dp))
            Spacer(Modifier.weight(1f))
            Text(controller.profileName.uppercase(), color = GoldBright, fontWeight = FontWeight.ExtraBold, letterSpacing = 2.sp)
        }
        if (s["hands"] == 0) {
            Spacer(Modifier.height(40.dp)); Text(tr("No hands played yet.", "Nog geen spellen gespeeld."), color = Ink); return@Column
        }
        StatSection(tr("Results", "Resultaten")) {
            StatLine(tr("Matches won – lost", "Matchen gewonnen – verloren"), "${s["matches_won"]} – ${s["matches_lost"]}", s.rate("matches_won", "matches"))
            StatLine(tr("Quick games won – lost – tied", "Snelle spellen gewonnen – verloren – gelijk"), "${s["games_won"]} – ${s["games_lost"]} – ${s["games_tied"]}", s.rate("games_won", "games"))
            StatLine(tr("Best match margin", "Grootste matchverschil"), "${s["best_match_margin"]}")
            StatLine(tr("Match win streak (best)", "Matchen op rij (record)"), "${s["match_streak"]} (${s["best_match_streak"]})")
        }
        StatSection(tr("Hands", "Spellen")) {
            StatLine(tr("Hands played", "Spellen gespeeld"), "${s["hands"]}")
            StatLine(tr("Won – lost – tied", "Gewonnen – verloren – gelijk"), "${s["hands_won"]} – ${s["hands_lost"]} – ${s["hands_tied"]}", s.rate("hands_won", "hands"))
            StatLine(tr("Hand win streak (best)", "Spellen op rij gewonnen (record)"), "${s["hand_streak"]} (${s["best_hand_streak"]})")
            StatLine(tr("Card points per hand (us – them)", "Kaartpunten per spel (wij – zij)"), "${s.avg("points_for", "hands")} – ${s.avg("points_against", "hands")}")
            StatLine(tr("Best hand (card points)", "Beste spel (kaartpunten)"), "${s["best_hand_points"]}")
            StatLine(tr("Match points scored (us – them)", "Matchpunten (wij – zij)"), "${s["score_for"]} – ${s["score_against"]}")
            StatLine(tr("All 8 tricks (for – against)", "Alle 8 slagen (voor – tegen)"), "${s["slams"]} – ${s["slams_against"]}")
        }
        StatSection(tr("Tricks", "Slagen")) {
            StatLine(tr("Tricks per hand (team)", "Slagen per spel (team)"), s.avg("tricks_team", "hands"))
            StatLine(tr("Tricks per hand (you)", "Slagen per spel (jij)"), s.avg("tricks_mine", "hands"))
            StatLine(tr("10s (manilles) won per hand", "10'en (manilles) per spel"), s.avg("manilles_team", "hands"))
        }
        StatSection(tr("Declaring", "Troef kiezen")) {
            StatLine(tr("Your team declared", "Jouw team koos troef"), hands(s["decl_hands"]), s.rate("decl_won", "decl_hands"))
            StatLine(tr("Opponents declared", "Tegenstanders kozen troef"), hands(s["def_hands"]), s.rate("def_won", "def_hands"))
            StatLine(tr("You declared", "Jij koos troef"), hands(s["my_decl"]), s.rate("my_decl_won", "my_decl"))
            StatLine(tr("You played mull", "Jij speelde mull"), hands(s["my_mull"]), s.rate("my_mull_won", "my_mull"))
            StatLine(tr("All mull hands", "Alle mull-spellen"), hands(s["mull_hands"]), s.rate("mull_won", "mull_hands"))
        }
        StatSection(tr("Knocking", "Kloppen")) {
            StatLine(tr("Your team knocked", "Jouw team klopte"), hands(s["mee_us"]), s.rate("mee_us_won", "mee_us"))
            StatLine(tr("  …you called it", "  …door jou"), hands(s["mee_mine"]), s.rate("mee_mine_won", "mee_mine"))
            StatLine(tr("Opponents knocked", "Tegenstanders klopten"), hands(s["mee_them"]), s.rate("mee_them_won", "mee_them"))
            StatLine(tr("Your team knocked back", "Jouw team klopte terug"), hands(s["ter_us"]), s.rate("ter_us_won", "ter_us"))
            StatLine(tr("  …you called it", "  …door jou"), hands(s["ter_mine"]), s.rate("ter_mine_won", "ter_mine"))
            StatLine(tr("Opponents knocked back", "Tegenstanders klopten terug"), hands(s["ter_them"]), s.rate("ter_them_won", "ter_them"))
            StatLine(tr("Hands at ×4 or more", "Spellen aan ×4 of meer"), hands(s["big_hands"]), s.rate("big_won", "big_hands"))
        }
        StatSection(tr("Other", "Overig")) {
            StatLine(tr("Hints used", "Hints gebruikt"), "${s["hints_used"]}")
        }
        if (s.history.isNotEmpty()) StatSection(tr("Last results", "Laatste resultaten")) {
            s.history.takeLast(12).reversed().forEach { h ->
                val us = h.optInt("us"); val them = h.optInt("them")
                StatLine(if (h.optString("mode") == "match") "Match" else tr(tr("Quick game", "Snel spel"), "Snel spel"), "$us – $them",
                    if (us > them) tr("won", "gewonnen") else if (us < them) tr("lost", "verloren") else tr("tied", "gelijk"))
            }
        }
        Spacer(Modifier.height(20.dp))
    }
}

// ----------------------------------------------------------------- game ----

private class Sizes(val handW: Dp, val trickW: Dp, val seatCardW: Dp, val compactSeats: Boolean) {
    val handH get() = handW * 1.45f
    val trickH get() = trickW * 1.45f
}

@Composable
private fun GameScene(controller: GameController) {
    val state = controller.board ?: return
    var confirmRestart by remember { mutableStateOf(false) }
    BoxWithConstraints(Modifier.fillMaxSize()) {
        val landscape = maxWidth > maxHeight
        if (landscape) {
            val panelW = 196.dp
            val leftW = maxWidth - panelW
            // a phone on its side is held to its height; a tablet or a desktop
            // window gets bigger cards, up to 120 dp, so they suit the table
            val handW = minOf(120.dp, maxHeight * .27f / 1.45f, (leftW - 24.dp) / 5.7f)
            val tableH = maxHeight - handW * 1.45f - 34.dp
            val trickW = minOf(handW * .86f, (tableH - 64.dp) / 2f / 1.45f)
            val sizes = Sizes(handW, trickW, trickW * .55f, compactSeats = true)
            Row(Modifier.fillMaxSize()) {
                Column(Modifier.width(leftW).fillMaxHeight()) {
                    Table(controller, state, sizes, centerScore = false, modifier = Modifier.fillMaxWidth().weight(1f))
                    HumanHand(controller, sizes, leftW, Modifier.fillMaxWidth().padding(bottom = 4.dp))
                }
                SidePanel(controller, state, Modifier.width(panelW).fillMaxHeight()) { confirmRestart = true }
            }
        } else {
            val fullW = maxWidth
            val compact = maxWidth < 390.dp || maxHeight < 700.dp
            val handW = minOf(78.dp, (maxWidth - 24.dp) / 5.7f)
            val trickW = if (compact) 52.dp else 60.dp
            val sizes = Sizes(handW, trickW, trickW * .55f, compactSeats = false)
            Column(Modifier.fillMaxSize()) {
                TopBar(controller, state) { confirmRestart = true }
                HandScoreBar(controller, state)
                Table(controller, state, sizes, centerScore = false, modifier = Modifier.fillMaxWidth().weight(1f).padding(top = 6.dp))
                BottomBar(controller, Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 4.dp))
                HumanHand(controller, sizes, fullW, Modifier.fillMaxWidth().padding(bottom = 6.dp))
            }
        }
        if (controller.phase == TablePhase.HAND_OVER || controller.phase == TablePhase.MATCH_OVER) HandOver(controller, state)
        if (confirmRestart) AlertDialog(
            onDismissRequest = { confirmRestart = false },
            confirmButton = { TextButton(onClick = { confirmRestart = false; controller.restart() }) { Text(tr("Restart", "Herbeginnen")) } },
            dismissButton = { TextButton(onClick = { confirmRestart = false }) { Text(tr("Cancel", "Annuleren")) } },
            title = { Text(tr("Restart?", "Herbeginnen?")) },
            text = { Text(if (controller.isMatch) tr("Start a new match from 0–0? The current match is abandoned.", "Nieuwe match vanaf 0–0? De huidige match wordt gestopt.") else tr("Deal a new quick game? This hand is abandoned.", "Nieuw snel spel delen? Dit spel wordt gestopt.")) },
        )
    }
}

/** ☰ menu: open cards, restart, lobby. */
@Composable
private fun GameMenu(controller: GameController, restart: () -> Unit) {
    var open by remember { mutableStateOf(false) }
    Box {
        Surface(modifier = Modifier.size(40.dp).clickable { open = true }, shape = CircleShape,
            color = Color.White.copy(alpha = .08f), border = BorderStroke(1.dp, Color.White.copy(alpha = .18f))) {
            Box(contentAlignment = Alignment.Center) { Text("☰", color = Ink, fontSize = 18.sp) }
        }
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            DropdownMenuItem(text = { Text(if (controller.showOpenCards) tr("Hide all cards", "Verberg alle kaarten") else tr("Show all cards", "Toon alle kaarten")) },
                onClick = { open = false; controller.toggleOpenCards() })
            DropdownMenuItem(text = { Text(if (controller.animations) tr("Animations: on", "Animaties: aan") else tr("Animations: off", "Animaties: uit")) },
                onClick = { controller.enableAnimations(!controller.animations) })
            DropdownMenuItem(text = { Text(if (controller.smileys) tr("Smileys: on", "Smileys: aan") else tr("Smileys: off", "Smileys: uit")) },
                onClick = { controller.enableSmileys(!controller.smileys) })
            if (controller.playoutNetAvailable) DropdownMenuItem(
                text = { Text(tr("Bot thinking: ", "Denkwerk: ") + pliesLabel(controller.netPlies) +
                    (controller.botMsMean?.let { " · $it ms" } ?: "")) },
                onClick = {
                    val c = RuleBot.PLY_CHOICES
                    controller.setPlayoutPlies(c[(c.indexOf(controller.netPlies) + 1) % c.size])
                })
            DropdownMenuItem(text = { Text(if (controller.dutch) "Language: English" else "Taal: Nederlands") },
                onClick = { controller.setLanguage(!controller.dutch) })
            DropdownMenuItem(text = { Text(tr("Restart", "Herbeginnen")) }, onClick = { open = false; restart() })
            DropdownMenuItem(text = { Text(tr("Back to lobby", "Terug naar de lobby")) }, onClick = { open = false; controller.lobby() })
        }
    }
}

/** Big Us / Them match score with progress to 101. */
@Composable
private fun Scoreboard(controller: GameController, modifier: Modifier, big: Boolean = true) = Row(modifier, verticalAlignment = Alignment.CenterVertically) {
    ScoreBox(tr("US", "WIJ"), controller.usScore, GoldBright, Modifier.weight(1f), big, WINNING_SCORE)
    Spacer(Modifier.width(8.dp))
    ScoreBox(tr("THEM", "ZIJ"), controller.themScore, Color(0xFFE07A6E), Modifier.weight(1f), big, WINNING_SCORE)
}

/** Trump (or ? / no trump) with the stake multiplier. */
@Composable
private fun TrumpChip(controller: GameController, state: BoardState, size: Dp = 52.dp) = Column(horizontalAlignment = Alignment.CenterHorizontally) {
    val declared = controller.phase != TablePhase.DECLARATION
    Box {
        Surface(modifier = Modifier.size(size), shape = CircleShape, color = Color(0xFFFFFDF6), border = BorderStroke(2.dp, Gold), shadowElevation = 6.dp) {
            Box(contentAlignment = Alignment.Center) {
                if (!declared) Text("?", color = Wood, fontSize = 24.sp, fontWeight = FontWeight.Bold)
                else if (state.trumpSuit < 0) Text("∅", color = Wood, fontSize = 24.sp, fontWeight = FontWeight.Bold)
                else Text(suitSymbol(state.trumpSuit), color = suitTint(state.trumpSuit, controller.fourColor), fontSize = (size.value * .55f).sp)
            }
        }
        if (state.multiplier > 1 && declared) Surface(modifier = Modifier.align(Alignment.TopEnd).offset(x = 8.dp, y = (-6).dp),
            shape = RoundedCornerShape(8.dp), color = CardRed) {
            Text("×${state.multiplier}", color = Color.White, fontSize = 11.sp, fontWeight = FontWeight.ExtraBold,
                modifier = Modifier.padding(horizontal = 5.dp, vertical = 1.dp))
        }
    }
    Text(if (!declared) tr("TRUMP?", "TROEF?") else if (state.trumpSuit < 0) tr("MULL", "MULL") else tr("TRUMP", "TROEF"), color = InkDim, fontSize = 8.sp, letterSpacing = 1.sp)
}

@Composable
private fun TopBar(controller: GameController, state: BoardState, restart: () -> Unit) = Surface(
    modifier = Modifier.fillMaxWidth(), color = Color(0xB8000000), border = BorderStroke(1.dp, Color.White.copy(alpha = .08f)),
) {
    Row(Modifier.fillMaxWidth().padding(horizontal = 10.dp, vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        GameMenu(controller, restart)
        Spacer(Modifier.width(10.dp))
        Scoreboard(controller, Modifier.weight(1f))
        Spacer(Modifier.width(14.dp))
        TrumpChip(controller, state, 48.dp)
        Spacer(Modifier.width(4.dp))
    }
}

/** Message line and the Hint button, just above the hand. */
@Composable
private fun BottomBar(controller: GameController, modifier: Modifier) = Row(
    modifier.then(if (LocalAppTheme.current.style == "cafe") Modifier.clip(RoundedCornerShape(16.dp)).background(Panel).padding(horizontal = 8.dp, vertical = 2.dp) else Modifier),
    verticalAlignment = Alignment.CenterVertically) {
    // always two lines tall: a message that wraps (or appears) must not
    // resize this bar, which would resize the table above it and shake
    // every card on it
    Text(controller.message, color = Ink, fontSize = 13.sp, lineHeight = 16.sp, minLines = 2, maxLines = 2,
        overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
    SmileyButton(controller)
    Spacer(Modifier.width(8.dp))
    LastTrickButton(controller)
    Spacer(Modifier.width(8.dp))
    HintButton(controller)
}


@Composable
private fun SmileyButton(controller: GameController) {
    var open by remember { mutableStateOf(false) }
    Box {
        Surface(modifier = Modifier.size(38.dp).clickable { open = true }, shape = CircleShape,
            color = Color.White.copy(alpha = .12f), border = BorderStroke(1.dp, Gold.copy(alpha = .7f))) {
            Box(contentAlignment = Alignment.Center) { Text("😀", fontSize = 18.sp) }
        }
        // lines first, smileys last: the menu opens upwards from the
        // button, so the smileys end up right by the thumb
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            controller.playerLines.forEach { (line, mood) ->
                DropdownMenuItem(text = { Text(line) }, onClick = { open = false; controller.sendReaction(line, mood) })
            }
            HorizontalDivider()
            controller.playerSmileys.chunked(6).forEach { row ->
                Row(Modifier.padding(horizontal = 6.dp)) {
                    row.forEach { e ->
                        Text(e, fontSize = 26.sp, modifier = Modifier.clickable { open = false; controller.sendReaction(e) }.padding(5.dp))
                    }
                }
            }
        }
    }
}

@Composable
private fun LastTrickButton(controller: GameController) {
    var open by remember { mutableStateOf(false) }
    val enabled = controller.lastTrickCards.isNotEmpty()
    Surface(
        modifier = Modifier.height(38.dp).then(if (enabled) Modifier.clickable { open = true } else Modifier),
        shape = RoundedCornerShape(19.dp), color = Color.White.copy(alpha = if (enabled) .12f else .05f),
        border = BorderStroke(1.dp, if (enabled) Gold.copy(alpha = .7f) else Color.White.copy(alpha = .1f)),
    ) {
        Row(Modifier.padding(horizontal = 12.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(tr("Last trick", "Vorige slag"), color = if (enabled) Ink else InkDim.copy(alpha = .5f), fontWeight = FontWeight.Bold, fontSize = 13.sp)
        }
    }
    if (open) LastTrickDialog(controller) { open = false }
}

@Composable
private fun LastTrickDialog(controller: GameController, close: () -> Unit) {
    val cards = controller.lastTrickCards; val players = controller.lastTrickPlayers
    val points = cards.sumOf { intArrayOf(0, 0, 0, 1, 2, 3, 4, 5)[it % 8] }
    // a real dialog window: drawn inside the bottom bar it squashed the bar
    Dialog(onDismissRequest = close) {
        Surface(modifier = Modifier.clickable(onClick = close), shape = RoundedCornerShape(18.dp),
            color = Felt1, border = BorderStroke(1.5.dp, Gold.copy(alpha = .8f)), shadowElevation = 14.dp) {
            Column(Modifier.padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                Text(tr("LAST TRICK", "VORIGE SLAG"), color = GoldBright, fontWeight = FontWeight.ExtraBold, letterSpacing = 2.sp)
                Spacer(Modifier.height(10.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    cards.zip(players).forEach { (c, p) ->
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CardFace(c, controller.fourColor, Modifier, 58.dp, 84.dp, recommended = p == controller.lastTrickWinner)
                            Text(controller.seatName(p), color = if (p == controller.lastTrickWinner) GoldBright else Ink, fontSize = 12.sp,
                                fontWeight = if (p == controller.lastTrickWinner) FontWeight.Bold else FontWeight.Normal)
                        }
                    }
                }
                Spacer(Modifier.height(8.dp))
                Text(tr("${controller.seatName(controller.lastTrickWinner)} won it · $points points", "${controller.seatName(controller.lastTrickWinner)} won ze · $points punten"), color = Ink, fontSize = 13.sp)
                Text(tr("tap to close", "tik om te sluiten"), color = InkDim, fontSize = 10.sp)
            }
        }
    }
}

@Composable
private fun HintButton(controller: GameController) {
    val cardHint = controller.phase == TablePhase.PLAYING && controller.board?.currentPlayer == 0 && !controller.trickPaused
    val callHint = controller.phase == TablePhase.DECLARATION || controller.phase == TablePhase.MEEGAAN || controller.phase == TablePhase.TERUG
    val enabled = cardHint || callHint
    val showing = controller.hints.isNotEmpty() || controller.declarationHints.isNotEmpty() || controller.callHints != null
    Surface(
        modifier = Modifier.height(38.dp).then(if (enabled) Modifier.clickable(onClick = controller::toggleContextHint) else Modifier),
        shape = RoundedCornerShape(19.dp), color = if (enabled) GoldBright else Color.White.copy(alpha = .08f),
        border = BorderStroke(1.dp, if (enabled) Gold else Color.White.copy(alpha = .12f)),
    ) {
        Row(Modifier.padding(horizontal = 16.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(if (showing) tr("Hide hint", "Verberg hint") else "Hint", color = if (enabled) Wood else InkDim.copy(alpha = .5f), fontWeight = FontWeight.Bold, fontSize = 14.sp)
        }
    }
}

@Composable
private fun SidePanel(controller: GameController, state: BoardState, modifier: Modifier, restart: () -> Unit) = Surface(
    modifier = modifier, color = Color(0xB8000000), border = BorderStroke(1.dp, Color.White.copy(alpha = .08f)),
) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(10.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            GameMenu(controller, restart); Spacer(Modifier.width(12.dp)); TrumpChip(controller, state, 44.dp)
        }
        Spacer(Modifier.height(8.dp))
        Scoreboard(controller, Modifier.fillMaxWidth(), big = false)
        Spacer(Modifier.height(8.dp))
        HandScore(controller, state)
        Spacer(Modifier.height(8.dp))
        // fixed height, so the buttons below never jump as the message changes
        Text(controller.message, color = Ink, fontSize = 12.sp, lineHeight = 15.sp, textAlign = TextAlign.Center,
            minLines = 3, maxLines = 3, overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis)
        Spacer(Modifier.height(8.dp))
        Row { LastTrickButton(controller); Spacer(Modifier.width(6.dp)); HintButton(controller) }
    }
}

/** This hand's score as a slim full-width bar (portrait), off the table. */
@Composable
private fun HandScoreBar(controller: GameController, state: BoardState) = Surface(
    modifier = Modifier.fillMaxWidth(), color = Color(0x99000000),
) {
    Row(Modifier.padding(horizontal = 14.dp, vertical = 5.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(if (controller.isMatch) tr("HAND ${controller.handNumber}", "SPEL ${controller.handNumber}") else tr("QUICK GAME", "SNEL SPEL"), color = InkDim, fontSize = 10.sp, letterSpacing = 1.sp)
        Spacer(Modifier.weight(1f))
        Text(tr("Us ", "Wij "), color = GoldBright, fontSize = 13.sp, fontWeight = FontWeight.Bold)
        Text("${state.team0Points}", color = Ink, fontSize = 20.sp, fontWeight = FontWeight.ExtraBold, style = androidx.compose.ui.text.TextStyle(fontFeatureSettings = "tnum"), textAlign = TextAlign.End, modifier = Modifier.widthIn(min = 26.dp))
        Text("  –  ", color = InkDim, fontSize = 14.sp)
        Text("${state.team1Points}", color = Ink, fontSize = 20.sp, fontWeight = FontWeight.ExtraBold, style = androidx.compose.ui.text.TextStyle(fontFeatureSettings = "tnum"), modifier = Modifier.widthIn(min = 26.dp))
        Text(tr(" Them", " Zij"), color = Color(0xFFE07A6E), fontSize = 13.sp, fontWeight = FontWeight.Bold)
        Spacer(Modifier.weight(1f))
        Text(tr("tricks ${state.team0Tricks}–${state.team1Tricks}", "slagen ${state.team0Tricks}–${state.team1Tricks}"), color = InkDim, fontSize = 11.sp, style = androidx.compose.ui.text.TextStyle(fontFeatureSettings = "tnum"))
    }
}

/** This hand's card points (and tricks), clearly. */
@Composable
private fun HandScore(controller: GameController, state: BoardState) = Surface(
    shape = RoundedCornerShape(14.dp), color = Color.Black.copy(alpha = .35f), border = BorderStroke(1.dp, Gold.copy(alpha = .35f)),
) {
    Column(Modifier.padding(horizontal = 14.dp, vertical = 6.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Text(tr("THIS HAND · ", "DIT SPEL · ") + (if (controller.isMatch) tr("HAND ", "SPEL ") + controller.handNumber else tr(tr("QUICK GAME", "SNEL SPEL"), "SNEL SPEL")), color = InkDim, fontSize = 8.sp, letterSpacing = 1.sp)
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(tr("Us ", "Wij "), color = GoldBright, fontSize = 12.sp, fontWeight = FontWeight.Bold)
            Text("${state.team0Points}", color = Ink, fontSize = 22.sp, fontWeight = FontWeight.ExtraBold, style = androidx.compose.ui.text.TextStyle(fontFeatureSettings = "tnum"), textAlign = TextAlign.End, modifier = Modifier.widthIn(min = 28.dp))
            Text("  –  ", color = InkDim, fontSize = 16.sp)
            Text("${state.team1Points}", color = Ink, fontSize = 22.sp, fontWeight = FontWeight.ExtraBold, style = androidx.compose.ui.text.TextStyle(fontFeatureSettings = "tnum"), modifier = Modifier.widthIn(min = 28.dp))
            Text(tr(" Them", " Zij"), color = Color(0xFFE07A6E), fontSize = 12.sp, fontWeight = FontWeight.Bold)
        }
        Text(tr("tricks ${state.team0Tricks} – ${state.team1Tricks}   ·   60 points in play", "slagen ${state.team0Tricks} – ${state.team1Tricks}   ·   60 punten in het spel"), color = InkDim, fontSize = 9.sp)
    }
}

@Composable
private fun TablePrompt(controller: GameController) {
    when (controller.phase) {
        TablePhase.DECLARATION -> DeclarationPrompt(controller)
        TablePhase.MEEGAAN -> CallPrompt(controller, tr("KNOCK?", "KLOPPEN?"), tr("Knock", "Kloppen")) { controller.answerMeegaan(it) }
        TablePhase.TERUG -> CallPrompt(controller, tr("KNOCK BACK?", "TERUGKLOPPEN?"), tr("Knock back", "Terugkloppen")) { controller.answerTerug(it) }
        else -> Unit
    }
}

@Composable
private fun Table(controller: GameController, state: BoardState, sizes: Sizes, centerScore: Boolean, modifier: Modifier) {
    FeltTable(modifier) {
            val cafe = LocalAppTheme.current.style == "cafe"
            if (cafe) TableDecor(controller.cheers)
            // the café pint stands in the player's corner: their bubbles go beside it
            val myBubbleX = if (cafe) 70.dp else 10.dp
            val seatH = if (sizes.compactSeats) 52.dp else 34.dp + 14.dp + sizes.seatCardW * 1.45f + 8.dp
            val seatW = if (sizes.compactSeats) 64.dp else 60.dp
            Seat(controller, state, 2, Modifier.align(Alignment.TopCenter).offset(y = 4.dp), horizontal = true, sizes = sizes)
            Seat(controller, state, 1, Modifier.align(Alignment.CenterStart).offset(x = 4.dp), horizontal = false, sizes = sizes)
            Seat(controller, state, 3, Modifier.align(Alignment.CenterEnd).offset(x = (-4).dp), horizontal = false, sizes = sizes)
            // the human's own calls: a bubble at the bottom-left of the table
            CallBubbles(controller, state, 0, Modifier.align(Alignment.BottomStart).offset(x = myBubbleX, y = (-10).dp))
            val mine = controller.reactions[0]
            AnimatedVisibility(visible = mine != null, enter = scaleIn() + fadeIn(), exit = scaleOut() + fadeOut(),
                modifier = Modifier.align(Alignment.BottomStart).offset(x = myBubbleX, y = (-52).dp)) { ReactionBubble(mine ?: "") }
            // the trick: a symmetric cross around the table centre, each card
            // on its player's side (squeezed together only when space is short)
            val dv = minOf(sizes.trickH * .62f, maxHeight / 2 - seatH - 6.dp - sizes.trickH / 2).coerceAtLeast(sizes.trickH * .3f)
            val dh = minOf(sizes.trickW * 1.05f + 4.dp, maxWidth / 2 - seatW - 10.dp - sizes.trickW / 2).coerceAtLeast(sizes.trickW * .6f)
            // where each seat sits, seen from the table centre: cards fly in
            // from there and the finished trick sweeps to the winner's
            val seatAt = mapOf(2 to DpOffset(0.dp, -(maxHeight / 2 - 30.dp)), 1 to DpOffset(-(maxWidth / 2 - 30.dp), 0.dp),
                3 to DpOffset(maxWidth / 2 - 30.dp, 0.dp), 0 to DpOffset(0.dp, maxHeight / 2 + 60.dp))
            val home = mapOf(2 to DpOffset(0.dp, -dv), 1 to DpOffset(-dh, 0.dp), 3 to DpOffset(dh, 0.dp), 0 to DpOffset(0.dp, dv))
            for (p in listOf(2, 1, 3, 0)) {
                val h = home.getValue(p)
                TrickCard(controller, state, p, Modifier.align(Alignment.Center).offset(x = h.x, y = h.y), sizes, h, seatAt)
            }
            if (centerScore) Box(Modifier.align(Alignment.Center)) { HandScore(controller, state) }
            // declaration / meegaan / terug: low on the table, between the side seats
            Box(Modifier.align(Alignment.Center).offset(y = maxHeight * .14f).widthIn(max = maxWidth - seatW * 2 - 8.dp)) { TablePrompt(controller) }
    }
}

/**
 * Speech bubbles for this deal's calls: the declarer's trump (or no trump),
 * a meegaan, a terug -- next to the seat that made them, for the whole hand.
 */
@Composable
private fun CallBubbles(controller: GameController, state: BoardState, player: Int, modifier: Modifier,
                        tail: Tail = Tail.BottomStart, reaction: String? = null) {
    if (controller.phase == TablePhase.DECLARATION || controller.phase == TablePhase.LOBBY) return
    val lines = mutableListOf<Pair<String, Color>>()
    if (player == state.declarer) lines += (if (state.trumpSuit < 0) tr("Mull! ×2", "Mull! ×2") else tr("${suitSymbol(state.trumpSuit)} trump!", "${suitSymbol(state.trumpSuit)} troef!")) to
        (if (state.trumpSuit < 0) Wood else suitTint(state.trumpSuit, controller.fourColor))
    if (controller.meegaanBy == player) lines += tr("Knock!", "Klop!") to CardRed
    if (controller.terugBy == player) lines += tr("Knock back!", "Terugklop!") to CardRed
    if (lines.isEmpty() && reaction == null) return
    val align = if (tail == Tail.BottomEnd) Alignment.End else Alignment.Start
    Column(modifier, verticalArrangement = Arrangement.spacedBy(5.dp), horizontalAlignment = align) {
        // a reaction stacks on top of the calls, growing away from the speaker
        AnimatedVisibility(visible = reaction != null, enter = scaleIn() + fadeIn(), exit = scaleOut() + fadeOut()) {
            ReactionBubble(reaction ?: "")
        }
        // only the bubble nearest the speaker carries the tail
        lines.forEachIndexed { i, (text, color) ->
            SpeechBubble(text, color, if (tail == Tail.Start || i == lines.lastIndex) tail else null)
        }
    }
}

@Composable
private fun Seat(controller: GameController, state: BoardState, player: Int, modifier: Modifier, horizontal: Boolean, sizes: Sizes) {
    val active = state.currentPlayer == player && controller.phase == TablePhase.PLAYING && !controller.trickPaused
    val reaction = controller.reactions[player]
    val avatar = @Composable {
        Box {
            Surface(modifier = Modifier.size(34.dp), shape = CircleShape, color = if (active) Gold else Felt1,
                border = BorderStroke(if (active) 2.dp else 1.dp, if (active) GoldBright else Color.White.copy(alpha = .18f)), shadowElevation = if (active) 6.dp else 0.dp) {
                Box(contentAlignment = Alignment.Center) { Text(controller.seatName(player).first().toString(), color = if (active) Wood else Ink, fontWeight = FontWeight.Bold, fontSize = 14.sp) }
            }
            // speech hangs off zero-size anchors so it never resizes (and so
            // shifts) the seat. West / East: above the head, growing towards
            // the table centre, tail at the face. North: to the right of it.
            when (player) {
                1 -> Box(Modifier.align(Alignment.TopCenter).offset(x = (-15).dp, y = (-7).dp).size(0.dp).wrapContentSize(Alignment.BottomStart, unbounded = true)) {
                    CallBubbles(controller, state, player, Modifier, Tail.BottomStart, reaction)
                }
                3 -> Box(Modifier.align(Alignment.TopCenter).offset(x = 15.dp, y = (-7).dp).size(0.dp).wrapContentSize(Alignment.BottomEnd, unbounded = true)) {
                    CallBubbles(controller, state, player, Modifier, Tail.BottomEnd, reaction)
                }
                else -> {
                    Box(Modifier.align(Alignment.CenterEnd).offset(x = 10.dp).size(0.dp).wrapContentSize(Alignment.CenterStart, unbounded = true)) {
                        CallBubbles(controller, state, player, Modifier, Tail.Start)
                    }
                    Box(Modifier.align(Alignment.CenterStart).offset(x = (-8).dp).size(0.dp).wrapContentSize(Alignment.CenterEnd, unbounded = true)) {
                        AnimatedVisibility(visible = reaction != null, enter = scaleIn() + fadeIn(), exit = scaleOut() + fadeOut()) {
                            ReactionBubble(reaction ?: "")
                        }
                    }
                }
            }
        }
    }
    val label = @Composable {
        Text(controller.seatName(player) + if (player == state.declarer) " · D" else "",
            color = if (player == state.declarer) GoldBright else Ink, fontSize = 12.sp, fontWeight = FontWeight.SemiBold, letterSpacing = .5.sp,
            style = androidx.compose.ui.text.TextStyle(shadow = androidx.compose.ui.graphics.Shadow(Color.Black.copy(alpha = .75f), blurRadius = 6f)))
    }
    Box(modifier) {
        if (sizes.compactSeats && !controller.showOpenCards) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                avatar(); label()
                Text(tr("${controller.handCounts.getOrElse(player) { 0 }} cards", "${controller.handCounts.getOrElse(player) { 0 }} kaarten"), color = InkDim, fontSize = 8.sp)
            }
        } else {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                avatar(); label()
                BotCards(controller, player, horizontal, sizes.seatCardW, sizes.seatCardW * 1.45f)
            }
        }
    }
}

@Composable
private fun BotCards(controller: GameController, player: Int, horizontal: Boolean, width: Dp, height: Dp) {
    val cards = cardsInMask(if (controller.showOpenCards) controller.playerHandMask(player) else 0L)
    val count = controller.handCounts.getOrElse(player) { 0 }.coerceAtMost(8)
    if (controller.showOpenCards) {
        if (horizontal) Row { cards.forEachIndexed { i, c -> CardFace(c, controller.fourColor, Modifier.offset(x = if (i == 0) 0.dp else (-width * .45f) * i), width, height, small = true) } }
        else Column { cards.forEachIndexed { i, c -> CardFace(c, controller.fourColor, Modifier.offset(y = if (i == 0) 0.dp else (-height * .55f) * i), width, height, small = true) } }
    } else Box(Modifier.padding(top = 3.dp, end = 7.dp)) {
        CardBack(Modifier, width, height)
        Surface(
            modifier = Modifier.size(19.dp).align(Alignment.TopEnd).offset(x = 8.dp, y = (-7).dp),
            shape = CircleShape, color = GoldBright, border = BorderStroke(1.dp, Wood),
        ) { Box(contentAlignment = Alignment.Center) { Text("$count", color = Wood, fontSize = 9.sp, fontWeight = FontWeight.ExtraBold) } }
    }
}

@Composable
private fun TrickCard(controller: GameController, state: BoardState, player: Int, modifier: Modifier, sizes: Sizes,
                      home: DpOffset, seatAt: Map<Int, DpOffset>) {
    val cards = if (controller.heldTrickCards.isNotEmpty()) controller.heldTrickCards else state.trickCards
    val players = if (controller.heldTrickPlayers.isNotEmpty()) controller.heldTrickPlayers else state.trickPlayers
    val card = cards.zip(players).firstOrNull { it.second == player }?.first ?: -1
    if (card < 0) return
    key(card) {
        val animate = controller.animations
        val from = seatAt.getValue(player)
        val collectTo = controller.trickCollectTo
        val to = seatAt[collectTo] ?: home
        // 0 -> 1 as the card glides in from its player's seat; a long way
        // (a big window's table) takes longer, so the speed stays calm
        val enter = remember { Animatable(if (animate) 0f else 1f) }
        LaunchedEffect(Unit) { enter.animateTo(1f, tween(glideMs(300, from, home, max = 550), easing = FastOutSlowInEasing)) }
        // the sweep must end within GameController's COLLECT_MS (520)
        val collect by animateFloatAsState(if (collectTo >= 0 && animate) 1f else 0f,
            tween(glideMs(350, home, to, max = 500), easing = FastOutSlowInEasing), label = "collect")
        val spin = if (player == 1 || player == 2) -1f else 1f
        CardFace(card, controller.fourColor, modifier.graphicsLayer {
            val p = enter.value
            translationX = ((from.x - home.x) * (1 - p) + (to.x - home.x) * collect).toPx()
            translationY = ((from.y - home.y) * (1 - p) + (to.y - home.y) * collect).toPx()
            rotationZ = (1 - p) * 14f * spin
            val sc = (.82f + .18f * p) * (1f - .45f * collect)
            scaleX = sc; scaleY = sc
            alpha = (.4f + .6f * p) * (1f - .7f * collect)
        }, sizes.trickW, sizes.trickH)
    }
}

@Composable
private fun HumanHand(controller: GameController, sizes: Sizes, available: Dp, modifier: Modifier) = Column(modifier, horizontalAlignment = Alignment.CenterHorizontally) {
    val cards = cardsInMask(controller.handMask)
    val w = sizes.handW; val h = sizes.handH
    // Overlap so the corner index (rank over suit) stays readable, with a
    // small gap between suits.
    val gap = w * .14f
    val suits = cards.zipWithNext().count { (a, b) -> a / 8 != b / 8 }
    val step = minOf(w * .78f, (available - 20.dp - w - gap * suits) / maxOf(1, cards.size - 1))
    val total = w + step * (cards.size - 1).coerceAtLeast(0) + gap * suits
    // A fan ("waaier"): each card turns a few degrees about its bottom edge
    // and the outer cards sit a little lower, like a hand held up.
    val per = if (cards.size > 1) minOf(3.5f, 22f / (cards.size - 1)) else 0f
    val radius = h.value * 1.6f
    val pulse = rememberInfiniteTransition(label = "hint")
    val beat by pulse.animateFloat(0f, 1f, infiniteRepeatable(tween(900, easing = FastOutSlowInEasing), RepeatMode.Reverse), label = "beat")
    // the cards slide to their new places (a card played, one turning
    // playable) rather than jump; read in the layout and draw phases only
    val slide = if (controller.animations) tween<Dp>(240, easing = FastOutSlowInEasing) else snap()
    val turn = if (controller.animations) tween<Float>(240, easing = FastOutSlowInEasing) else snap()
    Box(Modifier.fillMaxWidth().height(h + 30.dp)) {
        var x = -total / 2 + w / 2
        cards.forEachIndexed { index, card -> key(card) {
            if (index > 0 && cards[index - 1] / 8 != card / 8) x += gap
            val playing = controller.phase == TablePhase.PLAYING
            val legal = playing && !controller.trickPaused && card in controller.legalCards
            val angle = (index - (cards.size - 1) / 2f) * per
            val drop = (radius * (1 - cos(angle.toDouble() * kotlin.math.PI / 180))).toFloat().dp
            val best = controller.hintPick == card
            val lift = (if (legal) (-14).dp else 0.dp) + (if (best) (-16).dp else 0.dp)
            val ax = animateDpAsState(x, slide, label = "x")
            val ay = animateDpAsState(lift + drop - 12.dp, slide, label = "y")
            val aAngle = animateFloatAsState(angle, turn, label = "angle")
            Column(
                Modifier.align(Alignment.BottomCenter)
                    .offset { androidx.compose.ui.unit.IntOffset(ax.value.roundToPx(), ay.value.roundToPx()) }
                    .graphicsLayer { rotationZ = aAngle.value; transformOrigin = TransformOrigin(0.5f, 1f) }
                    .then(if (best) Modifier.zIndex(1f) else Modifier),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                // hint badges float above the card (no layout space, so the
                // table keeps its size): the pick in the accent with a star
                val pct = controller.hints[card]?.let { "${(it * 10f).roundToInt() / 10f}%" }
                Box(Modifier.size(0.dp).wrapContentSize(Alignment.BottomCenter, unbounded = true)) {
                    if (pct != null) Surface(Modifier.offset(y = (-3).dp), shape = RoundedCornerShape(10.dp),
                        color = if (best) GoldBright else Color.Black.copy(alpha = .55f),
                        border = BorderStroke(1.dp, if (best) GoldBright else Gold.copy(alpha = .4f)), shadowElevation = if (best) 6.dp else 0.dp) {
                        Text(if (best) "★ $pct" else pct, color = if (best) Wood else Ink.copy(alpha = .85f), fontSize = if (best) 12.sp else 10.sp,
                            fontWeight = FontWeight.ExtraBold, modifier = Modifier.padding(horizontal = 6.dp, vertical = 1.dp))
                    }
                }
                CardFace(card, controller.fourColor, if (best) Modifier.graphicsLayer { val k = 1f + .035f * beat; scaleX = k; scaleY = k } else Modifier,
                    w, h, enabled = !playing || legal, recommended = best, glow = legal,
                    onClick = if (legal) ({ controller.playHuman(card) }) else null)
            }
            x += step
        } }
    }
}

@Composable
private fun DeclarationPrompt(controller: GameController) = FloatingPrompt(tr("YOU DEAL", "JIJ DEELT"), tr("Choose trump, or mull ×2", "Kies troef, of mull ×2")) {
    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        DISPLAY_SUIT_ORDER.forEach { suit ->
            SuitButton(suitSymbol(suit), suitTint(suit, controller.fourColor), controller.declarationHints[suit], controller.declarationHintPick == suit) { controller.declare(suit) }
        }
    }
    Spacer(Modifier.height(4.dp))
    controller.declarationHints[4]?.let { Text("${(it * 10).roundToInt() / 10f}%", color = GoldBright, fontSize = 10.sp, fontWeight = FontWeight.Bold) }
    Button(
        onClick = { controller.declare(-1) },
        contentPadding = PaddingValues(horizontal = 14.dp, vertical = 4.dp), modifier = Modifier.height(36.dp),
        colors = ButtonDefaults.buttonColors(containerColor = Gold, contentColor = Wood),
        border = if (controller.declarationHintPick == 4) BorderStroke(3.dp, GoldBright) else null,
    ) { Text(tr("∅  Mull ×2", "∅  Mull ×2"), fontWeight = FontWeight.Bold, maxLines = 1) }
}

@Composable
private fun SuitButton(symbol: String, color: Color, pct: Float?, recommended: Boolean, action: () -> Unit) = Column(horizontalAlignment = Alignment.CenterHorizontally) {
    Text(pct?.let { "${(it * 10).roundToInt() / 10f}%" } ?: " ", color = GoldBright, fontSize = 10.sp, fontWeight = FontWeight.Bold)
    Surface(modifier = Modifier.size(40.dp).clickable(onClick = action), shape = CircleShape, color = Color(0xFFFFFDF6),
        border = BorderStroke(if (recommended) 3.dp else 1.dp, if (recommended) GoldBright else Gold), shadowElevation = 4.dp) {
        Box(contentAlignment = Alignment.Center) { Text(symbol, color = color, fontSize = 23.sp) }
    }
}

@Composable
private fun CallPrompt(controller: GameController, title: String, acceptLabel: String, answer: (Boolean) -> Unit) = FloatingPrompt(title, "") {
    val hint = controller.callHints
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.Bottom) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            hint?.let { Text("${(it.first * 10).roundToInt() / 10f}%", color = InkDim, fontSize = 10.sp, fontWeight = FontWeight.Bold) }
            Button(
                onClick = { answer(false) },
                colors = ButtonDefaults.buttonColors(containerColor = Color.White.copy(alpha = .12f), contentColor = Ink),
                border = if (hint != null && !controller.callHintPick) BorderStroke(2.dp, GoldBright) else null,
                contentPadding = PaddingValues(horizontal = 14.dp, vertical = 6.dp),
            ) { ButtonLabel(tr("Pass", "Passen")) }
        }
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            hint?.let { Text("${(it.second * 10).roundToInt() / 10f}%", color = GoldBright, fontSize = 10.sp, fontWeight = FontWeight.Bold) }
            Button(
                onClick = { answer(true) },
                colors = ButtonDefaults.buttonColors(containerColor = GoldBright, contentColor = Wood),
                border = if (hint != null && controller.callHintPick) BorderStroke(2.dp, GoldBright) else null,
                contentPadding = PaddingValues(horizontal = 14.dp, vertical = 6.dp),
            ) { ButtonLabel(acceptLabel, bold = true) }
        }
    }
}

@Composable
private fun HandOver(controller: GameController, state: BoardState) {
    val matchOver = controller.phase == TablePhase.MATCH_OVER
    val us = state.team0Points; val them = state.team1Points
    val won = if (matchOver) controller.usScore > controller.themScore else us > them
    val tied = !matchOver && us == them
    val title = when {
        matchOver -> if (won) tr("YOU WIN THE MATCH!", "JULLIE WINNEN DE MATCH!") else tr("MATCH LOST", "MATCH VERLOREN")
        tied -> tr("HAND TIED", "GELIJKSPEL")
        won -> tr("HAND WON", "SPEL GEWONNEN")
        else -> tr("HAND LOST", "SPEL VERLOREN")
    }
    val icon = when { matchOver && won -> "🏆"; tied -> "🤝"; won -> "🎉"; else -> "😞" }
    val (dUs, dThem) = remember(state) { controller.handScores() }
    val usCol = GoldBright; val themCol = Color(0xFFE07A6E)
    // the numbers count up as the sheet appears
    val shown = remember { Animatable(0f) }
    LaunchedEffect(Unit) { shown.animateTo(1f, tween(900, easing = FastOutSlowInEasing)) }
    Box(Modifier.fillMaxSize().background(Color(0xB0000000)), contentAlignment = Alignment.Center) {
        Surface(modifier = Modifier.padding(18.dp).widthIn(max = 380.dp), shape = RoundedCornerShape(22.dp),
            color = Color.Transparent, border = BorderStroke(1.5.dp, Gold.copy(alpha = .85f)), shadowElevation = 16.dp) {
            Column(Modifier.background(Brush.verticalGradient(listOf(Felt1.copy(alpha = .97f), Felt0.copy(alpha = .98f))))
                .padding(horizontal = 18.dp, vertical = 16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                Text(icon, fontSize = 34.sp)
                Text(title, color = GoldBright, fontWeight = FontWeight.ExtraBold, fontSize = 21.sp, letterSpacing = 2.sp, textAlign = TextAlign.Center)
                Spacer(Modifier.height(12.dp))
                // the hand's score is the headline
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(tr("US", "WIJ"), color = usCol, fontWeight = FontWeight.Bold, fontSize = 12.sp, letterSpacing = 2.sp)
                        Text("${(dUs * shown.value).roundToInt()}", color = if (dUs > 0) Ink else InkDim, fontSize = 44.sp, fontWeight = FontWeight.ExtraBold, lineHeight = 46.sp)
                    }
                    Text("–", color = InkDim, fontSize = 30.sp, modifier = Modifier.padding(top = 18.dp))
                    Column(Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(tr("THEM", "ZIJ"), color = themCol, fontWeight = FontWeight.Bold, fontSize = 12.sp, letterSpacing = 2.sp)
                        Text("${(dThem * shown.value).roundToInt()}", color = if (dThem > 0) Ink else InkDim, fontSize = 44.sp, fontWeight = FontWeight.ExtraBold, lineHeight = 46.sp)
                    }
                }
                Spacer(Modifier.height(8.dp))
                // card points: one bar of 60, the 30 line marked
                Box(Modifier.fillMaxWidth().height(10.dp).clip(RoundedCornerShape(5.dp)).background(Color.White.copy(alpha = .1f))) {
                    Box(Modifier.fillMaxHeight().fillMaxWidth((us / 60f * shown.value).coerceIn(0f, 1f)).background(usCol))
                    Box(Modifier.align(Alignment.CenterEnd).fillMaxHeight().fillMaxWidth((them / 60f * shown.value).coerceIn(0f, 1f)).background(themCol))
                    Box(Modifier.align(Alignment.Center).width(2.dp).fillMaxHeight().background(Color.White.copy(alpha = .8f)))
                }
                Row(Modifier.fillMaxWidth()) {
                    Text("$us", color = Ink, fontSize = 13.sp, fontWeight = FontWeight.Bold)
                    Spacer(Modifier.weight(1f))
                    Text(tr("card points · tricks ${state.team0Tricks} – ${state.team1Tricks}", "kaartpunten · slagen ${state.team0Tricks} – ${state.team1Tricks}"),
                        color = InkDim, fontSize = 11.sp)
                    Spacer(Modifier.weight(1f))
                    Text("$them", color = Ink, fontSize = 13.sp, fontWeight = FontWeight.Bold)
                }
                Spacer(Modifier.height(10.dp))
                // the stakes
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically) {
                    Chip(if (state.trumpSuit < 0) tr("Mull", "Mull") else suitSymbol(state.trumpSuit) + " " + tr("trump", "troef"),
                        if (state.trumpSuit < 0) Ink else suitTint(state.trumpSuit, controller.fourColor, onDark = true))
                    controller.meegaanName?.let { Chip(tr("$it: knock", "$it: klop"), Color(0xFFE07A6E)) }
                    controller.terugName?.let { Chip(tr("$it: knock back", "$it: terugklop"), Color(0xFFE07A6E)) }
                    Chip("×${state.multiplier}", GoldBright)
                }
                if (controller.isMatch) {
                    Spacer(Modifier.height(10.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        ScoreBox(tr("US", "WIJ"), controller.usScore, usCol, Modifier.width(120.dp), big = true, target = WINNING_SCORE)
                        Spacer(Modifier.width(10.dp))
                        ScoreBox(tr("THEM", "ZIJ"), controller.themScore, themCol, Modifier.width(120.dp), big = true, target = WINNING_SCORE)
                    }
                }
                Spacer(Modifier.height(14.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    Button(onClick = controller::lobby, colors = ButtonDefaults.buttonColors(containerColor = Color.White.copy(alpha = .1f), contentColor = Ink)) {
                        Text("Lobby")
                    }
                    Button(onClick = { if (matchOver) controller.start(true) else controller.newDeal() },
                        colors = ButtonDefaults.buttonColors(containerColor = GoldBright, contentColor = Wood)) {
                        Text(if (matchOver) tr("Play again", "Opnieuw spelen") else tr("Next hand", "Volgend spel"), fontWeight = FontWeight.Bold)
                    }
                }
            }
        }
    }
}

private fun hands(n: Int) = if (n == 1) tr("1 hand", "1 spel") else tr("$n hands", "$n spellen")

private val DISPLAY_SUIT_ORDER = listOf(3, 1, 2, 0) // spades, hearts, clubs, diamonds
private val DISPLAY_SUIT_POSITION = DISPLAY_SUIT_ORDER.withIndex().associate { it.value to it.index }

private fun cardsInMask(mask: Long): List<Int> = (0 until 32)
    .filter { mask and (1L shl it) != 0L }
    .sortedWith(compareBy({ DISPLAY_SUIT_POSITION.getValue(it / 8) }, { it % 8 }))

/** Manille's card numbers: suit * 8 + rank, suits ♦ ♥ ♣ ♠, ranks 7 8 9 J Q K A 10. */
internal val ManilleDeck = Deck { card ->
    PlayingCard(MANILLE_SUITS[card / 8], arrayOf(Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE, Rank.TEN)[card % 8])
}
private val MANILLE_SUITS = arrayOf(Suit.DIAMONDS, Suit.HEARTS, Suit.CLUBS, Suit.SPADES)

/** A Manille suit number (-1: mull) as the symbol. */
private fun suitSymbol(suit: Int) = suitSymbol(if (suit < 0) null else MANILLE_SUITS[suit])

@Composable
private fun suitTint(suit: Int, fourColor: Boolean = false, onDark: Boolean = false): Color =
    suitTint(if (suit < 0) null else MANILLE_SUITS[suit], fourColor, onDark)
