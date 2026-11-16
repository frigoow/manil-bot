package com.manille.shared

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

private val Felt0 @Composable get() = LocalAppTheme.current.feltEdge
private val Felt1 @Composable get() = LocalAppTheme.current.feltMid
private val Felt2 @Composable get() = LocalAppTheme.current.feltCenter
private val Ink @Composable get() = LocalAppTheme.current.ink
private val InkDim @Composable get() = LocalAppTheme.current.inkDim
private val Gold @Composable get() = LocalAppTheme.current.accent
private val GoldBright @Composable get() = LocalAppTheme.current.accentBright
private val CardRed = Color(0xFFC0282C)
private val CardBlack = Color(0xFF1C1A16)
private val CardBlue = Color(0xFF1D5BB0)
private val CardGreen = Color(0xFF1E7B3C)
private val Wood @Composable get() = LocalAppTheme.current.rim
private val Panel @Composable get() = LocalAppTheme.current.panel

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
        ) {
            ManilleApp(controller)
        }
    }
}

@Composable
private fun ManilleApp(controller: GameController) {
    var screen by remember { mutableStateOf("lobby") }
    val theme = LocalAppTheme.current
    val weave = Gold.copy(alpha = .035f)
    val vignette = Color.Black.copy(alpha = .55f)
    val stars = remember { val r = kotlin.random.Random(7); List(150) { floatArrayOf(r.nextFloat(), r.nextFloat(), r.nextFloat()) } }
    Box(
        Modifier.fillMaxSize()
            .background(theme.bgTop?.let { top -> Brush.verticalGradient(listOf(top, theme.bgBottom ?: top)) }
                        ?: Brush.radialGradient(listOf(Felt2, Felt1, Felt0), radius = 1900f))
            .drawBehind {
                if (theme.style == "cafe") {
                    surrealBackdrop()
                    // stars only in the upper sky
                    stars.forEach { (fx, fy, b) ->
                        if (fy < .5f) drawCircle(Color.White.copy(alpha = (.10f + b * .4f) * (1 - fy * 1.6f).coerceAtLeast(0f)),
                            (.6f + b * .8f).dp.toPx(), Offset(fx * size.width, fy * size.height))
                    }
                } else if (theme.stars) {
                    // a quiet night sky: small stars, a few brighter ones
                    stars.forEach { (fx, fy, b) ->
                        val r = (if (b > .93f) 1.6f else .7f + b * .6f).dp.toPx()
                        drawCircle(Color.White.copy(alpha = .10f + b * (if (b > .93f) .55f else .25f)), r, Offset(fx * size.width, fy * size.height))
                    }
                } else {
                    // a faint diagonal weave, like cloth under the table
                    val step = 26.dp.toPx()
                    var x = -size.height
                    while (x < size.width) {
                        drawLine(weave, Offset(x, 0f), Offset(x + size.height, size.height), strokeWidth = 1f)
                        drawLine(weave, Offset(x + size.height, 0f), Offset(x, size.height), strokeWidth = 1f)
                        x += step
                    }
                }
                if (theme.glow != Color.Transparent) drawRect(Brush.radialGradient(listOf(theme.glow, Color.Transparent),
                    center = Offset(size.width * .5f, 0f), radius = size.width * 1.1f))
                drawRect(Brush.radialGradient(listOf(Color.Transparent, Color.Transparent, vignette),
                    center = center, radius = maxOf(size.width, size.height) * .75f))
            }
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
                        LanguageChip("English", !controller.dutch) { controller.setLanguage(false) }
                        Spacer(Modifier.width(6.dp))
                        LanguageChip("Nederlands", controller.dutch) { controller.setLanguage(true) }
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
                    ThemePicker(controller)
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
    if (profileDialog) ProfileDialog(controller) { profileDialog = false }
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
            LanguageChip(pliesLabel(plies), controller.netPlies == plies) { controller.setPlayoutPlies(plies) }
        }
    }
    val mean = controller.botMsMean
    Text(if (mean == null) tr("Thinking time: play a few moves to measure it", "Denktijd: speel enkele zetten om te meten")
         else tr("Thinking time: $mean ms per move (slowest ${controller.botMsMax} ms)", "Denktijd: $mean ms per zet (traagste ${controller.botMsMax} ms)"),
        color = InkDim, fontSize = 11.sp, modifier = Modifier.padding(top = 4.dp))
}

@Composable
private fun LanguageChip(label: String, selected: Boolean, onClick: () -> Unit) = Surface(
    modifier = Modifier.clickable(onClick = onClick), shape = RoundedCornerShape(16.dp),
    color = if (selected) GoldBright else Color.White.copy(alpha = .08f),
    border = BorderStroke(1.dp, if (selected) GoldBright else Gold.copy(alpha = .5f)),
) {
    Text(label, color = if (selected) Wood else Ink, fontSize = 12.sp, fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp))
}

@Composable
private fun ThemePicker(controller: GameController) = Column(Modifier.widthIn(max = 480.dp).fillMaxWidth()) {
    Text(tr("Table theme", "Tafelthema"), color = Ink, fontSize = 13.sp)
    Spacer(Modifier.height(6.dp))
    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        THEMES.forEach { t ->
            val selected = t.id == controller.theme.id
            Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.clickable { controller.chooseTheme(t.id) }) {
                Box(
                    Modifier.size(70.dp, 52.dp).clip(RoundedCornerShape(10.dp))
                        .background(Brush.radialGradient(listOf(t.feltCenter, t.feltMid, t.feltEdge)))
                        .border(if (selected) 3.dp else 1.dp, if (selected) t.accentBright else t.rim, RoundedCornerShape(10.dp)),
                    contentAlignment = Alignment.Center,
                ) {
                    Row(horizontalArrangement = Arrangement.spacedBy((-8).dp)) {
                        Box(Modifier.size(20.dp, 29.dp).clip(RoundedCornerShape(3.dp))
                            .background(Brush.linearGradient(listOf(t.backA, t.backB))).border(1.dp, t.accent, RoundedCornerShape(3.dp)),
                            contentAlignment = Alignment.Center) { Text(t.emblem, color = t.accentBright, fontSize = 9.sp) }
                        Box(Modifier.size(20.dp, 29.dp).clip(RoundedCornerShape(3.dp))
                            .background(Brush.verticalGradient(listOf(t.paperTop, t.paperBottom))).border(1.dp, t.accent.copy(alpha = .6f), RoundedCornerShape(3.dp)),
                            contentAlignment = Alignment.Center) { Text("A", color = CardRed, fontSize = 11.sp, fontWeight = FontWeight.Bold) }
                    }
                }
                Text(if (controller.dutch) t.nameNl else t.name, color = if (selected) GoldBright else InkDim, fontSize = 11.sp, fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal)
            }
        }
    }
}

@Composable
private fun ProfileDialog(controller: GameController, close: () -> Unit) {
    var newName by remember { mutableStateOf("") }
    var renaming by remember { mutableStateOf<Profile?>(null) }
    var renameText by remember { mutableStateOf("") }
    var deleting by remember { mutableStateOf<Profile?>(null) }
    AlertDialog(
        onDismissRequest = close,
        confirmButton = { TextButton(onClick = close) { Text(tr("Done", "Klaar")) } },
        title = { Text(tr("Profiles", "Profielen")) },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState())) {
                controller.profiles.forEach { p ->
                    Row(Modifier.fillMaxWidth().clickable { controller.selectProfile(p.id) }.padding(vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically) {
                        Text(if (p.id == controller.activeProfileId) "●" else "○", fontSize = 18.sp)
                        Spacer(Modifier.width(10.dp))
                        Text(p.name, Modifier.weight(1f), fontWeight = if (p.id == controller.activeProfileId) FontWeight.Bold else FontWeight.Normal)
                        TextButton(onClick = { renaming = p; renameText = p.name }) { Text(tr("Rename", "Hernoemen")) }
                        if (controller.profiles.size > 1) TextButton(onClick = { deleting = p }) { Text(tr("Delete", "Verwijderen")) }
                    }
                }
                Spacer(Modifier.height(10.dp))
                OutlinedTextField(value = newName, onValueChange = { newName = it.take(24) }, singleLine = true,
                    label = { Text(tr("New profile name", "Naam nieuw profiel")) }, modifier = Modifier.fillMaxWidth())
                TextButton(onClick = { controller.createProfile(newName); newName = "" }) { Text(tr("Create and switch to it", "Aanmaken en wisselen")) }
                Text(tr("Each profile has its own saved game and statistics.", "Elk profiel heeft een eigen bewaard spel en statistieken."), fontSize = 12.sp)
            }
        },
    )
    renaming?.let { p ->
        AlertDialog(
            onDismissRequest = { renaming = null },
            confirmButton = { TextButton(onClick = { controller.renameProfile(p.id, renameText); renaming = null }) { Text(tr("Rename", "Hernoemen")) } },
            dismissButton = { TextButton(onClick = { renaming = null }) { Text(tr("Cancel", "Annuleren")) } },
            title = { Text(tr("Rename profile", "Profiel hernoemen")) },
            text = { OutlinedTextField(value = renameText, onValueChange = { renameText = it.take(24) }, singleLine = true) },
        )
    }
    deleting?.let { p ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            confirmButton = { TextButton(onClick = { controller.deleteProfile(p.id); deleting = null }) { Text(tr("Delete", "Verwijderen")) } },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text(tr("Cancel", "Annuleren")) } },
            title = { Text(tr("Delete ${p.name}?", "${p.name} verwijderen?")) },
            text = { Text(tr("Its saved game and all its statistics are removed. This cannot be undone.", "Het bewaarde spel en alle statistieken worden verwijderd. Dit kan niet ongedaan gemaakt worden.")) },
        )
    }
}

@Composable
private fun LobbyCard(step: String, title: String, name: String, sub: String, onClick: (() -> Unit)? = null) = Surface(
    modifier = Modifier.fillMaxWidth().widthIn(max = 480.dp).then(if (onClick != null) Modifier.clickable(onClick = onClick) else Modifier),
    color = Panel, shape = RoundedCornerShape(14.dp), border = BorderStroke(1.dp, Color.White.copy(alpha = .12f)),
) {
    Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
        Surface(modifier = Modifier.size(28.dp), shape = CircleShape, color = Gold.copy(alpha = .14f), border = BorderStroke(1.dp, Gold.copy(alpha = .55f))) {
            Box(contentAlignment = Alignment.Center) { Text(step, color = GoldBright, fontWeight = FontWeight.Bold) }
        }
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f)) {
            Text(title, color = InkDim, fontSize = 10.sp, letterSpacing = 1.sp)
            Text(name, color = Ink, fontWeight = FontWeight.Bold, fontSize = 17.sp)
            Text(sub, color = GoldBright, fontSize = 11.sp)
        }
        if (onClick != null) Text("›", color = GoldBright, fontSize = 26.sp)
    }
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

@Composable
private fun StatCell(label: String, value: String) = Column(horizontalAlignment = Alignment.CenterHorizontally) {
    Text(value, color = GoldBright, fontWeight = FontWeight.Bold, fontSize = 17.sp)
    Text(label, color = InkDim, fontSize = 10.sp)
}

@Composable
private fun ModeButton(title: String, sub: String, modifier: Modifier, onClick: () -> Unit) = Button(
    modifier = modifier.height(92.dp), onClick = onClick,
    colors = ButtonDefaults.buttonColors(containerColor = Gold.copy(alpha = .14f), contentColor = Ink),
    border = BorderStroke(1.dp, Gold.copy(alpha = .65f)), shape = RoundedCornerShape(14.dp),
) { Column(horizontalAlignment = Alignment.CenterHorizontally) {
    Text(title, color = GoldBright, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center)
    Text(sub, color = InkDim, fontSize = 10.sp, textAlign = TextAlign.Center)
} }

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

@Composable
private fun StatSection(title: String, content: @Composable ColumnScope.() -> Unit) {
    Spacer(Modifier.height(12.dp))
    Surface(modifier = Modifier.widthIn(max = 560.dp).fillMaxWidth(), color = Panel, shape = RoundedCornerShape(14.dp),
        border = BorderStroke(1.dp, Color.White.copy(alpha = .12f))) {
        Column(Modifier.padding(horizontal = 16.dp, vertical = 12.dp)) {
            Text(title.uppercase(), color = InkDim, fontSize = 10.sp, letterSpacing = 1.sp)
            Spacer(Modifier.height(4.dp)); content()
        }
    }
}

@Composable
private fun StatLine(label: String, value: String, extra: String? = null) = Row(Modifier.fillMaxWidth().padding(vertical = 3.dp),
    verticalAlignment = Alignment.CenterVertically) {
    Text(label, color = Ink, fontSize = 13.sp, modifier = Modifier.weight(1f))
    Text(value, color = GoldBright, fontSize = 13.sp, fontWeight = FontWeight.Bold)
    if (extra != null) Text("  $extra", color = InkDim, fontSize = 12.sp, modifier = Modifier.widthIn(min = 44.dp), textAlign = TextAlign.End)
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

@Composable
private fun FloatingPrompt(title: String, sub: String, content: @Composable ColumnScope.() -> Unit) =
    Surface(shape = RoundedCornerShape(16.dp), color = Color.Transparent, border = BorderStroke(1.5.dp, Gold.copy(alpha = .85f)), shadowElevation = 12.dp) {
        // the theme's panel over a hint of its table colours
        Column(Modifier.background(Brush.verticalGradient(listOf(Panel.copy(alpha = .96f), Felt0.copy(alpha = .97f))))
            .border(.8.dp, GoldBright.copy(alpha = .25f), RoundedCornerShape(16.dp))
            .padding(horizontal = 12.dp, vertical = 8.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Text(title, color = GoldBright, fontWeight = FontWeight.ExtraBold, fontSize = 15.sp, letterSpacing = 1.5.sp)
            if (sub.isNotBlank()) { Spacer(Modifier.height(2.dp)); Text(sub, color = Ink, textAlign = TextAlign.Center, fontSize = 11.sp, maxLines = 2) }
            Spacer(Modifier.height(6.dp)); content()
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
    ScoreBox(tr("US", "WIJ"), controller.usScore, GoldBright, Modifier.weight(1f), big)
    Spacer(Modifier.width(8.dp))
    ScoreBox(tr("THEM", "ZIJ"), controller.themScore, Color(0xFFE07A6E), Modifier.weight(1f), big)
}

@Composable
private fun ScoreBox(label: String, score: Int, color: Color, modifier: Modifier, big: Boolean) = Surface(
    modifier = modifier, shape = RoundedCornerShape(12.dp), color = Color.Black.copy(alpha = .35f),
    border = BorderStroke(1.dp, color.copy(alpha = .45f)),
) {
    Column(Modifier.padding(horizontal = 10.dp, vertical = 5.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Text(label, color = color, fontSize = 10.sp, fontWeight = FontWeight.Bold, letterSpacing = 2.sp)
        Text("$score", color = Ink, fontSize = if (big) 30.sp else 24.sp, fontWeight = FontWeight.ExtraBold, lineHeight = if (big) 32.sp else 26.sp, style = androidx.compose.ui.text.TextStyle(fontFeatureSettings = "tnum"))
        Box(Modifier.fillMaxWidth().height(4.dp).clip(RoundedCornerShape(2.dp)).background(Color.White.copy(alpha = .12f))) {
            Box(Modifier.fillMaxWidth((score / 101f).coerceIn(0f, 1f)).height(4.dp).background(color))
        }
        Text(tr("to 101", "tot 101"), color = InkDim, fontSize = 8.sp)
    }
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

/**
 * The café theme's table props, drawn (no stock art): the player's own pint
 * and a cup of tea float above the desert, their shadows on the sand; a music
 * box plays on the sand below the bare tree. Each change of [cheers] (a
 * trick made of a 7, 8, 9 and 10) raises the pint for a toast.
 */
@Composable
private fun androidx.compose.foundation.layout.BoxScope.TableDecor(cheers: Int) {
    val anim = rememberInfiniteTransition(label = "decor")
    val time by anim.animateFloat(0f, 1f, infiniteRepeatable(tween(4000, easing = LinearEasing)), label = "time")
    // the toast: lift, clink, set down. The count seen at first composition
    // does not play, so coming back to the table does not toast again.
    val schol = remember { Animatable(0f) }
    var toasted by remember { mutableStateOf(cheers) }
    LaunchedEffect(cheers) {
        if (cheers == toasted) return@LaunchedEffect
        toasted = cheers
        schol.snapTo(0f); schol.animateTo(1f, tween(1300, easing = LinearEasing)); schol.snapTo(0f)
    }
    val p = schol.value
    fun ease(x: Float) = x.coerceIn(0f, 1f).let { it * it * (3 - 2 * it) }
    val up = when { p <= 0f -> 0f; p < .25f -> ease(p / .25f); p < .7f -> 1f; else -> ease((1f - p) / .3f) }
    // the clink: a quick forward jolt once the glass is up
    val clink = if (p in .25f..0.7f) kotlin.math.sin((p - .25f) / .45f * 2 * kotlin.math.PI.toFloat() * 1.5f) * (1f - (p - .25f) / .45f) else 0f
    // the player's pint, at their own corner of the table, on a coaster
    Box(Modifier.align(Alignment.BottomStart).offset(x = 8.dp, y = (-6).dp).size(58.dp, 72.dp)) {
        Canvas(Modifier.matchParentSize()) {
            drawOval(Color(0x55180C04).copy(alpha = .33f * (1f - .7f * up)), Offset(size.width * .06f, size.height * .80f), androidx.compose.ui.geometry.Size(size.width * .92f, size.height * .2f))
            drawOval(Brush.radialGradient(listOf(Color(0xFFD9C3A0), Color(0xFFB89A70))), Offset(size.width * .04f, size.height * .76f),
                androidx.compose.ui.geometry.Size(size.width * .92f, size.height * .2f))
        }
        Canvas(Modifier.align(Alignment.TopCenter).size(38.dp, 62.dp).graphicsLayer {
            // raised toward the table centre, pivoting on the glass's foot
            transformOrigin = TransformOrigin(.5f, 1f)
            translationY = -22.dp.toPx() * up
            translationX = 6.dp.toPx() * up + 3.dp.toPx() * clink
            rotationZ = 14f * up + 6f * clink
            scaleX = 1f + .14f * up; scaleY = scaleX
        }) { drawBeer(time, splash = if (p in .25f..0.75f) (p - .25f) / .5f else 0f) }
        if (p > 0f) {
            val a = ease((p - .22f) / .1f) * ease((.95f - p) / .15f)
            Text(tr("Cheers!", "Schol!"), color = Gold, fontSize = 15.sp, fontWeight = FontWeight.Black,
                modifier = Modifier.align(Alignment.TopStart).offset(x = 38.dp, y = (-58).dp)
                    .wrapContentSize(unbounded = true)
                    .graphicsLayer { alpha = a; scaleX = .7f + .3f * a; scaleY = scaleX; rotationZ = -8f })
        }
    }
    Box(Modifier.align(Alignment.TopEnd).offset(x = (-8).dp, y = 14.dp).size(64.dp, 56.dp)) {
        Canvas(Modifier.matchParentSize()) {
            drawOval(Color(0x55180C04), Offset(size.width * .02f, size.height * .82f), androidx.compose.ui.geometry.Size(size.width * .98f, size.height * .18f))
        }
        Canvas(Modifier.matchParentSize()) { drawTea(time) }
    }
    Canvas(Modifier.align(Alignment.TopStart).offset(x = 10.dp, y = 8.dp).size(60.dp, 66.dp)) { drawMusicBox(time) }
    Canvas(Modifier.align(Alignment.BottomEnd).offset(x = (-8).dp, y = (-6).dp).size(104.dp, 66.dp)) { drawClockOnBook() }
    // the elephant crosses slowly below the tea
    val walk by anim.animateFloat(0f, 1f, infiniteRepeatable(tween(26000, easing = LinearEasing), RepeatMode.Reverse), label = "walk")
    Canvas(Modifier.align(Alignment.TopEnd).offset(x = (-14 - 30 * walk).dp, y = 78.dp).size(42.dp, 58.dp)
        .graphicsLayer { scaleX = if (walk > .5f) 1f else -1f }) { drawStiltElephant(time * 4f % 1f) }
}

/** [splash] runs 0..1 over a toast: foam drops jump off the rim (0 = none). */
private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawBeer(time: Float, splash: Float = 0f) {
    val w = size.width; val h = size.height
    val topY = h * .04f; val botY = h * .97f; val inset = w * .11f
    fun xl(y: Float) = inset * (y - topY) / (botY - topY)
    val glass = androidx.compose.ui.graphics.Path().apply {
        moveTo(0f, topY); lineTo(w, topY); lineTo(w - inset, botY); lineTo(inset, botY); close()
    }
    // beer body
    val beerTop = h * .25f
    val beer = androidx.compose.ui.graphics.Path().apply {
        moveTo(xl(beerTop) + 1.5f, beerTop); lineTo(w - xl(beerTop) - 1.5f, beerTop); lineTo(w - inset - 1.5f, botY - h * .07f); lineTo(inset + 1.5f, botY - h * .07f); close()
    }
    drawPath(beer, Brush.verticalGradient(listOf(Color(0xFFF6C35A), Color(0xFFE09A22), Color(0xFFB06A10)), startY = beerTop, endY = botY))
    drawPath(beer, Brush.horizontalGradient(listOf(Color(0x66401E00), Color.Transparent, Color(0x33FFF3C0), Color.Transparent, Color(0x77401E00))))
    // rising bubbles
    for (k in 0 until 9) {
        val bx = w * (.25f + .5f * ((k * 37) % 9) / 9f)
        val phase = (time + k / 9f) % 1f
        val by = botY - h * .1f - phase * (botY - beerTop - h * .12f)
        drawCircle(Color.White.copy(alpha = .55f * (1 - phase) + .15f), 0.9f + (k % 3) * .4f, Offset(bx, by))
    }
    // foam head: a band with a bubbly top edge
    val foamTop = h * .10f
    drawRect(Brush.verticalGradient(listOf(Color(0xFFFFFCF2), Color(0xFFF1E3C4)), startY = foamTop, endY = beerTop + 2f),
        Offset(xl(foamTop) + 1f, foamTop), androidx.compose.ui.geometry.Size(w - 2 * xl(foamTop) - 2f, beerTop - foamTop + 2f))
    for (k in 0..5) drawCircle(Color(0xFFFFFDF6), w * .1f, Offset(w * (.1f + .16f * k), foamTop + h * .005f))
    drawCircle(Color(0xFFFFFDF6), w * .12f, Offset(w * .78f, foamTop - h * .035f))
    // glass: thick base, rim, highlights, condensation
    drawRect(Color(0x55DDEEFF), Offset(inset, botY - h * .07f), androidx.compose.ui.geometry.Size(w - 2 * inset, h * .07f))
    drawPath(glass, Color.White.copy(alpha = .55f), style = androidx.compose.ui.graphics.drawscope.Stroke(1.4f))
    drawLine(Color.White.copy(alpha = .55f), Offset(w * .2f, h * .3f), Offset(w * .24f, h * .86f), strokeWidth = w * .06f)
    drawLine(Color.White.copy(alpha = .25f), Offset(w * .78f, h * .32f), Offset(w * .74f, h * .8f), strokeWidth = w * .03f)
    for (k in 0 until 7) drawCircle(Color.White.copy(alpha = .45f), 1.1f, Offset(w * (.3f + .07f * ((k * 5) % 7)), h * (.4f + .07f * k)))
    if (splash > 0f) for (k in 0 until 6) {
        // each drop flies up and out, then falls back, fading
        val vx = (k - 2.5f) * .09f + .12f
        val x = w * (.5f + vx * splash * 2.2f)
        val y = topY - h * (.42f * splash - .5f * splash * splash) * (1f + .25f * (k % 3))
        drawCircle(Color(0xFFFFFDF6).copy(alpha = (1f - splash) * .95f), w * (.045f + .015f * (k % 2)), Offset(x, y))
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawTea(time: Float) {
    val w = size.width; val h = size.height
    val sz = { a: Float, b: Float -> androidx.compose.ui.geometry.Size(a, b) }
    // saucer
    drawOval(Brush.verticalGradient(listOf(Color(0xFFFFFFFF), Color(0xFFD8D4CC)), startY = h * .78f, endY = h), Offset(0f, h * .78f), sz(w, h * .2f))
    drawOval(Color(0xFFC9A54E), Offset(0f, h * .78f), sz(w, h * .2f), style = androidx.compose.ui.graphics.drawscope.Stroke(1.2f))
    drawOval(Color(0x22000000), Offset(w * .25f, h * .82f), sz(w * .5f, h * .1f))
    // cup body
    val l = w * .14f; val r = w * .78f; val top = h * .36f; val foot = h * .86f
    val body = androidx.compose.ui.graphics.Path().apply {
        moveTo(l, top)
        cubicTo(l, h * .72f, w * .3f, foot, (l + r) / 2, foot)
        cubicTo(w * .62f, foot, r, h * .72f, r, top)
        close()
    }
    drawPath(body, Brush.horizontalGradient(listOf(Color(0xFFCFCBC2), Color(0xFFFFFFFF), Color(0xFFF4F1EA), Color(0xFFBDB8AE)), startX = l, endX = r))
    drawLine(Color(0xFFC9A54E), Offset(l + w * .03f, h * .5f), Offset(r - w * .03f, h * .5f), strokeWidth = 1.2f)
    // handle
    drawArc(Color(0xFFE6E2DA), -70f, 160f, false, Offset(r - w * .1f, h * .42f), sz(w * .22f, h * .26f),
        style = androidx.compose.ui.graphics.drawscope.Stroke(w * .05f))
    // tea surface and gilded rim
    drawOval(Brush.radialGradient(listOf(Color(0xFFC07A30), Color(0xFF8A4A14)), center = Offset((l + r) / 2, top), radius = (r - l) / 2),
        Offset(l + 1.5f, top - h * .06f), sz(r - l - 3f, h * .12f))
    drawOval(Color(0xFFD4AF5A), Offset(l, top - h * .065f), sz(r - l, h * .13f), style = androidx.compose.ui.graphics.drawscope.Stroke(1.6f))
    drawOval(Color.White.copy(alpha = .35f), Offset(l + w * .1f, top - h * .03f), sz(w * .16f, h * .03f))
    // steam
    for (k in 0..2) {
        val phase = (time + k / 3f) % 1f
        val x0 = w * (.32f + .14f * k); val y0 = top - h * .08f
        val path = androidx.compose.ui.graphics.Path()
        for (i in 0..12) {
            val f = i / 12f
            val x = x0 + w * .05f * kotlin.math.sin((f * 6 + phase * 6.28f + k).toDouble()).toFloat()
            val y = y0 - f * h * .34f - phase * h * .05f
            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(path, Color.White.copy(alpha = .35f * (1 - phase) + .05f), style = androidx.compose.ui.graphics.drawscope.Stroke(2.2f, cap = androidx.compose.ui.graphics.StrokeCap.Round))
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawMusicBox(time: Float) {
    val w = size.width; val h = size.height
    val sz = { a: Float, b: Float -> androidx.compose.ui.geometry.Size(a, b) }
    val bx = w * .08f; val by = h * .55f; val bw = w * .78f; val bh = h * .34f
    // shadow on the sand
    drawOval(Color(0x442A1E14), Offset(bx - w * .02f, by + bh - h * .03f), sz(bw * 1.2f, h * .1f))
    // open lid, tilted back, with its mirror
    val lid = androidx.compose.ui.graphics.Path().apply {
        moveTo(bx, by); lineTo(bx + w * .06f, by - h * .38f); lineTo(bx + bw + w * .06f, by - h * .38f); lineTo(bx + bw, by); close()
    }
    drawPath(lid, Brush.verticalGradient(listOf(Color(0xFF6E431F), Color(0xFF4A2A12)), startY = by - h * .38f, endY = by))
    val mirror = androidx.compose.ui.graphics.Path().apply {
        moveTo(bx + w * .06f, by - h * .03f); lineTo(bx + w * .1f, by - h * .33f); lineTo(bx + bw - w * .0f, by - h * .33f); lineTo(bx + bw - w * .05f, by - h * .03f); close()
    }
    drawPath(mirror, Brush.linearGradient(listOf(Color(0xFFDCE6F0), Color(0xFF8FA3BA), Color(0xFFE8EEF4))))
    // the box
    drawRoundRect(Brush.verticalGradient(listOf(Color(0xFF9A6434), Color(0xFF5E3A1A)), startY = by, endY = by + bh),
        Offset(bx, by), sz(bw, bh), cornerRadius = androidx.compose.ui.geometry.CornerRadius(3f, 3f))
    drawRect(Color(0xFFD4AF5A), Offset(bx, by + bh * .12f), sz(bw, 1.6f))
    drawRect(Color(0xFFD4AF5A), Offset(bx, by + bh * .82f), sz(bw, 1.6f))
    drawCircle(Color(0xFF2A1A0C), 1.8f, Offset(bx + bw / 2, by + bh * .45f))
    drawLine(Color(0xFF2A1A0C), Offset(bx + bw / 2, by + bh * .45f), Offset(bx + bw / 2, by + bh * .6f), strokeWidth = 1.4f)
    // winding key
    drawLine(Color(0xFFD4AF5A), Offset(bx + bw, by + bh * .5f), Offset(bx + bw + w * .08f, by + bh * .5f), strokeWidth = 2f)
    drawOval(Color(0xFFD4AF5A), Offset(bx + bw + w * .06f, by + bh * .3f), sz(w * .06f, bh * .4f))
    // the dancer, turning (a squeezed width fakes the spin)
    val cx = bx + bw / 2; val spin = kotlin.math.cos(time * 6.283f * 2).toFloat()
    val sw = (.35f + .65f * kotlin.math.abs(spin))
    drawLine(Color(0xFFF2E6D6), Offset(cx, by - h * .02f), Offset(cx, by - h * .14f), strokeWidth = 1.6f)
    val tutu = androidx.compose.ui.graphics.Path().apply {
        moveTo(cx - w * .12f * sw, by - h * .14f); lineTo(cx + w * .12f * sw, by - h * .14f); lineTo(cx, by - h * .22f); close()
    }
    drawPath(tutu, Color(0xFFF4C6CF))
    drawLine(Color(0xFFF2E6D6), Offset(cx, by - h * .2f), Offset(cx, by - h * .28f), strokeWidth = 1.6f)
    drawLine(Color(0xFFF2E6D6), Offset(cx - w * .07f * sw, by - h * .31f), Offset(cx + w * .07f * sw, by - h * .31f), strokeWidth = 1.2f)
    drawCircle(Color(0xFFF2E6D6), w * .035f, Offset(cx, by - h * .31f))
    // a few notes drifting up
    for (k in 0..1) {
        val phase = (time + k * .5f) % 1f
        val nx = bx + bw * (.75f + .2f * k) + w * .08f * kotlin.math.sin(phase * 6.283f).toFloat()
        val ny = by - h * .1f - phase * h * .45f
        val a = (1 - phase) * .9f
        drawCircle(Color(0xFF2A1A0C).copy(alpha = a), 2.4f, Offset(nx, ny))
        drawLine(Color(0xFF2A1A0C).copy(alpha = a), Offset(nx + 2.2f, ny), Offset(nx + 2.2f, ny - h * .1f), strokeWidth = 1.2f)
    }
}

@Composable
private fun Table(controller: GameController, state: BoardState, sizes: Sizes, centerScore: Boolean, modifier: Modifier) {
    val shape = RoundedCornerShape(48.dp)
    val melting = LocalAppTheme.current.style == "cafe"
    val rim = Wood
    Box(modifier.padding(horizontal = 10.dp)
        .then(if (melting) Modifier.drawWithContent { drawContent(); meltingEdge(rim) } else Modifier)
        .shadow(20.dp, shape).clip(shape)
        .background(Brush.radialGradient(listOf(Felt2, Felt1, Felt0)))
        .then(if (LocalAppTheme.current.style == "cafe") Modifier.drawBehind { woodGrain() } else Modifier)
        .border(8.dp, Wood, shape).padding(8.dp)) {
        BoxWithConstraints(Modifier.fillMaxSize().border(1.dp, Gold.copy(alpha = .28f), RoundedCornerShape(38.dp))) {
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

private enum class Tail { BottomStart, BottomEnd, Start }

@Composable
private fun SpeechBubble(text: String, color: Color, tail: Tail? = Tail.BottomStart) = Box(Modifier.keepOnScreen()) {
    // tail: a small rotated square peeking out towards the speaker's avatar
    val tailMod = when (tail) {
        Tail.BottomStart -> Modifier.align(Alignment.BottomStart).offset(x = 10.dp, y = 4.dp)
        Tail.BottomEnd -> Modifier.align(Alignment.BottomEnd).offset(x = (-10).dp, y = 4.dp)
        Tail.Start -> Modifier.align(Alignment.CenterStart).offset(x = (-4).dp)
        null -> null
    }
    if (tailMod != null) Box(tailMod.size(10.dp).rotate(45f).background(Color(0xFFFFFDF6)))
    Surface(shape = RoundedCornerShape(14.dp), color = Color(0xFFFFFDF6), shadowElevation = 5.dp) {
        Text(text, color = color, fontWeight = FontWeight.ExtraBold, fontSize = 13.sp,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp))
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

/**
 * Keeps a bubble inside the screen: it measures where it landed (before its
 * own shift, so there is no feedback) and slides back in by the overflow.
 */
@Composable
private fun Modifier.keepOnScreen(margin: Dp = 22.dp): Modifier {
    val density = LocalDensity.current
    val screenW = androidx.compose.ui.platform.LocalWindowInfo.current.containerSize.width.toFloat()
    val m = with(density) { margin.toPx() }
    var shift by remember { mutableStateOf(0) }
    return this
        .onGloballyPositioned { c ->
            val x = c.positionInWindow().x; val w = c.size.width
            val s = when {
                w > screenW - 2 * m -> (m - x)
                x < m -> m - x
                x + w > screenW - m -> screenW - m - x - w
                else -> 0f
            }.roundToInt()
            if (s != shift) shift = s
        }
        .offset { androidx.compose.ui.unit.IntOffset(shift, 0) }
}

/** A reaction: a single emoji in a round badge, a line of text in a speech bubble. */
@Composable
private fun ReactionBubble(text: String) {
    val emojiOnly = text.count { !it.isLowSurrogate() } <= 2
    if (emojiOnly) Surface(modifier = Modifier.keepOnScreen(), shape = CircleShape, color = Color(0xFFFFFDF6), shadowElevation = 6.dp, border = BorderStroke(1.dp, Gold)) {
        Text(text, fontSize = 24.sp, modifier = Modifier.padding(6.dp))
    } else Surface(modifier = Modifier.keepOnScreen(), shape = RoundedCornerShape(14.dp), color = Color(0xFFFFFDF6), shadowElevation = 6.dp, border = BorderStroke(1.dp, Gold)) {
        Text(text, color = Color(0xFF1C1A16), fontSize = 13.sp, fontWeight = FontWeight.SemiBold,
            modifier = Modifier.widthIn(max = 170.dp).padding(horizontal = 10.dp, vertical = 5.dp))
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

/**
 * How long a card takes from [a] to [b]: [base] ms over a phone table's
 * distances (up to ~180 dp), then growing with the square root of the
 * distance, up to [max] ms on a large window, so a card never races across
 * a big table.
 */
private fun glideMs(base: Int, a: DpOffset, b: DpOffset, max: Int): Int {
    val d = kotlin.math.hypot((a.x - b.x).value, (a.y - b.y).value)
    return (base * kotlin.math.sqrt(maxOf(1f, d / 180f))).roundToInt().coerceAtMost(max)
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

/** A one-line button label: longer words get a smaller size instead of wrapping. */
@Composable
private fun ButtonLabel(text: String, bold: Boolean = false) = Text(
    text, maxLines = 1, softWrap = false, fontWeight = if (bold) FontWeight.Bold else FontWeight.Medium,
    fontSize = when { text.length <= 7 -> 15.sp; text.length <= 10 -> 13.sp; else -> 12.sp },
)

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
                        ScoreBox(tr("US", "WIJ"), controller.usScore, usCol, Modifier.width(120.dp), big = true)
                        Spacer(Modifier.width(10.dp))
                        ScoreBox(tr("THEM", "ZIJ"), controller.themScore, themCol, Modifier.width(120.dp), big = true)
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

@Composable
private fun Chip(text: String, color: Color) = Surface(shape = RoundedCornerShape(10.dp), color = Color.Black.copy(alpha = .3f),
    border = BorderStroke(1.dp, color.copy(alpha = .6f))) {
    Text(text, color = color, fontSize = 12.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp))
}

@Composable
private fun Overlay(title: String, sub: String, content: @Composable ColumnScope.() -> Unit) = Box(Modifier.fillMaxSize().background(Color(0xB0000000)), contentAlignment = Alignment.Center) {
    Surface(modifier = Modifier.padding(18.dp).widthIn(max = 380.dp), shape = RoundedCornerShape(18.dp), color = Panel.copy(alpha = .97f),
        border = BorderStroke(1.5.dp, Gold.copy(alpha = .8f)), shadowElevation = 14.dp) {
        Column(Modifier.padding(horizontal = 18.dp, vertical = 18.dp), horizontalAlignment = Alignment.CenterHorizontally, content = {
            Text(title, color = GoldBright, fontWeight = FontWeight.ExtraBold, fontSize = 20.sp, letterSpacing = 2.sp, textAlign = TextAlign.Center)
            Spacer(Modifier.height(6.dp)); Text(sub, color = Ink, textAlign = TextAlign.Center, fontSize = 12.sp)
            Spacer(Modifier.height(14.dp)); content()
        })
    }
}

// ---------------------------------------------------------------- cards ----

@Composable
private fun CardBack(modifier: Modifier, width: Dp, height: Dp) {
    val t = LocalAppTheme.current
    Box(
        modifier.size(width, height).shadow(3.dp, RoundedCornerShape(5.dp)).clip(RoundedCornerShape(5.dp))
            .background(Brush.linearGradient(listOf(t.backA, t.backB)))
            .border(1.5.dp, t.paperTop, RoundedCornerShape(5.dp))
            .padding(3.dp)
            .drawBehind {
                if (t.style == "cafe") {
                    // a dream in miniature: night sky over sand, an eye at the horizon
                    val hz = size.height * .66f
                    drawRect(Brush.verticalGradient(listOf(Color(0xFF0F1D36), Color(0xFF3D5F8A), Color(0xFFE6B987)), endY = hz),
                        size = androidx.compose.ui.geometry.Size(size.width, hz))
                    drawRect(Brush.verticalGradient(listOf(Color(0xFFD8A866), Color(0xFF9A6A35)), startY = hz, endY = size.height),
                        topLeft = Offset(0f, hz), size = androidx.compose.ui.geometry.Size(size.width, size.height - hz))
                    drawCircle(Color(0xFFF3E9D2), size.width * .07f, Offset(size.width * .75f, size.height * .14f))
                    surrealEye(Offset(size.width / 2, size.height * .45f), size.width * .7f, Color(0xFF3D5F8A))
                } else {
                    // diamond lattice
                    val step = 7.dp.toPx()
                    var x = -size.height
                    while (x < size.width) {
                        drawLine(t.backLine, Offset(x, 0f), Offset(x + size.height, size.height), strokeWidth = .8f)
                        drawLine(t.backLine, Offset(x + size.height, 0f), Offset(x, size.height), strokeWidth = .8f)
                        x += step
                    }
                }
            }
            .border(1.dp, t.accent.copy(alpha = .6f), RoundedCornerShape(3.dp)),
        contentAlignment = Alignment.Center,
    ) {
        if (t.style != "cafe") Surface(shape = CircleShape, color = t.backB, border = BorderStroke(1.dp, t.accent.copy(alpha = .8f))) {
            Text(t.emblem, color = t.accentBright, fontSize = (width.value * .30f).sp, lineHeight = (width.value * .32f).sp,
                modifier = Modifier.padding(horizontal = 3.dp))
        }
    }
}

/** Pip positions (fractions of the pip area) of the number cards, as on a real deck. */
private val PIPS: Map<Int, List<Pair<Float, Float>>> = mapOf(
    0 to listOf(.25f to .0f, .75f to .0f, .5f to .25f, .25f to .5f, .75f to .5f, .25f to 1f, .75f to 1f),               // 7
    1 to listOf(.25f to .0f, .75f to .0f, .5f to .25f, .25f to .5f, .75f to .5f, .5f to .75f, .25f to 1f, .75f to 1f),  // 8
    2 to listOf(.25f to .0f, .75f to .0f, .25f to .33f, .75f to .33f, .5f to .5f, .25f to .67f, .75f to .67f, .25f to 1f, .75f to 1f), // 9
    7 to listOf(.25f to .0f, .75f to .0f, .5f to .17f, .25f to .33f, .75f to .33f, .25f to .67f, .75f to .67f, .5f to .83f, .25f to 1f, .75f to 1f), // 10
)

/**
 * A real-looking card: corner indices (rank over suit) top-left and, upside
 * down, bottom-right; the classic pip layout for 7-10, one large pip for the
 * ace, a framed court card with a crown for J / Q / K.
 */
@Composable
private fun CardFace(card: Int, fourColor: Boolean, modifier: Modifier, width: Dp, height: Dp, small: Boolean = false,
                     enabled: Boolean = true, recommended: Boolean = false, glow: Boolean = false, onClick: (() -> Unit)? = null) {
    val shape = RoundedCornerShape(if (small) 4.dp else 7.dp)
    val interaction = if (enabled && onClick != null) Modifier.clickable(onClick = onClick) else Modifier
    val suit = card / 8; val rank = card % 8
    val color = suitTint(suit, fourColor)
    val idx: TextUnit = (width.value * .26f).coerceIn(9f, 20f).sp
    val idxPip: TextUnit = (width.value * .30f).coerceIn(9f, 22f).sp
    val t = LocalAppTheme.current
    Surface(
        modifier = modifier.then(Modifier.size(width, height)).then(interaction)
            .shadow(if (recommended) 14.dp else if (glow) 9.dp else 4.dp, shape,
                ambientColor = if (glow || recommended) t.accentBright else Color.Black,
                spotColor = if (glow || recommended) t.accentBright else Color.Black),
        shape = shape, color = Color.Transparent,
        border = BorderStroke(if (recommended) 3.dp else if (glow) 1.5.dp else 1.dp,
            if (recommended) t.accentBright else if (glow) t.accent else Color.Black.copy(alpha = .18f)),
    ) {
        Box(Modifier.fillMaxSize()
            .background(if (enabled) Brush.verticalGradient(listOf(t.paperTop, t.paperBottom))
                        else Brush.verticalGradient(listOf(Color(0xFFCFCBC1), Color(0xFFBDB8AD))))) {
            // fine inner frame in the theme's accent
            if (!small) Box(Modifier.matchParentSize().padding(2.5.dp).border(.8.dp, t.accent.copy(alpha = .35f), RoundedCornerShape(5.dp)))
            val corner = @Composable { m: Modifier ->
                Column(m.padding(horizontal = if (small) 2.dp else 3.dp, vertical = if (small) 1.dp else 2.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(cardRank(card), color = color, fontWeight = FontWeight.ExtraBold, fontSize = idx, lineHeight = idx)
                    Text(suitSymbol(suit), color = color, fontSize = idxPip, lineHeight = idxPip)
                }
            }
            corner(Modifier.align(Alignment.TopStart))
            if (!small) {
                corner(Modifier.align(Alignment.BottomEnd).rotate(180f))
                // the face itself: inside the corner indices
                val fx = width * .24f; val fy = height * .12f
                val faceW = width - fx * 2; val faceH = height - fy * 2
                Box(Modifier.offset(x = fx, y = fy).size(faceW, faceH)) {
                    when (rank) {
                        6 -> if (t.style == "cafe") CafeAce(suit, color, faceW, faceH)
                             else Text(suitSymbol(suit), Modifier.align(Alignment.Center), color = color, fontSize = (width.value * .62f).sp)
                        3, 4, 5 -> if (t.style == "cafe") CafeCourt(rank, suit, color, faceW, faceH) else CourtFace(rank, suit, color, faceW, faceH)
                        else -> {
                            val pip = (width.value * .2f).coerceAtLeast(7f)
                            val pipBox = (pip * 1.25f).dp
                            PIPS.getValue(rank).forEach { (px, py) ->
                                Text(suitSymbol(suit), color = color, fontSize = pip.sp, lineHeight = pip.sp, textAlign = TextAlign.Center,
                                    modifier = Modifier.offset(x = faceW * px - pipBox / 2, y = (faceH - pipBox) * py)
                                        .size(pipBox).then(if (py > .5f) Modifier.rotate(180f) else Modifier))
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * The café theme's court cards, after the surrealists: a little dream landscape with
 * the jack's upturned moustache, the queen's eye, the king's melting crown.
 */
@Composable
private fun CafeCourt(rank: Int, suit: Int, color: Color, w: Dp, h: Dp) {
    val t = LocalAppTheme.current
    val shape = RoundedCornerShape(3.dp)
    Box(Modifier.size(w, h).clip(shape).border(1.dp, t.accent.copy(alpha = .8f), shape)
        .drawBehind {
            val hz = size.height * .62f
            drawRect(Brush.verticalGradient(listOf(Color(0xFF1B2E4F), Color(0xFF5B7DA6), Color(0xFFE9C38E)), endY = hz), size = androidx.compose.ui.geometry.Size(size.width, hz))
            drawRect(Brush.verticalGradient(listOf(Color(0xFFD8A866), Color(0xFFB07A3B)), startY = hz, endY = size.height),
                topLeft = Offset(0f, hz), size = androidx.compose.ui.geometry.Size(size.width, size.height - hz))
            // a long shadow across the sand
            drawLine(Color(0x552A1E14), Offset(size.width * .5f, hz + size.height * .1f), Offset(size.width * .05f, size.height * .97f), strokeWidth = size.width * .06f)
            val c = Offset(size.width / 2, size.height * .42f)
            when (rank) {
                3 -> surrealMoustache(c, size.width * .82f, Color(0xFF17110C))
                4 -> surrealEye(c, size.width * .84f, color)
                else -> meltingCrown(c, size.width * .74f)
            }
        }) {
        Text(suitSymbol(suit), color = Color(0xFFFFFBF2), fontSize = (w.value * .26f).sp,
            modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 1.dp))
    }
}

/** The café theme's ace: the suit symbol, slowly dripping. */
@Composable
private fun CafeAce(suit: Int, color: Color, w: Dp, h: Dp) {
    Box(Modifier.size(w, h).drawBehind {
        val cx = size.width / 2; val top = size.height * .62f
        val drips = listOf(-.16f to .22f, -.02f to .34f, .12f to .16f)
        drips.forEach { (dx, len) ->
            val x = cx + dx * size.width; val r = size.width * .035f
            drawRoundRect(color, Offset(x - r, top - r), androidx.compose.ui.geometry.Size(r * 2, len * size.height),
                cornerRadius = androidx.compose.ui.geometry.CornerRadius(r, r))
            drawCircle(color, r * 1.35f, Offset(x, top + len * size.height - r))
        }
    }, contentAlignment = Alignment.Center) {
        Text(suitSymbol(suit), color = color, fontSize = (w.value * .62f).sp, modifier = Modifier.offset(y = -h * .06f))
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.surrealMoustache(c: Offset, w: Float, ink: Color) {
    for (side in listOf(-1f, 1f)) {
        val p = androidx.compose.ui.graphics.Path()
        p.moveTo(c.x, c.y)
        p.cubicTo(c.x + side * w * .14f, c.y + w * .16f, c.x + side * w * .34f, c.y + w * .12f, c.x + side * w * .42f, c.y - w * .30f)
        p.cubicTo(c.x + side * w * .32f, c.y - w * .02f, c.x + side * w * .16f, c.y + w * .06f, c.x, c.y - w * .05f)
        p.close()
        drawPath(p, ink)
    }
    drawCircle(ink, w * .05f, Offset(c.x, c.y - w * .02f))
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.surrealEye(c: Offset, w: Float, iris: Color) {
    val ink = Color(0xFF17110C)
    val p = androidx.compose.ui.graphics.Path()
    p.moveTo(c.x - w * .45f, c.y)
    p.quadraticBezierTo(c.x, c.y - w * .38f, c.x + w * .45f, c.y)
    p.quadraticBezierTo(c.x, c.y + w * .38f, c.x - w * .45f, c.y)
    p.close()
    drawPath(p, Color(0xFFFFFBF2))
    drawCircle(iris, w * .15f, c)
    drawCircle(ink, w * .065f, c)
    drawCircle(Color.White, w * .025f, Offset(c.x + w * .04f, c.y - w * .04f))
    drawPath(p, ink, style = androidx.compose.ui.graphics.drawscope.Stroke(w * .03f))
    for (k in 0..4) {
        val f = -.32f + k * .16f
        val x = c.x + f * w; val y = c.y - w * .19f * (1 - (f / .45f) * (f / .45f)) - w * .005f
        drawLine(ink, Offset(x, y), Offset(x + f * w * .25f, y - w * .12f), strokeWidth = w * .022f)
    }
    // a single tear
    val tx = c.x + w * .08f; val ty = c.y + w * .30f
    drawCircle(Color(0xFFBFD8F0), w * .045f, Offset(tx, ty))
    drawLine(Color(0xFFBFD8F0), Offset(tx, ty - w * .1f), Offset(tx, ty), strokeWidth = w * .03f)
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.meltingCrown(c: Offset, w: Float) {
    val gold = Color(0xFFD9A93E); val ink = Color(0xFF4A3618)
    val base = c.y + w * .12f; val l = c.x - w * .45f; val r = c.x + w * .45f
    val p = androidx.compose.ui.graphics.Path()
    p.moveTo(l, base)
    p.lineTo(l, c.y - w * .22f); p.lineTo(c.x - w * .22f, c.y - w * .02f)
    p.lineTo(c.x, c.y - w * .36f); p.lineTo(c.x + w * .22f, c.y - w * .02f)
    p.lineTo(r, c.y - w * .22f); p.lineTo(r, base)
    // the rim melts: soft drips hanging from the band
    p.cubicTo(r - w * .05f, base + w * .30f, r - w * .14f, base + w * .30f, r - w * .17f, base + w * .06f)
    p.cubicTo(c.x + w * .05f, base + w * .1f, c.x + w * .06f, base + w * .48f, c.x - w * .04f, base + w * .44f)
    p.cubicTo(c.x - w * .10f, base + w * .40f, c.x - w * .10f, base + w * .08f, c.x - w * .2f, base + w * .07f)
    p.cubicTo(l + w * .12f, base + w * .2f, l + w * .04f, base + w * .2f, l, base)
    p.close()
    drawPath(p, gold)
    drawPath(p, ink, style = androidx.compose.ui.graphics.drawscope.Stroke(w * .025f))
    for (x in listOf(l, c.x, r)) drawCircle(Color(0xFFF6EEDD), w * .05f, Offset(x, if (x == c.x) c.y - w * .36f else c.y - w * .22f))
    drawCircle(Color(0xFFB0302A), w * .06f, Offset(c.x, base - w * .07f))
}

/**
 * The café theme's world behind the table, after the surrealists (all drawn here): a
 * vast dusk sky, a low horizon with cliffs and a far bare tree, sand with
 * long shadows. The table stands in it.
 */
private fun androidx.compose.ui.graphics.drawscope.DrawScope.surrealBackdrop() {
    val w = size.width; val h = size.height; val hz = h * .78f
    val sz = { a: Float, b: Float -> androidx.compose.ui.geometry.Size(a, b) }
    drawRect(Brush.verticalGradient(listOf(Color(0xFF0B1528), Color(0xFF1E3558), Color(0xFF557AA6), Color(0xFFE8BC88)), endY = hz), size = sz(w, hz))
    drawRect(Brush.verticalGradient(listOf(Color(0xFFD9AA68), Color(0xFFB27C40), Color(0xFF7A4E26)), startY = hz, endY = h),
        topLeft = Offset(0f, hz), size = sz(w, h - hz))
    val moon = Offset(w * .84f, h * .07f)
    drawCircle(Brush.radialGradient(listOf(Color(0x44F3E9D2), Color.Transparent), center = moon, radius = w * .16f), w * .16f, moon)
    drawCircle(Color(0xFFF3E9D2), w * .045f, moon)
    drawOval(Color(0x26FFFFFF), Offset(w * .08f, h * .5f), sz(w * .4f, h * .006f))
    drawOval(Color(0x1EFFFFFF), Offset(w * .5f, h * .62f), sz(w * .34f, h * .005f))
    val cliff = androidx.compose.ui.graphics.Path().apply {
        moveTo(0f, hz); lineTo(0f, hz - h * .06f); lineTo(w * .07f, hz - h * .075f); lineTo(w * .15f, hz - h * .04f)
        lineTo(w * .22f, hz - h * .05f); lineTo(w * .34f, hz); close()
    }
    drawPath(cliff, Color(0xFF6E482E))
    val far = androidx.compose.ui.graphics.Path().apply {
        moveTo(w * .72f, hz); lineTo(w * .82f, hz - h * .02f); lineTo(w * .9f, hz - h * .016f); lineTo(w, hz - h * .03f); lineTo(w, hz); close()
    }
    drawPath(far, Color(0xFF65607A))

}

/** Polished walnut: long, gently wavy grain lines with a few light streaks. */
private fun androidx.compose.ui.graphics.drawscope.DrawScope.woodGrain() {
    val grain = Color(0x1F1E0E05); val light = Color(0x14FFE6C8)
    var y = 0f; var k = 0
    while (y < size.height) {
        val path = androidx.compose.ui.graphics.Path()
        for (i in 0..24) {
            val x = size.width * i / 24f
            val yy = y + 5.dp.toPx() * kotlin.math.sin(i * .55 + k * 1.7).toFloat()
            if (i == 0) path.moveTo(x, yy) else path.lineTo(x, yy)
        }
        drawPath(path, if (k % 3 == 0) light else grain, style = androidx.compose.ui.graphics.drawscope.Stroke((1 + k % 2).dp.toPx()))
        y += (9 + (k * 7) % 11).dp.toPx(); k++
    }
    // a soft sheen, as on a varnished top
    drawRect(Brush.linearGradient(listOf(Color(0x18FFFFFF), Color.Transparent, Color(0x10FFFFFF)), start = Offset.Zero, end = Offset(size.width, size.height)))
}

/** The table's bottom edge melts: slow drips of wood hanging under the rim. */
private fun androidx.compose.ui.graphics.drawscope.DrawScope.meltingEdge(rim: Color) {
    val bottom = size.height
    listOf(.16f to 14f, .27f to 7f, .62f to 9f, .71f to 18f, .86f to 8f).forEach { (fx, len) ->
        val x = size.width * fx; val r = 4.dp.toPx(); val l = len.dp.toPx()
        drawRoundRect(rim, Offset(x - r, bottom - 6.dp.toPx()), androidx.compose.ui.geometry.Size(r * 2, l + 6.dp.toPx()),
            cornerRadius = androidx.compose.ui.geometry.CornerRadius(r, r))
        drawCircle(rim, r * 1.3f, Offset(x, bottom + l - r * .3f))
        drawCircle(Color(0x33FFFFFF), r * .4f, Offset(x - r * .4f, bottom + l - r * .6f))
    }
}

/** A closed book on the table, a soft clock draped over its edge (with ants). */
private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawClockOnBook() {
    val w = size.width; val h = size.height
    val sz = { a: Float, b: Float -> androidx.compose.ui.geometry.Size(a, b) }
    val bx = w * .38f; val by = h * .45f; val bw = w * .6f; val bh = h * .3f
    drawOval(Color(0x55180C04), Offset(bx - w * .04f, by + bh - h * .05f), sz(bw * 1.12f, h * .14f))
    drawRoundRect(Color(0xFF7A1F22), Offset(bx, by), sz(bw, bh), cornerRadius = androidx.compose.ui.geometry.CornerRadius(3f, 3f))
    drawRect(Color(0xFFF1E6CC), Offset(bx + 2f, by + bh * .18f), sz(bw - 4f, bh * .64f))
    for (k in 1..4) drawLine(Color(0x33000000), Offset(bx + 2f, by + bh * (.18f + .13f * k)), Offset(bx + bw - 2f, by + bh * (.18f + .13f * k)), strokeWidth = .8f)
    drawRect(Color(0xFFD4AF5A), Offset(bx + bw * .1f, by + bh * .05f), sz(bw * .5f, 1.4f))
    // the clock: face up on the cover, its left half pouring over the spine edge
    val face = Offset(bx + bw * .45f, by - h * .01f); val rx = bw * .42f; val ry = h * .07f
    val clock = androidx.compose.ui.graphics.Path().apply {
        moveTo(face.x + rx, face.y)
        cubicTo(face.x + rx, face.y - ry * 1.5f, face.x - rx * .6f, face.y - ry * 1.5f, face.x - rx, face.y - ry * .2f)
        cubicTo(face.x - rx * 1.25f, face.y + ry, face.x - rx * 1.3f, by + bh * .9f, face.x - rx * 1.05f, by + bh * 1.25f)
        cubicTo(face.x - rx * .85f, by + bh * 1.35f, face.x - rx * .6f, by + bh * .8f, face.x - rx * .55f, by + 2f)
        cubicTo(face.x - rx * .1f, face.y + ry, face.x + rx * .7f, face.y + ry * 1.1f, face.x + rx, face.y)
        close()
    }
    drawPath(clock, Brush.verticalGradient(listOf(Color(0xFFF6EED2), Color(0xFFDCC88F)), startY = face.y - ry * 1.5f, endY = by + bh * 1.3f))
    drawPath(clock, Color(0xFFB08A3A), style = androidx.compose.ui.graphics.drawscope.Stroke(3f))
    drawPath(clock, Color(0xFF5A4220), style = androidx.compose.ui.graphics.drawscope.Stroke(1f))
    for (k in 0 until 12) {
        val a = kotlin.math.PI * 2 * k / 12
        val px = face.x + rx * .78f * kotlin.math.cos(a).toFloat(); val py = face.y - ry * .3f + ry * .75f * kotlin.math.sin(a).toFloat()
        if (px > face.x - rx * .55f) drawCircle(Color(0xFF3A2A14), if (k % 3 == 0) 1.8f else 1f, Offset(px, py))
    }
    val hub = Offset(face.x + rx * .05f, face.y - ry * .3f)
    drawLine(Color(0xFF3A2A14), hub, Offset(hub.x + rx * .45f, hub.y - ry * .3f), strokeWidth = 2f)
    drawLine(Color(0xFF3A2A14), hub, Offset(hub.x - rx * .3f, hub.y + ry * .45f), strokeWidth = 1.4f)
    drawCircle(Color(0xFF3A2A14), 2f, hub)
    for (k in 0 until 5) drawCircle(Color(0xFF17110C), 1.4f, Offset(face.x - rx * (1.0f + .05f * (k % 2)), by + bh * (.1f + .2f * k)))
}

/** A small elephant on impossibly long legs, walking across the table. */
private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawStiltElephant(step: Float) {
    val w = size.width; val h = size.height; val ink = Color(0xFF1E1511)
    val sz = { a: Float, b: Float -> androidx.compose.ui.geometry.Size(a, b) }
    val foot = h * .95f; val body = h * .22f
    drawOval(Color(0x44180C04), Offset(w * .05f, foot - h * .03f), sz(w * .9f, h * .07f))
    drawOval(ink, Offset(w * .22f, body - h * .09f), sz(w * .5f, h * .17f))
    drawCircle(ink, w * .11f, Offset(w * .75f, body - h * .04f))
    drawLine(ink, Offset(w * .82f, body), Offset(w * .86f, body + h * .16f), strokeWidth = 2.2f)
    drawOval(Color(0xFFE9C38E), Offset(w * .28f, body - h * .14f), sz(w * .3f, h * .07f))   // the obelisk saddle
    val sw = kotlin.math.sin(step * 6.283f).toFloat()
    listOf(.28f, .38f, .55f, .65f).forEachIndexed { i, fx ->
        val swing = (if (i % 2 == 0) sw else -sw) * w * .04f
        drawLine(ink, Offset(w * fx, body + h * .06f), Offset(w * fx + swing, foot), strokeWidth = 1.2f)
    }
}


/** J / Q / K: a framed panel tinted in the suit colour with a crown and the letter. */
@Composable
private fun CourtFace(rank: Int, suit: Int, color: Color, w: Dp, h: Dp) = Box(
    Modifier.size(w, h).clip(RoundedCornerShape(4.dp))
        .background(Brush.verticalGradient(listOf(color.copy(alpha = .16f), Color(0xFFFFF4D6), color.copy(alpha = .16f))))
        .border(1.dp, color.copy(alpha = .55f), RoundedCornerShape(4.dp)),
    contentAlignment = Alignment.Center,
) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(arrayOf("♝", "♛", "♚")[rank - 3] + "︎", color = Color(0xFFB8892E), fontSize = (w.value * .42f).sp, lineHeight = (w.value * .44f).sp)
        Text(arrayOf("J", "Q", "K")[rank - 3], color = color, fontWeight = FontWeight.ExtraBold, fontSize = (w.value * .40f).sp, lineHeight = (w.value * .42f).sp)
        Text(suitSymbol(suit), color = color, fontSize = (w.value * .28f).sp, lineHeight = (w.value * .3f).sp)
    }
}

private fun hands(n: Int) = if (n == 1) tr("1 hand", "1 spel") else tr("$n hands", "$n spellen")

private val DISPLAY_SUIT_ORDER = listOf(3, 1, 2, 0) // spades, hearts, clubs, diamonds
private val DISPLAY_SUIT_POSITION = DISPLAY_SUIT_ORDER.withIndex().associate { it.value to it.index }

private fun cardsInMask(mask: Long): List<Int> = (0 until 32)
    .filter { mask and (1L shl it) != 0L }
    .sortedWith(compareBy({ DISPLAY_SUIT_POSITION.getValue(it / 8) }, { it % 8 }))
private fun cardRank(card: Int) = arrayOf("7", "8", "9", "J", "Q", "K", "A", "10")[card % 8]
// U+FE0E asks for the text glyph: without it ♥ ♦ ♣ render as colour emoji
// that ignore the suit colour (the four-colour deck showed red diamonds).
private fun suitSymbol(suit: Int) = if (suit < 0) "∅" else arrayOf("♦", "♥", "♣", "♠")[suit] + "\uFE0E"

/**
 * Suit colour: red suits red and black suits black, as on a real deck, but in
 * distinct shades so they never blur together: spades jet black, clubs a dark
 * slate, hearts crimson, diamonds a warmer orange-red.
 */
@Suppress("UNUSED_PARAMETER")
@Composable
private fun suitTint(suit: Int, fourColor: Boolean = false, onDark: Boolean = false): Color = when {
    suit < 0 -> GoldBright
    suit == 1 -> Color(0xFFD0142E)
    suit == 0 -> Color(0xFFE2581C)
    suit == 2 -> if (onDark) Color(0xFFC6D2E8) else Color(0xFF2D3E60)
    else -> if (onDark) Ink else Color(0xFF0E0E0E)
}
