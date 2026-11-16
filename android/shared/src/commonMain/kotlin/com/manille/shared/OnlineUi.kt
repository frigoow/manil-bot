package com.manille.shared

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/*
 * The online screens, between the lobby and the table: the server to play
 * on, a new table or one to join by its code, and the chairs at that table
 * until the host starts. Once a deal is on, the usual table takes over.
 */

private val Warn = Color(0xFFE07A6E)

@Composable
internal fun OnlineScreen(controller: GameController) {
    val theme = LocalAppTheme.current
    val state = controller.online ?: return
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 24.dp, vertical = 30.dp),
        horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center,
    ) {
        Text("ONLINE", color = theme.accentBright, fontSize = 34.sp, fontWeight = FontWeight.ExtraBold, letterSpacing = 6.sp)
        Text(tr("Play with friends at one table; bots take the free chairs.", "Speel met vrienden aan één tafel; bots nemen de vrije stoelen."),
            color = theme.inkDim, fontSize = 13.sp, textAlign = TextAlign.Center)
        Spacer(Modifier.height(22.dp))
        val info = state.info
        when {
            state.status == null -> ServerForm(controller)
            info == null -> TablePicker(controller, state.status)
            else -> TableChairs(controller, info)
        }
        state.problem?.let {
            Spacer(Modifier.height(12.dp))
            Text(it, color = Warn, fontSize = 13.sp, textAlign = TextAlign.Center)
        }
        Spacer(Modifier.height(22.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (state.status != null && info == null) TextButton(onClick = controller::disconnect) {
                Text(tr("Other server", "Andere server"), color = theme.inkDim)
            }
            TextButton(onClick = controller::closeOnline) { Text(tr("Back to offline play", "Terug naar offline spelen"), color = theme.inkDim) }
        }
    }
}

@Composable
private fun OnlinePanel(content: @Composable () -> Unit) {
    val theme = LocalAppTheme.current
    Surface(Modifier.widthIn(max = 480.dp).fillMaxWidth(), shape = RoundedCornerShape(16.dp), color = theme.panel,
        border = BorderStroke(1.dp, theme.accent.copy(alpha = .3f))) {
        Column(Modifier.padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) { content() }
    }
}

@Composable
private fun fieldColors() = LocalAppTheme.current.let { t ->
    OutlinedTextFieldDefaults.colors(focusedTextColor = t.ink, unfocusedTextColor = t.ink, focusedBorderColor = t.accentBright,
        unfocusedBorderColor = t.accent.copy(alpha = .5f), focusedLabelColor = t.accentBright, unfocusedLabelColor = t.inkDim, cursorColor = t.accentBright)
}

@Composable
private fun GoldButton(text: String, modifier: Modifier = Modifier, enabled: Boolean = true, onClick: () -> Unit) {
    val theme = LocalAppTheme.current
    Button(onClick = onClick, enabled = enabled, modifier = modifier,
        colors = ButtonDefaults.buttonColors(containerColor = theme.accentBright, contentColor = theme.rim)) {
        Text(text, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center)
    }
}

@Composable
private fun ServerForm(controller: GameController) = OnlinePanel {
    val theme = LocalAppTheme.current
    var url by remember { mutableStateOf(controller.serverUrl) }
    var name by remember { mutableStateOf(controller.onlineName) }
    OutlinedTextField(name, { name = it.take(20) }, singleLine = true, colors = fieldColors(), modifier = Modifier.fillMaxWidth(),
        label = { Text(tr("Your name at the table", "Je naam aan tafel")) })
    Spacer(Modifier.height(8.dp))
    OutlinedTextField(url, { url = it.trim() }, singleLine = true, colors = fieldColors(), modifier = Modifier.fillMaxWidth(),
        label = { Text(tr("Server address", "Serveradres")) }, placeholder = { Text("ws://192.168.1.20:8080/play", color = theme.inkDim) })
    Spacer(Modifier.height(10.dp))
    val valid = url.startsWith("ws://") || url.startsWith("wss://")
    GoldButton(tr("Connect", "Verbinden"), Modifier.fillMaxWidth(), enabled = valid && name.isNotBlank()) { controller.connect(url, name) }
}

@Composable
private fun TablePicker(controller: GameController, status: RemoteSession.Status) = OnlinePanel {
    val theme = LocalAppTheme.current
    when (status) {
        RemoteSession.Status.CONNECTING -> Text(tr("Connecting to ${controller.serverUrl}…", "Verbinden met ${controller.serverUrl}…"), color = theme.inkDim, fontSize = 13.sp)
        RemoteSession.Status.OFFLINE -> Text(tr("The server cannot be reached; trying again…", "De server is niet bereikbaar; opnieuw proberen…"), color = Warn, fontSize = 13.sp, textAlign = TextAlign.Center)
        RemoteSession.Status.ONLINE -> {
            Text(tr("NEW TABLE", "NIEUWE TAFEL"), color = theme.accentBright, fontWeight = FontWeight.Bold, letterSpacing = 2.sp, fontSize = 12.sp)
            Spacer(Modifier.height(8.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                GoldButton(tr("Quick game", "Snel spel"), Modifier.weight(1f)) { controller.createTable(false) }
                GoldButton(tr("Match to 101", "Match tot 101"), Modifier.weight(1f)) { controller.createTable(true) }
            }
            Spacer(Modifier.height(18.dp))
            Text(tr("JOIN A TABLE", "AAN TAFEL GAAN"), color = theme.accentBright, fontWeight = FontWeight.Bold, letterSpacing = 2.sp, fontSize = 12.sp)
            Spacer(Modifier.height(8.dp))
            var code by remember { mutableStateOf("") }
            Row(verticalAlignment = Alignment.CenterVertically) {
                OutlinedTextField(code, { code = it.uppercase().filter { c -> c.isLetterOrDigit() }.take(5) }, singleLine = true,
                    colors = fieldColors(), modifier = Modifier.weight(1f), label = { Text(tr("Table code", "Tafelcode")) },
                    keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Characters))
                Spacer(Modifier.width(10.dp))
                GoldButton(tr("Join", "Meedoen"), enabled = code.length == 5) { controller.joinTable(code) }
            }
        }
    }
}

/** The four chairs: yours at the bottom, your partner across, the opponents left and right. */
@Composable
private fun TableChairs(controller: GameController, info: ServerMessage.TableInfo) = OnlinePanel {
    val theme = LocalAppTheme.current
    Text(tr("TABLE CODE", "TAFELCODE"), color = theme.inkDim, fontSize = 11.sp, letterSpacing = 2.sp)
    Text(info.code, color = theme.accentBright, fontSize = 40.sp, fontWeight = FontWeight.ExtraBold, fontFamily = FontFamily.Monospace, letterSpacing = 8.sp)
    Text(tr("Share it: friends join with this code.", "Deel hem: vrienden doen mee met deze code.") + " · " +
        (if (info.match) tr("match to 101", "match tot 101") else tr("quick game", "snel spel")), color = theme.inkDim, fontSize = 12.sp, textAlign = TextAlign.Center)
    Spacer(Modifier.height(16.dp))
    val seat = { rel: Int -> (info.you + rel) % 4 }
    Chair(info, seat(2), tr("partner", "partner"), !info.started, controller)
    Spacer(Modifier.height(8.dp))
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Chair(info, seat(1), tr("opponent", "tegenstander"), !info.started, controller, Modifier.weight(1f))
        Chair(info, seat(3), tr("opponent", "tegenstander"), !info.started, controller, Modifier.weight(1f))
    }
    Spacer(Modifier.height(8.dp))
    Chair(info, info.you, tr("you", "jij"), false, controller)
    Spacer(Modifier.height(16.dp))
    val players = info.seats.count { !it.bot }
    when {
        info.started -> Text(tr("The game is on…", "Het spel is bezig…"), color = theme.inkDim, fontSize = 13.sp)
        info.host -> GoldButton(
            if (players == 4) tr("Start", "Starten") else tr("Start · bots take ${4 - players} chair${if (players == 3) "" else "s"}", "Starten · bots nemen ${4 - players} stoel${if (players == 3) "" else "en"}"),
            Modifier.fillMaxWidth()) { controller.startOnline() }
        else -> Text(tr("Waiting for the host to start…", "Wachten tot de gastheer start…"), color = theme.inkDim, fontSize = 13.sp)
    }
    Spacer(Modifier.height(6.dp))
    TextButton(onClick = controller::lobby) { Text(tr("Leave table", "Tafel verlaten"), color = theme.inkDim) }
}

@Composable
private fun Chair(info: ServerMessage.TableInfo, seat: Int, role: String, canSit: Boolean, controller: GameController, modifier: Modifier = Modifier.fillMaxWidth(.6f)) {
    val theme = LocalAppTheme.current
    val chair = info.seats[seat]
    val free = chair.name == null
    val mine = seat == info.you
    Surface(
        modifier.let { if (free && canSit) it.clickable { controller.sit(seat) } else it },
        shape = RoundedCornerShape(12.dp),
        color = if (mine) theme.accent.copy(alpha = .22f) else Color.Black.copy(alpha = .22f),
        border = BorderStroke(1.dp, if (free) theme.inkDim.copy(alpha = .4f) else theme.accent.copy(alpha = .6f)),
    ) {
        Column(Modifier.padding(horizontal = 12.dp, vertical = 8.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Text(role.uppercase(), color = theme.inkDim, fontSize = 10.sp, letterSpacing = 1.5.sp)
            Text(
                when {
                    free -> if (canSit) tr("Free · tap to sit", "Vrij · tik om te zitten") else tr("Bot", "Bot")
                    chair.bot -> "${chair.name} 🤖"
                    chair.away -> "${chair.name} · " + tr("away", "weg")
                    else -> chair.name!!
                },
                color = if (free) theme.inkDim else theme.ink, fontSize = 15.sp, fontWeight = if (free) FontWeight.Normal else FontWeight.Bold,
                textAlign = TextAlign.Center,
            )
        }
    }
}
