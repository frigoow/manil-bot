plugins {
    id("org.jetbrains.kotlin.jvm")
    id("org.jetbrains.kotlin.plugin.serialization")
    application
}

// The online game's server: the tables, run by the same referee, rules
// engine (the C++ JNI library) and bots as the app, behind one WebSocket
// (/play, messages in :shared's Protocol.kt). Tables live in memory.
//
//   ./gradlew :server:run      a server on port 8080 with the desktop build's
//                               engine and nets (desktop/build/appResources)
//
// Settings come from the environment: PORT, MANILLE_RESOURCES (a folder with
// native/ and models/), MANILLE_GRACE_SECONDS (how long a dropped player's
// chair waits before a bot plays it), MANILLE_DB (the SQLite file; default
// manille.db in the working directory), MANILLE_ADMIN_TOKEN (turns on the
// admin API; see Api.kt).

val ktor = "3.6.1"

dependencies {
    implementation(project(":shared"))
    implementation("io.ktor:ktor-server-core:$ktor")
    implementation("io.ktor:ktor-server-cio:$ktor")
    implementation("io.ktor:ktor-server-websockets:$ktor")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.11.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.11.0")
    implementation("org.slf4j:slf4j-simple:2.0.17")
    // players, statistics and games: one SQLite file (natives for Linux, Windows, macOS, x64 and arm64)
    implementation("org.xerial:sqlite-jdbc:3.49.1.0")
    testImplementation(kotlin("test"))
    testImplementation("io.ktor:ktor-client-cio:$ktor")
}

application {
    mainClass = "com.manille.server.MainKt"
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED")
}

val desktopResources = rootProject.file("desktop/build/appResources/common")

tasks.named<JavaExec>("run") {
    dependsOn(":desktop:buildNative", ":desktop:syncModels")
    environment("MANILLE_RESOURCES", desktopResources.path)
}

tasks.test {
    dependsOn(":desktop:buildNative", ":desktop:syncModels")
    systemProperty("manille.resources", desktopResources.path)
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}
