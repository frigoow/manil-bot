import org.jetbrains.compose.desktop.application.dsl.TargetFormat

plugins {
    id("org.jetbrains.kotlin.jvm")
    id("org.jetbrains.kotlin.plugin.compose")
    id("org.jetbrains.compose")
}

// The desktop shell (Windows, macOS on Apple Silicon, Linux): the shared game
// and screens from :shared, the C++ engine as a JNI library built here with
// CMake for the OS that runs the build, and the nets from the Android assets.
//
//   ./gradlew :desktop:run                  play it
//   ./gradlew :desktop:packageDistributionForCurrentOS
//                                           .msi / .dmg / .deb (needs a full JDK with jpackage)
//
// A package can only be made on its own OS: .github/workflows/desktop.yml
// builds all three. The same Gradle properties as the Android app rename the
// bots and the café theme (app/build.gradle.kts).

val repoRoot: File = rootDir.parentFile
val appVersion: String = Regex("""versionName = "([^"]+)"""")
    .find(file("../app/build.gradle.kts").readText())!!.groupValues[1]
val botNames = (findProperty("manille.botNames") as String?)
    ?: "Jef,Mieke,Staf,Lieve,Rik,Greet,Fons,Marie,Warre,Leen"
val cafeThemeName = (findProperty("manille.cafeThemeName") as String?) ?: "Café"

dependencies {
    implementation(project(":manille"))
    implementation(compose.desktop.currentOs)
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-swing:1.11.0")
    testImplementation(kotlin("test"))
}

// ---- the app's resources: models and the JNI library ----------------------
// Compose ships build/appResources/common with the app and tells it where
// through the compose.application.resources.dir system property.
val appResources = layout.buildDirectory.dir("appResources")

val syncModels by tasks.registering(Sync::class) {
    description = "Copies the nets from the Android assets (scripts/fetch_models.sh) for the desktop app."
    from(file("../app/src/main/assets/models")) { exclude("README.md") }
    into(appResources.map { it.dir("common/models") })
}

val buildNative by tasks.registering(CMakeJniLibrary::class) {
    description = "Builds the C++ engine's JNI library for this OS."
    sources.set(File(repoRoot, "cpp"))
    extraSources.set(File(repoRoot, "third_party"))
    workDir.set(layout.buildDirectory.dir("cmake"))
    outputDir.set(appResources.map { it.dir("common/native") })
    cmake.set(provider { findCmake() })
    jniInclude.set(provider { findJniInclude() })
    target.set("manille_mobile")
    definitions.set(appResources.map { listOf("MANILLE_JNI=ON", "MANILLE_JNI_OUT=${it.dir("common/native").asFile.path}") })
}

compose.desktop {
    application {
        mainClass = "com.manille.desktop.MainKt"
        jvmArgs += listOf(
            "-Dmanille.botNames=$botNames", "-Dmanille.cafeThemeName=$cafeThemeName",
            // System.load of the engine and Skia's own libraries (Java 24+ warns without it)
            "--enable-native-access=ALL-UNNAMED",
            // Memory: the game's own heap is small (~20 MB live). The serial
            // collector has the smallest footprint and gives free heap back
            // to the system; without a cap the JVM starts at 1/64 of the RAM
            // and may grow to a quarter of it.
            "-XX:+UseSerialGC", "-Xms32m", "-Xmx512m",
            "-XX:MinHeapFreeRatio=10", "-XX:MaxHeapFreeRatio=30",
            // Skia's GPU cache (Direct3D on Windows, OpenGL/Metal elsewhere):
            // 256 MB by default, far more than a card table needs
            "-Dskiko.gpu.resourceCacheLimit=64M",
        )
        nativeDistributions {
            targetFormats(TargetFormat.Msi, TargetFormat.Dmg, TargetFormat.Deb)
            packageName = "Manille"
            packageVersion = appVersion
            description = "Manille, the Flemish card game, against strong bots"
            vendor = "Manille"
            appResourcesRootDir.set(appResources)
            modules("jdk.unsupported")
            linux { iconFile.set(file("icons/manille.png")) }
            windows {
                iconFile.set(file("icons/manille.ico"))
                menuGroup = "Manille"
                shortcut = true
                // fixed, so a new version installs over the old one
                upgradeUuid = "6f1d3a9e-2c47-4b8e-9a51-3e0c7d2b8f14"
            }
            macOS {
                iconFile.set(file("icons/manille.icns"))
                bundleID = "com.manille.desktop"
            }
        }
    }
}

tasks.matching { it.name == "prepareAppResources" }.configureEach { dependsOn(syncModels, buildNative) }

// The app offscreen, saved as PNGs (ScreenshotTest): ./gradlew :desktop:test -Pmanille.screenshots=/some/dir
tasks.test {
    dependsOn(syncModels, buildNative)
    systemProperty("compose.application.resources.dir", appResources.get().dir("common").asFile.path)
    systemProperty("manille.screenshots", (findProperty("manille.screenshots") as String?) ?: "")
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    outputs.upToDateWhen { false }
}
