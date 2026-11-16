import org.jetbrains.compose.desktop.application.dsl.TargetFormat
import javax.inject.Inject

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
    implementation(project(":shared"))
    implementation(compose.desktop.currentOs)
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-swing:1.11.0")
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

/** Runs CMake on cpp/ with MANILLE_JNI=ON and collects manille_mobile for this OS. */
abstract class CMakeJniLibrary @Inject constructor(private val exec: ExecOperations) : DefaultTask() {
    @get:InputDirectory @get:PathSensitive(PathSensitivity.RELATIVE) abstract val sources: DirectoryProperty
    @get:InputDirectory @get:PathSensitive(PathSensitivity.RELATIVE) abstract val thirdParty: DirectoryProperty
    @get:Internal abstract val workDir: DirectoryProperty
    @get:OutputDirectory abstract val outputDir: DirectoryProperty
    @get:Input abstract val cmake: Property<String>
    @get:Input abstract val jniInclude: ListProperty<String>

    @TaskAction fun build() {
        val work = workDir.get().asFile
        val ninja = File(cmake.get()).parentFile?.resolve("ninja")?.takeIf { it.canExecute() }
        exec.exec {
            commandLine(buildList {
                add(cmake.get()); add("-S"); add(sources.get().asFile.path); add("-B"); add(work.path)
                if (ninja != null) { add("-G"); add("Ninja"); add("-DCMAKE_MAKE_PROGRAM=${ninja.path}") }
                add("-DCMAKE_BUILD_TYPE=Release"); add("-DMANILLE_JNI=ON")
                add("-DMANILLE_JNI_OUT=${outputDir.get().asFile.path}")
                add("-DJNI_INCLUDE_DIRS=${jniInclude.get().joinToString(";")}")
            })
        }
        exec.exec { commandLine(cmake.get(), "--build", work.path, "--config", "Release", "--target", "manille_mobile", "--parallel") }
    }
}

val isWindows = System.getProperty("os.name").lowercase().contains("win")

/** cmake: -Pmanille.cmake, else PATH, else the newest one in the Android SDK. */
fun findCmake(): String {
    (findProperty("manille.cmake") as String?)?.let { return it }
    val exe = if (isWindows) "cmake.exe" else "cmake"
    System.getenv("PATH").orEmpty().split(File.pathSeparator)
        .map { File(it, exe) }.firstOrNull { it.canExecute() }?.let { return it.path }
    val sdk = System.getenv("ANDROID_HOME") ?: "${System.getProperty("user.home")}/Android/Sdk"
    return File(sdk, "cmake").listFiles().orEmpty().sortedByDescending { it.name }
        .map { File(it, "bin/$exe") }.firstOrNull { it.canExecute() }?.path
        ?: throw GradleException("cmake not found: install it, or pass -Pmanille.cmake=/path/to/cmake")
}

/**
 * jni.h: from a full JDK (the one running Gradle, JAVA_HOME, the usual
 * install places). A runtime-only JDK such as Android Studio's has no
 * headers; then the NDK's self-contained jni.h, which has the same ABI.
 */
fun findJniInclude(): List<String> {
    val homes = buildList {
        add(File(System.getProperty("java.home")))
        System.getenv("JAVA_HOME")?.let { add(File(it)) }
        addAll(File(System.getProperty("user.home"), ".jdks").listFiles().orEmpty())
        addAll(File("/usr/lib/jvm").listFiles().orEmpty())
        addAll(File("/Library/Java/JavaVirtualMachines").listFiles().orEmpty().map { File(it, "Contents/Home") })
    }
    homes.firstOrNull { File(it, "include/jni.h").isFile }?.let { home ->
        val include = File(home, "include")
        return listOf(include.path) + include.listFiles().orEmpty().filter { it.isDirectory }.map { it.path }
    }
    val sdk = System.getenv("ANDROID_HOME") ?: "${System.getProperty("user.home")}/Android/Sdk"
    val ndkJni = File(sdk, "ndk").listFiles().orEmpty().sortedByDescending { it.name }
        .flatMap { File(it, "toolchains/llvm/prebuilt").listFiles().orEmpty().toList() }
        .map { File(it, "sysroot/usr/include/jni.h") }.firstOrNull { it.isFile }
        ?: throw GradleException("No jni.h: install a full JDK (e.g. openjdk-21-jdk-headless) or set JAVA_HOME to one.")
    // only jni.h itself: the rest of the NDK sysroot is Android's libc
    val dir = layout.buildDirectory.dir("jni-include").get().asFile
    dir.mkdirs(); ndkJni.copyTo(File(dir, "jni.h"), overwrite = true)
    return listOf(dir.path)
}

val buildNative by tasks.registering(CMakeJniLibrary::class) {
    description = "Builds the C++ engine's JNI library for this OS."
    sources.set(File(repoRoot, "cpp"))
    thirdParty.set(File(repoRoot, "third_party"))
    workDir.set(layout.buildDirectory.dir("cmake"))
    outputDir.set(appResources.map { it.dir("common/native") })
    cmake.set(provider { findCmake() })
    jniInclude.set(provider { findJniInclude() })
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
