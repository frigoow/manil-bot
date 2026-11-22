plugins {
    id("org.jetbrains.kotlin.multiplatform")
    id("com.android.kotlin.multiplatform.library")
    id("org.jetbrains.kotlin.plugin.compose")
}

// Everything that is not platform plumbing: the game session, the bots'
// Kotlin side, storage formats and the Compose screens. Platform code (the
// C++ engine bridge, ONNX Runtime, files) comes in through Platform.kt's
// interfaces. The iOS targets are declared so common code stays common; they
// only compile on a Mac.
//
// Android and the desktop are both JVMs: they share src/jvmShared (JNI to the
// C++ engine, the nets on ONNX Runtime, file storage), compiled into each
// target against that target's ONNX Runtime build.
kotlin {
    androidLibrary {
        namespace = "com.manille.shared"
        compileSdk = 37
        minSdk = 26
        withHostTestBuilder {}
    }
    jvm("desktop")
    iosArm64()
    iosSimulatorArm64()

    sourceSets {
        commonMain.dependencies {
            api(project(":platform"))
            api("org.jetbrains.compose.runtime:runtime:1.12.2")
            api("org.jetbrains.compose.foundation:foundation:1.12.2")
            api("org.jetbrains.compose.ui:ui:1.12.2")
            api("org.jetbrains.compose.material3:material3:1.12.0-alpha03")
            implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.11.0")
            implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.11.0")
        }
        androidMain {
            kotlin.srcDir("src/jvmShared/kotlin")
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.11.0")
                // 1.27.0: first tested release that honours mlas.disable_kleidiai for Gemm and
                // GRU (see OrtSessions.kt). 1.28.0 - 1.32.0 crashed at startup under the
                // emulator's arm64 translation, so test on a device before moving past it.
                implementation("com.microsoft.onnxruntime:onnxruntime-android:1.27.0")
            }
        }
        val desktopMain by getting {
            kotlin.srcDir("src/jvmShared/kotlin")
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-swing:1.11.0")
                // the desktop build: native runtimes for Windows, macOS and Linux in one jar
                implementation("com.microsoft.onnxruntime:onnxruntime:1.27.0")
            }
        }
        commonTest.dependencies {
            implementation(kotlin("test"))
        }
    }
}

// The desktop tests play real games: the C++ engine's JNI library and the
// nets, as the desktop app ships them (android/desktop builds both).
tasks.matching { it.name == "desktopTest" }.configureEach {
    dependsOn(":desktop:buildNative", ":desktop:syncModels")
    (this as Test).systemProperty("manille.resources", rootProject.file("desktop/build/appResources/common").path)
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}
