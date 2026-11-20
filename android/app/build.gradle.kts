plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.plugin.compose")
}

// Personal touches stay out of git: a local build can rename the bots and the
// café theme by setting, in ~/.gradle/gradle.properties,
//   manille.botNames=Ann,Bob,Cas,...      (at least three, comma separated)
//   manille.cafeThemeName=Our café
// Released builds use the defaults below.
val botNames = (findProperty("manille.botNames") as String?)
    ?: "Jef,Mieke,Staf,Lieve,Rik,Greet,Fons,Marie,Warre,Leen"
val cafeThemeName = (findProperty("manille.cafeThemeName") as String?) ?: "Café"

// Release signing comes from the environment (the release workflow decodes
// the keystore from a repository secret). Without it a release build falls
// back to the debug key, which is fine for local testing.
val releaseKeystore = System.getenv("MANILLE_KEYSTORE")?.let { file(it) }?.takeIf { it.exists() }

android {
    namespace = "com.manille.androidapp"
    compileSdk = 37
    ndkVersion = "30.0.16138531"

    defaultConfig {
        applicationId = "com.manille.androidapp"
        minSdk = 26
        targetSdk = 37
        versionCode = 15
        versionName = "1.2.2"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        buildConfigField("String", "BOT_NAMES", "\"$botNames\"")
        buildConfigField("String", "CAFE_THEME_NAME", "\"$cafeThemeName\"")
        // The connected Android device is arm64; avoiding three unused native
        // runtimes keeps the debug APK practical to install over USB.
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++17", "-O3")
            }
        }
    }

    signingConfigs {
        if (releaseKeystore != null) create("release") {
            storeFile = releaseKeystore
            storePassword = System.getenv("MANILLE_KEYSTORE_PASSWORD")
            keyAlias = System.getenv("MANILLE_KEY_ALIAS")
            keyPassword = System.getenv("MANILLE_KEY_PASSWORD")
        }
    }

    buildTypes {
        release {
            // No R8: the JNI bridge reaches Kotlin by name, and the APK is
            // almost all model files anyway.
            isMinifyEnabled = false
            signingConfig = signingConfigs.findByName("release") ?: signingConfigs.getByName("debug")
        }
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
        }
    }
    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
    }
    androidResources {
        noCompress += "onnx"
    }
}

dependencies {
    // the game, its screens and Compose come in through :shared
    implementation(project(":shared"))
    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.11.0")
    // ONNX Runtime (onnxruntime-android) comes in through :shared, with the nets
}
