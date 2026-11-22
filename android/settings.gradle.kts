pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "Manille"
// The shared building blocks (profiles, themes, cards, the table...) are the
// :platform module of cardbot-framework, a git submodule at the repository root.
include(":platform")
project(":platform").projectDir = file("../cardbot-framework/platform")
include(":manille", ":desktop")
// The Android app needs the NDK just to be configured; desktop-only builds
// (CI's tests, the desktop packages) leave it out: -Pmanille.skipAndroidApp=true
if (providers.gradleProperty("manille.skipAndroidApp").orNull != "true") include(":app")
