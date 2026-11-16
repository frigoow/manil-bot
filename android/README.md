# Manille apps: Android and desktop

Kotlin and Compose Multiplatform render the game; a JNI adapter
(`cpp/jni/jni_bridge.cpp`) calls the C++ engine and the rule agent through
the plain C interface in `cpp/mobile/manille_mobile.h`; ONNX Runtime runs
the networks. The same game and screens run on Android (`app/`) and on
Windows, macOS (Apple Silicon) and Linux (`desktop/`). No server.

## Build

Needs the Android SDK with platform 37, NDK 30.0.16138531 and CMake 3.22.1
(Android Studio installs them), plus the exported model files in
`app/src/main/assets/models/`. Either unzip `manille-app-models.zip` from the
release there, or export them from the PyTorch weights at the repository root:

```bash
PYTHONPATH=. python -m learn.export_mobile --rule-assets --verify
```

Then:

```bash
cd android
./build.sh assembleDebug      # -> app/build/outputs/apk/debug/app-debug.apk
```

For the desktop app the same model files are enough, plus CMake and a C++
compiler (GCC or Clang; MSVC on Windows):

```bash
./gradlew :desktop:run                              # play it
./gradlew :desktop:packageDistributionForCurrentOS  # .deb / .dmg / .msi
```

The desktop build compiles the engine's JNI library with CMake
(`cpp/CMakeLists.txt`, `MANILLE_JNI=ON`) for the OS it runs on; packaging
needs a full JDK with `jpackage`. Without the NDK installed, add
`-Pmanille.skipAndroidApp=true` (leaves `:app` out). Each package can only be made on its own
OS: `.github/workflows/desktop.yml` builds all three. Profiles and saved
games live in `%APPDATA%\Manille`, `~/Library/Application Support/Manille` or
`~/.local/share/manille`.

`build.sh` refuses a JRE-only installation and picks the JDK bundled with
Android Studio on common Linux paths; set `JAVA_HOME` to a directory with
`bin/javac` to use another. The model files are generated assets and are not
in git.

## What runs where

| Piece | Where |
|---|---|
| Rules, legality, scoring | `cpp/manille_game.cpp` -- the only rules implementation; the UI asks it |
| Bot card play | `cpp/rule_agent.cpp` + `cpp/rule_playout.cpp` (+ `cpp/fast_policy.cpp`), called by `shared/.../RuleBot.kt` |
| World sampling | `cpp/sampler/` via `manille_mobile_sample_worlds` |
| Belief / Q / decision networks | ONNX Runtime: `app/.../BeliefNet.kt`, `QNet.kt`, `DeclarationNet.kt`, `AnnounceNet.kt` |
| The table: deals, calls, bots' turns, score | `shared/.../LocalSession.kt` behind `TableSession` (`Table.kt`) |
| Table on screen, hints, stats, profiles | `shared/.../ManilleUi.kt`, `GameController.kt`, `Persistence.kt` |
| JNI, the nets on ONNX Runtime, files (Android + desktop) | `shared/src/jvmShared/` (`com.manille.jvm`) |

The code is split in two Gradle modules:

- `shared/` (Kotlin Multiplatform + Compose Multiplatform): the game session,
  the bots' Kotlin side, storage and every screen. It has no Android or JVM
  calls: `shared/.../Platform.kt` lists what it needs from the platform
  (the C++ engine, the nets, files), and it also compiles for iOS
  (`./gradlew :shared:compileKotlinIosArm64` works on Linux too).
- `shared/src/jvmShared`: what Android and the desktop share as JVMs -- JNI
  to the C++ engine, the nets on ONNX Runtime, file storage (`JvmPlatform`);
  compiled into both targets against each one's ONNX Runtime.
- `app/`: the Android shell -- `MainActivity`, `AndroidPlatform` (where the
  models and files are) and the native build.
- `desktop/`: the desktop shell -- `Main.kt`, the CMake build of the JNI
  library, packaging.

The game is split in two: a **session** (`TableSession`; offline,
`LocalSession`) is the referee -- it deals, checks every move, runs the bots
and keeps the score, and reports what happens as events with a `TableView`
(what seat 0 may see: never another hand). `GameController` turns those into
the screen: messages, the held trick, the bots' reactions, statistics and the
saved game. A networked table will be another `TableSession`.

`./gradlew :shared:testAndroidHostTest` runs the shared tests (among them:
files written by earlier versions still load); `./gradlew :shared:desktopTest`
runs them on the desktop JVM too, plus whole games through `LocalSession` with
the real engine and nets.

The local player is seat 0. A move is sampled, weighed and played out on a
worker thread from a copy of the game, so the UI never blocks on the bot.
If a model file is missing the app says which and how to export it, rather
than silently playing a weaker bot.
