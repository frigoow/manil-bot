# Manille Android app

Kotlin/Jetpack Compose renders the game; a JNI adapter
(`app/src/main/cpp/jni_bridge.cpp`) calls the C++ engine and the rule agent
through the plain C interface in `cpp/mobile/manille_mobile.h`; ONNX Runtime
Android runs the networks. No server, no desktop process.

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
| Table, hints, stats, profiles | `shared/.../ManilleUi.kt`, `GameController.kt`, `Persistence.kt` |

The code is split in two Gradle modules:

- `shared/` (Kotlin Multiplatform + Compose Multiplatform): the game session,
  the bots' Kotlin side, storage and every screen. It has no Android or JVM
  calls: `shared/.../Platform.kt` lists what it needs from the platform
  (the C++ engine, the nets, files), and it also compiles for iOS
  (`./gradlew :shared:compileKotlinIosArm64` works on Linux too).
- `app/`: the Android shell -- `MainActivity`, `AndroidPlatform` (JNI to the
  C++ engine, ONNX Runtime, file storage) and the native build.

`./gradlew :shared:testAndroidHostTest` runs the shared tests (among them:
files written by earlier versions still load).

The local player is seat 0. A move is sampled, weighed and played out on a
worker thread from a copy of the game, so the UI never blocks on the bot.
If a model file is missing the app says which and how to export it, rather
than silently playing a weaker bot.
