# Manille

**Manille on your phone**, against bots that play it properly. Everything runs
on the device -- the rules engine, the bots and their networks -- so there is
no account, no server and no connection needed.

### [Download Manille.apk](https://github.com/frigoow/manil-bot/releases/latest/download/Manille.apk)

Android 8.0 or newer, about 330 MB (almost all of it the bots' networks).
Older versions are on the [releases page](https://github.com/frigoow/manil-bot/releases).

To install: open the downloaded file on your phone and allow your browser or
file manager to install unknown apps when Android asks. New versions install
over the old one and keep your profiles, statistics and saved game.

---

## What's in it

- **Matches to 101** with the real calls -- trump or *mull* (no trump, ×2),
  *kloppen* (knock, ×2) and *terugkloppen* (knock back, ×2) -- or quick single
  deals.
- **Dutch or English**, Dutch by default, with the game's own words.
- **Hints**: the bot's own analysis of your position -- every legal card with
  how much it likes it -- and the same for declarations and knocks.
- The **last trick**, **statistics** per profile, a saved game you can resume,
  six table themes, and bots that react to the game -- including a partner who
  lets you know when you threw away a ten.

---

## The game

Manille is played by four players in two partnerships with a 32-card deck.
The card ranking is unusual: **the 10 (the *manille*) is the highest card**,
above the Ace.

| Card | 10 | A | K | Q | J | 9 | 8 | 7 |
|---|---|---|---|---|---|---|---|---|
| Points | 5 | 4 | 3 | 2 | 1 | 0 | 0 | 0 |

60 points are in play per deal. You must follow suit; if you cannot, you must
trump if able, and you must overtrump if able. The dealer's team chooses trump
or **mull** (no trump, ×2); the other team may **knock** (*kloppen*, ×2), and
the dealer's team may **knock back** (*terugkloppen*, ×2). A deal scores
`(winner's points − 30) × multiplier`, so 30–30 scores nothing. First team to
101 wins the match.

---

## How the bots play

Every move, for whichever bot is to play:

1. **Track** what is public: the cards played, who showed out of which suit,
   the calls.
2. **Sample worlds**: 400 deals of the hidden cards consistent with everything
   seen so far. Each is weighted by a **belief network** (which seat holds
   which unseen card) and by how well the play so far fits the rule policy.
   After a knock, worlds are reweighted by how likely each hand was to make
   that call.
3. **Rule tiers** propose a card: cash winners, keep the partner's trick, don't
   feed the opponents' trick, lead the suit that develops tricks -- with a
   separate rule set for no-trump play.
4. **Play-out**: for every legal card, each world is played to the end of the
   deal by a fast rule policy, which gives the expected final score of every
   card.
5. **Blend** each card's play-out value with a learned **Q-network** (60/40),
   and overrule the tiers' proposal only when the paired evidence across worlds
   clears a z-test.

A move takes about a tenth of a second on one CPU core.

The calls -- trump or mull, knock, knock back -- come from a separate
**decision layer**. It plays for the *match*, not for the points of one deal:
a margin network predicts how the deal will end for each option, and each
outcome is scored by the exact probability of winning the match from the
resulting score. The same hand mulls at 50–90 behind and plays safe at 90–50
ahead.

The trump/mull choice comes from a version of the decision layer refit to
these bots' own deals; the knocks from the original. On the 5000 deals of
22,000 where the two disagreed, the refit chose trump as well (match
equity −0.04 pp, and +1.4 points per deal) but knocked worse (−2.2 pp).

### How strong

Points per deal, from duplicate deals -- every deal replayed with the teams
swapped, so the comparison is how two players handled *the same cards*:

| The phone bot against | Points per deal (95% CI) |
|---|---|
| ISMCTS-5000, the classic search baseline | **+2.29** (4000 deals) |
| A DMC self-play network playing on its own | **+1.7** (4000 deals) |
| The same network, with real declarations and knocks (as in the app) | **+1.43** [+1.18, +1.67] (3000 deals) |
| SPARTA, a GPU search over that network | +0.14 [−0.22, +0.50] -- level (1200 deals) |

A deal splits 60 card points, so a point or two per deal is a large edge. The
bots have never been formally measured against people.

---

## Build it yourself

Needs the Android SDK with platform 37, NDK 30.0.16138531 and CMake 3.22.1
(Android Studio installs them) and a JDK 17+.

```bash
git clone https://github.com/frigoow/manil-bot.git && cd manil-bot
scripts/fetch_models.sh                 # the networks, into android/app/src/main/assets/models
cd android && ./build.sh assembleDebug  # -> app/build/outputs/apk/debug/app-debug.apk
```

`build.sh` uses Android Studio's bundled JDK on common Linux paths; set
`JAVA_HOME` to use another. The C++ can also be built and tested on a desktop,
no phone needed:

```bash
cmake -B build -S cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build
```

**Your own names.** The bots are called Jef, Mieke, Staf and friends. To give
them other names in your own builds, put this in `~/.gradle/gradle.properties`
(it never enters the repository):

```properties
manille.botNames=Ann,Bob,Cas,Dirk
manille.cafeThemeName=Our café
```

---

## Releases

Pushing a tag `vX.Y.Z` runs `.github/workflows/release.yml`: it fetches the
networks, builds and signs the APK, and publishes it as release `vX.Y.Z`,
which `releases/latest/download/Manille.apk` then points at. To release:

1. bump `versionCode` (+1) and `versionName` in `android/app/build.gradle.kts`,
2. commit, then `git tag vX.Y.Z && git push origin main vX.Y.Z`.

The workflow refuses a tag that does not match `versionName`. Signing uses
four repository secrets: `MANILLE_KEYSTORE_B64` (the keystore, base64),
`MANILLE_KEYSTORE_PASSWORD`, `MANILLE_KEY_ALIAS`, `MANILLE_KEY_PASSWORD`.
Android only installs an update signed with the same key as the installed
app, so that key must never change.

The networks are versioned separately as release `models-v1`
(`manille-app-models.zip`). A new set of networks goes into a new models
release, and `MODELS_TAG` in the workflow and `scripts/fetch_models.sh` moves
to it.

---

## Repository layout

```
android/        the app: Kotlin/Compose UI, JNI bridge, ONNX Runtime
cpp/            the rules engine, world samplers, rule agent and play-out
cpp/mobile/     the plain C interface the app calls through JNI
scripts/        fetch_models.sh
third_party/    the game interface from ismcsolver (MIT)
```

**The engine is the only rules implementation.** Legality, trick resolution
and scoring live in `cpp/manille_game.cpp`; the UI, the samplers and the bots
all ask it.

The networks were trained by self-play (Deep Monte Carlo: a Q-network trained
on full-episode returns, separate networks for trump and no-trump deals) in a
research codebase that is not part of this repository, and exported to ONNX.
The C++ rule agent is a port of a Python reference implementation from that
codebase, parity-tested card for card. Code comments that mention `learn/` or
`gui/` refer to that original code.

---

## License

Apache 2.0 -- see [LICENSE](LICENSE).

Includes the game interface header of
[ismcsolver](https://github.com/sfranzen/ismcsolver) by Steven Franzen (MIT,
`third_party/ismcsolver`). The app uses
[ONNX Runtime](https://github.com/microsoft/onnxruntime) (MIT) from Maven.
