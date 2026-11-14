#!/usr/bin/env bash
set -euo pipefail

# Gradle needs a JDK, not merely a JRE. Prefer an explicitly configured JDK,
# then the bundled Android Studio JBR on common Linux installations.
candidate_jdks=()
if [[ -n "${JAVA_HOME:-}" ]]; then
    candidate_jdks+=("$JAVA_HOME")
fi
if [[ -n "${ANDROID_STUDIO_JDK:-}" ]]; then
    candidate_jdks+=("$ANDROID_STUDIO_JDK")
fi
candidate_jdks+=(
    "/snap/android-studio/current/jbr"
    "/opt/android-studio/jbr"
)

for candidate_jdk in "${candidate_jdks[@]}"; do
    if [[ -x "$candidate_jdk/bin/javac" ]]; then
        export JAVA_HOME="$candidate_jdk"
        break
    fi
done

if [[ ! -x "${JAVA_HOME:-}/bin/javac" ]]; then
    echo "A full JDK is required (javac was not found)." >&2
    echo "Install a JDK or set JAVA_HOME to one; Android Studio's bundled JBR works." >&2
    exit 1
fi

if [[ -z "${ANDROID_HOME:-}" && -d "$HOME/Android/Sdk" ]]; then
    export ANDROID_HOME="$HOME/Android/Sdk"
fi

exec ./gradlew "$@"
