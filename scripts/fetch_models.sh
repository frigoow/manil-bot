#!/usr/bin/env bash
# Download the app's networks (not in git: ~320 MB) into the Android assets.
#   scripts/fetch_models.sh            # release models-v2
#   MODELS_TAG=models-v1 scripts/fetch_models.sh
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${MODELS_TAG:-models-v2}"
URL="https://github.com/frigoow/manil-bot/releases/download/$TAG/manille-app-models.zip"
DEST=android/app/src/main/assets/models
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "Fetching $URL"
curl -fL --retry 3 -o "$TMP/models.zip" "$URL"
unzip -o -q "$TMP/models.zip" -d "$DEST"
echo "Models in $DEST:"
ls -1 "$DEST"
