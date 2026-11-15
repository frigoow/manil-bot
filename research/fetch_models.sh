#!/usr/bin/env bash
# Download the research networks (not in git: ~480 MB) into research/artifacts
# and research/runs, the paths the code and its defaults use.
#   research/fetch_models.sh                  # release research-models-v1
#   MODELS_TAG=research-models-v2 research/fetch_models.sh
set -euo pipefail
cd "$(dirname "$0")"
TAG="${MODELS_TAG:-research-models-v1}"
URL="https://github.com/frigoow/manil-bot/releases/download/$TAG/manille-research-models.zip"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "Fetching $URL"
curl -fL --retry 3 -o "$TMP/models.zip" "$URL"
unzip -o -q "$TMP/models.zip" -d .
ls -1 artifacts runs
