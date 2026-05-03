#!/usr/bin/env bash
set -euo pipefail

RUFFLE_BIN="${RUFFLE_BIN:-/Applications/Ruffle.app/Contents/MacOS/ruffle}"
SWF="${1:-game/Falsh原版魔塔合集/魔塔V1.12.swf}"

if [[ ! -x "$RUFFLE_BIN" ]]; then
  echo "Ruffle executable not found: $RUFFLE_BIN" >&2
  exit 1
fi
if [[ ! -f "$SWF" ]]; then
  echo "SWF not found: $SWF" >&2
  exit 1
fi

echo "Launching Ruffle for visual compatibility check:"
echo "  $RUFFLE_BIN $SWF"
"$RUFFLE_BIN" "$SWF"
