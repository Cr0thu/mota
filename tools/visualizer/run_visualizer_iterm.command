#!/bin/zsh
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
RUNNER="$ROOT_DIR/tools/visualizer/run_visualizer.command"

if [[ ! -d /Applications/iTerm.app ]]; then
  echo "[visualizer] /Applications/iTerm.app not found."
  echo "[visualizer] Falling back to direct launch."
  exec "$RUNNER"
fi

osascript <<OSA
tell application "iTerm"
  activate
  set newWindow to (create window with default profile)
  tell current session of newWindow
    write text "cd " & quoted form of "$ROOT_DIR" & "; " & quoted form of "$RUNNER"
  end tell
end tell
OSA
