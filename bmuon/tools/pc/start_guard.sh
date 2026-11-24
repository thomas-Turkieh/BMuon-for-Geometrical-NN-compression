#!/usr/bin/env bash
# macOS / Linux: starts the night guard in the background (it survives closing the terminal or
# VS Code) and keeps the Mac awake while it runs (caffeinate). Refuses if a guard already runs.
#   bash bmuon/tools/pc/start_guard.sh [PLAN] [OUT]
# macOS needs bash >= 4 and GNU coreutils (timeout):  brew install bash coreutils
# Keep the Mac plugged in and the lid open: closing the lid sleeps it anyway.
set -eu
cd "$(dirname "$0")/../../.."
PLAN="${1:-bmuon/experiments/wide_arch_runs.yml}"
OUT="${2:-bmuon/results/wide_arch_guard.out}"
if pgrep -f "night_guard[.]sh" > /dev/null; then
  echo "guard already running: $(pgrep -f 'night_guard[.]sh' | tr '\n' ' ')"; exit 1
fi
for d in /opt/homebrew/opt/coreutils/libexec/gnubin /usr/local/opt/coreutils/libexec/gnubin; do
  if [ -d "$d" ]; then PATH="$d:$PATH"; fi
done
for d in /opt/homebrew/bin /usr/local/bin; do
  if [ -x "$d/bash" ]; then PATH="$d:$PATH"; break; fi
done
export PATH
if ! bash -c '[ "${BASH_VERSINFO[0]}" -ge 4 ]'; then echo "bash >= 4 required: brew install bash"; exit 1; fi
if ! command -v timeout > /dev/null; then echo "timeout missing: brew install coreutils"; exit 1; fi
CAF=""
if command -v caffeinate > /dev/null; then CAF="caffeinate -ims"; fi
INTERVAL=600 WINDOW=1 nohup $CAF bash bmuon/experiments/night_guard.sh "$PLAN" >> "$OUT" 2>&1 < /dev/null &
echo "guard started, pid $!, log $OUT"
