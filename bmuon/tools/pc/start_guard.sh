#!/usr/bin/env bash
# macOS / Linux: starts the night guard in the background (it survives closing the terminal or
# VS Code) and keeps the Mac awake while it runs (caffeinate). Refuses if a guard already runs.
#   bash bmuon/tools/pc/start_guard.sh [PLAN] [OUT]
# Needs: bash >= 4 (macOS ships 3.2: bash bmuon/tools/pc/mac_install_bash.sh puts one in
# ~/.local/bin), a python with pyyaml (the venv ~/.venvs/ea, or VENV=...), and a timeout command
# (GNU coreutils if present, else the stand-in in bmuon/tools/pc/macbin).
# Keep the Mac plugged in and the lid open: closing the lid sleeps it anyway.
set -eu
cd "$(dirname "$0")/../../.."
PLAN="${1:-bmuon/experiments/wide_arch_runs.yml}"
OUT="${2:-bmuon/results/wide_arch_guard.out}"
if pgrep -f "night_guard[.]sh" > /dev/null; then
  echo "guard already running: $(pgrep -f 'night_guard[.]sh' | tr '\n' ' ')"; exit 1
fi
# python of the guard (reads the plan with yaml): the project venv first
VENV="${VENV:-$HOME/.venvs/ea}"
if [ -x "$VENV/bin/python" ]; then PATH="$VENV/bin:$PATH"; fi
# bash >= 4: ~/.local/bin (mac_install_bash.sh), else Homebrew if present
for d in /usr/local/bin /opt/homebrew/bin "$HOME/.local/bin"; do
  if [ -x "$d/bash" ] && "$d/bash" -c '[ "${BASH_VERSINFO[0]}" -ge 4 ]' 2> /dev/null; then PATH="$d:$PATH"; fi
done
# timeout: GNU coreutils if installed, else the python stand-in
for d in /opt/homebrew/opt/coreutils/libexec/gnubin /usr/local/opt/coreutils/libexec/gnubin; do
  if [ -d "$d" ]; then PATH="$d:$PATH"; fi
done
export PATH
if ! command -v timeout > /dev/null; then
  chmod +x bmuon/tools/pc/macbin/timeout; PATH="$PWD/bmuon/tools/pc/macbin:$PATH"
fi
BASH4=$(command -v bash)
if ! "$BASH4" -c '[ "${BASH_VERSINFO[0]}" -ge 4 ]'; then
  echo "bash >= 4 required: bash bmuon/tools/pc/mac_install_bash.sh"; exit 1
fi
if ! python -c 'import yaml' 2> /dev/null; then
  echo "python with pyyaml required (venv $VENV not found or incomplete)"; exit 1
fi
if ! timeout 5 true; then echo "timeout command not working"; exit 1; fi
CAF=""
if command -v caffeinate > /dev/null; then CAF="caffeinate -ims"; fi
INTERVAL=600 WINDOW=1 nohup $CAF "$BASH4" bmuon/experiments/night_guard.sh "$PLAN" >> "$OUT" 2>&1 < /dev/null &
echo "guard started, pid $!, bash $("$BASH4" -c 'echo $BASH_VERSION'), python $(command -v python), log $OUT"
