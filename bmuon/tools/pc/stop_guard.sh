#!/usr/bin/env bash
# macOS / Linux: stops the night guard (and its caffeinate). The runs on the cluster keep going;
# nothing restarts them until a guard runs again. Never run two guards at once (two computers).
#   bash bmuon/tools/pc/stop_guard.sh
pids=$(pgrep -f "night_guard[.]sh" || true)
if [ -z "$pids" ]; then echo "no guard running"; exit 0; fi
echo "stopping $(echo $pids)"
pkill -f "night_guard[.]sh"
