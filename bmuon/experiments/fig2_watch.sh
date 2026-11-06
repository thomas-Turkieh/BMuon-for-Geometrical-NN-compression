#!/usr/bin/env bash
# Redraws Figure 2 from the latest snapshots every few minutes until the deadline, then (or as
# soon as every run is DONE) a final pass on more structures. Started by `night.sh watch HOST`.
#   fig2_watch.sh DEADLINE [PLAN]
# Outputs (shared home): bmuon/results/fig2/{fig2.png, fig2_points.csv, STATUS.md}, and
# bmuon/results/fig2_final/ for the final pass.
set -u
DEADLINE="$1"; PLAN="${2:-bmuon/experiments/fig2_runs.yml}"
USER="${USER:-$(id -un)}"
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source "/tmp/$USER/nequix-venv/bin/activate"
export BMUON_DATA="/tmp/$USER/bmuon-data" XLA_PYTHON_CLIENT_PREALLOCATE=false
log() { echo "[fig2_watch $(date '+%F %T')] $*"; }
names=$(python -c "import yaml; print(' '.join(r['name'] for r in yaml.safe_load(open('$PLAN', encoding='utf-8'))['runs'] if r['kind'] != 'reference'))")
all_done() { for n in $names; do [ -f "bmuon/runs/$n/DONE" ] || return 1; done; }
fig2() {  # $1 = n structures, $2 = out dir
  left=$(( DEADLINE - $(date +%s) )); [ "$left" -gt 60 ] || return 1
  timeout -k 30 "$left" python -m bmuon.experiments.fig2 --plan "$PLAN" \
    --data "$BMUON_DATA/mptrj-aselmdb" --n-structures "$1" --out "$2"
}
while [ $(( DEADLINE - $(date +%s) )) -gt 600 ]; do
  # final pass (1000 structures) when every run is DONE, or from 2 h before the deadline; repeated
  # until every run is DONE (cached per snapshot, so each repeat only evaluates new snapshots)
  if all_done || [ $(( DEADLINE - $(date +%s) )) -lt 7200 ]; then
    all_done && log "every run is DONE: final pass (1000 structures)" || log "deadline close: final pass with what exists"
    fig2 1000 bmuon/results/fig2_final && log "final figure: bmuon/results/fig2_final/fig2.png"
    all_done && break
  else
    log "live pass (300 structures)"
    fig2 300 bmuon/results/fig2 || log "live pass failed or cut by the deadline"
  fi
  sleep 300
done
log "exit"
