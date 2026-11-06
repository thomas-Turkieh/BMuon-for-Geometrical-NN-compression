#!/usr/bin/env bash
# One rank of a multi-machine run, restarted on its own until the run is DONE or the deadline.
# Started by night.sh on every host (do not call by hand):
#   night_rank.sh NAME CONFIG RANK NPROC COORD DEADLINE [overrides...]
#
# DEADLINE is an absolute time (seconds since epoch). Guarantees:
#   * python gets SIGTERM at DEADLINE and SIGKILL 45 s later (timeout -k), data workers included;
#   * a watchdog kills every bmuon.train of this user on this host at DEADLINE + 60 s;
#   * no (re)start when less than 5 min are left.
# Restarts (no SSH, no cron, no PC needed): after a crash (another host died, network or NFS
# blip...) the rank waits 60 s and starts again; train.py resumes from out_dir/state.pkl
# (saved every save_every_steps). If out_dir/log.jsonl (written by rank 0 on the shared home)
# has not moved for STALE seconds, the run is considered hung and every rank restarts.
set -u
NAME="$1"; CONFIG="$2"; RANK="$3"; NPROC="$4"; COORD="$5"; DEADLINE="$6"; shift 6
USER="${USER:-$(id -un)}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
OUT="bmuon/runs/$NAME"
mkdir -p "$OUT"
STALE="${BMUON_STALE_SECS:-1500}"
MINE="python -m [b]muon[.]train .*out_dir=$OUT( |$)"   # processes of this run only
log() { echo "[night_rank $RANK $(hostname -s) $(date '+%F %T')] $*"; }

# shellcheck disable=SC1091
source "/tmp/$USER/nequix-venv/bin/activate"
export BMUON_DATA="/tmp/$USER/bmuon-data" XLA_PYTHON_CLIENT_PREALLOCATE=false

( sleep $(( DEADLINE + 60 - $(date +%s) )) 2>/dev/null
  pkill -9 -u "$USER" -f 'python -m [b]muon[.]train' ) &
WATCHDOG=$!

attempt=0
while :; do
  [ -f "$OUT/DONE" ] && { log "run DONE"; break; }
  left=$(( DEADLINE - $(date +%s) ))
  [ "$left" -lt 300 ] && { log "deadline: no more (re)start"; break; }
  attempt=$((attempt + 1))
  pkill -9 -u "$USER" -f "$MINE" 2>/dev/null   # leftovers of an earlier attempt of this run
  log "attempt $attempt, $left s before the deadline"
  echo "$(hostname -s) rank $RANK $(date -Is) attempt $attempt" >> "$OUT/hosts.log"
  started=$(date +%s)
  timeout -s TERM -k 45 "$left" python -m bmuon.train "$CONFIG" \
    --coordinator "$COORD" --num-processes "$NPROC" --process-id "$RANK" \
    --set out_dir="$OUT" "$@" &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    m=$(stat -c %Y "$OUT/log.jsonl" 2>/dev/null || echo 0)
    ref=$(( m > started ? m : started ))
    if [ $(( $(date +%s) - ref )) -gt "$STALE" ]; then
      log "log.jsonl has not moved for ${STALE}s: run hung, restarting this rank"
      kill -TERM "$pid" 2>/dev/null; sleep 50; kill -KILL "$pid" 2>/dev/null
      pkill -9 -u "$USER" -f "$MINE" 2>/dev/null
      break
    fi
  done
  wait "$pid"; rc=$?
  log "python exited with code $rc"
  # After a crash (abort, kill), the data-loader workers survive and keep the coordinator port
  # open: the next attempt's connections would land on them. `timeout` leads its own process
  # group, so kill that whole group, then whatever is left of this run on this host.
  kill -KILL -- "-$pid" 2>/dev/null
  pkill -9 -u "$USER" -f "$MINE" 2>/dev/null
  sleep 2
  [ "$rc" -eq 0 ] && { log "normal end"; break; }
  sleep 60
done
pkill -P "$WATCHDOG" 2>/dev/null; kill "$WATCHDOG" 2>/dev/null
log "exit"
