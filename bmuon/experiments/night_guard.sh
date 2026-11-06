#!/usr/bin/env bash
# Keeps tonight's Figure 2 runs alive from the PC (Git Bash), e.g. across the lab reboots
# (kernel updates, ~00:30): every 5 min, every run of the plan that is not DONE and has lost
# a rank is stopped and started again on the same hosts with night.sh, so it resumes from its
# last state.pkl. A host that stays unreachable for 15 min is replaced by an idle host of the
# same GPU type. The live-figure watcher is restarted the same way. Stops at STOP_AT.
#
#   bash bmuon/experiments/night_guard.sh [PLAN]        (log: bmuon/results/night_guard.log)
#   DRY=1 bash bmuon/experiments/night_guard.sh         one check, prints what it would do
set -u
cd "$(dirname "$0")/../.."
PLAN="${1:-bmuon/experiments/fig2_runs.yml}"
NIGHT="bash bmuon/experiments/night.sh"
CU="${CLUSTER_USER:-thomas.turkieh}"
LOG=bmuon/results/night_guard.log
mkdir -p bmuon/results
SSHO=(-n -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR)
r() { local h="$1"; shift; timeout 40 ssh "${SSHO[@]}" -l "$CU" "$h.polytechnique.fr" "$@" 2>/dev/null; }
log() { echo "[$(date '+%a %H:%M:%S')] $*" | tee -a "$LOG"; }
declare -A down   # host -> consecutive checks unreachable

dline=$($NIGHT deadline)   # "runs started now stop at Wed 04 Nov 07:25:00 (513 min from now)"
deadline=$(echo "$dline" | sed -n 's/.*stop at \(.*\) (.*/\1/p')
mins=$(echo "$dline" | sed -n 's/.*(\([0-9]*\) min from now).*/\1/p')
now0=$(r "${GATEWAY:-aerides}" 'date +%s')
[ -n "$mins" ] && [ -n "$now0" ] || { log "cannot get the deadline (daytime or gateway down): $dline"; exit 1; }
dl_s=$((now0 + 60 * mins))
log "guard started, plan $PLAN, runs stop at $deadline"

runs() { python -c "import yaml
for r in yaml.safe_load(open('$PLAN', encoding='utf-8'))['runs']:
    if r['kind'] != 'reference': print(r['name'], r['config'])" | tr -d '\r'; }

replace_host() {   # replace_host NAME OLD -> prints the new host (same GPU type, idle, provisioned)
  local name="$1" old="$2" type used new
  type=$(awk -v h="$old" '$1 == h {print $2}' bmuon/experiments/machines.txt)
  used=$(r "${GATEWAY:-aerides}" 'cat ~/nequix/bmuon/runs/fig2-*/hosts.txt ~/nequix/bmuon/results/fig2/WATCHER 2>/dev/null' | tr '\n' ' ')
  new=$($NIGHT free "$type" | awk '$3 == "gpu-free" && $4 == "cpu-ok" && /ours=0/ {print $2}' \
        | while read -r h; do [[ " $used mayenne ardennes indre manche " == *" $h "* ]] || echo "$h"; done | head -1)
  [ -n "$new" ] || return 1
  $NIGHT provision "$new" >> "$LOG" 2>&1
  echo "$new"
}

check_once() {
  local now; now=$(r "${GATEWAY:-aerides}" 'date +%s')
  [ -n "$now" ] && [ $((dl_s - now)) -lt 1800 ] && { log "less than 30 min before the deadline: guard stops"; exit 0; }
  while read -r name config; do
    st=$(r "${GATEWAY:-aerides}" "cd ~/nequix/bmuon/runs/$name 2>/dev/null && { [ -f DONE ] && echo DONE || cat hosts.txt; }")
    [ -z "$st" ] && continue              # not launched (or gateway down)
    [ "$st" = DONE ] && continue
    hosts=($st); missing=(); unreach=()
    for h in "${hosts[@]}"; do
      c=$(r "$h" "pgrep -u \$USER -fc 'night_rank[.]sh $name '")
      if [ -z "$c" ]; then unreach+=("$h"); down[$h]=$(( ${down[$h]:-0} + 1 ))
      else down[$h]=0; [ "$c" -ge 1 ] || missing+=("$h"); fi
    done
    [ ${#missing[@]} -eq 0 ] && [ ${#unreach[@]} -eq 0 ] && continue
    if [ ${#unreach[@]} -gt 0 ]; then
      long=(); for h in "${unreach[@]}"; do [ "${down[$h]}" -ge 3 ] && long+=("$h"); done
      if [ ${#long[@]} -eq 0 ]; then log "$name: unreachable ${unreach[*]} (rebooting?), waiting"; continue; fi
      for h in "${long[@]}"; do
        if [ -n "${DRY:-}" ]; then log "$name: would replace $h"; continue 2; fi
        new=$(replace_host "$name" "$h") || { log "$name: no idle host to replace $h, waiting"; continue 2; }
        log "$name: $h unreachable for 15 min, replaced by $new"
        hosts=("${hosts[@]/$h/$new}")
      done
    fi
    log "$name: ranks missing on ${missing[*]} ${unreach[*]}: restart on ${hosts[*]} (resume from state.pkl)"
    [ -n "${DRY:-}" ] && continue
    $NIGHT stop "${hosts[*]}" >> "$LOG" 2>&1
    $NIGHT stop "$name" >> "$LOG" 2>&1
    sleep 5
    if ! $NIGHT start "$name" "$config" "${hosts[*]}" >> "$LOG" 2>&1; then
      # a host may have been taken by someone else: swap the busy ones and retry next check
      for h in "${hosts[@]}"; do
        m=$(r "$h" "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1")
        if [ -n "$m" ] && [ "$m" -ge 3000 ]; then
          new=$(replace_host "$name" "$h") && { log "$name: $h busy (${m} MiB), replaced by $new";
            r "${GATEWAY:-aerides}" "sed -i 's/\\b$h\\b/$new/' ~/nequix/bmuon/runs/$name/hosts.txt"; }
        fi
      done
      log "$name: start refused, will retry"
    fi
  done < <(runs)
  # live-figure watcher
  w=$(r "${GATEWAY:-aerides}" 'cat ~/nequix/bmuon/results/fig2/WATCHER 2>/dev/null')
  if [ -n "$w" ]; then
    c=$(r "$w" "pgrep -u \$USER -fc 'fig2_watch[.]sh'")
    if [ "${c:-0}" = 0 ] && [ -z "${DRY:-}" ]; then
      if [ -n "$c" ]; then $NIGHT watch "$w" "$PLAN" >> "$LOG" 2>&1 && log "watcher restarted on $w"
      else log "watcher host $w unreachable"; fi
    fi
  fi
}

if [ -n "${DRY:-}" ]; then check_once; exit 0; fi
while :; do
  check_once
  sleep 300
done
