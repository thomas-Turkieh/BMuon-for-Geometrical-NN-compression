#!/usr/bin/env bash
# Launch the 8 Figure 2 runs (8 GPUs each) + the live-figure watcher.
#
#   bash bmuon/experiments/fig2_launch.sh            dry run: picks the hosts, prints the commands
#   bash bmuon/experiments/fig2_launch.sh --go       provisions the hosts that need it, then starts
#
# Hosts: every lab machine whose GPU is idle right now (night.sh free). GPUs are pooled by speed
# (TYPES: pools separated by spaces, types of a pool joined by +) and a run only uses GPUs of one
# pool, so a fast run is never slowed down by a much slower card. Default: 4000ada+a5000+3090
# (fast, 20-24 GB), then a4000 (~1.4x slower), then 2000ada (~1.5-2x slower). mayenne, ardennes and indre are skipped
# (100 Mb/s network cards). The watcher takes one more host.
set -u
cd "$(dirname "$0")/../.."
NIGHT="bash bmuon/experiments/night.sh"
PER_RUN="${PER_RUN:-8}"
EXCLUDE=" mayenne ardennes indre manche "
TYPES="${TYPES:-4000ada+a5000+3090 a4000 2000ada}"
RUNS=()   # from fig2_runs.yml, in its order (most important first)
while read -r n c; do RUNS+=("$n $c"); done < <(python -c "import yaml
for r in yaml.safe_load(open('bmuon/experiments/fig2_runs.yml', encoding='utf-8'))['runs']:
    if r['kind'] != 'reference': print(r['name'], r['config'])" | tr -d '\r')   # most important first
GO=0; [ "${1:-}" = "--go" ] && GO=1
# safe to re-run: runs already launched tonight (hosts.txt in their folder) are skipped, and the
# watcher is started only once (bmuon/results/fig2/WATCHER holds its host)
launched=$(ssh -n -o BatchMode=yes -o LogLevel=ERROR -l "${CLUSTER_USER:-thomas.turkieh}" "${GATEWAY:-aerides}".polytechnique.fr   'cd ~/nequix; for d in bmuon/runs/fig2-*/hosts.txt; do [ -f "$d" ] && basename "$(dirname "$d")"; done; cat bmuon/results/fig2/WATCHER 2>/dev/null | sed "s/^/WATCHER:/"' 2>/dev/null | grep -v -i quantum)
watcher=$(echo "$launched" | sed -n 's/^WATCHER://p')
[ -n "$watcher" ] && EXCLUDE="$EXCLUDE$watcher "   # its GPU looks idle between two passes
todo_runs=(); for r in "${RUNS[@]}"; do [[ " $(echo $launched) " == *" ${r%% *} "* ]] && echo "already launched: ${r%% *}" || todo_runs+=("$r"); done
RUNS=("${todo_runs[@]}")
[ ${#RUNS[@]} -eq 0 ] && { echo "every run is already launched"; [ -n "$watcher" ] && exit 0; }

echo "== looking for idle GPUs..."
free_out=$($NIGHT free)
groups=(); spare=""
for pool in $TYPES; do
  hosts=$(for t in ${pool//+/ }; do
            echo "$free_out" | awk -v t="$t" '$1 == t && $3 == "gpu-free" && $0 ~ /ours=0/ {print $2}'
          done | while read -r h; do [[ "$EXCLUDE" == *" $h "* ]] || echo "$h"; done)
  arr=($hosts)
  i=0
  while [ $((i + PER_RUN)) -le ${#arr[@]} ] && [ ${#groups[@]} -lt ${#RUNS[@]} ]; do
    groups+=("${arr[*]:i:PER_RUN}"); i=$((i + PER_RUN))
  done
  [ -z "$spare" ] && [ $i -lt ${#arr[@]} ] && spare="${arr[$i]}"
done
[ -z "$spare" ] && spare=$(echo "$free_out" | awk '$3 == "gpu-free" && $0 ~ /ours=0/ {print $2}' | tail -1)
[ -n "$watcher" ] && spare=""   # already running

echo "== plan: ${#groups[@]}/${#RUNS[@]} runs of $PER_RUN GPUs, watcher on ${watcher:-${spare:-NONE}}"
for i in "${!groups[@]}"; do
  set -- ${RUNS[$i]}
  echo "  $1  ($2)  on: ${groups[$i]}"
done
for ((i=${#groups[@]}; i<${#RUNS[@]}; i++)); do echo "  NOT STARTED (not enough idle GPUs): ${RUNS[$i]%% *}"; done
$NIGHT deadline

if [ $GO = 0 ]; then echo "(dry run: add --go to provision and start)"; exit 0; fi

echo "== provisioning (only hosts not ready yet)"
all="${groups[*]} $spare"
todo=($all)   # `provision` skips the hosts that are already ready
for ((i=0; i<${#todo[@]}; i+=8)); do $NIGHT provision "${todo[*]:i:8}"; done

echo "== starting"
for i in "${!groups[@]}"; do
  set -- ${RUNS[$i]}
  $NIGHT start "$1" "$2" "${groups[$i]}" || echo ">>> $1 NOT started"
done
if [ -n "$spare" ]; then
  $NIGHT watch "$spare" && ssh -n -o BatchMode=yes -o LogLevel=ERROR -l "${CLUSTER_USER:-thomas.turkieh}" "${GATEWAY:-aerides}".polytechnique.fr     "mkdir -p ~/nequix/bmuon/results/fig2 && echo $spare > ~/nequix/bmuon/results/fig2/WATCHER" 2>/dev/null
fi
echo "== follow with:  bash bmuon/experiments/night.sh status   and   ssh ablette cat ~/nequix/bmuon/results/fig2/STATUS.md"
