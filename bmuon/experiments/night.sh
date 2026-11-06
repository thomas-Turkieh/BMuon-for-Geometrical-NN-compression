#!/usr/bin/env bash
# Night runs on the lab machines, driven from the PC (Git Bash) or from any lab machine.
# Needs no SSH key between lab machines: every command SSHes from here to each host, and the
# runs then live on their own until the deadline (see night_rank.sh).
#
#   bash bmuon/experiments/night.sh free [4000ada a5000 ...]     free GPUs right now (GPU idle, CPU not saturated)
#   bash bmuon/experiments/night.sh provision "h1 h2 ..."        env + data into /tmp/$USER (from ~/bmuon-env.tar.zst)
#   bash bmuon/experiments/night.sh check "h1 h2 ..."            env / data / GPU / our processes on each host
#   bash bmuon/experiments/night.sh start NAME CONFIG "h1 h2 ..." [--set k=v ...]
#   bash bmuon/experiments/night.sh status [NAME ...]            hosts alive + last epoch of each run (all runs by default)
#   bash bmuon/experiments/night.sh stop NAME | "h1 h2 ..."      kills the run (wrappers + python)
#   bash bmuon/experiments/night.sh watch HOST [PLAN]            live Figure 2 (fig2_watch.sh) on a spare GPU host
#   bash bmuon/experiments/night.sh deadline                     when tonight's runs will stop
#
# Rules:
#   * start is refused from 30 min before STOP_AT until 18:30, and every process stops by itself
#     (SIGTERM at STOP_AT, default 07:25, SIGKILL 45 s later, watchdog 60 s later). The DSI window
#     ends at 07:30 on weekdays (window.sh): set STOP_AT=HH:MM to change it;
#   * global batch kept at 256: batch_size/grad_accum_steps are set from the number of hosts
#     (1, 2, 4, 8, 16 or 32) unless given with --set;
#   * a crashed or hung run restarts by itself from its last state.pkl (save_every_steps=1000).
# Runs are written to ~/nequix/bmuon/runs/NAME (shared home), with the host list in hosts.txt.
set -u
CLUSTER_USER="${CLUSTER_USER:-thomas.turkieh}"
REMOTE_REPO="${REMOTE_REPO:-nequix}"          # relative to the remote home
STOP_AT="${STOP_AT:-07:25}"
_m=$((10#${STOP_AT%:*} * 60 + 10#${STOP_AT#*:} - 30)); REFUSE_FROM=$(( (_m / 60) * 100 + _m % 60 ))  # HHMM
GATEWAY="${GATEWAY:-aerides}"                 # host used to read the shared home
SSH_OPTS=(-n -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR)

fqdn() { case "$1" in *.*) echo "$1";; *) echo "$1.polytechnique.fr";; esac; }
rsh() { local h="$1"; shift; ssh "${SSH_OPTS[@]}" -l "$CLUSTER_USER" "$(fqdn "$h")" "$@" 2>&1 | grep -v -i "post-quantum\|store now, decrypt later\|openssh.com/pq.html"; return "${PIPESTATUS[0]}"; }
here="$(cd "$(dirname "$0")" && pwd)"
machines() { grep -v '^#' "$here/machines.txt" | awk 'NF{print $1, $2}'; }

# seconds-since-epoch of the next STOP_AT, computed on the gateway (cluster clock); refuses daytime
deadline() {
  rsh "$GATEWAY" "now=\$(date +%s); hm=\$((10#\$(date +%H%M))); s=\$(date -d 'today $STOP_AT' +%s);
    if [ \$hm -ge $REFUSE_FROM ] && [ \$hm -lt 1830 ]; then echo REFUSE; exit; fi
    [ \$now -ge \$s ] && s=\$(date -d 'tomorrow $STOP_AT' +%s); echo \$s"
}

cmd="${1:-}"; shift || true
case "$cmd" in

free)
  types="${*:-}"
  tmp=$(mktemp -d)
  while read -r h g; do
    [ -n "$types" ] && ! [[ " $types " == *" $g "* ]] && continue
    ( r=$(timeout 25 ssh "${SSH_OPTS[@]}" -l "$CLUSTER_USER" "$(fqdn "$h")" '
        q=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d " ")
        u=${q%%,*}; m=${q##*,}; c=$(nproc); l=$(cut -d" " -f1 /proc/loadavg)
        ours=$(pgrep -u $USER -fc "python -m [b]muon[.]train"); env=$([ -f /tmp/$USER/nequix-venv/.bmuon_ready ] && [ -f /tmp/$USER/bmuon-data/mptrj-aselmdb/.complete ] && echo ready || echo -)
        cpu=$(awk -v l=$l -v c=$c "BEGIN{print (l < 0.5*c) ? \"cpu-ok\" : \"cpu-busy\"}")
        gpu=$([ -n "$u" ] && [ "$m" -lt 3000 ] && echo gpu-free || echo gpu-busy)
        echo "$gpu $cpu load=$l/$c gpu=${u}%,${m}MiB env=$env ours=$ours"' 2>/dev/null | grep -v -i quantum) || r="unreachable"
      echo "$g $h ${r:-unreachable}" > "$tmp/$h" ) &
    sleep 0.05
  done < <(machines)
  wait
  all=$(cat "$tmp"/* | sort); rm -rf "$tmp"
  echo "== FREE (GPU idle, CPU ok)";             echo "$all" | grep "gpu-free cpu-ok"   | column -t
  echo "== GPU idle but CPU saturated (slower)"; echo "$all" | grep "gpu-free cpu-busy" | column -t
  echo "== summary"; for g in $(echo "$all" | awk '{print $1}' | sort -u); do
    echo "  $g: $(echo "$all" | grep "^$g .*gpu-free cpu-ok" | wc -l) free, $(echo "$all" | grep "^$g .*gpu-free cpu-busy" | wc -l) cpu-busy, $(echo "$all" | grep -c "^$g ") total"; done
  echo "  free hosts: $(echo "$all" | grep "gpu-free cpu-ok" | awk '{print $2}' | tr '\n' ' ')"
  ;;

provision)
  hosts=($1)
  rsh "$GATEWAY" 'test -f ~/bmuon-env.tar.zst' || { echo "~/bmuon-env.tar.zst missing on the shared home"; exit 1; }
  for h in "${hosts[@]}"; do
    ( out=$(rsh "$h" 'set -e; R=/tmp/$USER
        if [ -f $R/nequix-venv/.bmuon_ready ] && [ -f $R/bmuon-data/mptrj-aselmdb/.complete ]; then echo "already ready"; exit 0; fi
        df -BG --output=avail /tmp | tail -1 | awk "{ if (\$1+0 < 20) { print \"not enough space in /tmp\"; exit 1 } }"
        mkdir -p $R; rm -rf $R/nequix-venv $R/uv-python $R/cuda12 $R/bmuon-data
        zstd -dc -T4 ~/bmuon-env.tar.zst | tar -x -C $R
        test -f $R/nequix-venv/.bmuon_ready && test -f $R/bmuon-data/mptrj-aselmdb/.complete && echo "installed"' | tail -1)
      echo "$h: ${out:-FAILED}" ) &
    sleep 1
  done
  wait
  ;;

check)
  for h in $1; do
    ( r=$(rsh "$h" 'R=/tmp/$USER; e=$([ -f $R/nequix-venv/.bmuon_ready ] && echo env || echo NO-ENV); d=$([ -f $R/bmuon-data/mptrj-aselmdb/.complete ] && echo data || echo NO-DATA)
        echo "$e $d ours=$(pgrep -u $USER -fc "python -m [b]muon[.]train") load=$(cut -d" " -f1 /proc/loadavg)/$(nproc) $(nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader | head -1)"' | tr '\n' ' ') || r="unreachable"
      echo "$h: $r" ) &
  done | sort
  wait
  ;;

start)
  name="$1"; config="$2"; hosts=($3); shift 3
  extra=("$@"); [ "${extra[0]:-}" = "--set" ] && extra=("${extra[@]:1}")
  n=${#hosts[@]}
  dl=$(deadline) || { echo "cannot reach $GATEWAY"; exit 1; }
  [ "$dl" = "REFUSE" ] && { echo "REFUSED: runs are only started between 18:30 and 30 min before $STOP_AT"; exit 1; }
  left=$(( dl - $(rsh "$GATEWAY" 'date +%s') ))
  [ "$left" -lt 1800 ] && { echo "REFUSED: less than 30 min before $STOP_AT"; exit 1; }
  # global batch 256
  auto=()
  if ! printf '%s\n' "${extra[@]}" | grep -q '^batch_size='; then
    case $n in
      1)  auto=(batch_size=64 grad_accum_steps=4);;
      2)  auto=(batch_size=64 grad_accum_steps=2);;
      4)  auto=(batch_size=64 grad_accum_steps=1);;
      8)  auto=(batch_size=32 grad_accum_steps=1 max_n_nodes=2048 max_n_edges=104000);;
      16) auto=(batch_size=16 grad_accum_steps=1 max_n_nodes=1024 max_n_edges=52000);;
      32) auto=(batch_size=8 grad_accum_steps=1 max_n_nodes=512 max_n_edges=26000);;
      *)  echo "REFUSED: $n hosts; use 1, 2, 4, 8, 16 or 32 (global batch 256) or give batch_size/grad_accum_steps"; exit 1;;
    esac
  fi
  # checkpoint defaults, only when neither the config nor --set gives them (a --set always wins
  # over the config, so adding them blindly would override e.g. snapshot_every_epochs: 1)
  have=$(rsh "$GATEWAY" "cd ~/$REMOTE_REPO && /tmp/\$USER/nequix-venv/bin/python -c 'import yaml,sys
sys.argv=[\"x\"]
from bmuon.train import load_config
c = load_config(\"$config\")
print(*[k for k in (\"save_every_steps\", \"snapshot_every_epochs\") if c.get(k)])' 2>/dev/null" | tail -1)
  defaults=()
  for kv in save_every_steps=1000 snapshot_every_epochs=5; do
    k=${kv%%=*}
    [[ " $have " == *" $k "* ]] || printf '%s\n' "${extra[@]}" | grep -q "^$k=" || defaults+=("$kv")
  done
  sets=("${defaults[@]}" "${auto[@]}" "${extra[@]}")
  # pre-flight: every host ready, GPU free, nothing of ours running
  bad=0
  for h in "${hosts[@]}"; do
    r=$(rsh "$h" 'R=/tmp/$USER; [ -f $R/nequix-venv/.bmuon_ready ] && [ -f $R/bmuon-data/mptrj-aselmdb/.complete ] || { echo NOT-PROVISIONED; exit; }
        [ "$(pgrep -u $USER -fc "python -m [b]muon[.]train")" = 0 ] || { echo OURS-RUNNING; exit; }
        m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$m" -lt 3000 ] || { echo "GPU-BUSY(${m}MiB)"; exit; }; echo ok') || r="unreachable"
    [ "$r" = ok ] || { echo "  $h: $r"; bad=1; }
  done
  [ $bad = 1 ] && { echo "REFUSED: fix or replace the hosts above"; exit 1; }
  port=$((12000 + $(echo -n "$name" | cksum | cut -d' ' -f1) % 1000))
  coord="$(fqdn "${hosts[0]}"):$port"
  rsh "${hosts[0]}" "mkdir -p ~/$REMOTE_REPO/bmuon/runs/$name && echo '${hosts[*]}' > ~/$REMOTE_REPO/bmuon/runs/$name/hosts.txt"
  echo "run $name: $n hosts, coordinator $coord, stop at $(rsh "$GATEWAY" "date -d @$dl '+%a %H:%M'")"
  echo "  config $config  --set ${sets[*]}"
  for i in "${!hosts[@]}"; do
    h="${hosts[$i]}"
    # Only the simple command goes to the background, with its own redirections: a backgrounded
    # `cd && nohup ...` list would be a subshell holding the ssh channel open until python exits.
    rsh "$h" "cd ~/$REMOTE_REPO || exit 1; nohup setsid bash bmuon/experiments/night_rank.sh $name $config $i $n $coord $dl ${sets[*]} \
      > /tmp/\$USER/night-$name-rank$i.log 2>&1 < /dev/null & echo '  rank $i on $h'"
  done
  ;;

status)
  names=("$@")
  if [ ${#names[@]} -eq 0 ]; then
    names=($(rsh "$GATEWAY" "cd ~/$REMOTE_REPO/bmuon/runs && for d in */hosts.txt; do r=\${d%/hosts.txt}; [ -f \$r/DONE ] || [ \$(( \$(date +%s) - \$(stat -c %Y \$d) )) -lt 50000 ] && echo \$r; done 2>/dev/null"))
  fi
  for name in "${names[@]}"; do
    hosts=($(rsh "$GATEWAY" "cat ~/$REMOTE_REPO/bmuon/runs/$name/hosts.txt"))
    alive=0; dead=""
    for h in "${hosts[@]}"; do
      c=$(rsh "$h" 'pgrep -u $USER -fc "python -m [b]muon[.]train"' 2>/dev/null | tail -1)
      [[ "$c" =~ ^[1-9] ]] && alive=$((alive+1)) || dead="$dead $h"
    done
    info=$(rsh "$GATEWAY" "cd ~/$REMOTE_REPO/bmuon/runs/$name; [ -f DONE ] && echo -n 'DONE | '; python3 - <<'EOF'
import json, time, os
ep = st = None
try:
    for l in open('log.jsonl'):
        d = json.loads(l)
        if 'epoch' in d: ep = d
        else: st = d
except FileNotFoundError:
    print('no log yet'); raise SystemExit
age = time.time() - os.path.getmtime('log.jsonl')
s = f'log {age/60:.0f} min ago'
if ep: s += f\" | epoch {ep['epoch']} done ({ep['time/epoch']/60:.1f} min) val F {1000*ep['val/force_mae']:.1f} meV/A E {1000*ep['val/energy_mae_per_atom']:.2f} meV/at\"
if st: s += f\" | step {st.get('step', '?')} loss {st.get('train/loss', float('nan')):.2f}\"
print(s)
EOF")
    echo "$name: $alive/${#hosts[@]} hosts running${dead:+ (not running:$dead)} | $info"
  done
  ;;

stop)
  if [[ "$1" != *" "* ]] && rsh "$GATEWAY" "test -f ~/$REMOTE_REPO/bmuon/runs/$1/hosts.txt"; then
    hosts=$(rsh "$GATEWAY" "cat ~/$REMOTE_REPO/bmuon/runs/$1/hosts.txt")
  else hosts="$1"; fi
  for h in $hosts; do
    ( rsh "$h" 'pkill -u $USER -f "[n]ight_rank.sh"; pkill -9 -u $USER -f "python -m [b]muon[.]train"; sleep 2; echo "left: $(pgrep -u $USER -fc "python -m [b]muon[.]train")"' | tail -1 | sed "s/^/$h: /" ) &
  done
  wait
  ;;

watch)
  # live Figure 2 on one spare GPU host (bmuon/experiments/fig2_watch.sh), stops by itself at STOP_AT
  h="$1"; plan="${2:-bmuon/experiments/fig2_runs.yml}"
  dl=$(deadline) || { echo "cannot reach $GATEWAY"; exit 1; }
  [ "$dl" = "REFUSE" ] && { echo "REFUSED: only between 18:30 and 30 min before $STOP_AT"; exit 1; }
  rsh "$h" "cd ~/$REMOTE_REPO || exit 1; nohup setsid bash bmuon/experiments/fig2_watch.sh $dl $plan     > /tmp/\$USER/fig2-watch.log 2>&1 < /dev/null & echo 'fig2 watcher started on $h (log $h:/tmp/\$USER/fig2-watch.log)'"
  ;;

deadline)
  dl=$(deadline); [ "$dl" = REFUSE ] && { echo "daytime: starting is refused now"; exit 0; }
  rsh "$GATEWAY" "echo \"runs started now stop at \$(date -d @$dl '+%a %d %b %H:%M:%S') (\$(( ($dl - \$(date +%s)) / 60 )) min from now)\""
  ;;

*) sed -n '2,24p' "$0"; exit 1 ;;
esac
