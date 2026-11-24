#!/usr/bin/env bash
# Per-host health of runs, from the PC (Git Bash): for each host of each run (hosts.txt on the
# cluster), prints the number of night_rank / train processes, GPU memory used and utilization,
# number of GPU processes, count of out-of-memory errors in the rank logs and the last log line.
# "DOWN" = host unreachable. Opens one ssh per host: use sparingly (SSH volume).
#   bash bmuon/tools/pc/hosts_check.sh RUN1 RUN2 ...
G="ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o LogLevel=ERROR -l thomas.turkieh"
for r in "$@"; do
  hosts=$($G aerides.polytechnique.fr "cat ~/nequix/bmuon/runs/$r/hosts.txt")
  echo "== $r"
  for h in $hosts; do
    ( o=$(timeout 25 $G $h.polytechnique.fr "echo \$(pgrep -u \$USER -fc 'night_rank[.]sh $r ') \$(pgrep -u \$USER -fc 'python -m [b]muon[.]train') \$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ') \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) \$(cat /tmp/\$USER/night-$r-rank*.log 2>/dev/null | grep -c RESOURCE_EXHAUSTED) \$(tail -1 /tmp/\$USER/night-$r-rank*.log 2>/dev/null | tail -1 | cut -c1-90)" 2>/dev/null)
      if [ -z "$o" ]; then echo "  $h DOWN"; else echo "  $h $o"; fi ) &
  done; wait
done
