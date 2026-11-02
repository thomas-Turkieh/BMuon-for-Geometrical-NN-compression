#!/usr/bin/env bash
# Launch a 16-machine data-parallel smoke run on the fish room and report per-epoch times.
# Thin wrapper around launch_dist.sh (clears stray processes first, waits for DONE).
#
#   bash bmuon/experiments/run_16.sh ["host1 ... host16"] [config.yml]
#
# Defaults to the 16 fish-room RTX 3090 machines and nequix-mp-1-bmuon on 0.3% of MPtrj.
# The global batch (batch_size x grad_accum_steps x n_hosts) is kept at 256 for 16 hosts
# with batch_size=16 grad_accum_steps=1.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1
USER="${USER:-$(id -un)}"
HOSTS="${1:-ablette anchois anguille barbeau brochet carrelet gardon gymnote labre lieu mulet murene piranha raie requin rouget}"
CONFIG="${2:-bmuon/configs/nequix-mp-1-bmuon.yml}"
NAME=essai16m

echo "=== check ==="
bash bmuon/experiments/launch_dist.sh check "$HOSTS"

echo "=== clear stray processes (avoid a jax 'different incarnation' abort) ==="
bash bmuon/experiments/launch_dist.sh stop "$HOSTS" 2>/dev/null || true
sleep 2
rm -rf "bmuon/runs/$NAME"

echo "=== start ==="
bash bmuon/experiments/launch_dist.sh start "$NAME" "$CONFIG" "$HOSTS" \
  --set train_subset_frac=0.003 valid_subset_frac=0.01 n_epochs=2 log_every=5 batch_size=16 grad_accum_steps=1

echo "=== wait for DONE (max ~30 min) ==="
for t in $(seq 1 180); do
  [ -f "bmuon/runs/$NAME/DONE" ] && { echo "DONE after ~$((t*10)) s"; break; }
  sleep 10
done

echo "=== per-epoch time ==="
"/tmp/$USER/nequix-venv/bin/python" - "$NAME" <<'PY'
import ast, sys
name = sys.argv[1]
try:
    rows = [d for d in map(ast.literal_eval, open(f'bmuon/runs/{name}/log.jsonl')) if 'time/epoch' in d]
    for d in rows:
        print(f"epoch {d.get('epoch')}: {d['time/epoch']:.1f} s | val/loss {d.get('val/loss'):.4f}")
    if not rows:
        print("no epoch finished - see /tmp/$USER/dist-%s-rank0.log" % name)
except FileNotFoundError:
    print("log.jsonl missing - run did not start")
PY
