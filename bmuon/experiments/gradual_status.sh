#!/usr/bin/env bash
# Status of the gradual-pruning runs, from the PC: hosts alive (night.sh status) + progress table
# (gradual_status.py, on the cluster) + the training-curve plot copied to bmuon/results/gradual/.
#   bash bmuon/experiments/gradual_status.sh
cd "$(dirname "$0")/../.."
GW="${GATEWAY:-aerides}"; CU="${CLUSTER_USER:-thomas.turkieh}"
names=$(python -c "import yaml; print(' '.join(r['name'] for r in yaml.safe_load(open('bmuon/experiments/gradual_runs.yml', encoding='utf-8'))['runs']))")
bash bmuon/experiments/night.sh status $names
ssh -n -o BatchMode=yes -o ConnectTimeout=15 -o LogLevel=ERROR -l "$CU" "$GW.polytechnique.fr" \
  "cd ~/nequix && source /tmp/\$USER/nequix-venv/bin/activate && python -m bmuon.experiments.gradual_status --plot bmuon/results/gradual/curves.png 2>&1 | grep -v -i warn"
mkdir -p bmuon/results/gradual
# bonus runs (lower priority, no host replacement), if any
if [ -f bmuon/experiments/bonus_runs.yml ]; then
  echo
  bnames=$(python -c "import yaml; print(' '.join(r['name'] for r in yaml.safe_load(open('bmuon/experiments/bonus_runs.yml', encoding='utf-8'))['runs']))")
  bash bmuon/experiments/night.sh status $bnames
  ssh -n -o BatchMode=yes -o ConnectTimeout=15 -o LogLevel=ERROR -l "$CU" "$GW.polytechnique.fr"     "cd ~/nequix && source /tmp/\$USER/nequix-venv/bin/activate && python -m bmuon.experiments.gradual_status --plan bmuon/experiments/bonus_runs.yml 2>&1 | grep -v -i warn"
fi
scp -q -o BatchMode=yes "$CU@$GW.polytechnique.fr:nequix/bmuon/results/gradual/curves.png" bmuon/results/gradual/curves.png 2>/dev/null
echo "guard: $(tail -1 bmuon/results/night_guard.log)"
