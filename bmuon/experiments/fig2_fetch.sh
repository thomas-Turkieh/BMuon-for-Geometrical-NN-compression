#!/usr/bin/env bash
# Copy the live Figure 2 (and the final one when it exists) from the cluster to this PC,
# then print the run status. Run from EA/code in Git Bash:
#   bash bmuon/experiments/fig2_fetch.sh
# Files land in bmuon/results/fig2_live/ (open fig2.png and STATUS.md in VS Code).
set -u
cd "$(dirname "$0")/../.."
dst=bmuon/results/fig2_live
mkdir -p "$dst"
src="${CLUSTER_USER:-thomas.turkieh}@${GATEWAY:-aerides}.polytechnique.fr:nequix/bmuon/results"
scp -q -o BatchMode=yes -o LogLevel=ERROR "$src/fig2/fig2.png" "$src/fig2/fig2_energy.png" "$src/fig2/STATUS.md" "$src/fig2/fig2_points.csv" "$dst/" \
  && echo "live figure  -> $dst/fig2.png"
if scp -q -o BatchMode=yes -o LogLevel=ERROR "$src/fig2_final/fig2.png" "$dst/fig2_final.png" 2>/dev/null; then
  scp -q -o BatchMode=yes -o LogLevel=ERROR "$src/fig2_final/fig2_points.csv" "$dst/fig2_final_points.csv"
  scp -q -o BatchMode=yes -o LogLevel=ERROR "$src/fig2_final/fig2_energy.png" "$dst/fig2_final_energy.png"
  echo "FINAL figure -> $dst/fig2_final.png"
fi
echo
cat "$dst/STATUS.md" 2>/dev/null
echo
bash bmuon/experiments/night.sh status
