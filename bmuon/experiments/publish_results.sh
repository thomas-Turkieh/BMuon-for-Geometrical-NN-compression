#!/usr/bin/env bash
# Commit and push the results of this machine to GitHub (as Octave Koenig, noreply address).
#   bash bmuon/experiments/publish_results.sh ["message"]
# Pushes bmuon/results/, the measured costs bmuon/configs/costs_*.json and the small files
# of bmuon/runs/ (see bmuon/.gitignore).
# Safe to run from several machines at once: it rebases and retries.
set -u
cd "$(dirname "$0")/../.."
MSG="${1:-results from $(hostname) $(date -Is)}"
GIT=(git -c user.name="Octave Koenig" -c user.email="OctaveKNG@users.noreply.github.com")
# only existing paths: git add aborts entirely on a pathspec that matches nothing
PATHS=()
for p in bmuon/results bmuon/runs bmuon/configs/costs_*.json; do [ -e "$p" ] && PATHS+=("$p"); done
[ ${#PATHS[@]} -gt 0 ] && "${GIT[@]}" add -A -- "${PATHS[@]}"
if "${GIT[@]}" diff --cached --quiet; then echo "nothing new to publish"; exit 0; fi
"${GIT[@]}" commit -q -m "$MSG" || exit 1
for i in 1 2 3 4 5 6; do
  if "${GIT[@]}" pull -q --rebase origin main && "${GIT[@]}" push -q origin main; then
    echo "published: $MSG"; exit 0
  fi
  sleep $((RANDOM % 30 + 10))
done
echo "push failed after 6 tries; the commit is kept locally, run this script again" >&2
exit 1
