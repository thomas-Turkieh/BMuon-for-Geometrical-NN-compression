# Cluster-side script, run from the PC (Git Bash) through the gateway; prints a tar of the
# config.yml and the validation/pruning lines of log.jsonl of each run given as argument.
#   mkdir -p ~/runs_cache && cd ~/runs_cache && \
#   ssh -o BatchMode=yes -l thomas.turkieh aerides.polytechnique.fr bash -s -- RUN1 RUN2 ... \
#       < <repo>/bmuon/tools/pc/pull_logs.sh | tar -xmf -
set -e
T=$(mktemp -d); cd ~/nequix/bmuon/runs
for r in "$@"; do
  mkdir -p $T/$r; cp $r/config.yml $T/$r/
  grep -E 'val/force_mae|prune/level' $r/log.jsonl > $T/$r/log.jsonl
  [ -f $r/DONE ] && touch $T/$r/DONE
done
tar -C $T -cf - . ; rm -rf $T
