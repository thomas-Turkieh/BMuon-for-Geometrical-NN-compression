# Cluster-side: copies the files of finished runs and the final models into a tar for the repo
# (edit the run lists / model copies at the bottom before use). Run from the PC:
#   ssh -o BatchMode=yes -l thomas.turkieh aerides.polytechnique.fr bash < bmuon/tools/pc/cluster_export_for_repo.sh | tar -C <staging dir> -xmf -
set -e
T=$(mktemp -d); R=~/nequix/bmuon/runs
put() { # experiment run
  d=$T/runs/$1/$2; mkdir -p $d
  for f in config.yml log.jsonl hosts.log hosts.txt DONE; do [ -f $R/$2/$f ] && cp $R/$2/$f $d/; done
  cp $R/$2/prune_epoch*.npz $d/ 2>/dev/null || true
  for e in 000 010 020 030 040 050 060 070 080 090 099; do [ -f $R/$2/alloc_epoch$e.npz ] && cp $R/$2/alloc_epoch$e.npz $d/; done
  true
}
for r in mp1-gradual-bmuon-mu-wd1e3 mp1-gradual-bmuon-mag-wd1e3 mp1-gradual-muon-mag-wd1e3 mp1-gradual-bmuon-mu-wd1e2; do put 2026-11-06_full-mptrj_gradual-pruning $r; done
for r in mp1-uniform-c14-wd1e3 mp1-uniform-c20-wd1e3 mp1-uniform-c60-wd1e3 mp1-uniform-c80-wd1e3; do put 2026-11-07_full-mptrj_small-models-from-scratch $r; done
for r in mp1-arch-learned-bmuon-mu-70pct-scratch mp1-arch-uniform-70pct-scratch; do put 2026-11-10_full-mptrj_learned-architecture-from-scratch $r; done
for k in 1.25 1.5 2 2.5 3; do put 2026-11-10_full-mptrj_wide-models-pruned-to-nequix-size mp1-wide-x$k-gradual-bmuon-mu; done
put 2026-11-22_full-mptrj_x1.5-pruned-architecture-from-scratch mp1-arch-x1.5-pruned-scratch
M=$T/models; mkdir -p $M/2026-11-10_full-mptrj_wide-models-pruned-to-nequix-size $M/2026-11-10_full-mptrj_learned-architecture-from-scratch $M/2026-11-06_full-mptrj_gradual-pruning
W=$M/2026-11-10_full-mptrj_wide-models-pruned-to-nequix-size
cp ~/export_models/x1.5.nqx $W/wide-x1.5-gradual-bmuon-mu_wd1e-3_epoch099_nequix-mp1-size.nqx
cp ~/export_models/x2.nqx   $W/wide-x2-gradual-bmuon-mu_wd1e-3_epoch099_nequix-mp1-size.nqx
cp ~/export_models/x2.5.nqx $W/wide-x2.5-gradual-bmuon-mu_wd1e-3_epoch099_nequix-mp1-size.nqx
L=$M/2026-11-10_full-mptrj_learned-architecture-from-scratch
cp $R/mp1-arch-learned-bmuon-mu-70pct-scratch/snap_epoch099.nqx $L/learned-architecture-bmuon-mu_30pct-cost_wd1e-3_epoch099.nqx
cp $R/mp1-arch-uniform-70pct-scratch/snap_epoch099.nqx $L/uniform-architecture_30pct-cost_wd1e-3_epoch099.nqx
cp $R/mp1-gradual-muon-mag-wd1e3/snap_epoch099.nqx $M/2026-11-06_full-mptrj_gradual-pruning/gradual-muon-mag_wd1e-3_epoch099_90pct-cost-removed.nqx
tar -C $T -cf - . ; rm -rf $T
