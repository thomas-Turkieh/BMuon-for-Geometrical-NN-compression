# Training logs

One folder per experiment, named `<start date>_<data>_<what is tested>`. Inside it, one folder per
training run, named by its run id. The run id is the `out_dir` of its config and the folder name
on the cluster, where training writes to `bmuon/runs/<run id>/` without the experiment level.

Each run folder holds:
- `config.yml`: the full resolved config;
- `log.jsonl`: training and validation metrics, one JSON line per log step;
- `hosts.log` / `hosts.txt`: the machines used;
- `alloc_epoch*.npz`: BMuon allocation μ every 10 epochs;
- `prune_epoch*.npz`: gradual-pruning masks, one per pruning event;
- `progress.md`: October runs only.

Checkpoints and optimizer states stay on the cluster. The trained models are in
[`../models/`](../models/) and the figures and tables are in [`../results/`](../results/), both
under the same experiment names.

Data: MPtrj, with the same 5 % validation split for every run. Model: Nequix-MP-1
(`128x0e + 64x1o + 32x2e + 32x3o`, 857 type-I channels), unless stated otherwise.

| Experiment | Principle | Runs | Status |
|---|---|---|---|
| [`2026-10_smoke-tests_distributed-launch`](2026-10_smoke-tests_distributed-launch/) | Infrastructure tests of the multi-machine launcher (1, 2 and 16 machines). No scientific result. | `essai*` | done |
| [`2026-10-13_full-mptrj_dense-training_bmuon-vs-uniform`](2026-10-13_full-mptrj_dense-training_bmuon-vs-uniform/) | Full MPtrj, up to 100 epochs. BMuon with a 30 % cost budget (weight decay 0.01, 0.001 or selective) against a small uniform model of the same cost (35 %) trained with Muon. Detailed registry: [`RUNS.md`](2026-10-13_full-mptrj_dense-training_bmuon-vs-uniform/RUNS.md). | `mp1-bmuon-wd1e2`, `mp1-bmuon-wd1e3`, `mp1-bmuon-selwd`, `mp1-selwd-late-wd0p1`, `mp1-uniform-shrink-wd1e2`, `mp1-uniform-shrink-wd1e3` | done or stopped (wd1e2 at ep. 84, wd1e3 at ep. 65) |
| [`2026-11-03_20pct-mptrj_pruning-methods-comparison`](2026-11-03_20pct-mptrj_pruning-methods-comparison/) | The paper's Figure 2 redone for Nequix at reduced scale: 20 % of MPtrj, 25 epochs. Muon, Muon + group lasso, NuMuon and BMuon are pruned after training; three variants are pruned gradually during training (levels in % of channels). Plan: `experiments/FIG2_PLAN.md`. | `fig2-bmuon`, `fig2-muon`, `fig2-muon-gl-2`, `fig2-muon-gl-0p5` (stopped at ep. 1), `fig2-numuon` (diverged at ep. 8), `fig2-gradual-bmuon-mu`, `fig2-gradual-bmuon-mag`, `fig2-gradual-muon-mag` | done |
| [`2026-11-06_full-mptrj_gradual-pruning`](2026-11-06_full-mptrj_gradual-pruning/) | Gradual pruning during training on full MPtrj, 100 epochs, same recipe as the October runs. 20/40/60/70/80/90 % of the cost is removed before epochs 30/39/48/57/66/75, then the model recovers until epoch 99. | `mp1-gradual-bmuon-mu-wd1e3`, `mp1-gradual-bmuon-mag-wd1e3`, `mp1-gradual-muon-mag-wd1e3`, `mp1-gradual-bmuon-mu-wd1e2` | training (logs copied on 2026-11-09) |
| [`2026-11-07_full-mptrj_small-models-from-scratch`](2026-11-07_full-mptrj_small-models-from-scratch/) | Baselines for the gradual-pruning figure: small uniform models trained from scratch at the pruned sizes (Muon, wd 0.001, 100 epochs). | `mp1-uniform-c20-wd1e3` (20 % cost), `mp1-uniform-c80-wd1e3` (80 %), `mp1-uniform-c60-wd1e3` (60 %), `mp1-uniform-c14-wd1e3` (14 %, smallest uniform) | c20 done, others training |

The night-guard output of the 2026-11-03 experiment is in its folder (`night_guard.out`). The
guard outputs of the two experiments still training are in `../results/*_guard.out` until they
end.
