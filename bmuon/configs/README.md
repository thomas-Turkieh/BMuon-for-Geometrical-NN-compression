# Training configs

Configs for `python -m bmuon.train --config FILE` (one GPU) or the multi-machine launcher
`bmuon/experiments/launch_dist.sh`. A config may start with `base_config: <path>` (relative to the
file): it is loaded first, then overridden key by key.

**Base configs** (this folder):

| File | Recipe |
|---|---|
| `nequix-mp-1-muon.yml` | Nequix-MP-1 with Muon, full MPtrj, 100 epochs: the published recipe. |
| `nequix-mp-1-bmuon.yml` | The same with BMuon on the type-I channels (30 % cost budget, annealed over the first 25 % of updates). |
| `nequix-mp-1-bmuon-selwd.yml` | The dense Nequix-MP-1 reference of the November experiments (`mp1-bmuon-selwd`). Its `selective_weight_decay_prune` key is at the top level, where the trainer does not read it (it expects it under `bmuon:`), so the run used plain weight decay 0.001. |
| `nequix-mp-1-uniform.yml`, `nequix-mp-1-uniform-wd1e3.yml` | Small uniform Nequix at 30 % of the cost (`40x0e + 16x1o + 8x2e + 8x3o`), Muon. |
| `small-muon.yml`, `small-bmuon.yml` | Development model (`128x0e + 64x1o`, 10 % of MPtrj, 20 epochs). |
| `toy-bmuon.yml` | CPU smoke test on a synthetic dataset (`bmuon/tests/make_toy_db.py`). |

**Experiment folders** — one per experiment, named like [`../runs/`](../runs/),
[`../models/`](../models/) and [`../results/`](../results/): `<start date>_<data>_<what is tested>`.
The run id of each config is its `out_dir`.

| Folder | Experiment |
|---|---|
| [`2026-11-03_20pct-mptrj_pruning-methods-comparison`](2026-11-03_20pct-mptrj_pruning-methods-comparison/) | Post-training and gradual pruning of Muon, Muon + group lasso, NuMuon and BMuon (20 % of MPtrj, 25 epochs). |
| [`2026-11-06_full-mptrj_gradual-pruning`](2026-11-06_full-mptrj_gradual-pruning/) | Gradual pruning during training to 90 % of the cost removed (full MPtrj, 100 epochs). |
| [`2026-11-07_full-mptrj_small-models-from-scratch`](2026-11-07_full-mptrj_small-models-from-scratch/) | Small uniform models trained from scratch at the pruned sizes. |
| [`2026-11-10_full-mptrj_learned-architecture-from-scratch`](2026-11-10_full-mptrj_learned-architecture-from-scratch/) | Architecture kept by BM-μ at 70 % of the cost removed vs a uniform one, both from scratch. |
| [`2026-11-10_full-mptrj_wide-models-pruned-to-nequix-size`](2026-11-10_full-mptrj_wide-models-pruned-to-nequix-size/) | 1.25× to 6× wider models gradually pruned to the size of Nequix-MP-1 (x4 and x6 stopped early). |
| [`2026-11-22_full-mptrj_x1.5-pruned-architecture-from-scratch`](2026-11-22_full-mptrj_x1.5-pruned-architecture-from-scratch/) | Architecture kept by the 1.5× wide model, retrained from scratch at the size of Nequix-MP-1. |

The October runs (`2026-10-13_full-mptrj_dense-training_bmuon-vs-uniform`) use the base configs
directly. The run plans read by the night guard are in `../experiments/*_runs.yml`.
