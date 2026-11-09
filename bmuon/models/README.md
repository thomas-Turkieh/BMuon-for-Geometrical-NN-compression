# Trained models

Nequix models (`.nqx`) trained by the group, one folder per experiment. The folder names match
[`../runs/`](../runs/) (training logs) and [`../results/`](../results/) (figures). Each model is
about 2.8 MB; the small uniform ones are smaller.

Load a model with the Nequix loader:

```python
from nequix.model import load_model
model, config = load_model("bmuon/models/2026-11-06_full-mptrj_gradual-pruning/gradual-bmuon-mu_wd1e-3_epoch074_80pct-cost-removed.nqx")
```

**File names.** `<method>_<weight decay>_epoch<NNN>[_<pruning>].nqx`
- `bmuon`: trained with BMuon; `budget30` means a 30 % cost budget.
- `muon`: trained with Muon.
- `uniform-small-costNN`: small uniform model trained with Muon at NN % of the full model's cost.
- `gradual-<optimizer>-<score>`: pruned during training. `mu` ranks channels by the BMuon
  allocation μ, `mag` by magnitude.
  - `NNpct-cost-removed`: NN % of the cost removed (cost model "paths").
  - `NNpct-channels-removed`: NN % of the 857 type-I channels removed.
- `_alloc.npz`: the BMuon allocation μ saved with the model at the same epoch. It is needed to
  prune that model by μ (`bmuon.prune`).

**Gradually pruned models are already pruned.** Their removed channels are exactly zero, so the
file is the pruned model at full shape. `bmuon.physical_prune` exports it to a smaller dense
model. Each gradually pruned model is the snapshot at the end of its phase, after recovery.

Validation errors are read from the training logs (MPtrj 5 % validation split).

### 2026-10-13_full-mptrj_dense-training_bmuon-vs-uniform

| Model | Training run | Epoch | Val force MAE (meV/Å) | Val energy MAE (meV/atom) |
|---|---|---|---|---|
| `bmuon-budget30_selective-wd-then-wd1e-1_epoch099.nqx` | `mp1-selwd-late-wd0p1` | 99 | 51.0 | 18.9 |
| `bmuon-budget30_selective-wd_epoch090.nqx` (+ `_alloc.npz`) | `mp1-bmuon-selwd` | 90 | 34.4 | 10.7 |
| `bmuon-budget30_wd1e-2_epoch080.nqx` (+ `_alloc.npz`) | `mp1-bmuon-wd1e2` | 80 | 38.6 | 13.3 |
| `bmuon-budget30_wd1e-3_epoch060.nqx` (+ `_alloc.npz`) | `mp1-bmuon-wd1e3` | 60 | 36.8 | 12.1 |
| `uniform-small-cost35_wd1e-2_epoch099.nqx` | `mp1-uniform-shrink-wd1e2` | 99 | 52.9 | 19.0 |
| `uniform-small-cost35_wd1e-3_epoch099.nqx` | `mp1-uniform-shrink-wd1e3` | 99 | 52.6 | 18.5 |

### 2026-11-03_20pct-mptrj_pruning-methods-comparison

| Model | Training run | Epoch | Val force MAE (meV/Å) | Val energy MAE (meV/atom) |
|---|---|---|---|---|
| `bmuon_epoch024.nqx` (+ `_alloc.npz`) | `fig2-bmuon` | 24 | 65.1 | 24.5 |
| `gradual-bmuon-mag_epoch006_20pct-channels-removed.nqx` | `fig2-gradual-bmuon-mag` | 6 | 103.5 | 64.1 |
| `gradual-bmuon-mag_epoch009_40pct-channels-removed.nqx` | `fig2-gradual-bmuon-mag` | 9 | 98.4 | 57.9 |
| `gradual-bmuon-mag_epoch012_55pct-channels-removed.nqx` | `fig2-gradual-bmuon-mag` | 12 | 92.5 | 51.5 |
| `gradual-bmuon-mag_epoch015_70pct-channels-removed.nqx` | `fig2-gradual-bmuon-mag` | 15 | 90.1 | 46.7 |
| `gradual-bmuon-mag_epoch018_80pct-channels-removed.nqx` | `fig2-gradual-bmuon-mag` | 18 | 88.5 | 43.2 |
| `gradual-bmuon-mag_epoch024_90pct-channels-removed.nqx` | `fig2-gradual-bmuon-mag` | 24 | 90.4 | 41.8 |
| `gradual-bmuon-mu_epoch006_20pct-channels-removed.nqx` | `fig2-gradual-bmuon-mu` | 6 | 89.9 | 50.8 |
| `gradual-bmuon-mu_epoch009_40pct-channels-removed.nqx` | `fig2-gradual-bmuon-mu` | 9 | 85.2 | 45.9 |
| `gradual-bmuon-mu_epoch012_55pct-channels-removed.nqx` | `fig2-gradual-bmuon-mu` | 12 | 79.4 | 41.0 |
| `gradual-bmuon-mu_epoch015_70pct-channels-removed.nqx` | `fig2-gradual-bmuon-mu` | 15 | 75.7 | 35.9 |
| `gradual-bmuon-mu_epoch018_80pct-channels-removed.nqx` | `fig2-gradual-bmuon-mu` | 18 | 75.3 | 33.8 |
| `gradual-bmuon-mu_epoch024_90pct-channels-removed.nqx` | `fig2-gradual-bmuon-mu` | 24 | 77.4 | 32.0 |
| `gradual-muon-mag_epoch006_20pct-channels-removed.nqx` | `fig2-gradual-muon-mag` | 6 | 88.7 | 51.1 |
| `gradual-muon-mag_epoch009_40pct-channels-removed.nqx` | `fig2-gradual-muon-mag` | 9 | 84.8 | 48.0 |
| `gradual-muon-mag_epoch012_55pct-channels-removed.nqx` | `fig2-gradual-muon-mag` | 12 | 80.7 | 42.0 |
| `gradual-muon-mag_epoch015_70pct-channels-removed.nqx` | `fig2-gradual-muon-mag` | 15 | 78.4 | 36.8 |
| `gradual-muon-mag_epoch018_80pct-channels-removed.nqx` | `fig2-gradual-muon-mag` | 18 | 78.6 | 34.5 |
| `gradual-muon-mag_epoch024_90pct-channels-removed.nqx` | `fig2-gradual-muon-mag` | 24 | 84.0 | 35.2 |
| `muon-group-lasso-2_epoch024.nqx` | `fig2-muon-gl-2` | 24 | 133.9 | 91.6 |
| `muon_epoch024.nqx` | `fig2-muon` | 24 | 60.8 | 22.0 |
| `numuon_epoch007_diverged-at-epoch008.nqx` | `fig2-numuon` | 7 | 86.8 | 48.9 |

### 2026-11-06_full-mptrj_gradual-pruning

| Model | Training run | Epoch | Val force MAE (meV/Å) | Val energy MAE (meV/atom) |
|---|---|---|---|---|
| `gradual-bmuon-mag_wd1e-3_epoch038_20pct-cost-removed.nqx` | `mp1-gradual-bmuon-mag-wd1e3` | 38 | 42.2 | 14.6 |
| `gradual-bmuon-mag_wd1e-3_epoch047_40pct-cost-removed.nqx` | `mp1-gradual-bmuon-mag-wd1e3` | 47 | 41.3 | 14.2 |
| `gradual-bmuon-mag_wd1e-3_epoch056_60pct-cost-removed.nqx` | `mp1-gradual-bmuon-mag-wd1e3` | 56 | 42.8 | 14.6 |
| `gradual-bmuon-mag_wd1e-3_epoch065_70pct-cost-removed.nqx` | `mp1-gradual-bmuon-mag-wd1e3` | 65 | 44.6 | 15.2 |
| `gradual-bmuon-mag_wd1e-3_epoch074_80pct-cost-removed.nqx` | `mp1-gradual-bmuon-mag-wd1e3` | 74 | 48.9 | 16.8 |
| `gradual-bmuon-mu_wd1e-2_epoch038_20pct-cost-removed.nqx` | `mp1-gradual-bmuon-mu-wd1e2` | 38 | 48.1 | 19.7 |
| `gradual-bmuon-mu_wd1e-2_epoch047_40pct-cost-removed.nqx` | `mp1-gradual-bmuon-mu-wd1e2` | 47 | 47.3 | 18.9 |
| `gradual-bmuon-mu_wd1e-2_epoch056_60pct-cost-removed.nqx` | `mp1-gradual-bmuon-mu-wd1e2` | 56 | 47.1 | 18.2 |
| `gradual-bmuon-mu_wd1e-2_epoch065_70pct-cost-removed.nqx` | `mp1-gradual-bmuon-mu-wd1e2` | 65 | 46.6 | 17.3 |
| `gradual-bmuon-mu_wd1e-2_epoch074_80pct-cost-removed.nqx` | `mp1-gradual-bmuon-mu-wd1e2` | 74 | 47.2 | 16.8 |
| `gradual-bmuon-mu_wd1e-3_epoch038_20pct-cost-removed.nqx` | `mp1-gradual-bmuon-mu-wd1e3` | 38 | 39.6 | 13.5 |
| `gradual-bmuon-mu_wd1e-3_epoch047_40pct-cost-removed.nqx` | `mp1-gradual-bmuon-mu-wd1e3` | 47 | 40.3 | 13.5 |
| `gradual-bmuon-mu_wd1e-3_epoch056_60pct-cost-removed.nqx` | `mp1-gradual-bmuon-mu-wd1e3` | 56 | 42.2 | 14.0 |
| `gradual-bmuon-mu_wd1e-3_epoch065_70pct-cost-removed.nqx` | `mp1-gradual-bmuon-mu-wd1e3` | 65 | 42.5 | 14.0 |
| `gradual-bmuon-mu_wd1e-3_epoch074_80pct-cost-removed.nqx` | `mp1-gradual-bmuon-mu-wd1e3` | 74 | 45.9 | 15.1 |
| `gradual-muon-mag_wd1e-3_epoch038_20pct-cost-removed.nqx` | `mp1-gradual-muon-mag-wd1e3` | 38 | 42.0 | 14.7 |
| `gradual-muon-mag_wd1e-3_epoch047_40pct-cost-removed.nqx` | `mp1-gradual-muon-mag-wd1e3` | 47 | 42.8 | 15.0 |
| `gradual-muon-mag_wd1e-3_epoch056_60pct-cost-removed.nqx` | `mp1-gradual-muon-mag-wd1e3` | 56 | 54.9 | 21.1 |
| `gradual-muon-mag_wd1e-3_epoch065_70pct-cost-removed.nqx` | `mp1-gradual-muon-mag-wd1e3` | 65 | 53.3 | 20.0 |
| `gradual-muon-mag_wd1e-3_epoch074_80pct-cost-removed.nqx` | `mp1-gradual-muon-mag-wd1e3` | 74 | 56.6 | 21.3 |

### 2026-11-07_full-mptrj_small-models-from-scratch

| Model | Training run | Epoch | Val force MAE (meV/Å) | Val energy MAE (meV/atom) |
|---|---|---|---|---|
| `uniform-small-cost20_wd1e-3_epoch099.nqx` | `mp1-uniform-c20-wd1e3` | 99 | 75.4 | 31.9 |

**Still to add when training ends:** the final models (epoch 99, 90 % of the cost removed) of the
four gradual-pruning runs, and the small models at 80 %, 60 % and 14 % of the cost.
The published Nequix-MP-1 (Muon, 100 % of the cost) is in `models/nequix-mp-1.nqx` at the
repository root.
