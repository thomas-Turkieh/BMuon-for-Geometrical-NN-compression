# BMuon for Nequix

Everything specific to this project lives in this folder `bmuon/`. The rest of the repository
is the upstream Nequix code (github.com/atomicarchitects/nequix, commit da0fb24, MIT license),
**unmodified** (`git diff da0fb24 -- . ':!bmuon'` is empty).

```
bmuon/
├── channels.py      map of the type-I channels (slices in linear_1, linear_2, radial MLP), costs
├── optimizer.py     BMuon as optax transformations + optax Muon/AdamW for the other parameters
├── model.py         PrunableConvolution (tensor product fed by fewer copies), save/load
├── prune.py         channel scores, global selection, export to a smaller dense model
├── train.py         training loop (Muon or BMuon), gradient accumulation, auto-resume
├── configs/         nequix-mp-1-{muon,bmuon}, small-{muon,bmuon} (dev), toy-bmuon (CPU smoke test)
├── experiments/
│   ├── setup_machine.sh one-command install on a lab machine (venv, CUDA root, OEQ, MPtrj)
│   ├── check_gpu.py     first run on a GPU machine: kernel == e3nn, slice map exact with kernels
│   ├── fragility.py     §6.1/6.3 accuracy after pruning, no fine-tuning
│   ├── speed.py         §6.2 timings and measured channel costs (costs_<gpu>.json)
│   ├── launch.py        spread runs over the lab machines by SSH
│   ├── run_forever.sh   one run, resumed until done, results pushed to GitHub
│   ├── launch_dist.sh   one run spread over several machines (data parallel)
│   ├── publish_results.sh  commit + push bmuon/results and run logs (rebase, retries)
│   ├── machines.txt     the 169 lab machines with their GPU
│   └── runs_*.yml       run plans (pilot, wave 3)
├── tests/           14 tests (see below)
├── runs/            training logs, one folder per experiment (<date>_<data>_<principle>), then
│                    one per run: config, log.jsonl, hosts.log, alloc every 10 epochs, pruning
│                    masks; checkpoints and optimizer states stay on the machine (runs/README.md)
├── results/         figures and tables, same experiment folders as runs/ (results/README.md)
├── models/          trained models (.nqx), same experiment folders (models/README.md)
└── data/            MPtrj, generated, git-ignored
```

All commands are run from the repository root.

## Install (pinned like Nequix)

On the lab machines (no CUDA toolkit, small home quota), one command per machine installs
everything on the local disk `/tmp/$USER` (~10 min): the environment, a CUDA 12 root assembled
from the NVIDIA pip wheels so that the OpenEquivariance extension compiles, and the MPtrj data
(downloaded and preprocessed into `/tmp/$USER/bmuon-data/mptrj-aselmdb`):

    bash bmuon/experiments/setup_machine.sh
    source /tmp/$USER/nequix-venv/bin/activate

`run_forever.sh` runs it automatically on a machine that is not set up yet.

The code was tested with jax 0.5.3, e3nn-jax 0.20.8, equinox 0.13.1, optax 0.2.6, lmdb 1.x
(the versions of `uv.lock`). lmdb 3.x breaks Nequix's forked data workers; jax ≥ 0.10 removed
`jax.device_put_replicated`, used by the training loop.

## Tests

    python -m pytest bmuon/tests

* the slice map is exact: zeroing a channel's `linear_1` column zeroes the gradient of all its
  other slices and only them;
* with the full budget, BMuon reproduces Nequix's `optax.contrib.muon` updates step by step;
* the allocation solver satisfies the cost budget and beats random feasible allocations;
* low-allocation channels move by at most ~mu^alpha;
* the exported pruned model equals the model with zeroed slices (energy, forces), whole irreps
  can disappear, the pruned model is O(3)-equivariant (rotations and inversion), save/load works.

## Typical commands

    python -m bmuon.experiments.check_gpu
    python -m bmuon.experiments.speed --kernel --out bmuon/results/speed_3090 --cost-json bmuon/configs/costs_3090.json
    python -m bmuon.experiments.fragility --model models/nequix-mp-1.nqx --data $BMUON_DATA/mptrj-aselmdb --n-structures 2000
    python -m bmuon.train bmuon/configs/small-bmuon.yml --set bmuon.budget=0.3 out_dir=bmuon/runs/test
    python -m bmuon.experiments.launch plan  bmuon/experiments/runs_wave3.yml bmuon/experiments/machines.txt
    python -m bmuon.experiments.launch start bmuon/experiments/runs_wave3.yml bmuon/experiments/machines.txt
    python -m bmuon.experiments.launch status bmuon/experiments/runs_wave3.yml

One run over several machines (global batch = batch_size x grad_accum_steps x hosts, keep 256):

    H="ablette anchois anguille barbeau"
    bash bmuon/experiments/launch_dist.sh check "$H"
    bash bmuon/experiments/launch_dist.sh start mp1-bmuon-30 bmuon/configs/nequix-mp-1-bmuon.yml "$H" --set batch_size=64 grad_accum_steps=1
    bash bmuon/experiments/launch_dist.sh status mp1-bmuon-30 "$H"

## Implementation notes

* e3nn-jax regroups the input of `linear_2`: there is **one matrix per output irrep of the
  tensor product** (rows = all (path, copy) pairs producing that irrep), not one per path.
  BMuon orthogonalizes that whole matrix with the channel scaling on its rows.
* e3nn `Linear` multiplies each weight by a constant depending on its fan-in; the export
  rescales `linear_2` rows so that the pruned model is exactly the zeroed one.
* Weights are stored `(in, out)` and applied as `x @ W`: a channel is a **column** of
  `linear_1` and of the radial layer, and a **row** of `linear_2`.
