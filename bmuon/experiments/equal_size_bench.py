"""Equal-size comparison: a wide model gradually pruned to the size of Nequix-MP-1 vs the dense
Nequix-MP-1 trained with the same recipe. Measurements for ``equal_size_fig.py``, on one GPU.

    python -m bmuon.experiments.equal_size_bench \
        --dense bmuon/runs/mp1-bmuon-selwd --wide bmuon/runs/mp1-wide-x1.5-gradual-bmuon-mu \
        --epoch 99 --out ~/equal_size_bench [--parts errors,inference,training] [--n-val N]

Parts (each writes its own file in --out):
  errors     per-structure errors of both final models on the whole validation split
             (the 5 % MPtrj split of every training run) -> errors.npz
  inference  time of energy + forces (jitted, OpenEquivariance kernels) on a fixed sample of
             validation structures, models timed in interleaved rounds -> inference.csv
  training   time of one training step (loss, gradients, BMuon update) at the per-GPU batch
             of the wide run, for the dense model, the wide model at full width (as trained:
             pruned channels are masked, not removed) and the physically pruned architecture
             of every pruning phase -> training.csv
  meta.json  parameters, tensor-product paths, GPU, phases, steps per epoch.

The wide final model is evaluated and timed after physical pruning (``bmuon.physical_prune``),
which gives the same function as the masked model; the script checks it on a sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import e3nn_jax as e3nn
import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax

from bmuon.channels import build_channel_spec, tp_input_irreps, tp_paths
from bmuon.experiments.common import bucket
from bmuon.experiments.fig2 import load_eval_set
from bmuon.model import apply_tp_config, load_pruned
from bmuon.physical_prune import kept_counts, physically_prune
from bmuon.train import build_optimizer
from nequix.data import AseDBDataset, DataLoader
from nequix.model import Nequix, load_model
from nequix.train import loss as nequix_loss


# ------------------------------------------------------------------ models


def n_params(model) -> int:
    return int(sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))))


def n_tp_paths(model) -> int:
    """Tensor-product paths fed by the input copies (the "paths" cost model of the pruning)."""
    sh = e3nn.s2_irreps(model.lmax)
    return int(sum(mul for conv in model.layers for _, mul in tp_paths(conv, sh)))


def load_dense(run: Path, epoch: int):
    path = run / f"snap_epoch{epoch:03d}.nqx"
    if not path.exists():  # older runs: no snapshot at the last epoch, last.nqx is the EMA model
        path = run / "last.nqx"
    print(f"dense model: {path}", flush=True)
    return load_model(str(path), kernel=True)


def load_wide(run: Path, epoch: int):
    """Final wide snapshot (masked, full width) and its physically pruned version."""
    model, config = load_pruned(str(run / f"snap_epoch{epoch:03d}.nqx"), kernel=True)
    spec = build_channel_spec(model, cost_model=config["bmuon"]["cost_model"])
    last = sorted(run.glob("prune_epoch*.npz"))[-1]
    keep = np.load(last)["keep"].astype(bool)
    pruned = physically_prune(model, spec, keep, kernel=True)
    return model, pruned, config, spec


def phase_irreps(wide_model, spec, keep) -> list:
    """tp_input_irreps (one string per block) of the wide model with only ``keep`` channels."""
    counts = kept_counts(spec, keep)
    out = []
    for t, conv in enumerate(wide_model.layers):
        out.append(" + ".join(f"{counts[(t, str(ir))]}x{ir}" for _, ir in tp_input_irreps(conv)))
    return out


def build_from_config(config: dict):
    """A freshly initialised model of the architecture described by a run config."""
    model = Nequix(
        jax.random.key(0),
        n_species=len(config["atomic_numbers"]),
        hidden_irreps=config["hidden_irreps"],
        lmax=config["lmax"],
        cutoff=config["cutoff"],
        n_layers=config["n_layers"],
        radial_basis_size=config["radial_basis_size"],
        radial_mlp_size=config["radial_mlp_size"],
        radial_mlp_layers=config["radial_mlp_layers"],
        radial_polynomial_p=config["radial_polynomial_p"],
        mlp_init_scale=config["mlp_init_scale"],
        index_weights=config["index_weights"],
        layer_norm=config["layer_norm"],
        shift=config["shift"],
        scale=config["scale"],
        avg_n_neighbors=config["avg_n_neighbors"],
        atom_energies=([{int(k): v for k, v in config["atom_energies"].items()}[int(n)]
                        for n in config["atomic_numbers"]]
                       if isinstance(config.get("atom_energies"), dict) else config.get("atom_energies")),
        kernel=True,
    )
    model, _ = apply_tp_config(model, {**config, "kernel": True}, jax.random.key(7))
    return model


# ------------------------------------------------------------------ inputs


def padded_batches(graphs, batch: int = 32, cap: int = 60000):
    """Graphs grouped by size into padded batches (as bmuon.experiments.fig2.predict)."""
    edges = [int(g.n_edge[0]) for g in graphs]
    order = list(np.argsort(edges))
    groups, cur, cur_e = [], [], 0
    for i in order:
        if cur and (len(cur) == batch or cur_e + edges[i] > cap):
            groups.append(cur)
            cur, cur_e = [], 0
        cur.append(i)
        cur_e += edges[i]
    if cur:
        groups.append(cur)
    out = []
    for idx in groups:
        gb = jraph.batch([graphs[i] for i in idx])
        n, m = int(np.sum(gb.n_node)), int(np.sum(gb.n_edge))
        gp = jraph.pad_with_graphs(gb, n_node=bucket(n + 1, 256), n_edge=bucket(m + 1, 4096),
                                   n_graph=batch + 1)
        out.append((idx, jax.device_put(gp)))
    return out


@eqx.filter_jit
def energy_forces(model, g):
    e, f, _ = model(g)
    return e, f


def predict(model, batches, graphs):
    es, fs = [None] * len(graphs), [None] * len(graphs)
    for idx, gp in batches:
        e, f = energy_forces(model, gp)
        e, f = np.asarray(e), np.asarray(f)
        off = 0
        for j, i in enumerate(idx):
            k = int(graphs[i].n_node[0])
            es[i], fs[i] = float(e[j]), f[off:off + k]
            off += k
    return np.array(es), fs


def per_structure(model, batches, graphs, e_true, f_true, n_atoms):
    e, f = predict(model, batches, graphs)
    f_err = np.abs(np.concatenate(f) - f_true).sum(axis=1)  # |dFx|+|dFy|+|dFz| per atom
    bounds = np.concatenate([[0], np.cumsum(n_atoms)])
    f_sum = np.add.reduceat(f_err, bounds[:-1])  # per structure
    return np.abs(e - e_true) / n_atoms, f_sum


def gpu_info() -> dict:
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu",
                            "--format=csv,noheader"], capture_output=True, text=True).stdout
        return {"nvidia_smi": q.strip()}
    except OSError:
        return {}


# ------------------------------------------------------------------ parts


def part_errors(dense, pruned, masked, data, config, out: Path, n_val):
    t0 = time.time()
    graphs, e_true, f_true, n_atoms = load_eval_set(data, config, n_val or 10**9)
    print(f"validation split: {len(graphs)} structures, {int(n_atoms.sum())} atoms "
          f"(loaded in {time.time() - t0:.0f} s)", flush=True)
    batches = padded_batches(graphs)
    res = {"n_atoms": n_atoms}
    for name, m in [("dense", dense), ("wide_pruned", pruned)]:
        t0 = time.time()
        res[f"{name}_e_abs_per_atom"], res[f"{name}_f_abs_sum"] = per_structure(
            m, batches, graphs, e_true, f_true, n_atoms)
        e_mae = res[f"{name}_e_abs_per_atom"].mean() * 1000
        f_mae = res[f"{name}_f_abs_sum"].sum() / (3 * n_atoms.sum()) * 1000
        print(f"{name:12s} energy MAE {e_mae:.3f} meV/atom, force MAE {f_mae:.3f} meV/A "
              f"({time.time() - t0:.0f} s)", flush=True)
    # masked (full-width) model on a sample: must give the same predictions as the pruned one
    k = min(2000, len(graphs))
    sub = padded_batches(graphs[:k])
    _, f_m = predict(masked, sub, graphs[:k])
    _, f_p = predict(pruned, sub, graphs[:k])
    dmax = float(np.max(np.abs(np.concatenate(f_m) - np.concatenate(f_p))))
    print(f"masked vs physically pruned, max |dF| on {k} structures: {dmax:.2e} eV/A", flush=True)
    res["masked_vs_pruned_max_dF"] = np.array(dmax)
    np.savez(out / "errors.npz", **res)


def part_inference(models: dict, data, config, out: Path, n: int = 2048, rounds: int = 20):
    graphs, *_ = load_eval_set(data, config, n, seed=1)
    batches = padded_batches(graphs)
    n_atoms = int(sum(int(g.n_node[0]) for g in graphs))

    def run(m):
        for _, gp in batches:
            e, f = energy_forces(m, gp)
        jax.block_until_ready((e, f))

    for m in models.values():  # compile every bucket
        run(m)
        run(m)
    rows = []
    names = list(models)
    for r in range(rounds):
        order = names if r % 2 == 0 else names[::-1]
        for name in order:
            t0 = time.perf_counter()
            run(models[name])
            dt = time.perf_counter() - t0
            rows.append({"round": r, "model": name, "seconds": dt, "n_structures": len(graphs),
                         "n_atoms": n_atoms, "ms_per_structure": 1000 * dt / len(graphs)})
        print(f"inference round {r}: " + ", ".join(
            f"{x['model']} {x['ms_per_structure']:.3f} ms" for x in rows[-len(names):]), flush=True)
    with open(out / "inference.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def part_training(arch_configs: dict, opt_configs: dict, data, config, out: Path,
                  n_batches: int = 30, rounds: int = 5):
    ds = AseDBDataset(data, config["atomic_numbers"], config["cutoff"], backend="jax")
    train_ds, _ = ds.split(valid_frac=config["valid_frac"])
    loader = DataLoader(train_ds, shuffle=True, seed=0, batch_size=config["batch_size"],
                        max_n_nodes=config["max_n_nodes"], max_n_edges=config["max_n_edges"],
                        avg_n_nodes=config["avg_n_nodes"], avg_n_edges=config["avg_n_edges"],
                        num_workers=8)
    batches = []
    for b in loader:
        batches.append(jax.device_put(b))
        if len(batches) == n_batches:
            break
    n_struct = int(sum(int(np.sum(np.asarray(jraph.get_graph_padding_mask(b)))) for b in batches))
    print(f"training batches: {len(batches)} of up to {config['batch_size']} structures "
          f"({n_struct} structures)", flush=True)
    total_steps = config["n_epochs"] * 5864
    schedule = optax.warmup_cosine_decay_schedule(1e-3, 1e-2, 1e-6, 586, total_steps)

    steppers = {}
    for name, arch in arch_configs.items():
        model = build_from_config(arch)
        optim, _ = build_optimizer(opt_configs[name], model, schedule, total_steps)
        opt_state = optim.init(eqx.filter(model, eqx.is_array))
        c = opt_configs[name]

        def make(optim=optim, c=c):
            @eqx.filter_jit
            def step(model, opt_state, batch):
                (l, _), g = eqx.filter_value_and_grad(nequix_loss, has_aux=True)(
                    model, batch, c["energy_weight"], c["force_weight"], c["stress_weight"],
                    c["loss_type"])
                updates, opt_state = optim.update(g, opt_state, eqx.filter(model, eqx.is_array))
                return eqx.apply_updates(model, updates), opt_state, l
            return step

        steppers[name] = [make(), model, opt_state, n_params(model), n_tp_paths(model)]
        print(f"{name}: {steppers[name][3]} parameters, {steppers[name][4]} TP paths", flush=True)

    def run(name, k=None):
        step, model, opt_state = steppers[name][:3]
        for b in batches[:k] if k else batches:
            model, opt_state, l = step(model, opt_state, b)
        jax.block_until_ready(l)
        steppers[name][1], steppers[name][2] = model, opt_state

    for name in steppers:  # compile + warm up
        t0 = time.time()
        run(name, 3)
        print(f"{name}: compiled in {time.time() - t0:.0f} s", flush=True)
    rows, names = [], list(steppers)
    for r in range(rounds):
        order = names if r % 2 == 0 else names[::-1]
        for name in order:
            t0 = time.perf_counter()
            run(name)
            dt = time.perf_counter() - t0
            rows.append({"round": r, "model": name, "seconds": dt, "steps": len(batches),
                         "ms_per_step": 1000 * dt / len(batches),
                         "params": steppers[name][3], "tp_paths": steppers[name][4]})
        print(f"training round {r}: " + ", ".join(
            f"{x['model']} {x['ms_per_step']:.1f} ms" for x in rows[-len(names):]), flush=True)
    with open(out / "training.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ main


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dense", required=True)
    p.add_argument("--wide", required=True)
    p.add_argument("--epoch", type=int, default=99)
    p.add_argument("--out", required=True)
    p.add_argument("--parts", default="errors,inference,training")
    p.add_argument("--n-val", type=int, default=0, help="0 = whole validation split")
    p.add_argument("--data", default=f"/tmp/{os.environ.get('USER')}/bmuon-data/mptrj-aselmdb")
    args = p.parse_args()
    parts = args.parts.split(",")
    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)
    dense_run, wide_run = Path(args.dense), Path(args.wide)

    dense, dense_cfg = load_dense(dense_run, args.epoch)
    masked, pruned, wide_cfg, spec = load_wide(wide_run, args.epoch)
    meta = {
        "gpu": gpu_info(), "jax_devices": [str(d) for d in jax.devices()],
        "epoch": args.epoch, "dense_run": dense_run.name, "wide_run": wide_run.name,
        "params": {"dense": n_params(dense), "wide_pruned": n_params(pruned),
                   "wide_full": n_params(masked)},
        "tp_paths": {"dense": n_tp_paths(dense), "wide_pruned": n_tp_paths(pruned),
                     "wide_full": n_tp_paths(masked)},
        "tp_irreps": {"dense": [str(tp_input_irreps(c)) for c in dense.layers],
                      "wide_pruned": [str(tp_input_irreps(c)) for c in pruned.layers],
                      "wide_full": [str(tp_input_irreps(c)) for c in masked.layers]},
    }
    print(json.dumps(meta, indent=1), flush=True)

    if "errors" in parts:
        part_errors(dense, pruned, masked, args.data, wide_cfg, out, args.n_val)
    if "inference" in parts:
        part_inference({"dense": dense, "wide_pruned": pruned, "wide_full_masked": masked},
                       args.data, wide_cfg, out)
    if "training" in parts:
        # phases of the wide run: full width, then the channels kept after each pruning event
        keeps = [np.ones(spec.num_channels, bool)] + [
            np.load(f)["keep"].astype(bool) for f in sorted(wide_run.glob("prune_epoch*.npz"))]
        events = wide_cfg["gradual"]["prune_after_epochs"]
        bounds = [0] + [e + 1 for e in events] + [wide_cfg["n_epochs"]]
        meta["phases"] = []
        base = {k: v for k, v in wide_cfg.items() if k not in ("tp_expand", "tp_input_irreps")}
        dense_arch = {**base}
        arch, opt = {"dense": dense_arch}, {"dense": {**dense_cfg, **{k: base[k] for k in (
            "batch_size", "max_n_nodes", "max_n_edges")}}}
        # phase0 = the wide model at full width, which is also what every phase computes
        # when pruning only masks channels (as in the training run)
        w = np.asarray(spec.weights, np.float64)
        for i, keep in enumerate(keeps):
            name = f"phase{i}"
            irreps = phase_irreps(masked, spec, keep)
            arch[name] = {**base, "tp_input_irreps": irreps}
            opt[name] = {**base, "tp_input_irreps": irreps}
            meta["phases"].append({"name": name, "epochs": [bounds[i], bounds[i + 1] - 1],
                                   "n_epochs": bounds[i + 1] - bounds[i],
                                   "cost_kept_paths": float(w[keep].sum() / w.sum()),
                                   "tp_irreps": irreps})
        part_training(arch, opt, args.data, wide_cfg, out)
    meta["gpu_end"] = gpu_info()
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
