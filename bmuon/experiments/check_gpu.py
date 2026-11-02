"""First thing to run on a new GPU machine: checks that BMuon is correct with the kernels.

    uv run python -m bmuon.experiments.check_gpu

1. the OpenEquivariance kernel gives the same energies/forces as plain e3nn on nequix-mp-1;
2. the channel map is exact on the kernel path: zeroing a channel's linear_1 column zeroes
   the gradient of exactly the slices the map assigns to that channel;
3. a pruned model run with the kernel equals the zeroed model run without it.
Prints PASS/FAIL for each and exits with status 1 if anything fails.
"""

from __future__ import annotations

import argparse
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ase.build import bulk

from bmuon.channels import build_channel_spec
from bmuon.prune import prune, select_channels, zero_channels
from bmuon.experiments.common import pad, to_graph
from nequix.model import Nequix, load_model

FAILED = []


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILED.append(name)


def structures(config):
    out = []
    for atoms in [bulk("Si", "diamond", a=5.43, cubic=True).repeat(2),
                  bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat(2)]:
        atoms.rattle(0.05, seed=1)
        g = to_graph(atoms, config["atomic_numbers"], config["cutoff"])
        out.append(pad(g, 128, 4096))
    return out


def energy_forces(model, g):
    e, f, _ = eqx.filter_jit(lambda m, g: m(g))(model, g)
    n = int(np.asarray(g.n_node)[0])
    return float(np.asarray(e)[0]), np.asarray(f)[:n]


def check_kernel_matches(model_path):
    ref, config = load_model(model_path, kernel=False)
    ker, _ = load_model(model_path, kernel=True)
    for i, g in enumerate(structures(config)):
        e0, f0 = energy_forces(ref, g)
        e1, f1 = energy_forces(ker, g)
        de, df = abs(e1 - e0), float(np.abs(f1 - f0).max())
        report(f"kernel == e3nn on nequix-mp-1, structure {i}", de < 1e-3 and df < 1e-3,
               f"(|dE|={de:.2e} eV, max|dF|={df:.2e} eV/A)")


def small_model(kernel):
    return Nequix(jax.random.key(0), n_species=89, lmax=3, cutoff=5.0,
                  hidden_irreps="16x0e+8x1o+8x2e+8x3o", n_layers=3, radial_basis_size=8,
                  radial_mlp_size=16, radial_mlp_layers=2, index_weights=False,
                  layer_norm=True, kernel=kernel)


def check_slice_map(kernel, n_channels=24):
    model = small_model(kernel)
    spec = build_channel_spec(model)
    config = {"atomic_numbers": list(range(1, 90)), "cutoff": 5.0}
    g = jax.tree.map(jnp.asarray, structures(config)[1])
    params, static = eqx.partition(model, eqx.is_array)
    flat, treedef = jax.tree_util.tree_flatten_with_path(params)
    keys = [jax.tree_util.keystr(p) for p, _ in flat]

    @eqx.filter_jit
    def grads(p):
        return jax.grad(lambda p: jnp.sum(eqx.combine(p, static)(g)[0]))(p)

    bad = 0
    rng = np.random.default_rng(0)
    for c in rng.choice(spec.num_channels, n_channels, replace=False):
        leaves = []
        for k, (_, x) in zip(keys, flat):
            s = next((s for s in spec.slices if s.keypath == k and s.role == "A"), None)
            if s is not None:
                x = x * jnp.asarray(s.channel_ids != c, x.dtype)[None, :]
            leaves.append(x)
        gr = grads(jax.tree_util.tree_unflatten(treedef, leaves))
        gflat = {jax.tree_util.keystr(p): x for p, x in jax.tree_util.tree_flatten_with_path(gr)[0]}
        for s in spec.slices:
            if s.role == "A":
                continue
            G = np.asarray(gflat[s.keypath])
            G = G if s.axis == 0 else G.T
            mine = s.channel_ids == c
            own = np.abs(G[mine]).max() if mine.any() else 0.0
            other = np.abs(G[~mine]).max() if (~mine).any() else 1.0
            if own > 1e-6 or other == 0.0:
                bad += 1
                print(f"   channel {spec.channels[c]} {s.role} {s.ir}: own={own:.1e} other={other:.1e}")
    report(f"slice map exact ({'kernel' if kernel else 'e3nn'}, {n_channels} channels)", bad == 0)


def check_pruned_kernel():
    model = small_model(kernel=False)
    spec = build_channel_spec(model)
    keep = select_channels(np.random.default_rng(3).uniform(size=spec.num_channels), spec, 0.5)
    zeroed = zero_channels(model, spec, keep)
    pruned = prune(model, spec, keep, kernel=True)
    config = {"atomic_numbers": list(range(1, 90)), "cutoff": 5.0}
    for i, g in enumerate(structures(config)):
        e0, f0 = energy_forces(zeroed, g)
        e1, f1 = energy_forces(pruned, g)
        de, df = abs(e1 - e0), float(np.abs(f1 - f0).max())
        report(f"pruned (kernel) == zeroed (e3nn), structure {i}", de < 1e-3 and df < 1e-3,
               f"(|dE|={de:.2e}, max|dF|={df:.2e})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/nequix-mp-1.nqx")
    p.add_argument("--no-kernel", action="store_true", help="CPU dry run of check 2 only")
    args = p.parse_args()
    print("devices:", jax.devices(), flush=True)
    if args.no_kernel:
        check_slice_map(kernel=False, n_channels=6)
    else:
        check_kernel_matches(args.model)
        check_slice_map(kernel=True)
        check_pruned_kernel()
    print("ALL PASSED" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
