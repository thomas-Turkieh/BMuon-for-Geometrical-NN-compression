"""Section 6.2: inference time of Nequix vs the multiplicities entering the tensor product.

    python -m bmuon.experiments.speed --kernel --out bmuon/results/speed_3090 \
        --cost-json bmuon/configs/costs_3090.json

Times jitted energy + forces on Si supercells (64, 216, 512 atoms) for
  * uniform shrinks of every multiplicity (fraction f of the copies kept),
  * one variant per (block, irrep) with half of its copies removed  -> cost of a copy.

Every variant is timed **interleaved with the base model** (base, variant, base, variant, ...)
and reported as a ratio to the base times measured around it, so that slow drifts of the GPU
(clock, temperature, memory) cancel. The first version timed variants one after the other and
the base once; the times drifted by up to +20 % over the run, more than the effects measured.
GPU clock and temperature are recorded with every variant.

Writes <out>.csv (one row per variant) and <out>.json (summary + {"channel_costs"}).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import e3nn_jax as e3nn
import equinox as eqx
import jax
import numpy as np
from ase.build import bulk

from bmuon.experiments.common import bucket, pad, platform, to_graph
from bmuon.model import with_tp_irreps
from bmuon.prune import tp_irreps_summary
from nequix.model import load_model


def structures(config, sizes):
    out = []
    for n in sizes:
        atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((n, n, n))
        atoms.rattle(0.02, seed=n)
        g = to_graph(atoms, config["atomic_numbers"], config["cutoff"])
        out.append((len(atoms), pad(g, bucket(len(atoms) + 1, 32), bucket(int(g.n_edge[0]) + 1, 256))))
    return out


@eqx.filter_jit
def _energy_forces(m, g):
    e, forces, _ = m(g)
    return e, forces


def _once(model, g) -> float:
    t0 = time.perf_counter()
    jax.block_until_ready(_energy_forces(model, g))
    return time.perf_counter() - t0


def warm(model, graphs, n: int = 3):
    for _, g in graphs:
        for _ in range(n):
            _once(model, g)


def time_model(model, graphs, reps: int, warmup: int = 3, stress: bool = False):
    """Median time per structure (kept for other scripts; not interleaved)."""
    warm(model, graphs, warmup)
    return [float(np.median([_once(model, g) for _ in range(reps)])) for _, g in graphs]


def interleaved(base, variant, graphs, rounds: int, reps: int):
    """Alternate base and variant; return (t_base, t_variant, ratio) per structure size.

    ratio = median over rounds of t_variant / mean(t_base before, t_base after)."""
    warm(variant, graphs)
    tb, tv, ratios = [], [], []
    for _, g in graphs:
        b = [np.median([_once(base, g) for _ in range(reps)])]
        v = []
        for _ in range(rounds):
            v.append(np.median([_once(variant, g) for _ in range(reps)]))
            b.append(np.median([_once(base, g) for _ in range(reps)]))
        r = [v[i] / (0.5 * (b[i] + b[i + 1])) for i in range(rounds)]
        tb.append(float(np.median(b)))
        tv.append(float(np.median(v)))
        ratios.append(float(np.median(r)))
    return tb, tv, ratios


def gpu_state() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip().replace(" ", "")
    except Exception:
        return ""


def scaled(irreps: e3nn.Irreps, factor: float) -> e3nn.Irreps:
    return e3nn.Irreps([(max(1, int(round(m * factor))), ir) for m, ir in irreps])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/nequix-mp-1.nqx")
    p.add_argument("--sizes", default="2,3,4", help="Si cubic supercell repeats (8 n^3 atoms)")
    p.add_argument("--rounds", type=int, default=5, help="base/variant alternations")
    p.add_argument("--reps", type=int, default=10, help="timed calls per measurement")
    p.add_argument("--fractions", default="0.875,0.75,0.625,0.5,0.375,0.25,0.125")
    p.add_argument("--no-per-irrep", action="store_true")
    p.add_argument("--kernel", action="store_true", help="use OpenEquivariance kernels (GPU)")
    p.add_argument("--out", default="bmuon/results/speed")
    p.add_argument("--cost-json", default=None)
    args = p.parse_args()

    model, config = load_model(args.model, kernel=False)
    base_irreps = [e3nn.Irreps(s) for s in tp_irreps_summary(model)]
    base = with_tp_irreps(model, base_irreps, kernel=args.kernel)
    graphs = structures(config, [int(s) for s in args.sizes.split(",")])
    atoms = [n for n, _ in graphs]
    print(f"{platform()}  structures: {atoms} atoms  gpu(clock MHz,temp C,power W): {gpu_state()}",
          flush=True)
    warm(base, graphs, 5)

    rows = []

    def run(name, irreps, removed=None):
        variant = with_tp_irreps(model, irreps, kernel=args.kernel)
        tb, tv, ratio = interleaved(base, variant, graphs, args.rounds, args.reps)
        row = {"variant": name, "tp_irreps": " | ".join(map(str, irreps)), "gpu_state": gpu_state()}
        for n, b, v, r in zip(atoms, tb, tv, ratio):
            row[f"base_{n}_ms"] = 1e3 * b
            row[f"t_{n}_ms"] = 1e3 * v
            row[f"ratio_{n}"] = r
        if removed:
            row["removed_layer"], row["removed_ir"], row["removed_copies"] = removed
        rows.append(row)
        print(f"{name:14s} " + "  ".join(f"{1e3 * v:7.2f} ms ({100 * (r - 1):+5.1f}%)"
                                        for v, r in zip(tv, ratio)) + f"  [{row['gpu_state']}]",
              flush=True)
        del variant
        return np.array(ratio)

    for f in [float(x) for x in args.fractions.split(",")]:
        run(f"uniform_{f:g}", [scaled(x, f) for x in base_irreps])

    marginal = {}
    if not args.no_per_irrep:
        for t, irreps in enumerate(base_irreps):
            for i, (mul, ir) in enumerate(irreps):
                if mul < 2:
                    continue
                k = mul // 2
                new = list(base_irreps)
                new[t] = e3nn.Irreps([(m - k if j == i else m, r) for j, (m, r) in enumerate(irreps)])
                ratio = run(f"b{t}_{ir}_half", new, (t, str(ir), k))
                # fraction of the base time saved per removed copy, averaged over sizes
                marginal[f"{t}:{ir}"] = float(np.mean((1.0 - ratio) / k))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.csv", "w", newline="") as fh:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "variant", k))
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    result = {
        "platform": platform(),
        "kernel": args.kernel,
        "atoms": atoms,
        "uniform": {r["variant"]: [r[f"ratio_{n}"] for n in atoms]
                    for r in rows if r["variant"].startswith("uniform")},
        "saved_fraction_per_copy": marginal,
    }
    if marginal:
        vals = np.array(list(marginal.values()))
        noisy = bool((vals <= 0).any())
        floor = max(vals[vals > 0].min() * 0.1, 1e-9) if (vals > 0).any() else 1.0
        costs = {k: max(v, floor) for k, v in marginal.items()}
        mean = float(np.mean(list(costs.values())))
        result["channel_costs"] = {k: v / mean for k, v in costs.items()}
        result["warning"] = ("some per-copy savings are <= 0: below measurement noise, "
                             "do not use these costs as is") if noisy else ""
        if args.cost_json and not noisy:
            with open(args.cost_json, "w") as fh:
                json.dump({"channel_costs": result["channel_costs"], "platform": platform()},
                          fh, indent=2)
        elif args.cost_json:
            print("per-copy costs are below noise: cost JSON not written", flush=True)
    with open(f"{out}.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
