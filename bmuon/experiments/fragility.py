"""Section 6.1 / 6.3: accuracy of a model after global type-I channel pruning, no fine-tuning.

    python -m bmuon.experiments.fragility --model models/nequix-mp-1.nqx \
        --data $BMUON_DATA/mptrj-aselmdb --n-structures 2000 --out bmuon/results/fragility_muon.csv

Without --data, measures only the deviation from the unpruned model on toy crystals.
For a BMuon run, pass --alloc bmuon/runs/<run>/alloc_epochNNN.npz --score alloc.
Channels are removed by zeroing their slices, which is exactly equivalent to the export
(tested in bmuon/tests/test_prune.py) and avoids recompiling per pruning level.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from bmuon.channels import build_channel_spec
from bmuon.prune import magnitude_scores, normalize_by_group, select_channels, zero_channels
from bmuon.experiments.common import platform, predict, to_graph, toy_structures
from nequix.model import load_model


def load_structures(args, config):
    if args.data is None:
        atoms = toy_structures(args.n_structures)
        return [to_graph(a, config["atomic_numbers"], config["cutoff"]) for a in atoms], None
    from nequix.data import AseDBDataset

    ds = AseDBDataset(args.data, config["atomic_numbers"], config["cutoff"], backend="jax")
    if args.valid_frac > 0:
        # same split as training (nequix.data.Dataset.split, seed 42): held-out structures only
        _, ds = ds.split(valid_frac=args.valid_frac)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), min(args.n_structures, len(ds)), replace=False)
    graphs = [ds[int(i)] for i in idx]
    labels = [(float(g.globals["energy"][0]), np.asarray(g.nodes["forces"])) for g in graphs]
    return graphs, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", default=None, help="aselmdb file or directory used for training")
    p.add_argument("--n-structures", type=int, default=32)
    p.add_argument("--valid-frac", type=float, default=0.05,
                   help="evaluate on the validation split used in training (0: whole --data)")
    p.add_argument("--fractions", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    p.add_argument("--by", choices=["count", "cost"], default="count")
    p.add_argument("--score", choices=["magnitude", "magnitude-norm", "alloc", "random"],
                   default="magnitude-norm",
                   help="magnitude-norm: coupled magnitude divided by its (block, irrep) median")
    p.add_argument("--selection", choices=["global", "per-group"], default="global",
                   help="per-group: same fraction removed in every (block, irrep)")
    p.add_argument("--alloc", default=None, help="alloc_epochNNN.npz from bmuon.train")
    p.add_argument("--cost-model", default="cg")
    p.add_argument("--min-keep", type=int, default=1)
    p.add_argument("--exclude-blocks", default="",
                   help="comma-separated blocks never pruned, e.g. 0 (species embedding)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="bmuon/results/fragility.csv")
    args = p.parse_args()

    model, config = load_model(args.model, kernel=False)
    spec = build_channel_spec(model, cost_model=args.cost_model)
    print(f"{args.model}: {spec.num_channels} type-I channels\n{spec.summary()}", flush=True)
    if args.score == "magnitude":
        scores = magnitude_scores(model, spec)
    elif args.score == "magnitude-norm":
        scores = normalize_by_group(magnitude_scores(model, spec), spec)
    elif args.score == "alloc":
        scores = np.load(args.alloc)["alloc"]
        assert len(scores) == spec.num_channels, "allocation does not match the model"
    else:
        scores = np.random.default_rng(args.seed).uniform(size=spec.num_channels)

    graphs, labels = load_structures(args, config)
    n_atoms = np.array([int(g.n_node[0]) for g in graphs])
    e_ref, f_ref = predict(model, graphs)

    rows = []
    for frac in [float(x) for x in args.fractions.split(",")]:
        excl = tuple(int(x) for x in args.exclude_blocks.split(",") if x != "")
        keep = select_channels(scores, spec, frac, by=args.by, min_keep=args.min_keep,
                               exclude_layers=excl, per_group=args.selection == "per-group")
        e, f = predict(zero_channels(model, spec, keep), graphs)
        row = {
            "fraction": frac,
            "by": args.by,
            "exclude_blocks": args.exclude_blocks,
            "selection": args.selection,
            "score": args.score,
            "kept_channels": int(keep.sum()),
            "kept_cost": float(spec.weights[keep].sum() / spec.weights.sum()),
            **{f"kept_b{t}_{ir}": int(sum(keep[i] for i, c in enumerate(spec.channels)
                                        if c.layer == t and c.ir == ir))
               for t, ir in sorted({(c.layer, c.ir) for c in spec.channels})},
            "dev_energy_mae_per_atom": float(np.mean(np.abs(e - e_ref) / n_atoms)),
            "dev_force_mae": float(np.mean(np.abs(np.concatenate(f) - np.concatenate(f_ref)))),
        }
        if labels is not None:
            e_true = np.array([x[0] for x in labels])
            f_true = np.concatenate([x[1] for x in labels])
            row["energy_mae_per_atom"] = float(np.mean(np.abs(e - e_true) / n_atoms))
            row["force_mae"] = float(np.mean(np.abs(np.concatenate(f) - f_true)))
        rows.append(row)
        print(row, flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({platform()})")


if __name__ == "__main__":
    main()
