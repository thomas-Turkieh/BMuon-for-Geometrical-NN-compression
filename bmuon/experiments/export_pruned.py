"""Export a gradually pruned wide model as a compact model with the pruned channels removed.

    python -m bmuon.experiments.export_pruned --run bmuon/runs/<run id> --epoch 99 --out FILE.nqx

During training the pruned channels are only masked (set to zero), so a snapshot keeps the wide
shape. This keeps the channels of the last pruning mask (``prune_epoch*.npz``), removes the others
(``bmuon.physical_prune``) and saves the result with ``bmuon.model.save_pruned``; load it with
``bmuon.model.load_pruned``. The exported model computes the same function as the snapshot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import equinox as eqx
import jax
import numpy as np

from bmuon.channels import build_channel_spec
from bmuon.model import load_pruned, save_pruned
from bmuon.physical_prune import physically_prune
from bmuon.experiments.equal_size_bench import n_params, n_tp_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epoch", type=int, default=99)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    model, config = load_pruned(str(run / f"snap_epoch{args.epoch:03d}.nqx"), kernel=False)
    spec = build_channel_spec(model, cost_model=config["bmuon"]["cost_model"])
    mask = sorted(run.glob("prune_epoch*.npz"))[-1]
    keep = np.load(mask)["keep"].astype(bool)
    pruned = physically_prune(model, spec, keep, kernel=False)
    cfg = {k: v for k, v in config.items() if k != "tp_expand"}
    cfg["exported_from"] = {"run": run.name, "epoch": args.epoch, "mask": mask.name}
    save_pruned(args.out, pruned, cfg)

    back, _ = load_pruned(args.out, kernel=False)
    same = all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(
        jax.tree_util.tree_leaves(eqx.filter(pruned, eqx.is_array)),
        jax.tree_util.tree_leaves(eqx.filter(back, eqx.is_array))))
    print(f"{args.out}: {n_tp_paths(back)} TP paths, {n_params(back)} parameters "
          f"(wide snapshot: {n_tp_paths(model)} paths, {n_params(model)} parameters), "
          f"mask {mask.name}, reload identical: {same}")


if __name__ == "__main__":
    main()
