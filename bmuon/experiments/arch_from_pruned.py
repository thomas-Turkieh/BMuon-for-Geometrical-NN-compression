"""Architecture of a gradually pruned model, to retrain it from scratch.

    python -m bmuon.experiments.arch_from_pruned SNAPSHOT.nqx

Reads the channels kept in each (block, irrep) of a masked snapshot (removed channels are
exactly zero) and prints two ``tp_input_irreps`` lists for bmuon/train.py:
  * learned: the copies kept by the pruned model;
  * uniform: the same fraction of copies kept in every (block, irrep), at the same cost.
Costs use the "paths" cost model, as the pruning.
"""

from __future__ import annotations

import sys

import numpy as np

from bmuon.channels import build_channel_spec
from bmuon.prune import magnitude_scores
from nequix.model import load_model


def irreps_per_block(spec, keep):
    blocks = []
    for t in sorted({c.layer for c in spec.channels}):
        irs = {}
        for i, c in enumerate(spec.channels):
            if c.layer == t:
                irs.setdefault(c.ir, 0)
                irs[c.ir] += int(keep[i])
        blocks.append(" + ".join(f"{m}x{ir}" for ir, m in irs.items()))
    return blocks


def main(path):
    model, _ = load_model(path, kernel=False)
    spec = build_channel_spec(model, cost_model="paths")
    w = np.asarray(spec.weights, float)
    keep = np.asarray(magnitude_scores(model, spec) > 1e-12)
    frac = w[keep].sum() / w.sum()

    # uniform: in every (block, irrep), keep the first round(f * m) copies; f tuned to the same cost
    groups = {}
    for i, c in enumerate(spec.channels):
        groups.setdefault((c.layer, c.ir), []).append(i)

    def uniform(f):
        u = np.zeros(len(w), bool)
        for ids in groups.values():
            u[ids[:max(1, int(round(f * len(ids))))]] = True
        return u

    fs = np.linspace(0.05, 1.0, 951)
    f_best = min(fs, key=lambda f: abs(w[uniform(f)].sum() / w.sum() - frac))
    uni = uniform(f_best)

    print(f"snapshot: {path}")
    print(f"learned: {int(keep.sum())}/{len(w)} channels, cost kept {frac:.4f}")
    print("  tp_input_irreps:")
    for b in irreps_per_block(spec, keep):
        print(f'    - "{b}"')
    print(f"uniform: {int(uni.sum())}/{len(w)} channels, cost kept {w[uni].sum() / w.sum():.4f} (f = {f_best:.3f})")
    print("  tp_input_irreps:")
    for b in irreps_per_block(spec, uni):
        print(f'    - "{b}"')


if __name__ == "__main__":
    main(sys.argv[1])
