"""Post-annealing hard pruning of a Nequix model trained with BMuon.

At the end of the budget annealing (epoch prune_at_epoch = int(anneal_fraction * n_epochs)),
the learned allocation mu_c cleanly separates active channels (mu_c ~ 1) from dying ones
(mu_c ~ eps). This module:

  compute_keep_mask   : global top-K ranking -> bool mask over channels.
  pruned_hidden_irreps: derives a uniform hidden_irreps string for the new model.
  copy_weights        : builds a new Nequix with the pruned architecture and copies
                        the kept weight slices from the trained model.
  hard_prune          : convenience wrapper calling all three steps.

The resulting model can be trained with standard Muon (budget_fraction=1.0) for the
remaining epochs (typically 75% of the run), saving ~35% wall-clock time.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bmuon.channels import ChannelSpec


def compute_keep_mask(
    spec: ChannelSpec,
    alloc: np.ndarray,
    k_target: int,
    min_per_ir: int = 4,
) -> np.ndarray:
    """Bool mask (num_channels,): True for the k_target highest-allocation channels.

    min_per_ir ensures at least this many copies of each irrep per layer are kept,
    even if their allocation is below the global threshold.
    """
    alloc = np.asarray(alloc)
    k_target = min(max(k_target, 0), len(alloc))
    threshold = float(np.sort(alloc)[::-1][k_target - 1]) if k_target > 0 else float("inf")
    keep = alloc >= threshold

    for layer in sorted({c.layer for c in spec.channels}):
        for ir in sorted({c.ir for c in spec.channels}):
            ids = [i for i, c in enumerate(spec.channels) if c.layer == layer and c.ir == ir]
            if not ids:
                continue
            n_kept = int(keep[ids].sum())
            if n_kept < min_per_ir:
                extra = min_per_ir - n_kept
                not_kept = sorted([i for i in ids if not keep[i]], key=lambda i: -alloc[i])
                for i in not_kept[:extra]:
                    keep[i] = True
    return keep


def pruned_hidden_irreps(spec: ChannelSpec, keep_mask: np.ndarray, min_per_ir: int = 4) -> str:
    """New hidden_irreps: one uniform count per ir type (floor of per-layer average).

    Using a uniform count across all layers keeps the Nequix architecture valid (it takes
    a single hidden_irreps for all interaction blocks).
    """
    layers = sorted({c.layer for c in spec.channels})
    ir_types = sorted(
        {c.ir for c in spec.channels},
        key=lambda ir: (int(ir[0]), 0 if "e" in ir else 1),
    )
    n_layers = len(layers)
    counts = {}
    for ir in ir_types:
        total_kept = sum(1 for i, c in enumerate(spec.channels) if c.ir == ir and keep_mask[i])
        counts[ir] = max(min_per_ir, total_kept // n_layers)
    return " + ".join(f"{counts[ir]}x{ir}" for ir in ir_types)


def copy_weights(old_model, new_model, spec: ChannelSpec, keep_mask: np.ndarray):
    """Replace new_model's random weights with kept slices of old_model.

    For each budgeted leaf (linear_1, linear_2, radial MLP last layer), selects the rows
    or columns corresponding to kept channels. Non-budgeted same-shape leaves are copied
    directly. Returns a new equinox model.
    """
    ir_types = sorted(
        {c.ir for c in spec.channels},
        key=lambda ir: (int(ir[0]), 0 if "e" in ir else 1),
    )
    layers = sorted({c.layer for c in spec.channels})
    n_layers = len(layers)

    # Uniform kept count per ir across layers.
    new_counts = {}
    for ir in ir_types:
        total = sum(1 for i, c in enumerate(spec.channels) if c.ir == ir and keep_mask[i])
        new_counts[ir] = max(4, total // n_layers)

    # Which global channel IDs to keep per (layer, ir).
    keep_in_layer: dict = defaultdict(set)
    for layer in layers:
        for ir in ir_types:
            k = new_counts[ir]
            ids = [i for i, c in enumerate(spec.channels)
                   if c.layer == layer and c.ir == ir]
            flagged = [i for i in ids if keep_mask[i]]
            not_flagged = [i for i in ids if not keep_mask[i]]
            for i in (flagged + not_flagged)[:k]:
                keep_in_layer[(layer, ir)].add(i)

    # Build slice_ops: keypath -> (axis, sorted keep-positions in that axis).
    slice_ops = {}
    for s in spec.slices:
        kept_global = set()
        for ir in ir_types:
            kept_global |= keep_in_layer[(s.layer, ir)]
        keep_pos = np.where(np.array([cid in kept_global for cid in s.channel_ids]))[0]
        slice_ops[s.keypath] = (s.axis, keep_pos)

    old_leaves = {
        jax.tree_util.keystr(p): v
        for p, v in jax.tree_util.tree_flatten_with_path(eqx.filter(old_model, eqx.is_array))[0]
        if eqx.is_array(v)
    }

    def apply_slice(path, new_leaf):
        if not eqx.is_array(new_leaf):
            return new_leaf
        key = jax.tree_util.keystr(path)
        if key not in old_leaves:
            return new_leaf
        old_leaf = old_leaves[key]
        if key in slice_ops:
            axis, keep_idx = slice_ops[key]
            if len(keep_idx) == 0:
                return new_leaf
            sliced = old_leaf[keep_idx, :] if axis == 0 else old_leaf[:, keep_idx]
            return jnp.asarray(sliced) if sliced.shape == new_leaf.shape else new_leaf
        return jnp.asarray(old_leaf) if old_leaf.shape == new_leaf.shape else new_leaf

    new_params_copied = jax.tree_util.tree_map_with_path(
        apply_slice,
        eqx.filter(new_model, eqx.is_array),
        is_leaf=lambda x: not eqx.is_array(x),
    )
    return eqx.combine(new_params_copied, eqx.filter(new_model, lambda x: not eqx.is_array(x)))


def hard_prune(
    old_model,
    spec: ChannelSpec,
    alloc: np.ndarray,
    config: dict,
    k_target: Optional[int] = None,
    min_per_ir: int = 4,
):
    """Prune old_model to its top-k_target channels. Returns (new_model, new_hidden_irreps).

    k_target defaults to round(bmuon.budget * spec.total_cost).
    The caller must build a new optimizer from the returned model and re-initialize its state.
    """
    from nequix.model import Nequix

    if k_target is None:
        budget = float(config.get("bmuon", {}).get("budget", 0.3))
        k_target = max(min_per_ir, round(budget * spec.total_cost))

    keep_mask = compute_keep_mask(spec, np.asarray(alloc), k_target, min_per_ir)
    new_irreps = pruned_hidden_irreps(spec, keep_mask, min_per_ir)

    shift = float(getattr(old_model, "shift", 0.0))
    scale = float(getattr(old_model, "scale", 1.0))
    new_model = Nequix(
        jax.random.key(0),
        n_species=len(config["atomic_numbers"]),
        hidden_irreps=new_irreps,
        lmax=config["lmax"],
        cutoff=config["cutoff"],
        n_layers=config["n_layers"],
        radial_basis_size=config["radial_basis_size"],
        radial_mlp_size=config["radial_mlp_size"],
        radial_mlp_layers=config["radial_mlp_layers"],
        radial_polynomial_p=config["radial_polynomial_p"],
        mlp_init_scale=config["mlp_init_scale"],
        index_weights=config["index_weights"],
        shift=shift,
        scale=scale,
        avg_n_neighbors=config.get("avg_n_neighbors", 1.0),
        layer_norm=config.get("layer_norm", False),
        kernel=config.get("kernel", False),
    )
    new_model = copy_weights(old_model, new_model, spec, keep_mask)
    print(
        f"hard_prune: {int(keep_mask.sum())}/{len(keep_mask)} channels -> {new_irreps}",
        flush=True,
    )
    return new_model, new_irreps
