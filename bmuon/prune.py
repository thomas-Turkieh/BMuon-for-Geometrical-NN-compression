"""Channel pruning for Nequix models trained with (B)Muon.

Part 1 - scores, selection and exact export (used by fragility.py, check_gpu.py, speed.py, tests):
  magnitude_scores / normalize_by_group : per-channel scores.
  select_channels                       : keep mask from scores (global or per group).
  prune                                 : smaller dense model, exactly equal to the zeroed one.
  zero_channels                         : same model with the dropped slices set to zero.

Part 2 - post-annealing hard pruning (added on 2026-10-14):
  compute_keep_mask   : global top-K ranking of the BMuon allocation -> bool mask.
  pruned_hidden_irreps: uniform hidden_irreps string for the new model.
  copy_weights        : copies the kept slices into a new Nequix.
  hard_prune          : the three steps above.

Both parts lived in separate versions of this file (commits 14eef38 and 020293b); they are
merged here because the tests and experiment scripts need Part 1.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import e3nn_jax as e3nn
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bmuon.channels import ChannelSpec, build_channel_spec
from bmuon.model import with_tp_irreps


def _leaves(tree) -> dict:
    return {jax.tree_util.keystr(p): x for p, x in jax.tree_util.tree_flatten_with_path(tree)[0]}


def _path_weights(model) -> dict:
    """keystr -> runtime multiplier of every e3nn Linear weight in the interaction blocks."""
    out = {}
    for t, conv in enumerate(model.layers):
        for attr in ("linear_1", "linear_2", "skip"):
            lin = getattr(conv, attr)._linear
            for ins in lin.instructions:
                if ins.i_in == -1:
                    continue
                name = f"w[{ins.i_in},{ins.i_out}] {lin.irreps_in[ins.i_in]},{lin.irreps_out[ins.i_out]}"
                out[f".layers[{t}].{attr}._weights['{name}']"] = float(ins.path_weight)
    return out


def magnitude_scores(model, spec: ChannelSpec) -> np.ndarray:
    """Coupled magnitude s_c = sqrt(sum of squared norms of all slices of channel c)."""
    leaves = _leaves(eqx.filter(model, eqx.is_array))
    sq = np.zeros(spec.num_channels)
    for s in spec.slices:
        w = np.asarray(leaves[s.keypath])
        per = (w**2).sum(axis=1 - s.axis)
        np.add.at(sq, s.channel_ids, per)
    return np.sqrt(sq)


def normalize_by_group(scores: np.ndarray, spec: ChannelSpec) -> np.ndarray:
    """Divide each score by the median score of its (block, irrep) group.

    Raw slice norms are not comparable across blocks and irreps (different fan-ins, e3nn path
    constants, RMSNorm in between); a global ranking of raw magnitudes empties whole groups
    first (e.g. the last block), which says more about the scales than about importance."""
    out = np.asarray(scores, float).copy()
    groups = defaultdict(list)
    for i, c in enumerate(spec.channels):
        groups[(c.layer, c.ir)].append(i)
    for ids in groups.values():
        med = np.median(out[ids])
        if med > 0:
            out[ids] = out[ids] / med
    return out


def select_channels(
    scores: np.ndarray,
    spec: ChannelSpec,
    remove_fraction: float,
    by: str = "count",
    min_keep: int | dict = 1,
    round_to: int = 1,
    exclude_layers: tuple = (),
    per_group: bool = False,
) -> np.ndarray:
    """Global ranking: drop the lowest-scored channels until ``remove_fraction`` is reached.

    ``per_group``: instead of a global ranking, remove the same fraction of copies in every
    (block, irrep) group, the lowest-scored ones (uniform layer-wise pruning).

    ``exclude_layers``: blocks whose channels are never removed; the fraction is then taken
    over the channels (or cost) of the other blocks only.

    ``by``: "count" (fraction of channels) or "cost" (fraction of the total cost w).
    ``min_keep``: minimum copies kept per (block, irrep); an int or a dict irrep -> int.
    ``round_to``: kept copies per (block, irrep) rounded up to a multiple (kernel friendly).
    Returns a boolean keep mask over channels.
    """
    n = spec.num_channels
    w = spec.weights if by == "cost" else np.ones(n)
    eligible = np.array([c.layer not in set(exclude_layers) for c in spec.channels])
    target = remove_fraction * w[eligible].sum()
    groups = defaultdict(list)
    for i, c in enumerate(spec.channels):
        groups[(c.layer, c.ir)].append(i)

    def floor(ir):
        return min_keep.get(ir, 1) if isinstance(min_keep, dict) else min_keep

    keep = np.ones(n, bool)
    if per_group:
        for g, ids in groups.items():
            if g[0] in set(exclude_layers):
                continue
            ids = np.array(ids)
            k = min(int(round(remove_fraction * len(ids))), len(ids) - min(floor(g[1]), len(ids)))
            if k > 0:
                keep[ids[np.argsort(scores[ids], kind="stable")[:k]]] = False
        return _round_groups(keep, scores, groups, round_to)

    left = {g: len(ids) for g, ids in groups.items()}
    removed = 0.0
    for i in np.argsort(scores, kind="stable"):
        if removed >= target - 1e-12:
            break
        if not eligible[i]:
            continue
        c = spec.channels[i]
        g = (c.layer, c.ir)
        if left[g] <= min(floor(c.ir), len(groups[g])):
            continue
        keep[i] = False
        left[g] -= 1
        removed += w[i]
    return _round_groups(keep, scores, groups, round_to)


def _round_groups(keep, scores, groups, round_to):
    """Round the kept copies of every group up to a multiple of ``round_to``."""
    if round_to > 1:
        for g, ids in groups.items():
            ids = np.array(ids)
            m = int(keep[ids].sum())
            want = min(len(ids), int(np.ceil(m / round_to) * round_to)) if m else 0
            best = ids[np.argsort(-scores[ids], kind="stable")][:want]
            keep[ids] = False
            keep[best] = True
    return keep


def prune(model, spec: ChannelSpec, keep: np.ndarray, kernel: bool | None = None):
    """Export a dense model without the dropped channels. Exactly equal to ``model`` with
    the slices of the dropped channels set to zero."""
    kept = defaultdict(list)  # (layer, ir) -> old copy indices, ascending
    for i, c in enumerate(spec.channels):
        if keep[i]:
            kept[(c.layer, c.ir)].append(c.copy)

    new_irreps = []
    for t, conv in enumerate(model.layers):
        old = e3nn.Irreps(getattr(conv, "tp_input_irreps", conv.linear_1.irreps_out))
        new_irreps.append(
            e3nn.Irreps([(len(kept[(t, str(ir))]), ir) for _, ir in old if kept[(t, str(ir))]])
        )
    new_model = with_tp_irreps(model, new_irreps, kernel)
    new_spec = build_channel_spec(new_model, cost_model="uniform")

    old_leaves = _leaves(eqx.filter(model, eqx.is_array))
    old_slice = {(s.layer, s.role, s.ir): s for s in spec.slices}
    pw_old, pw_new = _path_weights(model), _path_weights(new_model)
    gathered = {}
    for s in new_spec.slices:
        o = old_slice[(s.layer, s.role, s.ir)]
        pos = {lab: k for k, lab in enumerate(o.labels)}
        idx = [pos[(pk, ir1, kept[(s.layer, ir1)][j])] for pk, ir1, j in s.labels]
        w = jnp.take(old_leaves[o.keypath], jnp.array(idx), axis=s.axis)
        # e3nn Linear multiplies each weight by a constant that depends on the fan-in
        # (number of input copies); keep the product weight * constant unchanged.
        if o.keypath in pw_old:
            w = w * (pw_old[o.keypath] / pw_new[s.keypath])
        gathered[s.keypath] = w

    params, static = eqx.partition(new_model, eqx.is_array)
    flat, treedef = jax.tree_util.tree_flatten_with_path(params)
    out = []
    for p, x in flat:
        k = jax.tree_util.keystr(p)
        y = gathered.get(k, old_leaves.get(k))
        if y is None or y.shape != x.shape:
            raise ValueError(f"cannot fill {k}: {None if y is None else y.shape} vs {x.shape}")
        out.append(y)
    return eqx.combine(jax.tree_util.tree_unflatten(treedef, out), static)


def zero_channels(model, spec: ChannelSpec, keep: np.ndarray):
    """Same model with every slice of a dropped channel set to zero (no shape change)."""
    params, static = eqx.partition(model, eqx.is_array)
    flat, treedef = jax.tree_util.tree_flatten_with_path(params)
    by_key = {s.keypath: s for s in spec.slices}
    out = []
    for p, x in flat:
        s = by_key.get(jax.tree_util.keystr(p))
        if s is not None:
            m = jnp.asarray(keep[s.channel_ids], x.dtype)
            x = x * (m[:, None] if s.axis == 0 else m[None, :])
        out.append(x)
    return eqx.combine(jax.tree_util.tree_unflatten(treedef, out), static)


def tp_irreps_summary(model) -> list:
    return [str(getattr(c, "tp_input_irreps", c.linear_1.irreps_out)) for c in model.layers]


# --- Part 2: post-annealing hard pruning ---


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
