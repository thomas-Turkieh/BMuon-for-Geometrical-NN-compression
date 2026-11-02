"""Map of the type-I channels of a Nequix model.

A type-I channel is one copy of an irrep at the output of ``linear_1`` in an interaction
block, i.e. one copy entering the tensor product. It owns three families of matrix slices
(code layout, weights stored as ``(in, out)`` and applied as ``x @ W``):

* role ``"A"``: column ``k`` of the ``linear_1`` weight of its irrep;
* role ``"B"``: for every path ``p = (l1, l2 -> l3)`` with ``l1`` = its irrep, row ``(p, k)``
  of the ``linear_2`` weight of irrep ``l3`` (e3nn-jax regroups the tensor-product output, so
  there is one ``linear_2`` matrix per output irrep, not one per path);
* role ``"C"``: for every such path, column ``(p, k)`` of the last layer of the radial MLP.

Zeroing all slices of a channel removes it exactly from the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import e3nn_jax as e3nn
import equinox as eqx
import jax
import numpy as np

PathKey = tuple  # (str(ir1), str(ir2), str(ir_out))


@dataclass(eq=False)
class SliceSpec:
    """One budgeted weight matrix and the channel owning each of its slices."""

    keypath: str  # jax keystr of the leaf in eqx.filter(model, eqx.is_array)
    layer: int
    role: str  # "A", "B" or "C"
    ir: Optional[str]  # output irrep of linear_1 (A) or of the path (B); None for C
    axis: int  # axis of the code-layout weight that is sliced (0: rows, 1: columns)
    channel_ids: np.ndarray  # (size along axis,) global channel id of each slice
    labels: list  # per slice: (path_key or None, ir1, copy); identifies it across pruning


@dataclass(frozen=True)
class Channel:
    layer: int
    ir: str  # irrep of the copy, e.g. "1o"
    copy: int  # index of the copy within its irrep
    n_paths: int  # number of tensor-product paths it feeds


@dataclass(eq=False)
class ChannelSpec:
    channels: list
    slices: list
    weights: np.ndarray  # cost w_c of each channel, normalized to mean 1
    paths: dict = field(default_factory=dict)  # layer -> list of (path_key, mul)

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    @property
    def keypaths(self) -> set:
        return {s.keypath for s in self.slices}

    @property
    def total_cost(self) -> float:
        return float(self.weights.sum())

    def ids_of(self, layer: int, ir: str) -> np.ndarray:
        return np.array(
            [i for i, c in enumerate(self.channels) if c.layer == layer and c.ir == ir], dtype=int
        )

    def summary(self) -> str:
        lines = []
        for t in sorted({c.layer for c in self.channels}):
            irs = {}
            for c in self.channels:
                if c.layer == t:
                    irs[c.ir] = irs.get(c.ir, 0) + 1
            lines.append(f"block {t}: " + " + ".join(f"{m}x{ir}" for ir, m in irs.items()))
        return "\n".join(lines)


# ---------------------------------------------------------------------------- costs


def cg_nnz(l1: int, l2: int, l3: int) -> int:
    """Number of nonzero real Clebsch-Gordan coefficients of a path (e3nn convention)."""
    cg = np.asarray(e3nn.clebsch_gordan(l1, l2, l3))
    return int((np.abs(cg) > 1e-8).sum())


def path_cost(path_key: PathKey, model: Union[str, dict, Callable] = "cg") -> float:
    """Cost of one copy flowing through one tensor-product path, per edge.

    ``model``: "paths" (1 per path, alias "uniform"), "cg" (nonzero CG coefficients), "dims"
    ((2l1+1)(2l2+1)(2l3+1)), a dict ``{path_key: cost}`` (e.g. measured), or a callable.
    """
    l1, l2, l3 = (e3nn.Irrep(s).l for s in path_key)
    if callable(model):
        return float(model(path_key))
    if isinstance(model, dict):
        return float(model[path_key])
    if model in ("uniform", "paths"):
        return 1.0
    if model == "cg":
        return float(cg_nnz(l1, l2, l3))
    if model == "dims":
        return float((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1))
    raise ValueError(f"unknown cost model {model}")


# ------------------------------------------------------------------------ the map


def tp_input_irreps(conv) -> e3nn.Irreps:
    """Irreps entering the tensor product of a convolution (output of linear_1)."""
    irreps = getattr(conv, "tp_input_irreps", None)
    if irreps is None:
        irreps = conv.linear_1.irreps_out
    return e3nn.Irreps(irreps)


def tp_paths(conv, sh_irreps: e3nn.Irreps) -> list:
    """Paths of the tensor product, in the order NequixConvolution creates them."""
    paths = []
    for mul, ir1 in tp_input_irreps(conv):
        for _, ir2 in e3nn.Irreps(sh_irreps):
            for ir_out in ir1 * ir2:
                if ir_out in conv.output_irreps:
                    paths.append(((str(ir1), str(ir2), str(ir_out)), mul))
    return paths


def _weight_names(linear) -> list:
    """(name, ir_in, ir_out) of each weight of an e3nn-jax equinox Linear."""
    lin = linear._linear
    out = []
    for ins in lin.instructions:
        if ins.i_in == -1:
            continue
        ir_in, ir_out = lin.irreps_in[ins.i_in], lin.irreps_out[ins.i_out]
        out.append((f"w[{ins.i_in},{ins.i_out}] {ir_in},{ir_out}", str(ir_in.ir), str(ir_out.ir)))
    return out


def build_channel_spec(model, cost_model: Union[str, dict, Callable] = "cg") -> ChannelSpec:
    """Build the type-I channel map of a (possibly pruned) Nequix model.

    ``cost_model``: "channel" gives every channel weight 1 (the budget of the BMuon paper);
    any path cost model (see ``path_cost``) gives w_c = sum of the costs of its paths.
    """
    sh_irreps = e3nn.s2_irreps(model.lmax)
    params = eqx.filter(model, eqx.is_array)
    existing = {jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(params)[0]}

    channels, slices, path_table = [], [], {}
    for t, conv in enumerate(model.layers):
        tp_in = tp_input_irreps(conv)
        start = {}
        for mul, ir in tp_in:
            if str(ir) in start:
                raise ValueError(f"block {t}: irrep {ir} appears twice in {tp_in}")
            start[str(ir)] = len(channels)
            channels.extend(Channel(t, str(ir), k, 0) for k in range(mul))

        paths = tp_paths(conv, sh_irreps)
        path_table[t] = paths
        tp_irreps = e3nn.Irreps([(mul, e3nn.Irrep(pk[2])) for pk, mul in paths])
        if tp_irreps != conv.tp_irreps:
            raise ValueError(f"block {t}: path reconstruction {tp_irreps} != {conv.tp_irreps}")
        _, _, inv = tp_irreps.sort()  # sorted entry j comes from path inv[j]
        sorted_paths = [paths[i] for i in inv]

        n_paths = {ir: 0 for ir in start}
        for pk, _ in paths:
            n_paths[pk[0]] += 1
        for i in range(len(channels)):
            c = channels[i]
            if c.layer == t:
                channels[i] = Channel(t, c.ir, c.copy, n_paths[c.ir])

        def ids(ir1, mul):
            return [start[ir1] + k for k in range(mul)]

        # A: linear_1, one weight per irrep, columns are the copies
        for name, _, ir_out in _weight_names(conv.linear_1):
            key = f".layers[{t}].linear_1._weights['{name}']"
            mul = dict((str(ir), m) for m, ir in tp_in)[ir_out]
            slices.append(
                SliceSpec(key, t, "A", ir_out, 1, np.array(ids(ir_out, mul)),
                          [(None, ir_out, k) for k in range(mul)])
            )

        # B: linear_2, one weight per output irrep of the tensor product (regrouped input)
        for name, ir_in, _ in _weight_names(conv.linear_2):
            key = f".layers[{t}].linear_2._weights['{name}']"
            cid, lab = [], []
            for pk, mul in sorted_paths:
                if pk[2] == ir_in:
                    cid += ids(pk[0], mul)
                    lab += [(pk, pk[0], k) for k in range(mul)]
            slices.append(SliceSpec(key, t, "B", ir_in, 0, np.array(cid), lab))

        # C: last layer of the radial MLP, one column per (path, copy) in sorted order
        last = len(conv.radial_mlp.layers) - 1
        key = f".layers[{t}].radial_mlp.layers[{last}].weights"
        cid, lab = [], []
        for pk, mul in sorted_paths:
            cid += ids(pk[0], mul)
            lab += [(pk, pk[0], k) for k in range(mul)]
        slices.append(SliceSpec(key, t, "C", None, 1, np.array(cid), lab))

    leaves = {jax.tree_util.keystr(p): x for p, x in jax.tree_util.tree_flatten_with_path(params)[0]}
    for s in slices:
        if s.keypath not in existing:
            raise KeyError(f"leaf {s.keypath} not found in model parameters")
        if leaves[s.keypath].shape[s.axis] != len(s.channel_ids):
            raise ValueError(
                f"{s.keypath}: axis {s.axis} has size {leaves[s.keypath].shape[s.axis]}, "
                f"map has {len(s.channel_ids)} slices"
            )

    weights = np.zeros(len(channels))
    if isinstance(cost_model, str) and cost_model.endswith(".json"):
        import json

        with open(cost_model) as f:
            cost_model = json.load(f)
    if cost_model == "channel":  # original BMuon: every channel costs 1
        return ChannelSpec(channels, slices, np.ones(len(channels)), path_table)
    if isinstance(cost_model, dict) and "channel_costs" in cost_model:
        # measured cost per copy, keyed "layer:irrep" (or "irrep" for every layer)
        table = cost_model["channel_costs"]
        for i, c in enumerate(channels):
            weights[i] = table.get(f"{c.layer}:{c.ir}", table.get(c.ir))
        return ChannelSpec(channels, slices, weights / weights.mean(), path_table)
    for t, paths in path_table.items():
        for pk, _ in paths:
            cost = path_cost(pk, cost_model)
            for i, c in enumerate(channels):
                if c.layer == t and c.ir == pk[0]:
                    weights[i] += cost
    weights = weights / weights.mean()
    return ChannelSpec(channels, slices, weights, path_table)
