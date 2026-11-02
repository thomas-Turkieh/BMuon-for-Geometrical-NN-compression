"""Physical pruning of type-I channels of a Nequix model.

A channel (see ``bmuon.channels``) is one copy of an irrep at the output of ``linear_1``, i.e. one
copy entering the tensor product of a block. Removing it physically means:

* ``linear_1``: keep only the columns of the kept copies (``irreps_in`` unchanged, ``irreps_out`` shrinks);
* tensor product (OpenEquivariance or e3nn): built on the kept copies only;
* radial MLP, last layer: keep the columns of the kept (path, copy);
* ``linear_2``: keep the rows of the kept (path, copy); e3nn normalises each output block by
  1/sqrt(fan-in), so the kept rows are multiplied by sqrt(new_rows / old_rows) to give exactly
  the same function as the masked model (``channel_sweep.mask_model``);
* skip connection, hidden width, layer norm, readout: unchanged.

The result equals the masked model, but runs a smaller tensor product.
"""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict

import e3nn_jax as e3nn
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bmuon.channels import ChannelSpec, tp_input_irreps
from nequix.layer_norm import RMSLayerNorm
from nequix.model import MLP, NequixConvolution, Sort

_WNAME = re.compile(r"\['(w\[\d+,\d+\]) [^']*'\]")


def _norm_key(key: str) -> str:
    """Drop the irrep sizes from the e3nn weight names (they change with pruning)."""
    return _WNAME.sub(r"['\1']", key)


class PrunedConvolution(NequixConvolution):
    """NequixConvolution whose tensor product only sees ``tp_in_irreps`` (a sub-selection of copies)."""

    def __init__(self, key, input_irreps, tp_in_irreps, output_irreps, sh_irreps, n_species,
                 radial_basis_size, radial_mlp_size, radial_mlp_layers, mlp_init_scale,
                 avg_n_neighbors, index_weights=True, layer_norm=False, kernel=False):
        self.output_irreps = output_irreps
        self.avg_n_neighbors = avg_n_neighbors
        self.index_weights = index_weights
        self.kernel = kernel

        irreps_out_tp, instructions = [], []
        for i, (mul, ir_in1) in enumerate(tp_in_irreps):
            for j, (_, ir_in2) in enumerate(sh_irreps):
                for ir_out in ir_in1 * ir_in2:
                    if ir_out in output_irreps:
                        k = len(irreps_out_tp)
                        irreps_out_tp.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        tp_irreps = e3nn.Irreps(irreps_out_tp)
        _, _, inv = tp_irreps.sort()
        self.tp_irreps = tp_irreps

        if kernel:
            try:
                import torch  # noqa: F401
            except ImportError:
                os.environ["OEQ_NOTORCH"] = "1"
            import openequivariance as oeq
            import openequivariance_extjax  # noqa: F401

            instructions = [instructions[i] for i in inv]
            problem = oeq.TPProblem(str(tp_in_irreps), str(sh_irreps), str(tp_irreps),
                                    instructions=instructions, shared_weights=False,
                                    internal_weights=False)
            self.tp_conv = oeq.jax.TensorProductConv(problem, deterministic=False)
        else:
            self.tp_conv = None

        self.sort = Sort(tp_irreps)
        tp_irreps = self.sort.irreps_sorted
        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.linear_1 = e3nn.equinox.Linear(irreps_in=input_irreps, irreps_out=tp_in_irreps, key=k1)
        self.radial_mlp = MLP(
            sizes=[radial_basis_size] + [radial_mlp_size] * radial_mlp_layers + [tp_irreps.num_irreps],
            activation=jax.nn.silu, use_bias=False, init_scale=mlp_init_scale, key=k2)

        gate_irreps = e3nn.Irreps(f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e")
        output_irreps = (output_irreps + gate_irreps).regroup()
        self.linear_2 = e3nn.equinox.Linear(irreps_in=tp_irreps, irreps_out=output_irreps, key=k3)
        self.skip = e3nn.equinox.Linear(
            irreps_in=input_irreps, irreps_out=output_irreps,
            linear_type="indexed" if index_weights else "vanilla",
            num_indexed_weights=n_species if index_weights else None,
            force_irreps_out=True, key=k4)
        self.layer_norm = (RMSLayerNorm(irreps=output_irreps, centering=False, std_balance_degrees=True)
                           if layer_norm else None)


def kept_counts(spec: ChannelSpec, keep) -> dict:
    counts = defaultdict(int)
    for i, c in enumerate(spec.channels):
        if keep[i]:
            counts[(c.layer, c.ir)] += 1
    return counts


def physically_prune(model, spec: ChannelSpec, keep, kernel=None):
    """Return a smaller Nequix equal to ``mask_model(model, spec, keep)`` (up to float rounding)."""
    keep = np.asarray(keep, bool)
    counts = kept_counts(spec, keep)
    layers = []
    keys = jax.random.split(jax.random.key(0), len(model.layers))
    for t, conv in enumerate(model.layers):
        tp_old = tp_input_irreps(conv)
        new = []
        for mul, ir in tp_old:
            n = counts[(t, str(ir))]
            if n == 0:
                raise ValueError(f"layer {t}: every copy of {ir} would be removed")
            new.append((n, ir))
        layers.append(PrunedConvolution(
            key=keys[t], input_irreps=conv.linear_1.irreps_in, tp_in_irreps=e3nn.Irreps(new),
            output_irreps=conv.output_irreps, sh_irreps=e3nn.s2_irreps(model.lmax),
            n_species=model.n_species, radial_basis_size=conv.radial_mlp.layers[0].weights.shape[0],
            radial_mlp_size=conv.radial_mlp.layers[0].weights.shape[1],
            radial_mlp_layers=len(conv.radial_mlp.layers) - 1, mlp_init_scale=1.0,
            avg_n_neighbors=conv.avg_n_neighbors, index_weights=conv.index_weights,
            layer_norm=conv.layer_norm is not None, kernel=conv.kernel if kernel is None else kernel))
    new_model = eqx.tree_at(lambda m: m.layers, model, layers)

    old_leaves = {_norm_key(jax.tree_util.keystr(p)): np.asarray(x) for p, x in
                  jax.tree_util.tree_flatten_with_path(eqx.filter(model, eqx.is_array))[0]}
    ops = defaultdict(list)
    for s in spec.slices:
        idx = np.array([p for p, cid in enumerate(s.channel_ids) if keep[cid]], dtype=int)
        ops[_norm_key(s.keypath)].append((s.axis, idx, s.role))

    def convert(path, leaf):
        if not eqx.is_array(leaf):
            return leaf
        k = _norm_key(jax.tree_util.keystr(path))
        x = old_leaves[k]
        for axis, idx, role in ops.get(k, []):
            old_n = x.shape[axis]
            x = np.take(x, idx, axis=axis)
            if role == "B":  # e3nn path weight 1/sqrt(fan-in) shrinks with the number of rows
                x = x * math.sqrt(len(idx) / old_n)
        if x.shape != leaf.shape:
            raise ValueError(f"{k}: converted shape {x.shape} != expected {leaf.shape}")
        return jnp.asarray(x, dtype=leaf.dtype)

    params, static = eqx.partition(new_model, eqx.is_array)
    params = jax.tree_util.tree_map_with_path(convert, params)
    return eqx.combine(params, static)
