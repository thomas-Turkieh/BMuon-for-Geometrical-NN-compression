"""Nequix interaction block whose tensor product takes fewer copies than the block input.

In Nequix, ``linear_1`` maps the block input to itself. After pruning type-I channels it
maps the block input to smaller multiplicities ``tp_input_irreps``; everything else (skip,
RMSNorm, gate, output irreps) is unchanged. The forward pass is inherited unchanged.
"""

from __future__ import annotations

import json
import os

import e3nn_jax as e3nn
import equinox as eqx
import jax

from nequix.layer_norm import RMSLayerNorm
from nequix.model import MLP, Nequix, NequixConvolution, Sort


class PrunableConvolution(NequixConvolution):
    tp_input_irreps: e3nn.Irreps = eqx.field(static=True)

    def __init__(
        self,
        key: jax.Array,
        input_irreps: e3nn.Irreps,
        tp_input_irreps: e3nn.Irreps,
        output_irreps: e3nn.Irreps,
        sh_irreps: e3nn.Irreps,
        n_species: int,
        radial_basis_size: int,
        radial_mlp_size: int,
        radial_mlp_layers: int,
        mlp_init_scale: float,
        avg_n_neighbors: float,
        index_weights: bool = True,
        layer_norm: bool = False,
        kernel: bool = False,
    ):
        # Same construction as NequixConvolution.__init__, with the tensor product fed by
        # tp_input_irreps instead of input_irreps.
        input_irreps = e3nn.Irreps(input_irreps)
        tp_input_irreps = e3nn.Irreps([(m, ir) for m, ir in e3nn.Irreps(tp_input_irreps) if m > 0])
        output_irreps = e3nn.Irreps(output_irreps)
        self.tp_input_irreps = tp_input_irreps
        self.output_irreps = output_irreps
        self.avg_n_neighbors = avg_n_neighbors
        self.index_weights = index_weights
        self.kernel = kernel

        irreps_out_tp, instructions = [], []
        for i, (mul, ir_in1) in enumerate(tp_input_irreps):
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
            if "OEQ_NOTORCH" not in os.environ:
                try:
                    import torch  # noqa: F401
                except ImportError:
                    os.environ["OEQ_NOTORCH"] = "1"
            import openequivariance as oeq
            import openequivariance_extjax  # noqa: F401

            instructions = [instructions[i] for i in inv]
            problem = oeq.TPProblem(
                str(tp_input_irreps),
                str(sh_irreps),
                str(tp_irreps),
                instructions=instructions,
                shared_weights=False,
                internal_weights=False,
            )
            self.tp_conv = oeq.jax.TensorProductConv(problem, deterministic=False)
        else:
            self.tp_conv = None

        self.sort = Sort(tp_irreps)
        tp_irreps = self.sort.irreps_sorted
        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.linear_1 = e3nn.equinox.Linear(
            irreps_in=input_irreps, irreps_out=tp_input_irreps, key=k1
        )
        self.radial_mlp = MLP(
            sizes=[radial_basis_size] + [radial_mlp_size] * radial_mlp_layers + [tp_irreps.num_irreps],
            activation=jax.nn.silu,
            use_bias=False,
            init_scale=mlp_init_scale,
            key=k2,
        )
        gate_irreps = e3nn.Irreps(f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e")
        output_irreps = (output_irreps + gate_irreps).regroup()
        self.linear_2 = e3nn.equinox.Linear(irreps_in=tp_irreps, irreps_out=output_irreps, key=k3)
        self.skip = e3nn.equinox.Linear(
            irreps_in=input_irreps,
            irreps_out=output_irreps,
            linear_type="indexed" if index_weights else "vanilla",
            num_indexed_weights=n_species if index_weights else None,
            force_irreps_out=True,
            key=k4,
        )
        self.layer_norm = (
            RMSLayerNorm(irreps=output_irreps, centering=False, std_balance_degrees=True)
            if layer_norm
            else None
        )


def conv_kwargs(model: Nequix, conv) -> dict:
    """Constructor arguments that rebuild ``conv`` (other than the tensor-product irreps)."""
    return dict(
        input_irreps=conv.skip.irreps_in,
        output_irreps=conv.output_irreps,
        sh_irreps=e3nn.s2_irreps(model.lmax),
        n_species=model.n_species,
        radial_basis_size=model.radial_basis_size,
        radial_mlp_size=conv.radial_mlp.layers[0].weights.shape[1],
        radial_mlp_layers=len(conv.radial_mlp.layers) - 1,
        mlp_init_scale=1.0,  # weights are overwritten
        avg_n_neighbors=conv.avg_n_neighbors,
        index_weights=conv.index_weights,
        layer_norm=conv.layer_norm is not None,
        kernel=conv.kernel,
    )


def with_tp_irreps(model: Nequix, tp_irreps: list, kernel: bool | None = None) -> Nequix:
    """Return ``model`` with each block rebuilt for the given tensor-product input irreps.

    Weights of the rebuilt blocks are random; use bmuon.prune to fill them.
    """
    convs = []
    for t, (conv, irreps) in enumerate(zip(model.layers, tp_irreps)):
        kw = conv_kwargs(model, conv)
        if kernel is not None:
            kw["kernel"] = kernel
        convs.append(PrunableConvolution(jax.random.key(t), tp_input_irreps=irreps, **kw))
    return eqx.tree_at(lambda m: m.layers, model, convs)


# ------------------------------------------------------------------ save / load


def save_pruned(path: str, model: Nequix, config: dict):
    """Save a pruned model: Nequix config + the tensor-product irreps of each block."""
    tp = [str(getattr(c, "tp_input_irreps", c.linear_1.irreps_out)) for c in model.layers]
    with open(path, "wb") as f:
        f.write((json.dumps({**config, "tp_input_irreps": tp}) + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_pruned(path: str, kernel: bool = False):
    from nequix.model import Nequix as _Nequix

    with open(path, "rb") as f:
        config = json.loads(f.readline().decode())
        model = _Nequix(
            key=jax.random.key(0),
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
            kernel=False,
        )
        model = with_tp_irreps(model, [e3nn.Irreps(s) for s in config["tp_input_irreps"]], kernel)
        model = eqx.tree_deserialise_leaves(f, model)
    return model, config
