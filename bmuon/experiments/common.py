"""Shared helpers for the experiment scripts: structures, padding, batched evaluation."""

from __future__ import annotations

import ase
import jax
import jraph
import numpy as np
from ase.build import bulk

from nequix.data import dict_to_graphstuple, preprocess_graph


def toy_structures(n: int = 32, seed: int = 0) -> list:
    """Rattled bulk crystals of common elements (no labels), for deviation measurements."""
    rng = np.random.default_rng(seed)
    protos = [
        ("Si", "diamond", 5.43, (2, 2, 2)),
        ("Cu", "fcc", 3.61, (2, 2, 2)),
        ("Fe", "bcc", 2.87, (3, 3, 3)),
        ("NaCl", "rocksalt", 5.64, (2, 2, 2)),
        ("MgO", "rocksalt", 4.21, (2, 2, 2)),
        ("Al", "fcc", 4.05, (2, 2, 2)),
        ("GaAs", "zincblende", 5.65, (2, 2, 2)),
        ("Ti", "hcp", 2.95, (3, 3, 2)),
    ]
    out = []
    for i in range(n):
        name, kind, a, rep = protos[i % len(protos)]
        kw = {"c": 4.68} if kind == "hcp" else {}
        atoms = bulk(name, kind, a=a, **kw).repeat(rep)
        atoms.rattle(0.05, seed=int(rng.integers(1 << 30)))
        atoms.set_cell(atoms.cell * (1 + 0.02 * rng.standard_normal()), scale_atoms=True)
        out.append(atoms)
    return out


def to_graph(atoms: ase.Atoms, atomic_numbers: list, cutoff: float, targets=False):
    idx = {z: i for i, z in enumerate(atomic_numbers)}
    return dict_to_graphstuple(preprocess_graph(atoms, idx, cutoff, targets))


def pad(graph, n_node: int, n_edge: int):
    """Pad one graph to fixed sizes so jit compiles once per size bucket."""
    return jraph.pad_with_graphs(graph, n_node=n_node, n_edge=n_edge, n_graph=2)


def bucket(x: int, base: int = 64) -> int:
    return int(base * 2 ** np.ceil(np.log2(max(x, base) / base)))


def make_eval_fn(model):
    import equinox as eqx

    @eqx.filter_jit
    def f(model, g):
        e, forces, stress = model(g)
        return e, forces

    return lambda m, g: f(m, g)


def predict(model, graphs: list):
    """Energies (per graph) and forces (per atom) for a list of unpadded graphs."""
    f = make_eval_fn(model)
    es, fs = [], []
    for g in graphs:
        n, m = int(g.n_node[0]), int(g.n_edge[0])
        gp = pad(g, bucket(n + 1, 32), bucket(m + 1, 256))
        e, forces = f(model, gp)
        es.append(float(np.asarray(e)[0]))
        fs.append(np.asarray(forces)[:n])
    return np.array(es), fs


def platform() -> str:
    d = jax.devices()[0]
    return f"{d.platform}:{getattr(d, 'device_kind', '')}"
