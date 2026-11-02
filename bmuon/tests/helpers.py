"""Small models and random graphs for the BMuon tests."""

import jax
import jax.numpy as jnp
import jraph
import numpy as np

from nequix.model import Nequix


def small_model(key=0, lmax=2, hidden="8x0e+4x1o+4x2e", n_layers=3, n_species=3, layer_norm=True):
    return Nequix(
        jax.random.key(key),
        n_species=n_species,
        lmax=lmax,
        cutoff=3.0,
        hidden_irreps=hidden,
        n_layers=n_layers,
        radial_basis_size=8,
        radial_mlp_size=16,
        radial_mlp_layers=2,
        index_weights=False,
        layer_norm=layer_norm,
    )


def random_graph(n_atoms=6, n_species=3, seed=0, cutoff=3.0, box=2.5):
    """A molecule-like graph: random positions, all pairs within cutoff, no cell."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, box, size=(n_atoms, 3)).astype(np.float32)
    species = rng.integers(0, n_species, size=n_atoms).astype(np.int32)
    s, r = [], []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j and np.linalg.norm(pos[i] - pos[j]) < cutoff:
                s.append(i)
                r.append(j)
    g = jraph.GraphsTuple(
        n_node=np.array([n_atoms]),
        n_edge=np.array([len(s)]),
        nodes={"species": species, "positions": pos, "forces": np.zeros_like(pos)},
        edges={"shifts": np.zeros((len(s), 3), np.float32)},
        senders=np.array(s, np.int32),
        receivers=np.array(r, np.int32),
        globals={"energy": np.zeros(1, np.float32), "cell": None},
    )
    return g


def energy(model, g):
    e, f, _ = model(g)
    return jnp.sum(e), f
