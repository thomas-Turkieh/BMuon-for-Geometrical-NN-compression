"""Pruning: exported model = zeroed model, stays equivariant, survives save/load."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from helpers import energy, random_graph, small_model  # noqa: E402

from bmuon.channels import build_channel_spec  # noqa: E402
from bmuon.model import load_pruned, save_pruned  # noqa: E402
from bmuon.prune import (  # noqa: E402
    magnitude_scores,
    prune,
    select_channels,
    tp_irreps_summary,
    zero_channels,
)

CONFIGS = [
    dict(lmax=2, hidden="8x0e+4x1o+4x2e", n_layers=3),
    dict(lmax=3, hidden="8x0e+4x1o+4x2e+4x3o", n_layers=4),
]


def _random_keep(spec, frac, seed, min_keep=1):
    scores = np.random.default_rng(seed).uniform(size=spec.num_channels)
    return select_channels(scores, spec, frac, min_keep=min_keep)


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("frac", [0.3, 0.7])
def test_export_equals_zeroed(cfg, frac):
    model = small_model(**cfg)
    spec = build_channel_spec(model)
    keep = _random_keep(spec, frac, seed=int(frac * 10))
    pruned = prune(model, spec, keep)
    zeroed = zero_channels(model, spec, keep)
    g = random_graph()
    e1, f1 = energy(zeroed, g)
    e2, f2 = energy(pruned, g)
    np.testing.assert_allclose(e1, e2, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(f1, f2, rtol=1e-4, atol=1e-5)
    # the export is genuinely smaller
    assert build_channel_spec(pruned).num_channels == keep.sum()


def test_whole_irrep_can_disappear():
    model = small_model(**CONFIGS[1])
    spec = build_channel_spec(model)
    keep = np.ones(spec.num_channels, bool)
    for i, c in enumerate(spec.channels):
        if c.layer == 1 and c.ir == "3o":
            keep[i] = False
    pruned = prune(model, spec, keep)
    assert "3o" not in tp_irreps_summary(pruned)[1]
    g = random_graph()
    np.testing.assert_allclose(
        energy(pruned, g)[0], energy(zero_channels(model, spec, keep), g)[0], rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("cfg", CONFIGS)
def test_pruned_model_is_equivariant(cfg):
    model = small_model(**cfg)
    spec = build_channel_spec(model)
    pruned = prune(model, spec, _random_keep(spec, 0.5, seed=3))
    g = random_graph()
    e, f = energy(pruned, g)
    for seed in range(3):
        R = Rotation.random(random_state=seed).as_matrix().astype(np.float32)
        if seed == 2:
            R = -R  # an improper rotation (rotation followed by inversion)
        gr = g._replace(nodes={**g.nodes, "positions": g.nodes["positions"] @ R.T})
        er, fr = energy(pruned, gr)
        np.testing.assert_allclose(er, e, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(fr, np.asarray(f) @ R.T, rtol=1e-3, atol=1e-4)


def test_selection_rules():
    model = small_model(**CONFIGS[1])
    spec = build_channel_spec(model)
    scores = magnitude_scores(model, spec)
    keep = select_channels(scores, spec, 0.9, min_keep=2)
    for t in range(4):
        for ir in {c.ir for c in spec.channels if c.layer == t}:
            ids = spec.ids_of(t, ir)
            assert keep[ids].sum() >= min(2, len(ids))
    keep = select_channels(scores, spec, 0.5, by="cost")
    assert spec.weights[~keep].sum() >= 0.5 * spec.weights.sum() - spec.weights.max()
    keep = select_channels(scores, spec, 0.5, round_to=4)
    for t in range(4):
        for ir in {c.ir for c in spec.channels if c.layer == t}:
            m = keep[spec.ids_of(t, ir)].sum()
            assert m % 4 == 0 or m == len(spec.ids_of(t, ir))


def test_save_load_roundtrip():
    model = small_model(**CONFIGS[0])
    spec = build_channel_spec(model)
    pruned = prune(model, spec, _random_keep(spec, 0.5, seed=1))
    config = dict(
        atomic_numbers=[1, 6, 8], hidden_irreps="8x0e+4x1o+4x2e", lmax=2, cutoff=3.0, n_layers=3,
        radial_basis_size=8, radial_mlp_size=16, radial_mlp_layers=2, radial_polynomial_p=2.0,
        mlp_init_scale=4.0, index_weights=False, layer_norm=True, shift=0.0, scale=1.0,
        avg_n_neighbors=1.0,
    )
    with tempfile.TemporaryDirectory() as d:
        save_pruned(f"{d}/m.nqx", pruned, config)
        loaded, _ = load_pruned(f"{d}/m.nqx")
    g = random_graph()
    np.testing.assert_allclose(energy(loaded, g)[0], energy(pruned, g)[0], rtol=1e-6)
    assert tp_irreps_summary(loaded) == tp_irreps_summary(pruned)
