"""Figure 2 methods: Muon+GL prox, NuMuon direction, gradual-pruning mask, SVD compression."""

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).parent))
from helpers import energy, random_graph, small_model  # noqa: E402

from bmuon.channels import build_channel_spec  # noqa: E402
from bmuon.optimizer import (  # noqa: E402
    BMuonState,
    bmuon,
    find_bmuon_state,
    lowrank_polar,
    map_bmuon_state,
    weight_decay_keys,
)
from bmuon.prune import magnitude_scores, select_channels  # noqa: E402


def _grads(params, seed):
    leaves, tdef = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    return jax.tree.unflatten(tdef, [jax.random.normal(k, x.shape) for k, x in zip(keys, leaves)])


def _leaves(tree):
    return {jax.tree_util.keystr(p): x for p, x in jax.tree_util.tree_flatten_with_path(tree)[0]}


def _channel_norms(params, spec):
    lv = _leaves(params)
    sq = np.zeros(spec.num_channels)
    for s in spec.slices:
        np.add.at(sq, s.channel_ids, (np.asarray(lv[s.keypath]) ** 2).sum(axis=1 - s.axis))
    return np.sqrt(sq)


def test_lowrank_polar_rank_and_full_rank():
    x = jax.random.normal(jax.random.key(0), (12, 8))
    for rel, k in [(1.0, 8), (0.25, 2), (0.5, 4)]:
        p = np.asarray(lowrank_polar(x, jnp.asarray(rel)))
        sv = np.linalg.svd(p, compute_uv=False)
        assert (sv > 1e-5).sum() == k
        np.testing.assert_allclose(sv[:k], 1.0, atol=1e-5)
    u, _, vt = np.linalg.svd(np.asarray(x), full_matrices=False)
    np.testing.assert_allclose(np.asarray(lowrank_polar(x, jnp.asarray(1.0))), u @ vt, atol=1e-5)


def test_group_lasso_prox_is_exact():
    """One step with the prox: every channel group equals the plain update shrunk by
    max(0, 1 - lr*lam/||g||); a huge lam zeroes every channel."""
    model = small_model()
    params = eqx.filter(model, eqx.is_array)
    spec = build_channel_spec(model, cost_model="channel")
    g = _grads(params, 1)
    lr, lam = 1e-2, 3.0
    plain = bmuon(lr, spec, 1.0)
    prox = bmuon(lr, spec, 1.0, group_lasso=lam)
    u0, _ = plain.update(g, plain.init(params), params)
    u1, _ = prox.update(g, prox.init(params), params)
    w0, w1 = optax.apply_updates(params, u0), optax.apply_updates(params, u1)
    n0 = _channel_norms(w0, spec)
    factor = np.maximum(0, 1 - lr * lam / n0)
    np.testing.assert_allclose(_channel_norms(w1, spec), n0 * factor, rtol=1e-4, atol=1e-6)
    huge = bmuon(lr, spec, 1.0, group_lasso=1e6)
    u2, _ = huge.update(g, huge.init(params), params)
    assert _channel_norms(optax.apply_updates(params, u2), spec).max() < 1e-6
    # leaves outside the channel matrices are untouched by the prox
    keys = spec.keypaths
    for k, a in _leaves(u0).items():
        if k not in keys:
            np.testing.assert_allclose(np.asarray(a), np.asarray(_leaves(u1)[k]), atol=1e-7)


def test_pruned_channels_stay_zero():
    """With keep = 0 for some channels and their slices zeroed, many BMuon updates (with weight
    decay and a decreasing budget) leave them exactly zero, and the others move."""
    model = small_model(hidden="8x0e+4x1o+4x2e")
    params = eqx.filter(model, eqx.is_array)
    spec = build_channel_spec(model, cost_model="channel")
    opt = bmuon(1e-2, spec, lambda t: 1.0 - 0.05 * t, weight_decay=1e-3,
                weight_decay_keys=weight_decay_keys(model))
    state = opt.init(params)
    keep = np.ones(spec.num_channels, bool)
    keep[np.random.default_rng(0).choice(spec.num_channels, spec.num_channels // 3, replace=False)] = False
    from bmuon.prune import zero_channels

    params = eqx.filter(zero_channels(model, spec, keep), eqx.is_array)
    state = map_bmuon_state(state, lambda st: st._replace(keep=jnp.asarray(keep, jnp.float32)))
    for t in range(12):
        u, state = opt.update(_grads(params, t), state, params)
        params = optax.apply_updates(params, u)
    norms = _channel_norms(params, spec)
    assert norms[~keep].max() == 0.0
    assert norms[keep].min() > 0.0
    bs = find_bmuon_state(state)
    assert isinstance(bs, BMuonState)
    assert np.all(np.asarray(bs.alloc)[~keep] <= 1e-6 + 1e-12)


def test_gradual_budget_schedule():
    from bmuon.train import gradual_budget_schedule

    g = {"prune_after_epochs": [1, 3], "levels": [0.2, 0.5]}
    f = gradual_budget_schedule(g, steps_per_epoch=10)  # phases [0,20) and [20,40)
    np.testing.assert_allclose(float(f(0)), 1.0)
    np.testing.assert_allclose(float(f(10)), 0.8)   # end of the first half: next level
    np.testing.assert_allclose(float(f(15)), 0.8)
    np.testing.assert_allclose(float(f(20)), 0.8)   # new phase starts at the survivors
    np.testing.assert_allclose(float(f(25)), 0.65)
    np.testing.assert_allclose(float(f(30)), 0.5)
    np.testing.assert_allclose(float(f(99)), 0.5)


def test_numuon_rank_schedule():
    from bmuon.train import numuon_rank_schedule

    r = numuon_rank_schedule({"final_rank": 0.25, "full_rank_frac": 0.1, "anneal_end_frac": 0.8}, 100)
    assert float(r(0)) == 1.0 and float(r(10)) == 1.0
    np.testing.assert_allclose(float(r(45)), 0.625, atol=1e-6)
    np.testing.assert_allclose(float(r(80)), 0.25, atol=1e-6)
    np.testing.assert_allclose(float(r(100)), 0.25, atol=1e-6)


def test_select_keeps_already_pruned_out():
    """Gradual selection: already-pruned channels (score -inf) are removed first and the
    level counts all channels."""
    model = small_model()
    spec = build_channel_spec(model, cost_model="channel")
    s = magnitude_scores(model, spec)
    dead = np.zeros(spec.num_channels, bool)
    dead[:5] = True
    s[dead] = -np.inf
    keep = select_channels(s, spec, 0.4, by="count", min_keep=1) & ~dead
    assert not keep[dead].any()
    assert abs((~keep).sum() - round(0.4 * spec.num_channels)) <= 2


def test_svd_compress_matches_params_and_full_rank_is_exact():
    from bmuon.experiments.fig2 import budgeted_params, slice_shapes, svd_compress

    model = small_model(hidden="16x0e+8x1o+8x2e")
    spec = build_channel_spec(model, cost_model="channel")
    slice_shapes(model, spec)
    total = budgeted_params(spec)
    m_full, p_full, _ = svd_compress(model, spec, total)
    g = random_graph()
    e0, f0 = energy(model, g)
    e1, f1 = energy(m_full, g)
    np.testing.assert_allclose(e1, e0, rtol=1e-4, atol=1e-4)
    assert p_full <= total
    m_half, p_half, r = svd_compress(model, spec, total // 2)
    assert p_half < total and r >= 1
    e2, _ = energy(m_half, g)
    assert np.isfinite(float(e2))


def test_lowrank_polar_rank_deficient_is_finite():
    """Zero rows / tiny singular values (e.g. unused species rows) must not give NaN."""
    x = np.array(jax.random.normal(jax.random.key(3), (40, 16)))
    x[:30] = 0.0                      # rank <= 10 < 16 columns
    x[30] *= 1e-6
    for rel in (1.0, 0.25):
        p = np.asarray(lowrank_polar(jnp.asarray(x), jnp.asarray(rel)))
        assert np.all(np.isfinite(p))
        assert np.linalg.svd(p, compute_uv=False).max() < 1 + 1e-4
    p = np.asarray(lowrank_polar(jnp.asarray(x.T), jnp.asarray(1.0)))
    assert np.all(np.isfinite(p))
