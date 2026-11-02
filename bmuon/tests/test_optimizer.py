"""BMuon optimizer: reduction to Muon, allocation solver, channel scaling."""

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from helpers import small_model  # noqa: E402

from bmuon.channels import build_channel_spec  # noqa: E402
from bmuon.optimizer import (  # noqa: E402
    bmuon,
    find_bmuon_state,
    newton_schulz,
    solve_allocation,
    weight_decay_keys,
)
from nequix.model import weight_decay_mask  # noqa: E402


def _random_grads(params, seed):
    leaves, tdef = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    return jax.tree.unflatten(tdef, [jax.random.normal(k, x.shape) for k, x in zip(keys, leaves)])


def _run(opt, params, steps=7):
    state = opt.init(params)
    out = []
    for t in range(steps):
        upd, state = opt.update(_random_grads(params, t), state, params)
        params = optax.apply_updates(params, upd)
        out.append(upd)
    return out, state


def test_newton_schulz_matches_optax():
    from optax.contrib._muon import orthogonalize_via_newton_schulz

    for shape in [(5, 9), (9, 5), (7, 7)]:
        x = jax.random.normal(jax.random.key(1), shape)
        ref = orthogonalize_via_newton_schulz(x, jnp.asarray((3.4445, -4.7750, 2.0315)), 5)
        np.testing.assert_allclose(newton_schulz(x), ref, rtol=1e-5, atol=1e-6)


def test_full_budget_is_muon():
    """With K = W (every mu = 1) the optimizer equals Nequix's optax.contrib.muon."""
    model = small_model(lmax=2, hidden="8x0e+4x1o+4x2e")
    params = eqx.filter(model, eqx.is_array)
    spec = build_channel_spec(model)
    ref = optax.contrib.muon(1e-2, weight_decay=1e-3, weight_decay_mask=weight_decay_mask(model))
    ours = bmuon(1e-2, spec, 1.0, weight_decay=1e-3, weight_decay_keys=weight_decay_keys(model))
    u_ref, _ = _run(ref, params)
    u_ours, state = _run(ours, params)
    for a, b in zip(u_ref, u_ours):
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(find_bmuon_state(state).alloc, 1.0)


@pytest.mark.parametrize("alpha", [0.25, 0.5])
def test_allocation_solver(alpha):
    """Closed form satisfies the budget and beats random feasible allocations."""
    rng = np.random.default_rng(0)
    n, eps = 40, 1e-3
    a = rng.exponential(size=n)
    w = rng.uniform(0.5, 3.0, size=n)
    w = w / w.mean()
    budget = 0.3 * w.sum()
    mu = np.asarray(solve_allocation(jnp.asarray(a), jnp.asarray(w), budget, alpha, eps))
    assert np.all(mu >= eps - 1e-9) and np.all(mu <= 1 + 1e-9)
    assert abs(w @ mu - budget) < 1e-3 * budget
    best = a @ mu**alpha
    for _ in range(2000):
        x = rng.uniform(eps, 1, size=n)
        for _ in range(50):  # rescale into the budget, respecting the box
            x = np.clip(x * budget / (w @ x), eps, 1)
        if abs(w @ x - budget) < 1e-3 * budget:
            assert a @ x**alpha <= best + 1e-6
    # cost-aware: at equal score, a cheaper channel gets at least as much allocation
    a2 = np.ones(n)
    mu2 = np.asarray(solve_allocation(jnp.asarray(a2), jnp.asarray(w), budget, alpha, eps))
    order = np.argsort(w)
    assert np.all(np.diff(mu2[order]) <= 1e-6)


def test_low_allocation_shrinks_channel_updates():
    """Rows/columns of a channel move by at most ~mu_c^alpha (times Muon's shape factor)."""
    model = small_model(lmax=2, hidden="8x0e+4x1o+4x2e")
    params = eqx.filter(model, eqx.is_array)
    spec = build_channel_spec(model, cost_model="cg")
    opt = bmuon(1.0, spec, 0.3, alpha=0.5, refresh_every=1)
    upd, state = _run(opt, params, steps=3)
    alloc = np.asarray(find_bmuon_state(state).alloc)
    w = spec.weights
    assert abs(w @ alloc - 0.3 * w.sum()) < 1e-3 * w.sum()  # the budget counts cost
    assert alloc.min() < 0.5  # the budget is binding
    flat = {jax.tree_util.keystr(p): x for p, x in jax.tree_util.tree_flatten_with_path(upd[-1])[0]}
    for s in spec.slices:
        d = np.asarray(flat[s.keypath]) / np.sqrt(max(1.0, flat[s.keypath].shape[1] / flat[s.keypath].shape[0]))
        norms = np.linalg.norm(d if s.axis == 0 else d.T, axis=1)
        bound = alloc[s.channel_ids] ** 0.5 * 1.25  # Newton-Schulz is only approximately polar
        assert np.all(norms <= bound + 1e-6), (s.keypath, norms.max(), bound.min())
