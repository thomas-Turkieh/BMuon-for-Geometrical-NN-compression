"""Budgeted Muon (BMuon) on the type-I channels of Nequix, as optax transformations.

Budgeted matrices (the slices listed in a ChannelSpec) get the BMuon direction

    Delta = D^a Polar(D^a M)   (slices along rows)   or   Polar(M D^a) D^a   (along columns),

where D = diag(mu) holds the allocation of the channel owning each slice. The allocation is
refreshed every ``refresh_every`` steps by one minorize-maximize step of

    max_{mu in [eps,1]^N, sum_c w_c mu_c = K}  Phi_M(mu)

with the closed form  mu_c = clip((alpha a_c / (lam w_c))^(1/(1-alpha)), eps, 1).
All other parameters keep the Muon / AdamW update of optax.contrib.muon, so the whole
optimizer reduces exactly to Nequix's Muon when the budget equals the total cost.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from optax.transforms import _masking

from bmuon.channels import ChannelSpec

ScalarOrSchedule = Union[float, Callable[[jax.Array], jax.Array]]

NS_COEFFS = (3.4445, -4.7750, 2.0315)


# ----------------------------------------------------------------------------- utils


def newton_schulz(x: jax.Array, steps: int = 5, coeffs=NS_COEFFS, eps: float = 1e-8):
    """Approximate polar factor, identical to optax.contrib.muon's default iteration."""
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (jnp.linalg.norm(x) + eps)
    a, b, c = coeffs

    def body(_, x):
        g = x @ x.T
        return a * x + (b * g + c * g @ g) @ x

    x = jax.lax.fori_loop(0, steps, body, x)
    return x.T if transposed else x


def lowrank_polar(x: jax.Array, rel_rank: jax.Array) -> jax.Array:
    """NuMuon direction: polar factor restricted to the top-k singular directions,
    k = max(1, round(rel_rank * min(shape))). Exact, through the eigendecomposition of the
    small Gram matrix (U_k V_k^T = X V_k S_k^-1 V_k^T; 2.5x faster than jnp.linalg.svd on
    these matrices); NuMuon's randomized block Krylov approximates the same quantity."""
    tall = x.shape[0] >= x.shape[1]
    y = x if tall else x.T
    w, v = jnp.linalg.eigh(y.T @ y)
    w, v = w[::-1], v[:, ::-1]  # decreasing singular values
    q = v.shape[1]
    k = jnp.maximum(1, jnp.round(rel_rank * q)).astype(jnp.int32)
    # directions with sigma^2 < 1e-6 sigma_max^2 are numerically null in float32 (the Gram
    # matrix squares the conditioning): drop them instead of dividing by ~0 (that gave NaN);
    # Newton-Schulz also sends such directions to ~0
    ok = (jnp.arange(q) < k) & (w > 1e-6 * jnp.maximum(w[0], 1e-30))
    inv = jnp.where(ok, 1.0 / jnp.sqrt(jnp.where(ok, w, 1.0)), 0.0)
    p = y @ (v * inv[None, :]) @ v.T
    return p if tall else p.T


def _scale(m: jax.Array, d: jax.Array, axis: int) -> jax.Array:
    return d[:, None] * m if axis == 0 else m * d[None, :]


def _as_schedule(x: ScalarOrSchedule) -> Callable[[jax.Array], jax.Array]:
    return x if callable(x) else (lambda _: jnp.asarray(x, jnp.float32))


def solve_allocation(scores, weights, budget, alpha, eps, iters: int = 80):
    """argmax_{mu in [eps,1]^N, w.mu = K} sum_c a_c mu_c^alpha  (closed form + bisection)."""
    a = jnp.maximum(scores, 1e-30)
    w = weights
    total = jnp.sum(w)
    p = 1.0 / (1.0 - alpha)

    def mu_of(log_lam):
        return jnp.clip((alpha * a / (jnp.exp(log_lam) * w)) ** p, eps, 1.0)

    log_lo = jnp.log(jnp.min(alpha * a / w)) - 1.0  # every mu = 1 below this
    log_hi = jnp.log(jnp.max(alpha * a / (w * eps ** (1.0 - alpha)))) + 1.0  # every mu = eps

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        too_big = jnp.sum(w * mu_of(mid)) > budget  # spend > K: raise lambda
        return jnp.where(too_big, mid, lo), jnp.where(too_big, hi, mid)

    lo, hi = jax.lax.fori_loop(0, iters, body, (log_lo, log_hi))
    mu = mu_of(0.5 * (lo + hi))
    mu = jnp.where(budget >= total * (1 - 1e-7), jnp.ones_like(mu), mu)
    mu = jnp.where(budget <= eps * total, jnp.full_like(mu, eps), mu)
    return mu


# ------------------------------------------------------------------- BMuon direction


class BMuonState(NamedTuple):
    count: jax.Array
    mu: Any  # momentum of the budgeted matrices
    alloc: jax.Array  # (num_channels,) current allocation
    scores: jax.Array  # (num_channels,) scores at the last refresh
    keep: jax.Array  # (num_channels,) 1 = alive, 0 = pruned for good (gradual pruning)


def scale_by_bmuon(
    spec: ChannelSpec,
    budget_fraction: ScalarOrSchedule,
    alpha: float = 0.5,
    refresh_every: int = 5,
    eps: float = 1e-6,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
    selective_wd=None,
    rank_schedule: Optional[Callable] = None,
) -> optax.GradientTransformation:
    """BMuon direction for the matrices of ``spec``, including Muon's shape rescaling.

    ``rank_schedule``: if given (step -> relative rank in (0, 1]), the polar factor is
    replaced by its truncation to the top singular directions (NuMuon direction).
    Pruned channels (``state.keep == 0``) get zero momentum, zero direction and the
    floor allocation, so their slices stay exactly zero once zeroed.

    Expects the update tree restricted to the budgeted leaves (other leaves may be
    MaskedNode, as inside ``optax.partition``).
    """
    n = spec.num_channels
    weights = jnp.asarray(spec.weights, jnp.float32)
    total = float(spec.weights.sum())
    frac = _as_schedule(budget_fraction)
    slices = {s.keypath: s for s in spec.slices}
    chan = {k: jnp.asarray(s.channel_ids) for k, s in slices.items()}

    def leaves_by_key(tree):
        flat = jax.tree_util.tree_flatten_with_path(
            tree, is_leaf=lambda x: isinstance(x, _masking.MaskedNode)
        )[0]
        return {
            jax.tree_util.keystr(p): x
            for p, x in flat
            if not isinstance(x, _masking.MaskedNode)
        }

    def map_by_key(fn, tree):
        def f(path, x):
            if isinstance(x, _masking.MaskedNode):
                return x
            return fn(jax.tree_util.keystr(path), x)

        return jax.tree_util.tree_map_with_path(
            f, tree, is_leaf=lambda x: isinstance(x, _masking.MaskedNode)
        )

    def init_fn(params):
        missing = set(slices) - set(leaves_by_key(params))
        if missing:
            raise KeyError(f"budgeted leaves missing from params: {sorted(missing)[:3]}")
        k0 = frac(jnp.zeros([], jnp.int32)) * total
        alloc = jnp.clip(jnp.full((n,), k0 / total, jnp.float32), eps, 1.0)  # uniform, w.mu = K
        return BMuonState(
            count=jnp.zeros([], jnp.int32),
            mu=optax.tree.zeros_like(params),
            alloc=alloc,
            scores=jnp.zeros(n),
            keep=jnp.ones((n,), jnp.float32),
        )

    def polar(m, count):
        if rank_schedule is None:
            return newton_schulz(m, ns_steps)
        return lowrank_polar(m, rank_schedule(count))

    def update_fn(updates, state, params=None):
        keep = state.keep
        updates = map_by_key(lambda k, g: _scale(g, keep[chan[k]], slices[k].axis), updates)
        mu = optax.tree.update_moment(updates, state.mu, beta, 1)
        count_inc = state.count + 1
        if nesterov:
            mu_hat = jax.tree.map(
                lambda m, g: beta * m + (1 - beta) * g,
                optax.tree.bias_correction(mu, beta, count_inc + 1),
                optax.tree.bias_correction(updates, beta, count_inc),
            )
        else:
            mu_hat = optax.tree.bias_correction(mu, beta, count_inc)
        mats = leaves_by_key(mu_hat)
        budget = frac(state.count) * total

        def refresh(_):
            scores = jnp.zeros(n)
            for k, m in mats.items():
                s = slices[k]
                d = state.alloc[chan[k]] ** alpha
                p_bar = polar(_scale(m, d, s.axis), state.count)
                per_slice = jnp.sum(m * p_bar, axis=1 - s.axis)
                scores = scores + jax.ops.segment_sum(per_slice, chan[k], num_segments=n)
            scores = scores * keep
            alloc = solve_allocation(scores, weights, budget, alpha, eps)
            return jnp.where(keep > 0, alloc, eps), scores

        alloc, scores = jax.lax.cond(
            state.count % refresh_every == 0,
            refresh,
            lambda _: (state.alloc, state.scores),
            None,
        )

        def direction(key, m):
            s = slices[key]
            d = alloc[chan[key]] ** alpha
            delta = _scale(polar(_scale(m, d, s.axis), state.count), d * keep[chan[key]], s.axis)
            fan_in, fan_out = m.shape
            return delta * jnp.sqrt(jnp.maximum(1.0, fan_out / fan_in))

        new_updates = map_by_key(direction, mu_hat)

        # Per-channel selective weight decay: wd_base on all, +wd_prune on dying channels.
        # Added before scale_by_learning_rate so final = -lr*(direction + wd*W).
        if selective_wd is not None and params is not None:
            wd_base_val, wd_prune_val = selective_wd
            mean_alloc = jnp.maximum(budget / total, eps)
            p_leaves = leaves_by_key(params)

            def add_selective_wd(key, upd):
                if key not in p_leaves:
                    return upd
                s = slices[key]
                mu_c = alloc[chan[key]]
                under_alloc = jnp.maximum(0.0, 1.0 - mu_c / mean_alloc) ** 2
                wd_vec = wd_base_val + wd_prune_val * under_alloc
                return upd + _scale(p_leaves[key], wd_vec, s.axis)

            new_updates = map_by_key(add_selective_wd, new_updates)

        return new_updates, BMuonState(count_inc, mu, alloc, scores, keep)

    return optax.GradientTransformation(init_fn, update_fn)


# ------------------------------------------------------------ group-lasso proximal step


class ProxGLState(NamedTuple):
    count: jax.Array


def prox_group_lasso(spec: ChannelSpec, lam: float, learning_rate: ScalarOrSchedule):
    """Exact proximal Group-Lasso step on the coupled slices of every channel (Muon+GL).

    Must be the last transformation of the chain: with w' = w + u (u = the update so far),
    every channel group g_c (all its slices) becomes g_c * max(0, 1 - lr * lam / ||g_c||).
    """
    n = spec.num_channels
    slices = {s.keypath: s for s in spec.slices}
    chan = {k: jnp.asarray(s.channel_ids) for k, s in slices.items()}
    lr = _as_schedule(learning_rate)
    is_masked = lambda x: isinstance(x, _masking.MaskedNode)  # noqa: E731

    def init_fn(params):
        del params
        return ProxGLState(jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("prox_group_lasso needs params")
        flat_p = {jax.tree_util.keystr(p): x for p, x in
                  jax.tree_util.tree_flatten_with_path(params, is_leaf=is_masked)[0]}
        flat_u = jax.tree_util.tree_flatten_with_path(updates, is_leaf=is_masked)[0]
        sq = jnp.zeros(n)
        for p, u in flat_u:
            k = jax.tree_util.keystr(p)
            if k in slices and not is_masked(u):
                w = flat_p[k] + u
                sq = sq + jax.ops.segment_sum(jnp.sum(w**2, axis=1 - slices[k].axis),
                                              chan[k], num_segments=n)
        factor = jnp.maximum(0.0, 1.0 - lr(state.count) * lam / jnp.maximum(jnp.sqrt(sq), 1e-12))

        def f(path, u):
            k = jax.tree_util.keystr(path)
            if is_masked(u) or k not in slices:
                return u
            w = flat_p[k] + u
            return _scale(w, factor[chan[k]], slices[k].axis) - flat_p[k]

        out = jax.tree_util.tree_map_with_path(f, updates, is_leaf=is_masked)
        return out, ProxGLState(state.count + 1)

    return optax.GradientTransformation(init_fn, update_fn)


# --------------------------------------------------------------- full optimizer


def _keystr_mask(keys: set, value_in, value_out):
    def labels(tree):
        return jax.tree_util.tree_map_with_path(
            lambda p, _: value_in if jax.tree_util.keystr(p) in keys else value_out, tree
        )

    return labels


def bmuon(
    learning_rate: ScalarOrSchedule,
    spec: ChannelSpec,
    budget_fraction: ScalarOrSchedule = 1.0,
    alpha: float = 0.5,
    refresh_every: int = 5,
    eps: float = 1e-6,
    weight_decay: Optional[float] = None,
    weight_decay_keys: Optional[set] = None,
    selective_weight_decay: Optional[float] = None,
    beta: float = 0.95,
    ns_steps: int = 5,
    rank_schedule: Optional[Callable] = None,
    group_lasso: Optional[float] = None,
) -> optax.GradientTransformation:
    """BMuon on the budgeted matrices, optax.contrib.muon (Muon + AdamW) on the rest.

    With ``budget_fraction=1`` the budgeted branch is exactly Muon; ``rank_schedule`` turns
    its direction into NuMuon's, ``group_lasso`` adds the exact Group-Lasso prox (Muon+GL).

    ``weight_decay_keys``: keystrs of the leaves that receive weight decay (Nequix decays
    every linear weight). Budgeted leaves always receive it when ``weight_decay`` is set.
    """
    budgeted = spec.keypaths
    decay_keys = set(weight_decay_keys or ()) - budgeted

    use_selective = selective_weight_decay is not None and weight_decay is not None
    sel_wd = (weight_decay, selective_weight_decay) if use_selective else None

    budget_branch = [
        scale_by_bmuon(
            spec, budget_fraction, alpha, refresh_every, eps, beta, ns_steps,
            selective_wd=sel_wd, rank_schedule=rank_schedule,
        )
    ]
    # selective_wd already handles weight decay on budgeted matrices;
    # only add uniform add_decayed_weights when selective_wd is not active.
    if weight_decay and not use_selective:
        budget_branch.append(optax.add_decayed_weights(weight_decay))
    budget_branch.append(optax.scale_by_learning_rate(learning_rate))
    if group_lasso:
        budget_branch.append(prox_group_lasso(spec, group_lasso, learning_rate))

    rest = optax.contrib.muon(
        learning_rate=learning_rate,
        beta=beta,
        ns_steps=ns_steps,
        weight_decay=weight_decay if weight_decay else None,
        weight_decay_mask=_keystr_mask(decay_keys, True, False),
    )
    return optax.partition(
        {"bmuon": optax.chain(*budget_branch), "rest": rest},
        _keystr_mask(budgeted, "bmuon", "rest"),
    )


def weight_decay_keys(model) -> set:
    """Keystrs of the leaves Nequix applies weight decay to."""
    import equinox as eqx

    from nequix.model import weight_decay_mask

    mask = weight_decay_mask(model)
    params = eqx.filter(model, eqx.is_array)
    mask = eqx.filter(mask, lambda x: isinstance(x, bool), replace=None)
    flat_p = jax.tree_util.tree_flatten_with_path(params)[0]
    flat_m = jax.tree_util.tree_flatten_with_path(mask)[0]
    m = {jax.tree_util.keystr(p): bool(v) for p, v in flat_m if isinstance(v, (bool, np.bool_))}
    return {jax.tree_util.keystr(p) for p, _ in flat_p if m.get(jax.tree_util.keystr(p), False)}


def map_bmuon_state(opt_state, fn):
    """opt_state with every BMuonState replaced by fn(state) (through chains, partitions,
    MultiSteps; works on the pmap-replicated state too)."""
    return jax.tree_util.tree_map(
        lambda x: fn(x) if isinstance(x, BMuonState) else x,
        opt_state,
        is_leaf=lambda x: isinstance(x, BMuonState),
    )


def find_bmuon_state(opt_state) -> Optional[BMuonState]:
    """Locate the BMuonState inside a (chained, partitioned, MultiSteps) optimizer state."""
    found = []

    def visit(x):
        if isinstance(x, BMuonState):
            found.append(x)
            return
        if isinstance(x, (tuple, list)):
            for y in x:
                visit(y)
        elif isinstance(x, dict):
            for y in x.values():
                visit(y)
        elif hasattr(x, "_fields"):
            for y in x:
                visit(y)
        elif hasattr(x, "inner_states"):
            visit(x.inner_states)
        elif hasattr(x, "inner_state"):
            visit(x.inner_state)

    visit(opt_state)
    return found[0] if found else None
