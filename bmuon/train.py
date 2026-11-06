"""Train Nequix with Muon or BMuon.

Usage:  python -m bmuon.train bmuon/configs/<config>.yml

The loop follows nequix/train.py (data, loss, EMA, evaluation are imported from Nequix)
with these additions:
  * optimizer "bmuon" (budgeted Muon on the type-I channels, Muon/AdamW elsewhere);
  * a budget schedule annealed from the full cost to ``bmuon.budget``;
  * gradient accumulation (``grad_accum_steps``) to keep Nequix's global batch on one GPU;
  * ``base_config`` inheritance, optional training subset, step-based lr schedule;
  * automatic resume from ``out_dir/state.pkl`` and local JSONL logs (wandb optional);
  * the allocation (one value per channel) is saved every epoch;
  * "numuon" (NuMuon low-rank direction), Muon+GL (``bmuon.group_lasso``) and gradual
    channel pruning (``gradual:``, channels zeroed for good at fixed epochs) for Figure 2;
  * data parallelism over several machines (``--coordinator host:port --num-processes N
    --process-id i``, see bmuon/experiments/launch_dist.sh). Every process reads its own
    shard of the data and runs the same number of micro-steps per epoch; gradients, metrics
    and validation are averaged over all processes; process 0 writes logs and checkpoints.
"""

from __future__ import annotations

import argparse
import atexit
import functools
import multiprocessing
import multiprocessing.util  # noqa: F401  (registers its exit hook before ours, see train)
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import yaml

from bmuon.channels import build_channel_spec
from bmuon.optimizer import bmuon, find_bmuon_state, map_bmuon_state, weight_decay_keys
from bmuon.prune import magnitude_scores, select_channels
from nequix.data import (
    AseDBDataset,
    ConcatDataset,
    DataLoader,
    IndexDataset,
    ParallelLoader,
    average_atom_energies,
    dataset_stats,
    prefetch,
)
from nequix.model import Nequix, save_model, weight_decay_mask
from nequix.train import load_training_state, loss, save_training_state


# ------------------------------------------------------------------------ config


def _expand(x):
    if isinstance(x, str):
        return os.path.expandvars(x)
    if isinstance(x, dict):
        return {k: _expand(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_expand(v) for v in x]
    return x


def apply_overrides(config: dict, overrides: list) -> dict:
    """``key=value`` or ``a.b=value`` overrides; values are parsed as YAML."""
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return config


def load_config(path: str) -> dict:
    """YAML config; ``base_config`` (relative to this file or absolute) is merged first.
    Environment variables in string values ($USER, $TMPDIR, ...) are expanded."""
    with open(path) as f:
        config = yaml.safe_load(f)
    base = config.pop("base_config", None)
    if base:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = (Path(path).parent / base_path).resolve()
            if not base_path.exists():
                base_path = Path(base)
        merged = load_config(str(base_path))
        for k, v in config.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
        config = merged
    return _expand(config)


def budget_schedule(cfg: dict, total_steps: int):
    """Fraction of the total cost available, linear from 1 to ``budget`` over the first
    ``anneal_fraction`` of training, then constant (as in the BMuon paper)."""
    target = float(cfg.get("budget", 1.0))
    n = max(1, int(cfg.get("anneal_fraction", 0.25) * total_steps))

    def frac(step):
        t = jnp.clip(step / n, 0.0, 1.0)
        return 1.0 + (target - 1.0) * t

    return frac


def gradual_budget_schedule(gradual: dict, steps_per_epoch: int):
    """Budget for gradual pruning (BMuon paper, App. E.4): in every phase between two
    pruning events, linear from the current kept fraction to the next one over the first
    half of the phase, then constant; after the last event, the final kept fraction."""
    bounds = [0] + [(e + 1) * steps_per_epoch for e in gradual["prune_after_epochs"]]
    keeps = [1.0] + [1.0 - lv for lv in gradual["levels"]]

    def frac(step):
        res = jnp.asarray(keeps[-1], jnp.float32)
        for i in range(len(bounds) - 1):
            s0, s1, k0, k1 = bounds[i], bounds[i + 1], keeps[i], keeps[i + 1]
            half = s0 + max(1, (s1 - s0) // 2)
            v = jnp.where(step < half, k0 + (k1 - k0) * (step - s0) / (half - s0), k1)
            res = jnp.where((step >= s0) & (step < s1), v, res)
        return res

    return frac


def numuon_rank_schedule(cfg: dict, total_steps: int):
    """NuMuon cosine rank schedule: full rank for the first ``full_rank_frac`` of training,
    cosine from 1 to ``final_rank`` until ``anneal_end_frac``, then constant."""
    r_end = float(cfg.get("final_rank", 0.25))
    a = float(cfg.get("full_rank_frac", 0.1))
    b = float(cfg.get("anneal_end_frac", 0.8))

    def rank(step):
        t = jnp.clip((step / total_steps - a) / (b - a), 0.0, 1.0)
        return r_end + (1.0 - r_end) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))

    return rank


def build_optimizer(config: dict, model, schedule, total_steps: int):
    wd = config["weight_decay"] if config["weight_decay"] != 0.0 else None
    clip = optax.clip_by_global_norm(config["grad_clip_norm"])
    name = config["optimizer"]
    spec = None
    if name == "muon":
        inner = optax.contrib.muon(
            learning_rate=schedule, weight_decay=wd, weight_decay_mask=weight_decay_mask(model)
        )
    elif name == "adamw":
        inner = optax.adamw(
            learning_rate=schedule,
            weight_decay=config["weight_decay"],
            mask=weight_decay_mask(model),
        )
    elif name in ("bmuon", "numuon"):
        cfg = config.get("bmuon", {})
        spec = build_channel_spec(model, cost_model=cfg.get("cost_model", "cg"))
        gradual = config.get("gradual")
        if name == "numuon":
            budget, ranks = 1.0, numuon_rank_schedule(config.get("numuon", {}), total_steps)
        elif gradual and gradual.get("anneal_budget", False):
            budget = gradual_budget_schedule(gradual, total_steps // config["n_epochs"])
            ranks = None
        else:
            budget, ranks = budget_schedule(cfg, total_steps), None
        inner = bmuon(
            learning_rate=schedule,
            spec=spec,
            budget_fraction=budget,
            rank_schedule=ranks,
            group_lasso=cfg.get("group_lasso"),
            alpha=cfg.get("alpha", 0.5),
            refresh_every=cfg.get("refresh_every", 5),
            eps=cfg.get("eps", 1e-6),
            weight_decay=wd,
            weight_decay_keys=weight_decay_keys(model),
            selective_weight_decay=cfg.get("selective_weight_decay_prune"),
        )
    else:
        raise ValueError(f"optimizer {name} not supported")
    optim = optax.chain(clip, inner)
    accum = int(config.get("grad_accum_steps", 1))
    if accum > 1:
        optim = optax.MultiSteps(optim, every_k_schedule=accum)
    return optim, spec


# ------------------------------------------------------------------------- data


def load_datasets(config):
    def ase(path):
        return AseDBDataset(
            file_path=path, atomic_numbers=config["atomic_numbers"], cutoff=config["cutoff"],
            backend="jax",
        )

    paths = config["train_path"]
    train_dataset = ConcatDataset([ase(p) for p in paths]) if isinstance(paths, list) else ase(paths)
    atomic_indices = getattr(train_dataset, "atomic_indices", None)
    if "valid_frac" in config:
        train_dataset, val_dataset = train_dataset.split(valid_frac=config["valid_frac"])
    else:
        val_dataset = ase(config["valid_path"])
    frac = config.get("train_subset_frac")
    if frac and frac < 1.0:
        rng = np.random.default_rng(config.get("subset_seed", 0))
        idx = np.sort(rng.choice(len(train_dataset), int(frac * len(train_dataset)), replace=False))
        train_dataset = IndexDataset(train_dataset, idx)
    vfrac = config.get("valid_subset_frac")
    if vfrac and vfrac < 1.0:
        rng = np.random.default_rng(1)
        idx = np.sort(rng.choice(len(val_dataset), int(vfrac * len(val_dataset)), replace=False))
        val_dataset = IndexDataset(val_dataset, idx)
    if atomic_indices is not None:  # needed by nequix.data.average_atom_energies after a split
        train_dataset.atomic_indices = atomic_indices
    return train_dataset, val_dataset


def shard(dataset, rank: int, n: int):
    """Every n-th structure starting at ``rank`` (all shards have n//N or n//N + 1 items)."""
    if n == 1:
        return dataset
    out = IndexDataset(dataset, np.arange(rank, len(dataset), n))
    if hasattr(dataset, "atomic_indices"):
        out.atomic_indices = dataset.atomic_indices
    return out


def prefetch_n(loader, n: int, queue_size: int = 4):
    """Yield exactly ``n`` items of a fresh iteration of ``loader``, built in a thread.

    Unlike nequix.data.prefetch the thread stops after n items, so the loader is never
    advanced concurrently with the next epoch's iteration."""
    import queue
    import threading

    q = queue.Queue(maxsize=queue_size)

    def worker():
        try:
            it = iter(loader)
            for _ in range(n):
                q.put(next(it))
        except StopIteration:
            q.put(RuntimeError(f"loader ran out before {n} batches"))
        except Exception as e:  # noqa: BLE001
            q.put(e)
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def evaluate_sums(model, dataloader, energy_weight, force_weight, stress_weight, loss_type):
    """As nequix.train.evaluate, but returns (sum of metric * n_graphs, n_graphs) so that
    the shards of several processes can be added."""
    totals, count = defaultdict(float), 0.0
    for batch in prefetch(dataloader):
        n_graphs = float(jnp.sum(jraph.get_graph_padding_mask(batch)))
        val_loss, metrics = loss(model, batch, energy_weight, force_weight, stress_weight,
                                 loss_type)
        totals["loss"] += float(val_loss) * n_graphs
        for k, v in metrics.items():
            totals[k] += float(v) * n_graphs
        count += n_graphs
    return dict(totals), count


def _load_state_with_fallback(path: Path):
    bak = path.with_name(path.name + ".bak")
    if path.exists():
        try:
            return load_training_state(path)
        except Exception as e:
            if bak.exists():
                print(f"WARNING: {path.name} corrupted ({type(e).__name__}), loading backup", flush=True)
                return load_training_state(bak)
            raise
    if bak.exists():
        print(f"WARNING: {path.name} missing, loading backup", flush=True)
        return load_training_state(bak)
    return None


def _atomic_dump(path: Path, fn):
    tmp = path.with_name(path.name + ".tmp")
    fn(tmp)
    bak = path.with_name(path.name + ".bak")
    if path.exists():
        os.replace(path, bak)
    os.replace(tmp, path)


# --------------------------------------------------------------------- logging


class Logger:
    def __init__(self, out_dir: Path, config: dict, resume_id=None):
        self.path = out_dir / "log.jsonl"
        self.wandb = None
        if config.get("wandb", False):
            import wandb

            kw = {"project": config.get("wandb_project", "bmuon-nequix"), "config": config,
                  "name": config.get("run_name"), "dir": str(out_dir)}
            if resume_id:
                kw.update(id=resume_id, resume="allow")
            wandb.init(**kw)
            self.wandb = wandb

    @property
    def run_id(self):
        return getattr(getattr(self.wandb, "run", None), "id", None)

    def log(self, logs: dict, step: int):
        with open(self.path, "a") as f:
            f.write(json.dumps({"step": int(step), **logs}) + "\n")
        if self.wandb:
            self.wandb.log(logs, step=int(step))
        print(f"step {int(step)}: {logs}", flush=True)


def allocation_logs(spec, alloc) -> dict:
    if spec is None or alloc is None:
        return {}
    alloc = np.asarray(alloc)
    logs = {
        "alloc/spent": float(spec.weights @ alloc / spec.weights.sum()),
        "alloc/frac_above_half": float((alloc > 0.5).mean()),
    }
    groups = defaultdict(list)
    for i, c in enumerate(spec.channels):
        groups[(c.layer, c.ir)].append(alloc[i])
    for (t, ir), v in groups.items():
        logs[f"alloc/b{t}_{ir}_mean"] = float(np.mean(v))
        logs[f"alloc/b{t}_{ir}_active"] = int(np.sum(np.asarray(v) > 0.5))
    return logs


def save_allocation(path: Path, spec, state):
    if spec is None or state is None:
        return
    np.savez(
        path,
        alloc=np.asarray(state.alloc),
        scores=np.asarray(state.scores),
        keep=np.asarray(state.keep),
        weights=spec.weights,
        layer=np.array([c.layer for c in spec.channels]),
        ir=np.array([c.ir for c in spec.channels]),
        copy=np.array([c.copy for c in spec.channels]),
    )


# ----------------------------------------------------------------------- train


def train(config_path: str, overrides: list | None = None, coordinator: str | None = None,
          num_processes: int = 1, process_id: int = 0):
    if num_processes > 1:
        jax.distributed.initialize(coordinator, num_processes, process_id)
        # jax.distributed installs a SIGTERM handler that the forked data-loader workers
        # inherit, so multiprocessing's exit hook (SIGTERM + join) would wait forever.
        atexit.register(lambda: [p.kill() for p in multiprocessing.active_children()])
    rank, n_proc = jax.process_index(), jax.process_count()
    main_process = rank == 0

    config = apply_overrides(load_config(config_path), overrides)
    out_dir = Path(config.get("out_dir", f"bmuon/runs/{Path(config_path).stem}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if main_process:
        with open(out_dir / "config.yml", "w") as f:
            yaml.safe_dump({**config, "num_processes": n_proc}, f)
    config["cache_dir"] = config.get("cache_dir") or os.environ.get("TMPDIR")

    train_dataset, val_dataset = load_datasets(config)
    n_train_total = len(train_dataset)
    if "atom_energies" in config:
        atom_energies = [config["atom_energies"][n] for n in config["atomic_numbers"]]
    else:
        atom_energies = average_atom_energies(train_dataset)
    keys = ["shift", "scale", "avg_n_neighbors", "max_n_edges", "max_n_nodes", "avg_n_nodes",
            "avg_n_edges"]
    stats = ({k: config[k] for k in keys} if all(k in config for k in keys)
             else dataset_stats(train_dataset, atom_energies))
    # statistics above are computed on the whole training set, identical in every process
    train_dataset = shard(train_dataset, rank, n_proc)
    val_dataset = shard(val_dataset, rank, n_proc)

    num_devices = jax.local_device_count()
    workers = config.get("num_workers", 16)
    loader_kw = dict(
        batch_size=config["batch_size"], max_n_nodes=stats["max_n_nodes"],
        max_n_edges=stats["max_n_edges"], avg_n_nodes=stats["avg_n_nodes"],
        avg_n_edges=stats["avg_n_edges"], num_workers=workers,
    )
    train_loader = ParallelLoader(DataLoader(train_dataset, shuffle=True, **loader_kw), num_devices)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kw)

    model = Nequix(
        jax.random.key(config.get("seed", 0)),
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
        shift=stats["shift"],
        scale=stats["scale"],
        avg_n_neighbors=stats["avg_n_neighbors"],
        atom_energies=atom_energies,
        kernel=config["kernel"],
    )
    config_to_save = {**config, **stats}

    accum = int(config.get("grad_accum_steps", 1))
    if n_proc > 1:
        # same number of micro-steps in every process (dynamic batching gives each shard a
        # different number of batches, and a missing step would hang the collectives); each
        # batch holds at most batch_size structures, so every shard has at least this many.
        micro_per_epoch = (n_train_total // n_proc) // (config["batch_size"] * num_devices)
    else:
        micro_per_epoch = len(train_dataset) // (config["batch_size"] * num_devices)
    steps_per_epoch = max(1, micro_per_epoch // accum)  # optimizer updates per epoch
    total_steps = config["n_epochs"] * steps_per_epoch
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config["learning_rate"] * config["warmup_factor"],
        peak_value=config["learning_rate"],
        end_value=1e-6,
        warmup_steps=int(config["warmup_epochs"] * steps_per_epoch),
        decay_steps=total_steps,
    )
    optim, spec = build_optimizer(config, model, schedule, total_steps)
    if spec is not None and main_process:
        print(f"BMuon channels ({spec.num_channels}):\n{spec.summary()}", flush=True)
    if n_proc > 1:
        print(f"process {rank}/{n_proc}: {len(train_dataset)} train structures, "
              f"{micro_per_epoch} micro-steps per epoch", flush=True)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    model = jax.device_put_replicated(model, jax.local_devices())
    opt_state = jax.device_put_replicated(opt_state, jax.local_devices())
    ema_model = jax.tree.map(lambda x: x.copy(), model)
    step, start_epoch, best_val_loss, run_id = jnp.array(0), 0, float("inf"), None

    save_every = int(config.get("save_every_steps", 0))  # 0 = checkpoint only at epoch end
    state_path = Path(config.get("state_path") or out_dir / "state.pkl")
    skip_micro = 0
    loaded = _load_state_with_fallback(state_path)
    if loaded is not None:
        (model, ema_model, _, opt_state, step, start_epoch, best_val_loss,
         run_id) = loaded
        # Rebuild optim from loaded model: weight_decay_mask must use the same
        # FunctionalLinear objects as the model, or jax.tree.map crashes on resume.
        optim, _ = build_optimizer(config, jax.tree.map(lambda x: x[0], model), schedule, total_steps)
        if save_every and n_proc > 1:
            # state.pkl may have been written mid-epoch: recover the epoch and how many of its
            # micro-steps were already done from the (fixed) global micro-step counter.
            start_epoch, skip_micro = divmod(int(step), micro_per_epoch)
        print(f"resumed from {state_path} at epoch {start_epoch}"
              + (f" (skip {skip_micro} micro-steps)" if skip_micro else ""), flush=True)
    logger = Logger(out_dir, config, run_id) if main_process else None

    psum = jax.pmap(lambda x: jax.lax.psum(x, "device"), axis_name="device")

    def all_sum(values: np.ndarray) -> np.ndarray:
        """Sum a host vector over all processes (zeros on the extra local devices)."""
        x = np.zeros((num_devices, len(values)), np.float32)
        x[0] = values
        return np.asarray(psum(x))[0]

    @functools.partial(eqx.filter_pmap, in_axes=(0, 0, None, 0, 0), axis_name="device")
    def train_step(model, ema_model, step, opt_state, batch):
        (total_loss, metrics), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
            model, batch, config["energy_weight"], config["force_weight"],
            config["stress_weight"], config["loss_type"],
        )
        grads = jax.lax.pmean(grads, axis_name="device")
        total_loss, metrics = jax.lax.pmean((total_loss, metrics), axis_name="device")
        metrics["grad_norm"] = optax.global_norm(grads)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        # EMA once per optimizer update (with accumulation, updates are zero in between)
        ema_step = step // accum
        decay = jnp.minimum(config["ema_decay"], (1 + ema_step) / (10 + ema_step))
        decay = jnp.where((step + 1) % accum == 0, decay, 1.0)
        ema_params, ema_static = eqx.partition(ema_model, eqx.is_array)
        new_ema = jax.tree.map(
            lambda ep, mp: ep * decay + mp * (1 - decay), ema_params, eqx.filter(model, eqx.is_array)
        )
        return model, eqx.combine(ema_static, new_ema), opt_state, total_loss, metrics

    def current_bmuon_state():
        s = find_bmuon_state(opt_state)
        return None if s is None else jax.tree.map(lambda x: x[0], s)

    gradual = config.get("gradual")
    by_key = {s.keypath: s for s in spec.slices} if spec is not None else {}

    def mask_slices(tree, keep):
        """Zero the slices of the dropped channels (leaves may carry the pmap device axis)."""
        def f(path, x):
            sl = by_key.get(jax.tree_util.keystr(path))
            if sl is None or not eqx.is_array(x):
                return x
            shape = [1] * x.ndim
            shape[x.ndim - 2 + sl.axis] = -1
            return x * jnp.asarray(keep[sl.channel_ids], x.dtype).reshape(shape)
        return jax.tree_util.tree_map_with_path(f, tree)

    def maybe_prune(epoch, model, ema_model, opt_state):
        """Gradual pruning: at the start of the epoch following each pruning event, remove
        (zero for good) the lowest-scored surviving channels down to the event's level.
        Idempotent, so it is a no-op after a resume (the mask lives in the optimizer state)."""
        if not gradual or spec is None:
            return model, ema_model, opt_state
        done = [lv for e, lv in zip(gradual["prune_after_epochs"], gradual["levels"]) if e < epoch]
        if not done:
            return model, ema_model, opt_state
        level = float(done[-1])
        bs = jax.tree.map(lambda x: x[0], find_bmuon_state(opt_state))
        keep_old = np.asarray(bs.keep) > 0
        if keep_old.sum() <= spec.num_channels - round(level * spec.num_channels):
            return model, ema_model, opt_state
        if gradual.get("score", "magnitude") == "alloc":
            scores = np.asarray(bs.alloc, np.float64).copy()
        else:
            scores = magnitude_scores(jax.tree.map(lambda x: x[0], model), spec)
        scores[~keep_old] = -np.inf
        keep = select_channels(scores, spec, level, by="count",
                               min_keep=int(gradual.get("min_keep", 1))) & keep_old
        keep_f = keep.astype(np.float32)
        model, ema_model = mask_slices(model, keep_f), mask_slices(ema_model, keep_f)

        def reset(st):
            return st._replace(
                mu=mask_slices(st.mu, keep_f),
                # paper: the allocation of every surviving channel is reset to one
                alloc=jnp.broadcast_to(jnp.where(keep, 1.0, 1e-6).astype(st.alloc.dtype),
                                       st.alloc.shape),
                keep=jnp.broadcast_to(jnp.asarray(keep_f), st.keep.shape),
            )

        opt_state = map_bmuon_state(opt_state, reset)
        if main_process:
            print(f"gradual pruning before epoch {epoch}: level {level:.2f}, "
                  f"{int(keep.sum())}/{spec.num_channels} channels kept", flush=True)
            np.savez(out_dir / f"prune_epoch{epoch:03d}.npz", keep=keep, scores=scores, level=level)
            logger.log({"prune/level": level, "prune/kept": int(keep.sum()), "epoch_start": epoch}, step)
        return model, ema_model, opt_state

    for epoch in range(start_epoch, config["n_epochs"]):
        model, ema_model, opt_state = maybe_prune(epoch, model, ema_model, opt_state)
        t0 = time.time()
        train_loader.loader.set_epoch(epoch)
        batches = prefetch_n(train_loader, micro_per_epoch) if n_proc > 1 else prefetch(train_loader)
        to_skip = skip_micro if epoch == start_epoch else 0
        for bi, batch in enumerate(batches):
            if bi < to_skip:  # micro-steps already done before a crash: advance the loader only
                continue
            model, ema_model, opt_state, total_loss, metrics = train_step(
                model, ema_model, step, opt_state, batch
            )
            step = step + 1
            if main_process and save_every and n_proc > 1 and int(step) % save_every == 0:
                _atomic_dump(state_path, lambda p: save_training_state(
                    p, model, ema_model, optim, opt_state, step, epoch, best_val_loss,
                    wandb_run_id=(logger.run_id if logger else None)))
            if step % config["log_every"] == 0 and main_process:
                logs = {"train/loss": total_loss.mean().item(),
                        "learning_rate": float(schedule(step // accum)),
                        "time/elapsed": time.time() - t0}
                logs.update({f"train/{k}": v.mean().item() for k, v in metrics.items()})
                bs = current_bmuon_state()
                if bs is not None:
                    logs.update(allocation_logs(spec, bs.alloc))
                logger.log(logs, step)

        ema_single = jax.tree.map(lambda x: x[0], ema_model)
        sums, count = evaluate_sums(ema_single, val_loader, config["energy_weight"],
                                    config["force_weight"], config["stress_weight"],
                                    config["loss_type"])
        names = sorted(sums)
        tot = all_sum(np.array([sums[k] for k in names] + [count], np.float64))
        val = {k: tot[i] / tot[-1] for i, k in enumerate(names)}
        if not main_process:
            print(f"process {rank}: epoch {epoch} done in {time.time() - t0:.0f} s", flush=True)
            continue
        if val["loss"] < best_val_loss:
            best_val_loss = float(val["loss"])
            save_model(out_dir / "checkpoint.nqx", ema_single, config_to_save)
        save_model(out_dir / "last.nqx", ema_single, config_to_save)
        snap_every = int(config.get("snapshot_every_epochs", 0))  # 0 = no periodic snapshots
        if snap_every and (epoch % snap_every == 0 or epoch == config["n_epochs"] - 1):
            # permanent (never overwritten) snapshot to prune/compare along the budget anneal
            save_model(out_dir / f"snap_epoch{epoch:03d}.nqx", ema_single, config_to_save)
        bs = current_bmuon_state()
        save_allocation(out_dir / f"alloc_epoch{epoch:03d}.npz", spec, bs)
        _atomic_dump(state_path, lambda p: save_training_state(
            p, model, ema_model, optim, opt_state, step, epoch + 1, best_val_loss,
            wandb_run_id=logger.run_id))
        logs = {f"val/{k}": float(v) for k, v in val.items()}
        logs.update({"epoch": epoch, "time/epoch": time.time() - t0})
        if bs is not None:
            logs.update(allocation_logs(spec, bs.alloc))
        logger.log(logs, step)

    if main_process:
        (out_dir / "DONE").touch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("--set", nargs="*", default=[], help="overrides, e.g. bmuon.budget=0.3")
    parser.add_argument("--coordinator", default=None, help="host:port of process 0")
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--process-id", type=int, default=0)
    args = parser.parse_args()
    train(args.config_path, args.set, args.coordinator, args.num_processes, args.process_id)


if __name__ == "__main__":
    main()
