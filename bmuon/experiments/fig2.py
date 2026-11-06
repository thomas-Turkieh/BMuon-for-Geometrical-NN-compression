"""Figure 2 of the BMuon paper, for Nequix: accuracy after post-training channel compression.

    python -m bmuon.experiments.fig2 --plan bmuon/experiments/fig2_runs.yml \
        --data $BMUON_DATA/mptrj-aselmdb --out bmuon/results/fig2 [--n-structures 1000]

For every entry of the plan, the latest snapshot of the run (snap_epochNNN.nqx) is
compressed at levels 0, 10, ..., 90, 95 % of the type-I channels, without any fine-tuning,
and evaluated on held-out structures (the validation split of training). Methods:
  mag   global pruning ranked by the coupled magnitude ||all slices of the channel||_2
  alloc global pruning ranked by the BMuon allocation mu of the same epoch
  svd   NuMuon: truncated SVD of every budgeted matrix, rank chosen so that the parameter
        count of these matrices matches the magnitude-pruned model at the same level
Channels already pruned during training (gradual runs) rank last; levels below the run's
own pruning level are skipped. Results are cached per (run, epoch, method), so the script
can be re-run every few minutes during the night (bmuon/experiments/fig2_watch.sh): it only
evaluates new snapshots and redraws fig2.png, fig2_points.csv and STATUS.md.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from bmuon.channels import build_channel_spec
from bmuon.experiments.common import bucket, pad, platform
from bmuon.prune import magnitude_scores, select_channels, zero_channels
from nequix.model import load_model

LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


# ------------------------------------------------------------------ inputs


def load_eval_set(data: str, config: dict, n: int, valid_frac: float = 0.05, seed: int = 0):
    """n held-out structures: the validation split used by every training run."""
    from nequix.data import AseDBDataset

    ds = AseDBDataset(data, config["atomic_numbers"], config["cutoff"], backend="jax")
    _, ds = ds.split(valid_frac=valid_frac)
    idx = np.random.default_rng(seed).choice(len(ds), min(n, len(ds)), replace=False)
    graphs = [ds[int(i)] for i in idx]
    e_true = np.array([float(g.globals["energy"][0]) for g in graphs])
    f_true = np.concatenate([np.asarray(g.nodes["forces"]) for g in graphs])
    n_atoms = np.array([int(g.n_node[0]) for g in graphs])
    return graphs, e_true, f_true, n_atoms


def latest_snapshot(run_dir: Path, epoch=None):
    snaps = sorted(glob.glob(str(run_dir / "snap_epoch*.nqx")))
    if epoch is not None:
        snaps = [s for s in snaps if int(re.search(r"(\d+)\.nqx$", s).group(1)) <= epoch]
    if not snaps:
        return None, None
    path = snaps[-1]
    return Path(path), int(re.search(r"(\d+)\.nqx$", path).group(1))


# ------------------------------------------------------------------ compression


def budgeted_params(spec, keep=None) -> int:
    """Parameters of the budgeted matrices left when only the channels in ``keep`` remain."""
    total = 0
    for s in spec.slices:
        ids = np.asarray(s.channel_ids)
        kept = len(ids) if keep is None else int(keep[ids].sum())
        total += kept * s.shape[1 - s.axis]  # set by slice_shapes
    return total


def slice_shapes(model, spec):
    leaves = {jax.tree_util.keystr(p): x for p, x in
              jax.tree_util.tree_flatten_with_path(eqx.filter(model, eqx.is_array))[0]}
    for s in spec.slices:
        s.shape = tuple(leaves[s.keypath].shape)


def svd_compress(model, spec, target_params: int, min_rank: int = 4):
    """Truncated SVD of every budgeted matrix with one common rank r (capped by the matrix
    size, at least min_rank), r the largest whose parameter count r*(m+n) (or m*n when that
    is smaller) does not exceed ``target_params``. Returns (model, params, r)."""
    shapes = [s.shape for s in spec.slices]

    def rank_of(r, m, n):
        return min(max(r, min_rank), m, n)

    def params(r):  # a matrix stays dense when its factorization would not be smaller
        return sum(min(m * n, rank_of(r, m, n) * (m + n)) for m, n in shapes)

    r = 1
    while r < max(max(sh) for sh in shapes) and params(r + 1) <= target_params:
        r += 1
    keys = {s.keypath for s in spec.slices}
    p, static = eqx.partition(model, eqx.is_array)
    flat, treedef = jax.tree_util.tree_flatten_with_path(p)
    out = []
    for path, x in flat:
        if jax.tree_util.keystr(path) in keys:
            m, n = x.shape
            k = rank_of(r, m, n)
            if k * (m + n) < m * n:
                u, sv, vt = jnp.linalg.svd(x, full_matrices=False)
                x = (u[:, :k] * sv[:k]) @ vt[:k]
        out.append(x)
    return eqx.combine(jax.tree_util.tree_unflatten(treedef, out), static), params(r), r


@eqx.filter_jit
def _energy_forces(model, g):
    e, forces, _ = model(g)
    return e, forces


def predict(model, graphs, batch: int = int(os.environ.get("FIG2_BATCH", 32))):
    """Energies and forces of a list of graphs, ``batch`` graphs per call (sorted by size,
    padded to power-of-two buckets). One module-level jit: every compressed model has the
    same pytree structure, so each bucket compiles once for the whole script."""
    import jraph

    edges = [int(g.n_edge[0]) for g in graphs]
    order = list(np.argsort(edges))
    cap = int(os.environ.get("FIG2_MAX_EDGES", 60000))  # bounds GPU memory per call
    batches, cur, cur_e = [], [], 0
    for i in order:
        if cur and (len(cur) == batch or cur_e + edges[i] > cap):
            batches.append(cur)
            cur, cur_e = [], 0
        cur.append(i)
        cur_e += edges[i]
    if cur:
        batches.append(cur)
    es, fs = [None] * len(graphs), [None] * len(graphs)
    for idx in batches:
        gb = jraph.batch([graphs[i] for i in idx])
        n, m = int(np.sum(gb.n_node)), int(np.sum(gb.n_edge))
        gp = jraph.pad_with_graphs(gb, n_node=bucket(n + 1, 256), n_edge=bucket(m + 1, 4096),
                                   n_graph=batch + 1)
        e, forces = _energy_forces(model, gp)
        e, forces = np.asarray(e), np.asarray(forces)
        off = 0
        for j, i in enumerate(idx):
            k = int(graphs[i].n_node[0])
            es[i], fs[i] = float(e[j]), forces[off:off + k]
            off += k
    return np.array(es), fs


def evaluate(model, ev):
    graphs, e_true, f_true, n_atoms = ev
    e, f = predict(model, graphs)
    return (float(np.mean(np.abs(e - e_true) / n_atoms)),
            float(np.mean(np.abs(np.concatenate(f) - f_true))))


def curve(entry, method, model_path, alloc_path, ev, cost_model="channel"):
    model, _ = load_model(str(model_path), kernel=False)
    spec = build_channel_spec(model, cost_model=cost_model)
    slice_shapes(model, spec)
    n = spec.num_channels
    mags = magnitude_scores(model, spec)
    dead = mags <= 1e-12  # channels already removed during training (gradual pruning)
    if method == "alloc":
        z = np.load(alloc_path)
        scores = np.asarray(z["alloc"], np.float64).copy()
        if "keep" in z:
            dead |= np.asarray(z["keep"]) <= 0
    else:
        scores = mags.copy()
    scores[dead] = -np.inf
    total_params = budgeted_params(spec)
    rows = []
    levels = LEVELS
    if entry["kind"] == "gradual":  # already pruned during training: start at its own level
        base_level = dead.sum() / n
        levels = sorted({round(base_level, 4)} | {lv for lv in LEVELS if lv > base_level + 1e-9})
    for lv in levels:
        keep = select_channels(scores, spec, lv, by="count", min_keep=1) & ~dead
        kept_params = budgeted_params(spec, keep)
        if method == "svd":
            m, used, rank = svd_compress(model, spec, kept_params)
            info = {"svd_rank": rank, "kept_params_frac": used / total_params}
        else:
            m = zero_channels(model, spec, keep)
            info = {"kept_params_frac": kept_params / total_params}
        e_mae, f_mae = evaluate(m, ev)
        rows.append({"run": entry["name"], "label": entry["label"], "kind": entry["kind"],
                     "method": method, "level": float(lv), "kept_channels": int(keep.sum()),
                     **info, "energy_mae_mev_atom": 1000 * e_mae, "force_mae_mev_A": 1000 * f_mae})
        print(f"  {entry['name']:28s} {method:5s} level {lv:.2f}: "
              f"F {1000 * f_mae:8.1f} meV/A  E {1000 * e_mae:8.2f} meV/at", flush=True)
    return rows


def phase_snapshots(run_dir: Path):
    """Gradual runs: (epoch, snapshot) at the end of every pruning phase, i.e. the last epoch
    before each new pruning event and the last epoch of training (error after recovery)."""
    try:
        cfg = yaml.safe_load(open(run_dir / "config.yml"))
    except Exception:  # noqa: BLE001
        return []
    g = cfg.get("gradual") or {}
    ends = list(g.get("prune_after_epochs", []))[1:] + [cfg["n_epochs"] - 1]
    return [(e, run_dir / f"snap_epoch{e:03d}.nqx") for e in ends if (run_dir / f"snap_epoch{e:03d}.nqx").exists()]


def phase_point(entry, epoch, model_path, ev):
    """One point of a gradual curve: the snapshot as trained (its own pruned channels only)."""
    model, _ = load_model(str(model_path), kernel=False)
    spec = build_channel_spec(model, cost_model="channel")
    slice_shapes(model, spec)
    keep = magnitude_scores(model, spec) > 1e-12
    e_mae, f_mae = evaluate(model, ev)
    row = {"run": entry["name"], "label": entry["label"], "kind": "phase", "method": "phase",
           "level": float(1 - keep.mean()), "kept_channels": int(keep.sum()), "epoch": epoch,
           "kept_params_frac": budgeted_params(spec, keep) / budgeted_params(spec),
           "energy_mae_mev_atom": 1000 * e_mae, "force_mae_mev_A": 1000 * f_mae}
    print(f"  {entry['name']:28s} phase end epoch {epoch}: level {row['level']:.2f} "
          f"F {row['force_mae_mev_A']:8.1f} meV/A", flush=True)
    return [row]


# ------------------------------------------------------------------ status


def run_status(run_dir: Path) -> dict:
    out = {"epochs": 0, "n_epochs": None, "last_epoch_min": None, "eta_h": None, "val_F": None,
           "val_E": None, "log_age_min": None, "done": (run_dir / "DONE").exists(), "kept": None}
    try:
        cfg = yaml.safe_load(open(run_dir / "config.yml"))
        out["n_epochs"] = cfg.get("n_epochs")
    except Exception:  # noqa: BLE001
        pass
    log = run_dir / "log.jsonl"
    if not log.exists():
        return out
    times = []
    for line in open(log):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "prune/kept" in d:
            out["kept"] = d["prune/kept"]
        if "epoch" in d and "time/epoch" in d:
            out["epochs"] = d["epoch"] + 1
            times.append(d["time/epoch"])
            out["val_F"] = 1000 * d.get("val/force_mae", float("nan"))
            out["val_E"] = 1000 * d.get("val/energy_mae_per_atom", float("nan"))
    out["log_age_min"] = (time.time() - os.path.getmtime(log)) / 60
    if times:
        recent = float(np.median(times[-3:])) / 60
        out["last_epoch_min"] = recent
        if out["n_epochs"]:
            out["eta_h"] = recent * (out["n_epochs"] - out["epochs"]) / 60
    return out


def write_status(plan, runs_dir: Path, out: Path, used: dict):
    lines = [f"# Figure 2 — run status ({time.strftime('%a %d %b %H:%M')})", "",
             "| run | curve(s) | epochs | min/epoch | ETA | val F (meV/Å) | val E (meV/at) | channels kept | last log | snapshot used |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for e in plan["runs"]:
        if e["kind"] == "reference":
            continue
        st = run_status(runs_dir / e["name"])
        fmt = lambda x, f="{:.1f}": "—" if x is None else f.format(x)  # noqa: E731
        label = e.get("label") or " / ".join(e.get("labels", {}).values())
        lines.append(
            f"| {e['name']} | {label} | {st['epochs']}/{st['n_epochs'] or '?'}{' DONE' if st['done'] else ''} | "
            f"{fmt(st['last_epoch_min'])} | {fmt(st['eta_h'], '{:.1f} h')} | {fmt(st['val_F'])} | {fmt(st['val_E'], '{:.2f}')} | "
            f"{st['kept'] or '—'} | {fmt(st['log_age_min'], '{:.0f} min')} | {used.get(e['name'], '—')} |")
    lines += ["", "`fig2.png` is redrawn from the latest snapshot of every run: until a run is finished,",
              "its curve is that of a partly trained model (epoch in the legend)."]
    (out / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ plot


STYLE = {
    "Muon": ("#6b7280", "o", "--"), "Muon+GL": ("#d97706", "s", "--"), "NuMuon": ("#2563eb", "^", "--"),
    "BM-mag": ("#059669", "D", "-"), "BM-μ": ("#be123c", "s", "-"),
}


def _style(label):
    if label.startswith("Muon+GL") and "0.5" in label:
        return ("#f5b041", "s", "--")
    base = next((k for k in ("Muon+GL", "NuMuon", "BM-mag", "BM-μ", "Muon") if label.startswith(k)), None)
    return STYLE.get(base, ("#7c3aed", "o", "-"))


def plot(rows, out: Path, title: str):
    """fig2.png (forces) and fig2_energy.png: left = post-training compression (the paper's
    Figure 2), right = gradual pruning (stars at the level reached in training, then extra
    post-training pruning), with the post-training Muon and BM-mu curves in light for context."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = []
    for r in rows:
        if (r["label"], r["kind"]) not in labels:
            labels.append((r["label"], r["kind"]))

    def pts(label, kind):
        p = sorted([r for r in rows if r["label"] == label and r["kind"] == kind], key=lambda r: r["level"])
        return [100 * r["level"] for r in p], p

    for key, name, fname in [("force_mae_mev_A", "Force MAE (meV/Å)", "fig2.png"),
                             ("energy_mae_mev_atom", "Energy MAE (meV/atom)", "fig2_energy.png")]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
        for label, kind in labels:
            x, p = pts(label, kind)
            y = [r[key] for r in p]
            color, marker, ls = _style(label)
            if kind == "post":
                ax1.plot(x, y, color=color, marker=marker, ls=ls, lw=2, ms=6, label=label,
                         markerfacecolor="white" if ls == "--" else color)
                if label.startswith(("Muon [", "BM-μ [")):
                    ax2.plot(x, y, color=color, ls=ls, lw=1.2, alpha=0.35, label=label + " (post)")
            elif kind == "reference":
                ax1.plot(x, y, color=color, ls=":", lw=1.2, alpha=0.6, label=label)
            elif kind == "phase":  # gradual: error at the end of each pruning phase (after recovery)
                ax2.plot(x, y, color=color, ls="-", lw=2, marker=marker, ms=7, label=label)
        ax1.set_title("Post-training compression, no fine-tuning (paper Fig. 2)", fontsize=10)
        ax2.set_title("Gradual pruning during training: error at the end of each phase", fontsize=10)
        for ax in (ax1, ax2):
            ax.set_yscale("log")
            ax.set_xlabel("Type-I channel compression (%)")
            ax.set_xticks([0, 20, 40, 60, 70, 80, 90, 95])
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=7.5, loc="upper left")
        ax1.set_ylabel(name + "  (lower is better)")
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="bmuon/experiments/fig2_runs.yml")
    ap.add_argument("--runs-dir", default="bmuon/runs")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-structures", type=int, default=1000)
    ap.add_argument("--epoch", type=int, default=None, help="use snapshots up to this epoch")
    ap.add_argument("--out", default="bmuon/results/fig2")
    args = ap.parse_args()

    plan = yaml.safe_load(open(args.plan))
    runs_dir, out = Path(args.runs_dir), Path(args.out)
    cache = out / f"cache_n{args.n_structures}"
    cache.mkdir(parents=True, exist_ok=True)
    ev = None
    rows, used = [], {}
    for entry in plan["runs"]:
        if entry["kind"] == "reference":
            model_path, epoch = Path(entry["model"]), entry.get("epoch", "ref")
        else:
            # max_epoch: last usable snapshot of a run that diverged later
            model_path, epoch = latest_snapshot(runs_dir / entry["name"], entry.get("max_epoch", args.epoch))
            if model_path is None:
                continue
        used[entry["name"]] = f"epoch {epoch}"
        for method in entry["methods"]:
            label = entry["label"] if len(entry["methods"]) == 1 else entry["labels"][method]
            e = {**entry, "label": label + ("" if entry["kind"] == "reference" else f" [ep. {epoch}]")}
            f = cache / f"{entry['name']}_e{epoch}_{method}.csv"
            if f.exists():
                with open(f) as fh:
                    part = list(csv.DictReader(fh))
                for r in part:
                    for k in r:
                        if k not in ("run", "label", "kind", "method"):
                            r[k] = float(r[k])
                    r["label"] = e["label"]
            else:
                if ev is None:
                    cfg_model, cfg = load_model(str(model_path), kernel=False)
                    print(f"loading {args.n_structures} held-out structures...", flush=True)
                    ev = load_eval_set(args.data, cfg, args.n_structures)
                alloc = None
                if method == "alloc":
                    alloc = (Path(entry["alloc"]) if entry["kind"] == "reference"
                             else runs_dir / entry["name"] / f"alloc_epoch{epoch:03d}.npz")
                    if not alloc.exists():
                        continue
                part = curve(e, method, model_path, alloc, ev, entry.get("cost_model", "channel"))
                tmp = f.with_suffix(".tmp")
                with open(tmp, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=sorted({k for r in part for k in r}))
                    w.writeheader()
                    w.writerows(part)
                os.replace(tmp, f)
            rows += part
        if entry["kind"] == "gradual":
            for ep, snap in phase_snapshots(runs_dir / entry["name"]):
                f = cache / f"{entry['name']}_e{ep}_phase.csv"
                if f.exists():
                    with open(f) as fh:
                        part = list(csv.DictReader(fh))
                    for r in part:
                        for k in r:
                            if k not in ("run", "label", "kind", "method"):
                                r[k] = float(r[k])
                else:
                    if ev is None:
                        _, cfg = load_model(str(snap), kernel=False)
                        print(f"loading {args.n_structures} held-out structures...", flush=True)
                        ev = load_eval_set(args.data, cfg, args.n_structures)
                    part = phase_point(entry, ep, snap, ev)
                    tmp = f.with_suffix(".tmp")
                    with open(tmp, "w", newline="") as fh:
                        w = csv.DictWriter(fh, fieldnames=sorted({k for r in part for k in r}))
                        w.writeheader()
                        w.writerows(part)
                    os.replace(tmp, f)
                for r in part:
                    r["label"] = entry["label"]
                rows += part
    if rows:
        with open(out / "fig2_points.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
        plot(rows, out, f"Nequix-MP-1 on MPtrj ({plan.get('description', '')}) — "
                        f"post-training (left) vs gradual pruning (right), {args.n_structures} held-out structures")
    write_status(plan, runs_dir, out, used)
    print(f"wrote {out}/fig2.png, fig2_points.csv, STATUS.md ({platform()})", flush=True)


if __name__ == "__main__":
    main()
