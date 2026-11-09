"""Gradual pruning (bmuon/experiments/gradual_runs.yml) vs classic post-training pruning of the
group's fully trained models, on the same held-out structures, with the same cost axis.

    python -m bmuon.experiments.gradual_fig --data $BMUON_DATA/mptrj-aselmdb \
        [--out bmuon/results/gradual_fig] [--n-structures 1000]

x axis: fraction of the channel cost removed, cost model "paths" (the one the gradual runs prune
with: levels 20/40/60/70/80/90 %).
  classic  a fully trained model (no pruning during training) is pruned once at every level,
           channels ranked by BMuon allocation mu ("alloc") or by coupled magnitude ("mag"),
           no fine-tuning;
  gradual  the snapshot at the end of every pruning phase (just before the next pruning event,
           i.e. after the recovery epochs), evaluated as trained;
  scratch  small uniform models trained from scratch (one point each).
Every evaluation is cached (one csv per model / epoch / method): re-running only adds the new
phase ends. Writes gradual_vs_classic.png (forces), gradual_vs_classic_energy.png, points.csv
and table.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import yaml


def _load_model_code():
    """Model and evaluation imports (JAX), loaded only when evaluating, so --plot-only runs anywhere."""
    global build_channel_spec, evaluate, load_eval_set, slice_shapes
    global magnitude_scores, select_channels, zero_channels, load_model
    from bmuon.channels import build_channel_spec
    from bmuon.experiments.fig2 import evaluate, load_eval_set, slice_shapes
    from bmuon.prune import magnitude_scores, select_channels, zero_channels
    from nequix.model import load_model

LEVELS = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9]
RUNS = Path("bmuon/runs")

# classic post-training pruning: fully trained models without pruning during training
CLASSIC = [
    {"name": "classic-bmuon-wd1e3", "wd": "0.001", "model": RUNS / "mp1-bmuon-wd1e3/snap_epoch060.nqx",
     "alloc": RUNS / "mp1-bmuon-wd1e3/alloc_epoch060.npz", "epoch": 60, "methods": ["alloc", "mag"],
     "labels": {"alloc": "BMuon, pruned by μ", "mag": "BMuon, pruned by magnitude"}},
    {"name": "classic-nequix-mp-1", "wd": "0.001", "model": Path("models/nequix-mp-1.nqx"), "epoch": None,
     "methods": ["mag"], "labels": {"mag": "Muon (published Nequix-MP-1), pruned by magnitude"}},
    {"name": "classic-bmuon-wd1e2", "wd": "0.01", "model": RUNS / "mp1-bmuon-wd1e2/snap_epoch080.nqx",
     "alloc": RUNS / "mp1-bmuon-wd1e2/alloc_epoch080.npz", "epoch": 80, "methods": ["alloc", "mag"],
     "labels": {"alloc": "BMuon, pruned by μ", "mag": "BMuon, pruned by magnitude"}},
]
# small models trained from scratch (uniform channel shrink)
SCRATCH = [
    {"name": "scratch-uniform-wd1e3", "wd": "0.001", "run": "mp1-uniform-shrink-wd1e3",
     "label": "small uniform model from scratch (Muon)"},
    {"name": "scratch-uniform-c20-wd1e3", "wd": "0.001", "run": "mp1-uniform-c20-wd1e3",
     "label": "small uniform model from scratch (Muon)"},
]
FULL_COST = None  # paths cost of the full Nequix-MP-1 architecture, set in main()


def cost_frac(model, keep=None):
    """(kept cost fraction w.r.t. the full architecture, spec) under the paths cost model."""
    spec = build_channel_spec(model, cost_model="paths")
    n_paths = np.array([c.n_paths for c in spec.channels], np.float64)  # unnormalized cost
    kept = n_paths.sum() if keep is None else n_paths[keep].sum()
    return kept / FULL_COST, spec


def cached(path: Path, fn):
    if path.exists():
        return list(csv.DictReader(open(path, encoding="utf-8")))
    rows = fn()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    return rows


def classic_curve(entry, method, ev):
    model, _ = load_model(str(entry["model"]), kernel=False)
    spec = build_channel_spec(model, cost_model="paths")
    slice_shapes(model, spec)
    scores = (np.asarray(np.load(entry["alloc"])["alloc"], np.float64) if method == "alloc"
              else magnitude_scores(model, spec))
    rows = []
    for lv in LEVELS:
        keep = select_channels(scores, spec, lv, by="cost", min_keep=1)
        kept, _ = cost_frac(model, keep)
        e, f = evaluate(zero_channels(model, spec, keep), ev)
        rows.append({"name": entry["name"], "kind": "classic", "method": method, "wd": entry["wd"],
                     "label": entry["labels"][method], "epoch": entry["epoch"] or "",
                     "level": lv, "cost_removed": 1 - kept, "kept_channels": int(keep.sum()),
                     "force_mae": 1000 * f, "energy_mae": 1000 * e})
        print(f"  {entry['name']} {method} level {lv:.2f}: cost removed {1 - kept:.3f}, F {1000 * f:.1f}", flush=True)
    return rows


def gradual_point(r, epoch, snap, ev):
    model, _ = load_model(str(snap), kernel=False)
    spec = build_channel_spec(model, cost_model="paths")
    slice_shapes(model, spec)
    keep = magnitude_scores(model, spec) > 1e-12  # channels removed in training are exactly zero
    kept, _ = cost_frac(model, keep)
    e, f = evaluate(model, ev)
    print(f"  {r['name']} epoch {epoch}: cost removed {1 - kept:.3f} ({int(keep.sum())} ch.), F {1000 * f:.1f}", flush=True)
    return [{"name": r["name"], "kind": "gradual", "method": r["score"], "wd": r["wd"], "label": r["label"],
             "epoch": epoch, "level": "", "cost_removed": 1 - kept, "kept_channels": int(keep.sum()),
             "force_mae": 1000 * f, "energy_mae": 1000 * e}]


def scratch_point(s, snap, epoch, ev):
    model, _ = load_model(str(snap), kernel=False)
    kept, _ = cost_frac(model)
    e, f = evaluate(model, ev)
    print(f"  {s['name']} epoch {epoch}: cost {kept:.3f}, F {1000 * f:.1f}", flush=True)
    return [{"name": s["name"], "kind": "scratch", "method": "", "wd": s["wd"], "label": s["label"],
             "epoch": epoch, "level": "", "cost_removed": 1 - kept, "kept_channels": "",
             "force_mae": 1000 * f, "energy_mae": 1000 * e}]


def dense_val(run: str) -> dict:
    """Validation force MAE (meV/Å) per epoch of a run, from its log (whole validation split)."""
    out = {}
    f = RUNS / run / "log.jsonl"
    if f.exists():
        for line in open(f):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "val/force_mae" in d and "epoch" in d:
                out[int(d["epoch"])] = 1000 * d["val/force_mae"]
    return out


COLORS = {"alloc": "#be123c", "magnitude": "#059669", "mag": "#059669", "muon": "#4b5563"}


def _color(name, method):
    if name.startswith("mp1-gradual-muon-mag") or "nequix" in name:
        return COLORS["muon"]
    return COLORS[method]


def plot(rows, out: Path, n: int, zero_f=None):
    """2 x 2: columns = weight decay 0.001 / 0.01; top = full range (log), bottom = zoom on the
    useful range (linear). Classic = dashed, open circles; gradual = solid squares, one per
    phase end; small models trained from scratch = stars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for key, ylab, fname, zoom, ticks in [
            ("force_mae", "Force MAE (meV/Å)", "gradual_vs_classic.png", (30, 90),
             [20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 700, 1000]),
            ("energy_mae", "Energy MAE (meV/atom)", "gradual_vs_classic_energy.png", (10, 40),
             [10, 15, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000, 10000])]:
        # same y scale on both columns of a row, so the two weight decays can be compared directly
        fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey="row")
        for col, wd in enumerate(["0.001", "0.01"]):
            sel = [r for r in rows if r["wd"] == wd]
            for row, ax in enumerate(axes[:, col]):
                for name in dict.fromkeys(r["name"] for r in sel if r["kind"] == "classic"):
                    for method in dict.fromkeys(r["method"] for r in sel if r["name"] == name):
                        p = sorted([r for r in sel if r["name"] == name and r["method"] == method],
                                   key=lambda r: r["cost_removed"])
                        ep = f", ep. {int(float(p[0]['epoch']))}" if p[0]["epoch"] != "" else ""
                        ax.plot([100 * r["cost_removed"] for r in p], [r[key] for r in p], color=_color(name, method),
                                ls="--", lw=1.6, marker="o", ms=5, markerfacecolor="white",
                                label=f"classic (after training): {p[0]['label']}{ep}")
                for name in dict.fromkeys(r["name"] for r in sel if r["kind"] == "gradual"):
                    p = sorted([r for r in sel if r["name"] == name], key=lambda r: r["cost_removed"])
                    eps = "/".join(str(int(float(r["epoch"]))) + ("*" if r.get("provisional") else "") for r in p)
                    ax.plot([100 * r["cost_removed"] for r in p], [r[key] for r in p], color=_color(name, p[0]["method"]),
                            ls="-", lw=2.4, marker="s", ms=7, label=f"gradual (during training): {p[0]['label']}, ep. {eps}")
                sc = [r for r in sel if r["kind"] == "scratch"]
                for r in sc:
                    ax.plot([100 * r["cost_removed"]], [r[key]], ls="none", marker="*", ms=15, color="#7c3aed",
                            label=f"small uniform model from scratch, {100 * (1 - r['cost_removed']):.0f} % cost, "
                                  f"ep. {int(float(r['epoch']))}" + ("" if int(float(r["epoch"])) >= 99 else " (still training)"))
                if key == "force_mae" and zero_f:
                    ax.axhline(zero_f, color="black", ls=":", lw=1, label=f"predicting zero forces ({zero_f:.0f} meV/Å)")
                ax.grid(alpha=0.3, which="both")
                ax.set_xticks([0, 20, 40, 60, 70, 80, 90])
                if row == 0:
                    ax.set_yscale("log")
                    lo, hi = min(r[key] for r in rows), max(r[key] for r in rows)
                    ax.set_yticks([t for t in ticks if 0.8 * lo <= t <= 1.25 * hi])
                    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                    ax.set_ylim(0.9 * lo, 1.15 * hi)
                    ax.set_title(f"weight decay {wd}: full range (log scale)", fontsize=10)
                else:
                    ax.set_ylim(*zoom)
                    ax.set_title(f"weight decay {wd}: zoom (linear scale)", fontsize=10)
                    ax.set_xlabel("Channel cost removed (%), paths cost model")
                    # legend below the panel, so it hides no point
                    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), framealpha=0.95)
                if col == 0:
                    ax.set_ylabel(ylab + "  (lower is better)")
                ax.tick_params(labelleft=True)   # keep y labels on the right column too (shared axis)
        fig.suptitle("Nequix-MP-1 on full MPtrj: gradual pruning during training vs classic pruning after training "
                     f"(no fine-tuning), {n} held-out structures\n"
                     "classic = the group's fully trained models pruned once; gradual = snapshot at the end of each pruning phase "
                     "(* = latest snapshot, phase not finished, still recovering)",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


def table(rows, out: Path, plan):
    refs = {r["name"]: dense_val(r["ref"]) for r in plan["runs"]}
    lines = ["| model | kind | epoch | cost removed | force MAE (meV/Å) | energy MAE (meV/atom) | dense run at that epoch (val. F) |",
             "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["wd"], r["kind"], r["name"], r["method"], r["cost_removed"])):
        ref = ""
        if r["kind"] == "gradual" and r["epoch"] != "" and int(r["epoch"]) in refs.get(r["name"], {}):
            ref = f"{refs[r['name']][int(r['epoch'])]:.1f}"
        lines.append(f"| {r['label']} (wd {r['wd']}) | {r['kind']}{' ' + r['method'] if r['kind'] == 'classic' else ''} | "
                     f"{r['epoch']} | {100 * r['cost_removed']:.1f} % | {r['force_mae']:.1f} | {r['energy_mae']:.2f} | {ref} |")
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    global FULL_COST
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="bmuon/experiments/gradual_runs.yml")
    ap.add_argument("--data")
    ap.add_argument("--n-structures", type=int, default=1000)
    ap.add_argument("--out", default="bmuon/results/gradual_fig")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figures from points.csv, no evaluation")
    args = ap.parse_args()
    out = Path(args.out)
    if args.plot_only:
        with open(out / "points.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            for k in ("cost_removed", "force_mae", "energy_mae"):
                r[k] = float(r[k])
            r["provisional"] = int(r["provisional"])
        zero_f = json.loads((out / "baseline.json").read_text())["zero_force_mae"]
        plot(rows, out, args.n_structures, zero_f)
        print(f"redrew {out}/gradual_vs_classic.png and gradual_vs_classic_energy.png", flush=True)
        return
    if not args.data:
        ap.error("--data is required unless --plot-only")
    _load_model_code()
    cache = out / f"cache_n{args.n_structures}"
    cache.mkdir(parents=True, exist_ok=True)
    plan = yaml.safe_load(open(args.plan, encoding="utf-8"))

    full, cfg = load_model("models/nequix-mp-1.nqx", kernel=False)
    FULL_COST = float(sum(c.n_paths for c in build_channel_spec(full, cost_model="paths").channels))
    print(f"loading {args.n_structures} held-out structures...", flush=True)
    ev = load_eval_set(args.data, cfg, args.n_structures)  # same split and seed as fig2.py
    zero_f = 1000 * float(np.mean(np.abs(ev[2])))  # MAE of predicting zero forces
    (out / "baseline.json").write_text(json.dumps({"zero_force_mae": zero_f}))
    print(f"predicting zero forces: F {zero_f:.1f} meV/A", flush=True)

    rows, phase_ends = [], {}
    for c in CLASSIC:
        for m in c["methods"]:
            rows += cached(cache / f"{c['name']}_{m}.csv", lambda c=c, m=m: classic_curve(c, m, ev))
    for r in plan["runs"]:
        rc = yaml.safe_load(open(RUNS / r["name"] / "config.yml"))
        g = rc["gradual"]
        info = {"name": r["name"], "label": r["label"].replace(" gradual", ""), "score": g["score"],
                "wd": f"{rc['weight_decay']:g}"}
        ends = [g["prune_after_epochs"][0]] + list(g["prune_after_epochs"][1:]) + [rc["n_epochs"] - 1]
        phase_ends[r["name"]] = set(ends)
        snaps = sorted((RUNS / r["name"]).glob("snap_epoch*.nqx"))
        latest = int(snaps[-1].stem[-3:]) if snaps else -1
        done_ends = [e for e in ends if e <= latest]
        # unfinished run: its latest snapshot as a provisional point (phase not finished yet)
        extra = [latest] if latest >= 0 and latest not in ends and not (RUNS / r["name"] / "DONE").exists() else []
        for ep in done_ends + extra:
            snap = RUNS / r["name"] / f"snap_epoch{ep:03d}.nqx"
            if snap.exists():
                rows += cached(cache / f"{r['name']}_e{ep}.csv", lambda info=info, ep=ep, snap=snap: gradual_point(info, ep, snap, ev))
    for s in SCRATCH:
        snaps = sorted((RUNS / s["run"]).glob("snap_epoch*.nqx"))
        if snaps:
            ep = int(snaps[-1].stem[-3:])
            rows += cached(cache / f"{s['name']}_e{ep}.csv", lambda s=s, sn=snaps[-1], ep=ep: scratch_point(s, sn, ep, ev))
    for r in rows:
        for k in ("cost_removed", "force_mae", "energy_mae"):
            r[k] = float(r[k])
        r["provisional"] = int(r["kind"] == "gradual" and int(float(r["epoch"])) not in phase_ends[r["name"]])
    with open(out / "points.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    plot(rows, out, args.n_structures, zero_f)
    table(rows, out, plan)
    print(f"wrote {out}/gradual_vs_classic.png, gradual_vs_classic_energy.png, points.csv, table.md", flush=True)


if __name__ == "__main__":
    main()
