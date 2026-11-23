"""Training curves of the wide models gradually pruned to the size of Nequix-MP-1 (plan
bmuon/experiments/wide_arch_runs.yml) against the dense Nequix-MP-1 (no pruning), from the logs.
Works on unfinished runs: every curve stops at the last validated epoch.

    python -m bmuon.experiments.wide_fig [--runs bmuon/runs]
        [--out bmuon/results/2026-11-10_full-mptrj_wide-models-pruned-to-nequix-size]

Top: validation force / energy MAE per epoch (whole validation split, log scale), with the
pruning events and the final error of the dense model. Bottom: tensor-product cost of each model
relative to Nequix-MP-1 (k x at the start, 1 x from epoch 75), solid up to the epoch reached,
faint for the rest of the schedule. Writes training_curves_partial.png and val_errors_per_epoch.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import yaml

WIDE = ["mp1-wide-x1.25-gradual-bmuon-mu", "mp1-wide-x1.5-gradual-bmuon-mu", "mp1-wide-x2-gradual-bmuon-mu",
        "mp1-wide-x2.5-gradual-bmuon-mu", "mp1-wide-x3-gradual-bmuon-mu"]
# dense Nequix-MP-1, same data / batch / lr / seed / validation split; BMuon budget only weights the
# updates, the evaluated network keeps all its channels
DENSE = [("mp1-bmuon-selwd", "Dense Nequix-MP-1, no pruning (BMuon optimizer, weight decay 0.001)", "#111827", "--"),
         ("mp1-bmuon-wd1e3", "Dense Nequix-MP-1, no pruning (BMuon optimizer, weight decay 0.001; run stopped at epoch 65)", "#9ca3af", "-.")]
N_EPOCHS = 100
# architecture kept by the 1.5x run (3,232 TP paths), retrained from scratch without pruning
SCRATCH = [("mp1-arch-x1.5-pruned-scratch", "Architecture found by pruning the 1.5× model, retrained from scratch (random init, no pruning, Nequix-MP-1 size)", "#0f9d8a")]


def read(run_dir: Path):
    val, prunes = {}, []
    for line in open(run_dir / "log.jsonl"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "val/force_mae" in d and "epoch" in d:
            val[int(d["epoch"])] = (1000 * d["val/force_mae"], 1000 * d["val/energy_mae_per_atom"])
        if "epoch_start" in d and "prune/level" in d:
            prunes.append(d)
    return val, prunes


def cost_schedule(k: float, gradual: dict):
    """[(first epoch, TP cost relative to Nequix-MP-1)] for the whole planned schedule."""
    sched = [(0, k)]
    for ep, lv in zip(gradual["prune_after_epochs"], gradual["levels"]):
        sched.append((ep + 1, k * (1 - lv)))
    return sched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="bmuon/runs")
    ap.add_argument("--out", default="bmuon/results/2026-11-10_full-mptrj_wide-models-pruned-to-nequix-size")
    args = ap.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    wide = []
    for name in WIDE:
        cfg = yaml.safe_load(open(runs / name / "config.yml", encoding="utf-8"))
        val, prunes = read(runs / name)
        k = float(cfg["tp_expand"])
        wide.append({"name": name, "k": k, "val": val, "sched": cost_schedule(k, cfg["gradual"]),
                     "events": [p + 1 for p in cfg["gradual"]["prune_after_epochs"]]})
    dense = [(name, label, c, ls, read(runs / name)[0]) for name, label, c, ls in DENSE if (runs / name).exists()]
    events = wide[0]["events"]
    colors = [plt.cm.plasma(x) for x in (0.82, 0.66, 0.5, 0.33, 0.12)]   # light = narrow, dark = wide

    fig, axes = plt.subplots(2, 2, figsize=(16, 9.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.15]})
    for col, (qi, ylab, ylim, ticks, unit) in enumerate([
            (0, "Validation force MAE (meV/Å)", (30, 100), [30, 35, 40, 45, 50, 60, 70, 80, 100], "meV/Å"),
            (1, "Validation energy MAE (meV/atom)", (9.5, 50), [10, 11, 12, 13, 15, 20, 25, 30, 40, 50], "meV/atom")]):
        ax, axc = axes[0, col], axes[1, col]
        # after the last pruning event every wide model has the size of Nequix-MP-1
        for a in (ax, axc):
            a.axvspan(events[-1], N_EPOCHS - 1, color="#e5e7eb", alpha=0.6, lw=0, zorder=0)
            for e in events:
                a.axvline(e, color="#6b7280", lw=0.8, ls=":", zorder=1)
        ax.text((events[-1] + N_EPOCHS - 1) / 2, ylim[1] / 1.04, "all wide models at\nNequix-MP-1 size",
                ha="center", va="top", fontsize=8.5, color="#374151")
        for name, label, c, ls, val in dense:
            eps = sorted(val)
            ax.plot(eps, [val[e][qi] for e in eps], color=c, ls=ls, lw=1.8, zorder=3, label=label)
        final = dense[0][4]
        last = max(final)
        if last >= N_EPOCHS - 1:
            ax.axhline(final[last][qi], color="#111827", lw=0.9, ls=(0, (1, 3)), zorder=2,
                       label=f"Target: final error of the dense Nequix-MP-1 (epoch {last}): "
                             f"forces {final[last][0]:.1f} meV/Å, energy {final[last][1]:.1f} meV/atom")
            ax.text(1, final[last][qi] / 1.012, f"target: dense final error, {final[last][qi]:.1f} {unit}",
                    va="top", fontsize=8.5, color="#111827")
        for name, label, c in SCRATCH:
            if not (runs / name / "log.jsonl").exists():
                continue
            val = read(runs / name)[0]
            eps = sorted(val)
            if not eps:
                continue
            ep = eps[-1]
            ax.plot(eps, [val[e][qi] for e in eps], color=c, lw=2.2, ls=(0, (5, 1.5)), zorder=4,
                    label=f"{label}; latest epoch {ep}: forces {val[ep][0]:.1f} meV/Å, energy {val[ep][1]:.1f} meV/atom")
            ax.plot([ep], [val[ep][qi]], "o", color=c, ms=6, zorder=5, markeredgecolor="white", markeredgewidth=0.8)
        for w, c in zip(wide, colors):
            eps = sorted(w["val"])
            if not eps:
                continue
            ep = eps[-1]
            ax.plot(eps, [w["val"][e][qi] for e in eps], color=c, lw=2.2, zorder=4,
                    label=f"{w['k']:g}× wider tensor products, gradually pruned to Nequix-MP-1 size (BM-μ); "
                          f"latest epoch {ep}: forces {w['val'][ep][0]:.1f} meV/Å, energy {w['val'][ep][1]:.1f} meV/atom")
            ax.plot([ep], [w["val"][ep][qi]], "o", color=c, ms=6, zorder=5, markeredgecolor="white", markeredgewidth=0.8)
            # cost: solid up to the epoch reached, faint for the remaining schedule
            xs, ys = [], []
            for i, (start, cost) in enumerate(w["sched"]):
                end = w["sched"][i + 1][0] if i + 1 < len(w["sched"]) else N_EPOCHS - 1
                xs += [start, end]
                ys += [cost, cost]
            axc.plot(xs, ys, color=c, lw=1.2, alpha=0.35, ls="--", zorder=3)
            done = [(x, y) for x, y in zip(xs, ys) if x <= ep] + [(ep, ys[max(i for i, x in enumerate(xs) if x <= ep)])]
            axc.plot([p[0] for p in done], [p[1] for p in done], color=c, lw=2.2, zorder=4)
        axc.axhline(1.0, color="#111827", ls="--", lw=1.6, zorder=3)
        axc.text(N_EPOCHS - 2, 0.92, "dense Nequix-MP-1 (1×)", fontsize=8, color="#111827", va="top", ha="right")

        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel(ylab + "  (lower is better)")
        ax.grid(alpha=0.3, which="major")
        ax.set_title(["Forces", "Energy"][qi], fontsize=11)
        axc.set_ylim(0, 3.3)
        axc.set_yticks([0, 1, 2, 3])
        axc.set_ylabel("Tensor-product cost\n(× Nequix-MP-1)")
        axc.set_xlabel("Epoch")
        axc.set_xlim(-1, N_EPOCHS)
        axc.grid(alpha=0.3)

    h, l = axes[0, 0].get_legend_handles_labels()
    order = ([i for i, s in enumerate(l) if s.startswith(("Dense", "Target"))]
             + [i for i, s in enumerate(l) if not s.startswith(("Dense", "Target"))])
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", ncol=2, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Wide Nequix-MP-1 models gradually pruned to the size of Nequix-MP-1 vs dense Nequix-MP-1, full MPtrj\n"
                 f"Partial results, runs in progress (last update {time.strftime('%d %b %Y')})", fontsize=12)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    caption = (
        "Validation mean absolute error per epoch on the whole held-out split (5 % of MPtrj, same split, batch, "
        "learning rate and seed for every run); log scale, lower is better.\n"
        f"Vertical dotted lines: the 6 pruning events (epochs {events[0]}–{events[-1]}). A k× model removes 1 − 1/k of "
        "its tensor-product cost, channels ranked by the BMuon allocation μ (BM-μ). The budget is annealed over the first "
        "half of each phase,\n"
        "so each cut removes channels that are already nearly inactive. Tensor-product cost: number "
        "of tensor-product paths fed by the kept channels, relative to Nequix-MP-1 (3,236 paths, 0.71 M parameters).")
    fig.text(0.5, -0.012, caption, ha="center", va="top", fontsize=8.5, color="#374151", linespacing=1.5)
    fig.savefig(out / "training_curves_partial.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    with open(out / "val_errors_per_epoch.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["run", "epoch", "force_mae_meV_per_A", "energy_mae_meV_per_atom"])
        for name, val in [(w["name"], w["val"]) for w in wide] + [(d[0], d[4]) for d in dense]:
            for e in sorted(val):
                wr.writerow([name, e, f"{val[e][0]:.2f}", f"{val[e][1]:.2f}"])
    print(f"wrote {out}/training_curves_partial.png and val_errors_per_epoch.csv")


if __name__ == "__main__":
    main()
