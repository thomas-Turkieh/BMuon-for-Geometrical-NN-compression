"""Training curves: the architecture found by pruning the 1.5x model, retrained from scratch,
against the dense Nequix-MP-1 (uniform architecture of the same tensor-product cost).

    python -m bmuon.experiments.scratch_vs_dense_fig [--runs bmuon/runs] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

DENSE = "mp1-bmuon-selwd"
SCRATCH = "mp1-arch-x1.5-pruned-scratch"
OUT = Path("bmuon/results/2026-11-22_full-mptrj_x1.5-pruned-architecture-from-scratch")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_DENSE, C_SCRATCH = "#2a78d6", "#0f9d8a"
N_EPOCHS = 100


def read(run_dir: Path) -> dict:
    val = {}
    for line in open(run_dir / "log.jsonl"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "val/force_mae" in d and "epoch" in d:
            val[int(d["epoch"])] = (1000 * d["val/force_mae"], 1000 * d["val/energy_mae_per_atom"])
    return val


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK2)
    ax.tick_params(colors=INK2, labelcolor=INK, length=3)
    ax.grid(True, color=GRID, lw=0.8, which="major")
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="bmuon/runs")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    dense, scratch = read(runs / DENSE), read(runs / SCRATCH)
    last = max(scratch)
    common = sorted(set(dense) & set(scratch))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.12, "wspace": 0.22})
    fig.subplots_adjust(top=0.77, bottom=0.17, left=0.07, right=0.975)

    panels = [(0, "Force", "meV/Å", (30, 100), [30, 35, 40, 45, 50, 60, 70, 80, 100]),
              (1, "Energy", "meV/atom", (9.5, 50), [10, 11, 12, 13, 15, 20, 25, 30, 40, 50])]
    for col, (q, name, unit, ylim, ticks) in enumerate(panels):
        ax, axd = axes[0, col], axes[1, col]
        style(ax)
        style(axd)
        e = sorted(dense)
        ax.plot(e, [dense[i][q] for i in e], color=C_DENSE, lw=2, zorder=3,
                label="Nequix-MP-1, dense (uniform architecture, 3,236 TP paths)")
        e = sorted(scratch)
        ax.plot(e, [scratch[i][q] for i in e], color=C_SCRATCH, lw=2.4, zorder=4,
                label="Architecture found by pruning the 1.5× model, retrained from scratch (3,232 TP paths)")
        ax.plot([last], [scratch[last][q]], "o", color=C_SCRATCH, ms=7, mec="white", mew=1.2, zorder=5)
        ax.annotate(f"epoch {last}: {scratch[last][q]:.2f}\n(dense: {dense[last][q]:.2f})",
                    (last, scratch[last][q]), xytext=(14, 22), textcoords="offset points",
                    fontsize=9.5, color=INK, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
        fin = max(dense)
        ax.annotate(f"dense, final (epoch {fin}): {dense[fin][q]:.2f}", (fin, dense[fin][q]),
                    xytext=(-8, 18), textcoords="offset points", ha="right", fontsize=9.5, color=INK)
        ax.axvspan(last + 0.5, N_EPOCHS - 1, color="#f1f0ec", lw=0, zorder=0)
        ax.text((last + N_EPOCHS) / 2, ylim[1] / 1.05, "from-scratch run: still training",
                ha="center", va="top", fontsize=9, color=INK2)
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel(f"Validation {name.lower()} MAE ({unit})", color=INK)
        ax.set_title(f"{'(a)' if col == 0 else '(b)'} {name} error", loc="left", fontweight="bold",
                     fontsize=12, color=INK)

        rel = [100 * (scratch[i][q] - dense[i][q]) / dense[i][q] for i in common]
        axd.axhline(0, color=INK2, lw=1)
        axd.fill_between(common, rel, 0, where=[r > 0 for r in rel], color="#d1d0cb", lw=0,
                         interpolate=True, zorder=1)
        axd.fill_between(common, rel, 0, where=[r <= 0 for r in rel], color="#bfe6df", lw=0,
                         interpolate=True, zorder=1)
        axd.plot(common, rel, color=C_SCRATCH, lw=2, zorder=3)
        axd.axvspan(last + 0.5, N_EPOCHS - 1, color="#f1f0ec", lw=0, zorder=0)
        m = max(3.0, max(abs(r) for r in rel) * 1.3)
        axd.set_ylim(-m, m)
        axd.text(1, m * 0.92, "from scratch worse ↑", va="top", fontsize=8.5, color=INK2)
        axd.text(1, -m * 0.92, "from scratch better ↓", va="bottom", fontsize=8.5, color=INK2)
        axd.set_ylabel("From scratch vs dense\n(% difference)", color=INK)
        axd.set_xlabel("Epoch", color=INK)
        axd.set_xlim(0, N_EPOCHS - 1)

    fig.text(0.07, 0.975, "Is the architecture found by pruning better on its own?\nThe 1.5×-pruned "
             "architecture retrained from scratch vs the dense Nequix-MP-1 (MPtrj)", linespacing=1.3,
             fontsize=13.5, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.07, 0.898,
             "Same tensor-product cost (3,232 vs 3,236 paths) and same training recipe (BMuon optimizer, "
             "weight decay 1e-3, 100 epochs, global batch 256,\nsame initialization seed and validation split), "
             "no pruning in either run. Only the per-block channel widths differ: uniform for Nequix-MP-1,\n"
             "those kept by BM-μ pruning in the 1.5× wide model for the from-scratch run. "
             f"Partial results: epoch {last} of 99.",
             fontsize=10, color=INK2, ha="left", va="top", linespacing=1.45)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.065, 0.83), ncol=2, frameon=False, fontsize=9.5)
    fig.text(0.07, 0.015,
             "Validation mean absolute error after each epoch (EMA weights) on the held-out split "
             "(5 % of MPtrj, 79,020 structures, never used for training), as logged during training; "
             "log scale, lower is better.\nForce MAE: mean over atoms and Cartesian components. "
             "Energy MAE: mean over structures of |ΔE| / number of atoms. Bottom panels: "
             "(from scratch − dense) / dense at the same epoch. One seed per run.\n"
             f"Runs: {DENSE} (dense), {SCRATCH} (from scratch).",
             fontsize=8.6, color=INK2, ha="left", va="bottom", linespacing=1.45)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"scratch_vs_dense_curves.{ext}", dpi=220)
    print(f"wrote {out / 'scratch_vs_dense_curves.png'}; epoch {last}: scratch "
          f"{scratch[last][0]:.2f}/{scratch[last][1]:.2f}, dense {dense[last][0]:.2f}/{dense[last][1]:.2f}")


if __name__ == "__main__":
    main()
