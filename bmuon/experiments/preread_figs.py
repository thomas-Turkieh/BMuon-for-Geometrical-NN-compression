"""Figures for the weekly meeting pre-read (wide-then-prune experiments), in a standard paper style.

    python -m bmuon.experiments.preread_figs --runs bmuon/runs --bench DIR [--out DIR]

--runs:  run folders with log.jsonl (+ config.yml for the wide runs).
--bench: output of equal_size_bench.py (errors.npz, inference.csv, meta.json).

Writes fig1..fig4 as PNG and PDF, and all figures in one PDF (pre-read_figures.pdf).
"""

from __future__ import annotations

import argparse
import json
import textwrap
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

from bmuon.experiments.equal_size_fig import bootstrap, mean_ci, read_csv  # noqa: E402

OUT = Path("bmuon/results/2026-11-24_full-mptrj_meeting-pre-read")
DENSE = "mp1-bmuon-selwd"
SCRATCH = "mp1-arch-x1.5-pruned-scratch"
WIDE = {1.25: "mp1-wide-x1.25-gradual-bmuon-mu", 1.5: "mp1-wide-x1.5-gradual-bmuon-mu",
        2.0: "mp1-wide-x2-gradual-bmuon-mu", 2.5: "mp1-wide-x2.5-gradual-bmuon-mu",
        3.0: "mp1-wide-x3-gradual-bmuon-mu"}
N_EPOCHS = 100
LAST = N_EPOCHS - 1
WIDTH_IN = 7.0  # double-column figure width (inches)

C_DENSE = "black"
C_SCRATCH = "tab:red"
C_WIDE = dict(zip(WIDE, plt.cm.viridis(np.linspace(0.0, 0.82, len(WIDE)))))
QTY = [(0, "Force MAE", "meV/Å"), (1, "Energy MAE", "meV/atom")]

PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "lines.linewidth": 1.3,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.edgecolor": "black",
    "legend.framealpha": 1.0,
    "patch.linewidth": 0.6,
    "savefig.dpi": 300,
}


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


def sgn(v, nd=1):
    return f"{v:+.{nd}f}".replace("-", "−")


def log_axis(ax, ylim, ticks):
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


CAPTIONS = False  # captions are written in the LaTeX pre-read (pre-read.tex)


def add_caption(fig, label, text, y=0.012, width=125):
    """Paper-style caption under the figure: bold label, then plain text."""
    if not CAPTIONS:
        return
    lines = textwrap.wrap(f"{label} {text}", width)
    fs = 8.5
    for i, line in enumerate(reversed(lines)):
        yy = y + i * fs * 1.35 / 72 / fig.get_figheight()
        if i == len(lines) - 1:  # first line: label in bold
            rest = line[len(label):]
            t = fig.text(0.07, yy, label, ha="left", va="bottom", fontsize=fs, fontweight="bold")
            fig.canvas.draw()
            bb = t.get_window_extent().transformed(fig.transFigure.inverted())
            fig.text(bb.x1, yy, rest, ha="left", va="bottom", fontsize=fs)
        else:
            fig.text(0.07, yy, line, ha="left", va="bottom", fontsize=fs)


# --------------------------------------------------------------------------- fig 1
def fig_curves(dense, wide, cfgs):
    events = [e + 1 for e in cfgs[1.5]["gradual"]["prune_after_epochs"]]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_IN, 5.0), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.08, "wspace": 0.28})
    fig.subplots_adjust(top=0.86, bottom=0.25, left=0.09, right=0.98)
    for col, (q, name, unit) in enumerate(QTY):
        ax, axc = axes[0, col], axes[1, col]
        for a in (ax, axc):
            a.axvspan(events[-1], LAST, color="0.93", lw=0, zorder=0)
            for e in events:
                a.axvline(e, color="0.6", lw=0.6, ls=":", zorder=1)
        e = sorted(dense)
        ax.plot(e, [dense[i][q] for i in e], color=C_DENSE, ls="--", zorder=5)
        for k, val in wide.items():
            e = sorted(val)
            ax.plot(e, [val[i][q] for i in e], color=C_WIDE[k], zorder=4)
            if e[-1] < LAST:
                ax.plot([e[-1]], [val[e[-1]][q]], "o", color=C_WIDE[k], ms=3.5, zorder=6)
            sched = [(0, k)] + [(p + 1, k * (1 - lv)) for p, lv in
                                zip(cfgs[k]["gradual"]["prune_after_epochs"], cfgs[k]["gradual"]["levels"])]
            xs, ys = [], []
            for i, (s, c) in enumerate(sched):
                xs += [s, sched[i + 1][0] if i + 1 < len(sched) else LAST]
                ys += [c, c]
            axc.plot(xs, ys, color=C_WIDE[k], zorder=4)
        axc.axhline(1, color=C_DENSE, ls="--", zorder=5)
        if q == 0:
            log_axis(ax, (31, 70), [32, 34, 36, 40, 45, 50, 60, 70])
        else:
            log_axis(ax, (9.6, 26), [10, 11, 12, 14, 16, 20, 25])
        ax.set_xlim(0, LAST)
        ax.set_ylabel(f"Validation {name.lower().replace('mae', 'MAE')} ({unit})")
        ax.set_title(f"({'ab'[col]}) {name.split()[0]}")
        axc.set_ylim(0, 3.3)
        axc.set_yticks([0, 1, 2, 3], ["0", "1×", "2×", "3×"])
        axc.set_ylabel("Model size\n(× Nequix-MP-1)")
        axc.set_xlabel("Epoch")
    handles = [Line2D([], [], color=C_DENSE, ls="--", label="Nequix-MP-1 (dense)")]
    handles += [Line2D([], [], color=C_WIDE[k],
                       label=f"{k:g}× wider, gradual pruning" + ("" if max(v) >= LAST else f" (ep. {max(v)})"))
                for k, v in wide.items()]
    handles += [Line2D([], [], color="0.6", lw=0.6, ls=":", label="Pruning step")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.535, 0.995), ncol=4)
    add_caption(fig, "Figure 1.",
                "Validation error during training of models whose tensor products are k× wider than Nequix-MP-1, "
                "gradually pruned with BM-μ (6 steps, epochs 30–75) to the tensor-product (TP) cost of "
                "Nequix-MP-1 (3,236 paths). Shaded: all models at Nequix-MP-1 size. MPtrj, same recipe for all "
                "runs (BMuon, weight decay 1e-3, batch 256, same seed and validation split of 79k structures); "
                f"EMA weights; log scale. Status as of {time.strftime('%d %b %Y')}.")
    return fig


# --------------------------------------------------------------------------- fig 2
def fig_final(dense, wide):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.4), gridspec_kw={"wspace": 0.3})
    fig.subplots_adjust(top=0.8, bottom=0.33, left=0.09, right=0.98)
    for ax, (q, name, unit) in zip(axes, QTY):
        ref = dense[LAST][q]
        ax.axhline(ref, color=C_DENSE, ls="--", lw=1, zorder=1)
        ax.plot([1.0], [ref], "s", color=C_DENSE, ms=5, zorder=4)
        ax.annotate(f"{ref:.2f}", (1.0, ref), xytext=(0, 6), textcoords="offset points", ha="center",
                    fontsize=7.5)
        for k, val in wide.items():
            ep = max(val)
            v = val[ep][q]
            done = ep >= LAST
            ax.plot([k], [v], "o", ms=5.5, color=C_WIDE[k] if done else "white", mec=C_WIDE[k], mew=1.2, zorder=4)
            rel = 100 * (v - ref) / ref
            txt = f"{v:.2f}\n({sgn(rel)}%)" if done else f"{v:.2f}\n(ep. {ep})"
            below = v < ref
            ax.annotate(txt, (k, v), xytext=(0, -7 if below else 6), textcoords="offset points", ha="center",
                        va="top" if below else "bottom", fontsize=7.5, linespacing=1.1)
        vals = [ref] + [val[max(val)][q] for val in wide.values()]
        span = max(vals) - min(vals)
        ax.set_ylim(min(vals) - 0.9 * span, max(vals) + 0.8 * span)
        ax.set_xticks([1.0] + list(wide), ["1"] + [f"{k:g}" for k in wide])
        ax.set_xlim(0.75, 3.25)
        ax.set_xlabel("Initial width (× Nequix-MP-1)")
        ax.set_ylabel(f"Validation {name.lower().replace('mae', 'MAE')} ({unit})")
        ax.set_title(f"({'ab'[q]}) {name.split()[0]}")
    handles = [Line2D([], [], marker="s", ls="--", color=C_DENSE, ms=5, lw=1, label="Nequix-MP-1 (dense)"),
               Line2D([], [], marker="o", ls="", color=C_WIDE[2.0], ms=5.5, label="Gradual pruning, finished (ep. 99)"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=C_WIDE[2.0], mew=1.2, ms=5.5,
                      label="Gradual pruning, still training")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.535, 0.995), ncol=3)
    add_caption(fig, "Figure 2.",
                "Validation error at the last epoch vs. initial tensor-product width; every model is pruned to "
                "the TP cost of Nequix-MP-1. Finished runs at epoch 99 (percentages relative to the dense model); "
                "open circles: runs still training, at their latest epoch. MPtrj validation split (79k "
                "structures), EMA weights, one seed per model.")
    return fig


# --------------------------------------------------------------------------- fig 3
def fig_equal_size(bench):
    meta = json.load(open(bench / "meta.json"))
    err = bootstrap(dict(np.load(bench / "errors.npz")))
    inf = read_csv(bench / "inference.csv")
    inf_ms = {m: mean_ci([float(r["ms_per_structure"]) for r in inf if r["model"] == m])
              for m in ("dense", "wide_pruned")}
    gpu = meta["gpu"]["nvidia_smi"].split(",")[0].replace("NVIDIA ", "").replace(" Generation", "")
    tp, pr = meta["tp_paths"], meta["params"]
    c_pr = C_WIDE[1.5]
    c_de = "0.55"

    fig = plt.figure(figsize=(WIDTH_IN, 6.2))
    gs = fig.add_gridspec(2, 3, left=0.09, right=0.98, top=0.95, bottom=0.27, wspace=0.5, hspace=0.42,
                          height_ratios=[1, 1.1])
    labels = ["Nequix-MP-1\n(dense)", "1.5× wider,\ngradually pruned"]

    for i, (key, name, unit) in enumerate([("f", "Force MAE", "meV/Å"), ("e", "Energy MAE", "meV/atom")]):
        ax = fig.add_subplot(gs[0, i])
        vals = [err[f"{key}_dense"], err[f"{key}_wide"]]
        for j, (m, lo, hi) in enumerate(vals):
            ax.errorbar(j, m, yerr=[[m - lo], [hi - m]], fmt="s" if j == 0 else "o", ms=5,
                        color=C_DENSE if j == 0 else c_pr, ecolor="black", capsize=3, lw=0.9, zorder=3)
            ax.annotate(f"{m:.2f}", (j, m), xytext=(7, 0), textcoords="offset points", va="center", fontsize=7.5)
        lo, hi = min(v[1] for v in vals), max(v[2] for v in vals)
        ax.set_ylim(lo - 0.5 * (hi - lo), hi + 0.9 * (hi - lo))
        ax.set_xticks([0, 1], labels)
        ax.set_xlim(-0.5, 1.6)
        ax.set_ylabel(f"Validation {name.lower().replace('mae', 'MAE')} ({unit})")
        ax.set_title(f"({'ab'[i]}) {name.split()[0]}")
        rel, rlo, rhi = err[f"{key}_rel"]
        ax.text(0.92, 0.95, f"Δ = {sgn(rel)}%\n95% CI [{sgn(rlo)}, {sgn(rhi)}]", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, linespacing=1.2)

    ax = fig.add_subplot(gs[0, 2])
    vals = [inf_ms["dense"], inf_ms["wide_pruned"]]
    for j, (m, lo, hi) in enumerate(vals):
        ax.bar(j, m, width=0.55, color=c_de if j == 0 else c_pr, ec="black", zorder=2)
        ax.errorbar(j, m, yerr=[[m - lo], [hi - m]], color="black", capsize=3, lw=0.9, zorder=3)
        ax.text(j, hi + 0.05, f"{m:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_ylim(0, max(v[2] for v in vals) * 1.3)
    ax.set_xticks([0, 1], labels)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("Inference time (ms / structure)")
    ax.set_title("(c) Inference")
    r = inf_ms["wide_pruned"][0] / inf_ms["dense"][0]
    ax.text(0.92, 0.95, f"Δ = {sgn(100 * (r - 1), 0)}%", transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5)

    # (d) channels per layer and irrep
    ax = fig.add_subplot(gs[1, :])

    def parse(s):
        return {int(t.split("x")[1][0]): int(t.split("x")[0]) for t in s.replace(" ", "").split("+")}
    n_layers = len(meta["tp_irreps"]["dense"])
    x, ticks, groups, bw = 0.0, [], [], 0.38
    for layer, (d, f, w) in enumerate(zip(meta["tp_irreps"]["dense"], meta["tp_irreps"]["wide_full"],
                                          meta["tp_irreps"]["wide_pruned"])):
        d, f, w = parse(d), parse(f), parse(w)
        last = layer == n_layers - 1  # the last layer outputs scalars only (energy readout)
        start = x
        for l in sorted(d):
            ax.bar(x - bw / 2, d[l], width=bw, color=c_de, ec="black", zorder=2)
            ax.bar(x + bw / 2, f[l], width=bw, fill=False, ec="black", ls="--", lw=0.6, zorder=2)
            ax.bar(x + bw / 2, w[l], width=bw, color=c_pr, ec="black", zorder=3)
            ax.text(x + bw / 2, max(w[l], f[l]) + 3, f"{w[l]}", ha="center", va="bottom", fontsize=7)
            ax.text(x - bw / 2, d[l] + 3, f"{d[l]}", ha="center", va="bottom", fontsize=7, color="0.3")
            paths = 1 if last else {0: 4, 1: 6, 2: 7, 3: 6}[l]
            ticks.append((x, f"$\\ell$={l}\n[{paths}]"))
            x += 1
        groups.append(((start + x - 1) / 2, f"Layer {layer + 1}" + (" (scalar output)" if last else "")))
        x += 0.7
    ax.set_xticks([t[0] for t in ticks], [t[1] for t in ticks])
    ax.tick_params(axis="x", which="both", top=False)
    for gx, g in groups:
        ax.text(gx, -0.24, g, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5)
    ax.set_xlim(-0.7, x - 0.4)
    ax.set_ylim(0, 265)
    ax.set_ylabel("Number of channels")
    ax.set_title("(d) Tensor-product input channels per layer and irrep")
    ax.legend(handles=[Patch(fc=c_de, ec="black", label="Nequix-MP-1"),
                       Patch(fill=False, ec="black", ls="--", label="1.5× wider, before pruning"),
                       Patch(fc=c_pr, ec="black", label="1.5× wider, after gradual pruning")],
              loc="upper left", ncol=3)

    add_caption(fig, "Figure 3.",
                f"1.5× wider model gradually pruned to Nequix-MP-1 size vs. dense Nequix-MP-1 at equal tensor-product cost "
                f"(MPtrj, epoch 99; {tp['wide_pruned']:,} vs {tp['dense']:,} TP paths, "
                f"{pr['wide_pruned'] / 1e3:.0f}k vs {pr['dense'] / 1e3:.0f}k parameters). (a, b) Whole validation "
                f"split ({err['n_structures']:,} structures); error bars: 95% bootstrap CI over structures; "
                "Δ: paired bootstrap; one seed per model. (c) Energy + forces, batches of 32, one "
                f"{gpu} GPU, mean of 20 rounds, pruned channels physically removed. (d) In brackets: TP paths "
                "per channel (its cost); the last layer outputs scalars only, so its channels feed one path each.")
    return fig, err, inf_ms


# --------------------------------------------------------------------------- fig 4
def fig_scratch(dense, scratch):
    last = max(scratch)
    common = sorted(set(dense) & set(scratch))
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_IN, 4.8), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08, "wspace": 0.28})
    fig.subplots_adjust(top=0.88, bottom=0.22, left=0.09, right=0.98)
    for col, (q, name, unit) in enumerate(QTY):
        ax, axd = axes[0, col], axes[1, col]
        e = sorted(dense)
        ax.plot(e, [dense[i][q] for i in e], color=C_DENSE, ls="--", zorder=3)
        e = sorted(scratch)
        ax.plot(e, [scratch[i][q] for i in e], color=C_SCRATCH, zorder=4)
        ax.plot([last], [scratch[last][q]], "o", color=C_SCRATCH, ms=3.5, zorder=5)
        ax.annotate(f"epoch {last}, {scratch[last][q]:.2f} vs {dense[last][q]:.2f}", (last, scratch[last][q]),
                    xytext=(10, 12), textcoords="offset points", fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", color="0.3", lw=0.6))
        for a in (ax, axd):
            a.axvspan(last + 0.5, LAST, color="0.93", lw=0, zorder=0)
        if q == 0:
            log_axis(ax, (30, 100), [30, 35, 40, 50, 60, 80, 100])
        else:
            log_axis(ax, (9.5, 50), [10, 12, 15, 20, 30, 40, 50])
        ax.set_ylabel(f"Validation {name.lower().replace('mae', 'MAE')} ({unit})")
        ax.set_title(f"({'ab'[col]}) {name.split()[0]}")
        rel = [100 * (scratch[i][q] - dense[i][q]) / dense[i][q] for i in common]
        axd.axhline(0, color="black", lw=0.6)
        axd.plot(common, rel, color=C_SCRATCH)
        m = max(3.0, max(abs(r) for r in rel) * 1.3)
        axd.set_ylim(-m, m)
        axd.set_ylabel("Rel. diff. to\ndense (%)")
        axd.set_xlabel("Epoch")
        axd.set_xlim(0, LAST)
    handles = [Line2D([], [], color=C_DENSE, ls="--", label="Nequix-MP-1 (dense, uniform channels)"),
               Line2D([], [], color=C_SCRATCH, label="Pruned 1.5× architecture, trained from scratch")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.535, 0.995), ncol=2)
    add_caption(fig, "Figure 4.",
                "Pruned 1.5× architecture (channels of Figure 3d) trained from scratch, without wide phase or "
                "pruning, vs. dense Nequix-MP-1: same TP cost (3,232 vs 3,236 paths), same recipe and seed. "
                f"Bottom: (from scratch − dense) / dense. Shaded: epochs not reached yet (run in progress, "
                f"epoch {last} of 99). MPtrj validation split, EMA weights, log scale.")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="bmuon/runs")
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PAPER_STYLE)

    dense = read(runs / DENSE)
    wide = {k: read(runs / n) for k, n in WIDE.items()}
    cfgs = {k: yaml.safe_load(open(runs / n / "config.yml", encoding="utf-8")) for k, n in WIDE.items()}
    scratch = read(runs / SCRATCH)

    f3, err, inf_ms = fig_equal_size(Path(args.bench))
    figs = {"fig1_gradual-pruning-of-wider-models_training-curves": fig_curves(dense, wide, cfgs),
            "fig2_final-error-vs-width": fig_final(dense, wide),
            "fig3_x1.5-pruned-vs-dense_equal-cost": f3,
            "fig4_pruned-architecture-from-scratch": fig_scratch(dense, scratch)}
    with PdfPages(out / "pre-read_figures.pdf") as pdf:
        for name, fig in figs.items():
            fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.03)
            fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.03)
            plt.close(fig)

    def rel(a, b):
        return [100 * (a[q] - b[q]) / b[q] for q in (0, 1)]
    meta = json.load(open(Path(args.bench) / "meta.json"))
    cuts = cfgs[1.5]["gradual"]["prune_after_epochs"]
    per_k = {}
    for k, v in wide.items():
        lv = [0.0] + cfgs[k]["gradual"]["levels"]
        per_k[f"{k:g}"] = {
            "epoch": max(v), "final": v[max(v)], "final_rel": rel(v[max(v)], dense[LAST]),
            "rel_before_first_cut": rel(v[cuts[0]], dense[cuts[0]]),
            "rel_before_last_cut": rel(v[cuts[-1]], dense[cuts[-1]]),
            "jump_last_cut": rel(v[cuts[-1] + 1], v[cuts[-1]]) if cuts[-1] + 1 in v else None,
            "step_fraction": 100 * (lv[1] - lv[0]) / (1 - lv[0]),
            "final_level": lv[-1]}
    summary = {"dense_epoch99": dense[LAST], "cuts": cuts, "per_k": per_k,
               "tp_paths": meta["tp_paths"], "params": meta["params"],
               "n_structures": err["n_structures"],
               "x1.5_bench_ci": {"f_rel": err["f_rel"], "e_rel": err["e_rel"]},
               "wide_latest": {f"{k:g}x": {"epoch": max(v), "force": v[max(v)][0], "energy": v[max(v)][1]}
                               for k, v in wide.items()},
               "scratch_latest": {"epoch": max(scratch), "force": scratch[max(scratch)][0],
                                  "energy": scratch[max(scratch)][1], "dense_same_epoch": dense[max(scratch)],
                                  "rel": rel(scratch[max(scratch)], dense[max(scratch)])},
               "x1.5_bench": {"force_rel": err["f_rel"], "energy_rel": err["e_rel"],
                              "f": [err["f_dense"], err["f_wide"]], "e": [err["e_dense"], err["e_wide"]],
                              "inference_ms": inf_ms}}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
