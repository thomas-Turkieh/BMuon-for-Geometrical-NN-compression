"""Figure: wide 1.5x model gradually pruned to the size of Nequix-MP-1 vs dense Nequix-MP-1.

    python -m bmuon.experiments.equal_size_fig --bench DIR [--out DIR]

Reads the measurements of ``equal_size_bench.py`` (errors.npz, inference.csv, training.csv,
meta.json) and draws four panels: inference cost, training compute, force error, energy error.

Uncertainties
  * errors: 95 % bootstrap confidence intervals over validation structures (10,000 resamples;
    a structure is the resampled unit, since the errors of its atoms are correlated). The
    difference between the models is bootstrapped *paired* (same resampled structures for
    both). One training seed per model: seed-to-seed variation is not included.
  * times: mean and 95 % t-interval over interleaved timing rounds on one GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import textwrap
from scipy import stats

OUT = Path("bmuon/results/2026-11-10_full-mptrj_wide-models-pruned-to-nequix-size")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_DENSE, C_WIDE = "#2a78d6", "#eb6834"
STEPS_PER_EPOCH, GLOBAL_BATCH, N_EPOCHS = 5864, 256, 100


def mean_ci(x, level=0.95):
    x = np.asarray(x, float)
    m = x.mean()
    if len(x) < 2:
        return m, m, m
    h = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def bootstrap(err, n_boot=10_000, seed=0, chunk=250):
    """MAEs of both models and their paired difference, with 95 % percentile intervals."""
    n_atoms = err["n_atoms"].astype(np.float64)
    e = {k: err[f"{k}_e_abs_per_atom"] * 1000 for k in ("dense", "wide_pruned")}
    f = {k: err[f"{k}_f_abs_sum"] * 1000 for k in ("dense", "wide_pruned")}
    n = len(n_atoms)
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in ("e_dense", "e_wide", "f_dense", "f_wide")}
    for start in range(0, n_boot, chunk):
        idx = rng.integers(0, n, size=(min(chunk, n_boot - start), n))
        na = n_atoms[idx].sum(axis=1) * 3
        samples["e_dense"].append(e["dense"][idx].mean(axis=1))
        samples["e_wide"].append(e["wide_pruned"][idx].mean(axis=1))
        samples["f_dense"].append(f["dense"][idx].sum(axis=1) / na)
        samples["f_wide"].append(f["wide_pruned"][idx].sum(axis=1) / na)
    s = {k: np.concatenate(v) for k, v in samples.items()}
    point = {
        "e_dense": e["dense"].mean(), "e_wide": e["wide_pruned"].mean(),
        "f_dense": f["dense"].sum() / (3 * n_atoms.sum()),
        "f_wide": f["wide_pruned"].sum() / (3 * n_atoms.sum()),
    }
    out = {"n_structures": n, "n_atoms": int(n_atoms.sum()), "n_boot": n_boot}
    for q in ("e", "f"):
        for m in ("dense", "wide"):
            lo, hi = np.percentile(s[f"{q}_{m}"], [2.5, 97.5])
            out[f"{q}_{m}"] = (point[f"{q}_{m}"], lo, hi)
        d = s[f"{q}_wide"] - s[f"{q}_dense"]
        rel = 100 * d / s[f"{q}_dense"]
        d0 = point[f"{q}_wide"] - point[f"{q}_dense"]
        out[f"{q}_diff"] = (d0, *np.percentile(d, [2.5, 97.5]))
        out[f"{q}_rel"] = (100 * d0 / point[f"{q}_dense"], *np.percentile(rel, [2.5, 97.5]))
        out[f"{q}_p_better"] = float(np.mean(d < 0))
    return out


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def training_hours(rows, meta):
    """Single-GPU compute of the 100 training epochs, per timing round (hours).

    micro-steps of ``batch_size`` structures per epoch = STEPS_PER_EPOCH * GLOBAL_BATCH / batch.
    """
    by = {}
    for r in rows:
        by.setdefault(r["model"], {})[int(r["round"])] = float(r["ms_per_step"]) / 1000
    rounds = sorted(by["dense"])
    micro = STEPS_PER_EPOCH * GLOBAL_BATCH / meta["batch_size"]
    phases = meta["phases"]
    dense = [N_EPOCHS * micro * by["dense"][r] / 3600 for r in rounds]
    as_run = [N_EPOCHS * micro * by["phase0"][r] / 3600 for r in rounds]
    physical = [sum(p["n_epochs"] * micro * by[p["name"]][r] for p in phases) / 3600 for r in rounds]
    return {"dense": dense, "as_run": as_run, "physical": physical}


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK2)
    ax.tick_params(colors=INK2, labelcolor=INK, length=3)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def bars(ax, labels, values, colors, hatches=None, fmt="{:.2f}", unit=""):
    x = np.arange(len(values))
    for i, (m, lo, hi) in enumerate(values):
        ax.bar(x[i], m, width=0.62, color=colors[i], edgecolor="white" if not hatches or not hatches[i] else colors[i],
               hatch=hatches[i] if hatches else None, fill=not (hatches and hatches[i]), lw=1.4, zorder=2)
        ax.errorbar(x[i], m, yerr=[[m - lo], [hi - m]], color=INK, capsize=4, lw=1.2, zorder=3)
        ax.text(x[i], hi + 0.02 * ax.get_ylim()[1], fmt.format(m) + unit, ha="center", va="bottom",
                color=INK, fontsize=10.5, fontweight="bold")
    ax.set_xticks(x, labels, fontsize=9.5)
    ax.set_xlim(-0.6, len(values) - 0.4)


def dots(ax, labels, values, colors, unit):
    x = np.arange(len(values))
    for i, (m, lo, hi) in enumerate(values):
        ax.errorbar(x[i], m, yerr=[[m - lo], [hi - m]], fmt="o", ms=9, color=colors[i],
                    mec="white", mew=1.5, ecolor=INK, capsize=5, lw=1.4, zorder=3)
        ax.text(x[i] + 0.1, m, f"{m:.2f}", ha="left", va="center", color=INK, fontsize=10.5,
                fontweight="bold")
    ax.set_xticks(x, labels, fontsize=9.5)
    ax.set_xlim(-0.5, len(values) - 0.5)
    lo = min(v[1] for v in values)
    hi = max(v[2] for v in values)
    pad = hi - lo
    ax.set_ylim(lo - 0.5 * pad, hi + 1.0 * pad)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()
    bench, out = Path(args.bench), Path(args.out)
    meta = json.load(open(bench / "meta.json"))
    meta.setdefault("batch_size", 32)
    err = bootstrap(dict(np.load(bench / "errors.npz")))
    inf = read_csv(bench / "inference.csv")
    inf_ms = {m: mean_ci([float(r["ms_per_structure"]) for r in inf if r["model"] == m])
              for m in ("dense", "wide_pruned", "wide_full_masked")}
    tr = training_hours(read_csv(bench / "training.csv"), meta)
    tr_ci = {k: mean_ci(v) for k, v in tr.items()}
    gpu = meta["gpu"]["nvidia_smi"].split(",")[0].replace("NVIDIA ", "").replace(" Generation", "")
    n_inf = int(inf[0]["n_structures"])
    avg_atoms = int(inf[0]["n_atoms"]) / n_inf

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(13.0, 12.6))
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.975, top=0.765, bottom=0.235, wspace=0.30, hspace=0.62)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    tp, pr = meta["tp_paths"], meta["params"]
    lab_d = f"Nequix-MP-1, dense\n{tp['dense']:,} TP paths\n{pr['dense'] / 1e3:.0f} k params"
    lab_w = (f"Wide 1.5×, pruned to\nNequix-MP-1 size\n{tp['wide_pruned']:,} TP paths\n"
             f"{pr['wide_pruned'] / 1e3:.0f} k params")
    title_kw = dict(loc="left", fontweight="bold", color=INK, fontsize=12, pad=26)

    def sub(ax, text):  # one-line subtitle under a panel title
        ax.text(0, 1.035, text, transform=ax.transAxes, ha="left", va="bottom", color=INK2, fontsize=9.5)

    # (a) inference
    ax = axes[0]
    style_axis(ax)
    vals = [inf_ms["dense"], inf_ms["wide_pruned"]]
    ax.set_ylim(0, max(v[2] for v in vals) * 1.22)
    bars(ax, [lab_d, lab_w], vals, [C_DENSE, C_WIDE], fmt="{:.2f}", unit=" ms")
    ax.set_ylabel("Inference time per structure (ms)", color=INK)
    ax.set_title("(a) Inference cost", **title_kw)
    r = inf_ms["wide_pruned"][0] / inf_ms["dense"][0]
    sub(ax, f"Energy + forces, 1× {gpu} · pruned / dense = {r:.2f}×")

    # (b) training compute
    ax = axes[1]
    style_axis(ax)
    vals = [tr_ci["dense"], tr_ci["as_run"], tr_ci["physical"]]
    ax.set_ylim(0, max(v[2] for v in vals) * 1.22)
    bars(ax, ["Nequix-MP-1,\ndense", "Wide 1.5×,\nas trained\n(pruned channels\nmasked)",
              "Wide 1.5×, projected:\nif pruned channels\nwere deleted from\nthe model at each cut"],
         vals, [C_DENSE, C_WIDE, C_WIDE], hatches=[None, None, "////"], fmt="{:.0f}", unit=" GPU-h")
    ax.set_ylabel(f"Training compute (GPU-hours on 1× {gpu})", color=INK)
    ax.set_title("(b) Training cost, 100 epochs", **title_kw)
    sub(ax, "Measured training-step time × number of steps")
    for i, k in enumerate(("as_run", "physical")):
        ax.text(i + 1, 0.05 * ax.get_ylim()[1], f"{tr_ci[k][0] / tr_ci['dense'][0]:.2f}× dense",
                ha="center", va="bottom", color="white" if k == "as_run" else INK,
                fontsize=9.5, fontweight="bold",
                bbox=None if k == "as_run" else dict(fc="white", ec="none", pad=2))

    # (c) forces, (d) energy
    for ax, q, title, unit in [(axes[2], "f", "(c) Force error", "meV/Å"),
                               (axes[3], "e", "(d) Energy error", "meV/atom")]:
        style_axis(ax)
        dots(ax, [lab_d, lab_w], [err[f"{q}_dense"], err[f"{q}_wide"]], [C_DENSE, C_WIDE], unit)
        name = "Force MAE" if q == "f" else "Energy MAE"
        ax.set_ylabel(f"{name} ({unit})", color=INK)
        ax.set_title(title, **title_kw)
        sub(ax, "MPtrj validation split (79k structures) · lower is better")
        d, dlo, dhi = err[f"{q}_diff"]
        rl = err[f"{q}_rel"][0]
        n_b = err["n_boot"]
        k_b = int(round(err[f"{q}_p_better"] * n_b))
        ax.text(0.97, 0.97, f"Pruned − dense: {d:+.2f} {unit} ({rl:+.1f} %)\n"
                f"paired 95 % CI [{dlo:+.2f}, {dhi:+.2f}] {unit}\n"
                f"pruned lower in {k_b:,} / {n_b:,} resamples",
                transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=9,
                linespacing=1.4, bbox=dict(fc="white", ec=GRID, boxstyle="round,pad=0.45"))

    def wrap(t, n):
        return "\n".join(textwrap.fill(par, n) for par in t.split("\n"))

    fig.text(0.075, 0.975, "Wide-then-prune vs. dense at equal tensor-product cost: "
             "a 1.5× wider Nequix gradually\npruned to the size of Nequix-MP-1, against the dense "
             "Nequix-MP-1 (MPtrj, 100 epochs)",
             ha="left", va="top", fontsize=14.5, fontweight="bold", color=INK, linespacing=1.3)
    fig.text(0.075, 0.918, wrap(
        "Both models are trained on MPtrj with the same recipe (BMuon optimizer and hyperparameters, "
        "weight decay 1e-3, 100 epochs, global batch 256, same initialization seed and validation split) "
        "and compared at their final epoch (99, EMA weights). The wide model has tensor products 1.5× "
        "wider; it is pruned 6 times, before epochs 30, 39, 48, 57, 66 and 75, keeping the channels "
        "with the largest BMuon allocation μ (BM-μ), until it has the tensor-product cost of Nequix-MP-1, "
        "then recovers until epoch 99. It is evaluated and timed after the pruned channels are removed "
        "(same predictions as the masked model: max |ΔF| < 2e-5 eV/Å).", 150),
        ha="left", va="top", fontsize=10, color=INK2, linespacing=1.4)
    notes = wrap(
        f"(a) Time for energy + forces (jitted JAX, OpenEquivariance kernels) on {n_inf:,} validation "
        f"structures (mean {avg_atoms:.0f} atoms) in batches of 32, on one {gpu}. Mean and 95 % t-interval "
        "over 20 interleaved timing rounds (the intervals are narrower than the line width). "
        "TP paths: tensor-product paths fed by the kept channels, the cost measure used for pruning. "
        f"The pruned model is {100 * (r - 1):.0f} % slower than the dense one at an equal number of TP "
        "paths, likely because it has 5 % more parameters and uneven channel counts across blocks and "
        "irreps, which the GPU kernels handle less efficiently.\n"
        f"(b) Measured time of one training step (loss, gradients, BMuon update; {meta['batch_size']} "
        f"structures, one {gpu}) × {STEPS_PER_EPOCH * GLOBAL_BATCH // meta['batch_size']:,} steps per "
        "epoch × 100 epochs; data loading, multi-GPU communication and validation are excluded. "
        "Mean and 95 % t-interval over 5 interleaved timing rounds. The actual runs used 16 (dense) and 8 "
        "(wide) GPUs; their wall-clock times are not comparable, so both are measured on the same GPU. "
        "Our implementation masks the pruned channels, so the wide run computed at full width for all 100 "
        "epochs (middle bar). The hatched bar sums the measured step times of the architecture of each "
        "phase between cuts: the cost if the channels were removed at each cut.\n"
        f"(c, d) Mean absolute error on the whole held-out validation split (5 % of MPtrj: "
        f"{err['n_structures']:,} structures, {err['n_atoms']:,} atoms, never used for training). "
        "Force MAE: mean over atoms and Cartesian components. Energy MAE: mean over structures of "
        "|ΔE| / number of atoms. Error bars: 95 % bootstrap confidence intervals over structures "
        f"({err['n_boot']:,} resamples). The difference uses a paired bootstrap (same resampled structures "
        "for both models), which is why it is significant although the two intervals overlap. "
        "One training seed per model: seed-to-seed variation is not included.\n"
        "Runs: mp1-bmuon-selwd (dense), mp1-wide-x1.5-gradual-bmuon-mu (wide).", 178)
    fig.text(0.075, 0.012, notes, ha="left", va="bottom", fontsize=8.6, color=INK2, linespacing=1.45)

    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"equal_size_comparison.{ext}", dpi=250)
    summary = {
        "inference_ms_per_structure": {k: list(v) for k, v in inf_ms.items()},
        "training_gpu_hours": {k: list(v) for k, v in tr_ci.items()},
        "errors": {k: (list(v) if isinstance(v, tuple) else v) for k, v in err.items()},
        "tp_paths": meta["tp_paths"], "params": meta["params"], "gpu": gpu,
    }
    with open(out / "equal_size_comparison.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
