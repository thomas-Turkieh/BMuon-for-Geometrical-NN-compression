"""Validation curves of the group's October runs (bmuon/results/octave_logs/val.csv).

    python bmuon/experiments/octave_training_plot.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d = Path("bmuon/results/octave_logs")
Path("bmuon/results/meeting").mkdir(parents=True, exist_ok=True)
runs = defaultdict(list)
for row in csv.reader(open(d / "val.csv")):
    name, ep, f, e = row[0], int(row[1]), float(row[2]), float(row[3])
    runs[name].append((ep, 1000 * f, 1000 * e))

STY = {
    "mp1-bmuon-selwd": ("BMuon, budget 30 %, selective wd", "#be123c", "-"),
    "mp1-bmuon-wd1e2": ("BMuon, budget 30 %, wd 0.01", "#9f1239", "-"),
    "mp1-bmuon-wd1e3": ("BMuon, budget 30 %, wd 0.001", "#fda4af", "-"),
    "mp1-uniform-shrink-wd1e3": ("Small uniform model (same 30 % cost), wd 0.001", "#374151", "--"),
    "mp1-uniform-shrink-wd1e2": ("Small uniform model (same 30 % cost), wd 0.01", "#9ca3af", "--"),
}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for k, (ax, name) in enumerate(zip(axes, ["Force MAE (meV/Å)", "Energy MAE (meV/atom)"])):
    for run, (label, color, ls) in STY.items():
        if run not in runs:
            continue
        p = sorted(runs[run])
        p = [q for q in p if q[0] < 100]
        ax.plot([q[0] for q in p], [q[1 + k] for q in p], color=color, ls=ls, lw=2, label=label)
        ax.annotate(f"{p[-1][1 + k]:.1f}", (p[-1][0], p[-1][1 + k]), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, va="center", color="#111827")
    ax.axhline(32.2 if k == 0 else 9.43, color="#6b7280", lw=1, ls=":")
    ax.text(1, 32.2 if k == 0 else 9.43, " published Nequix-MP-1 (100 % cost, Muon)", fontsize=8,
            va="bottom", color="#4b5563")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(name + " — validation")
    ax.grid(alpha=0.3, which="both")
    ax.set_ylim(top=(150 if k == 0 else 80))
axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Training curves on the full MPtrj: BMuon (30 % cost budget) vs small uniform model of the same cost",
             fontsize=11)
fig.tight_layout()
fig.savefig(Path("bmuon/results/meeting") / "training_curves.png", dpi=160)
print("wrote", Path("bmuon/results/meeting") / "training_curves.png")
