"""Meeting figure from bmuon/results/fig2_octave/fig2_points.csv (see fig2_octave.yml).

    python -m bmuon.experiments.fig2_octave_plot bmuon/results/fig2_octave

Left: the paper's Figure 2 question (Muon vs BMuon, pruned by magnitude or by mu, no
fine-tuning), with the small uniform models trained from scratch as single points.
Right: BM-mu pruning for the three weight decays (what makes unfunded channels die).
Writes fig2_meeting.png (forces), fig2_meeting_energy.png and fig2_meeting_table.md.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MEET = Path("bmuon/results/meeting")
GRAY, GREEN, RED = "#6b7280", "#059669", "#be123c"
REDS = {"wd 0.001": "#fda4af", "selective wd": "#f43f5e", "wd 0.01": "#9f1239"}


def load(out: Path):
    """Every cached curve (one csv per model and method, written by fig2.py)."""
    rows = []
    for f in sorted(out.glob("cache_n*/*.csv")):
        rows += list(csv.DictReader(open(f, encoding="utf-8")))
    for r in rows:
        for k in ("level", "kept_channels", "energy_mae_mev_atom", "force_mae_mev_A"):
            r[k] = float(r[k])
    return rows


def series(rows, run, method):
    p = sorted([r for r in rows if r["run"] == run and r["method"] == method], key=lambda r: r["level"])
    return p


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "bmuon/results/fig2_octave")
    rows = load(out)
    full_channels = max(r["kept_channels"] for r in rows if r["level"] == 0)
    uniform = []
    for run, wd in (("ref-uniform-wd1e3", "wd 0.001"), ("ref-uniform-wd1e2", "wd 0.01")):
        p = series(rows, run, "mag")
        if p:
            uniform.append((wd, 100 * (1 - p[0]["kept_channels"] / full_channels), p[0]))

    for key, name, fname in [("force_mae_mev_A", "Force MAE (meV/Å)", "pruning_forces.png"),
                             ("energy_mae_mev_atom", "Energy MAE (meV/atom)", "pruning_energy.png")]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for run, method, label, color, marker, ls in [
            ("ref-nequix-mp-1", "mag", "Muon + magnitude pruning (Nequix-MP-1)", GRAY, "o", "--"),
            ("ref-bmuon-wd1e2", "mag", "BMuon + magnitude pruning (BM-mag)", GREEN, "D", "-"),
            ("ref-bmuon-wd1e2", "alloc", "BMuon + allocation (μ) pruning (BM-μ)", RED, "s", "-"),
        ]:
            p = series(rows, run, method)
            if p:
                ax1.plot([100 * r["level"] for r in p], [r[key] for r in p], color=color, marker=marker,
                         ls=ls, lw=2, ms=7, label=label, markerfacecolor="white" if ls == "--" else color)
        for wd, x, r in uniform:
            ax1.plot([x], [r[key]], marker="*", ms=16, ls="none", color="black" if wd == "wd 0.01" else "#9ca3af",
                     markeredgecolor="black", label=f"Small uniform model trained from scratch ({wd})")
        ax1.axvline(38, color=RED, lw=1, ls=":")
        ax1.text(38.8, 0.03, "38 % of channels switched\noff by BMuon (μ < 0.01)", transform=ax1.get_xaxis_transform(),
                 va="bottom", fontsize=8, color="#4b5563")
        ax1.set_title("Post-training compression, no fine-tuning (≈ paper Fig. 2)", fontsize=10)

        for run, wd in (("ref-bmuon-wd1e3", "wd 0.001"), ("ref-bmuon-selwd", "selective wd"),
                        ("ref-bmuon-wd1e2", "wd 0.01")):
            for method, ls, marker in (("alloc", "-", "s"),):
                p = series(rows, run, method)
                if p:
                    ax2.plot([100 * r["level"] for r in p], [r[key] for r in p], color=REDS[wd], ls=ls,
                             marker=marker, ms=5, lw=1.8,
                             label=f"{'BM-μ' if method == 'alloc' else 'BM-mag'}, {wd}")
        p = series(rows, "ref-nequix-mp-1", "mag")
        if p:
            ax2.plot([100 * r["level"] for r in p], [r[key] for r in p], color=GRAY, ls="--", lw=1.2,
                     alpha=0.5, label="Muon (reference)")
        for x, wd in ((38, "wd 0.01"), (18, "wd 0.001 / selective")):
            ax2.axvline(x, color=REDS["wd 0.01" if x == 38 else "selective wd"], lw=1, ls=":")
            ax2.text(x + 0.8, 0.03, f"{x} % switched off\n({wd})", transform=ax2.get_xaxis_transform(),
                     va="bottom", fontsize=8, color="#4b5563")
        ax2.set_title("Weight decay switches off the channels BMuon stops funding", fontsize=10)

        for ax in (ax1, ax2):
            ax.set_yscale("log")
            ax.set_xlabel("Type-I channels removed (%)")
            ax.set_xticks([0, 20, 40, 60, 70, 80, 90, 95])
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=8, loc="upper left")
        ax1.set_ylabel(name + " (lower is better)")
        fig.suptitle("Nequix-MP-1 on MPtrj: error after channel pruning, 1000 validation structures",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(MEET / fname, dpi=160)
        plt.close(fig)

    # table of the key numbers
    lines = ["| model | pruning | 0 % | 50 % | 70 % | 80 % | 90 % |", "|---|---|---|---|---|---|---|"]
    names = {"ref-nequix-mp-1": "Muon (Nequix-MP-1)", "ref-bmuon-wd1e2": "BMuon wd 0.01",
             "ref-bmuon-selwd": "BMuon selective wd", "ref-bmuon-wd1e3": "BMuon wd 0.001"}
    for run, nm in names.items():
        for method in ("mag", "alloc"):
            p = {round(r["level"], 2): r for r in series(rows, run, method)}
            if p:
                cells = [f"{p[lv]['force_mae_mev_A']:.0f}" if lv in p else "—" for lv in (0.0, 0.5, 0.7, 0.8, 0.9)]
                lines.append(f"| {nm} | {'μ' if method == 'alloc' else 'magnitude'} | " + " | ".join(cells) + " |")
    for wd, x, r in uniform:
        lines.append(f"| Small uniform model {wd} | trained small ({x:.0f} % fewer channels) | "
                     f"{r['force_mae_mev_A']:.0f} | | | | |")
    (MEET / "table_forces.md").write_text("Force MAE (meV/Å)\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
