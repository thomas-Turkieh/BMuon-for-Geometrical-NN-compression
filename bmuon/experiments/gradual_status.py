"""Status of the gradual-pruning runs (bmuon/experiments/gradual_runs.yml), read from their logs.

    python -m bmuon.experiments.gradual_status [--plan PLAN] [--plot bmuon/results/2026-11-06_full-mptrj_gradual-pruning/training_curves.png]

For every run: epoch reached, current phase and pruning level (channels and cost kept), last
validation errors, the same epoch of the post-training reference run (no gradual pruning), the
minutes per epoch and the expected end. With --plot, the validation force error per epoch of every
run and reference, with the pruning events marked.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

RUNS = Path("bmuon/runs")


def read_log(name: str):
    val, prunes, last_t, step = {}, [], None, 0
    f = RUNS / name / "log.jsonl"
    if not f.exists():
        return val, prunes, None, 0
    for line in open(f):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = max(step, int(d.get("step", 0) or 0))
        if "val/force_mae" in d and "epoch" in d:
            val[int(d["epoch"])] = d
        if "prune/level" in d:
            prunes.append(d)
    last_t = f.stat().st_mtime
    return val, prunes, last_t, step


def phase_of(epoch: int, gradual: dict) -> str:
    ends = [e + 1 for e in gradual["prune_after_epochs"]]
    for i, e in enumerate(ends):
        if epoch < e:
            return f"phase {i + 1}/{len(ends)} (prune at ep. {e})"
    return "recovery"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="bmuon/experiments/gradual_runs.yml")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()
    plan = yaml.safe_load(open(args.plan, encoding="utf-8"))
    now = time.time()
    out = []
    for r in plan["runs"]:
        cfg = yaml.safe_load(open(r["config"], encoding="utf-8"))
        base_f = Path(r["config"]).parent / "base.yml"
        base = yaml.safe_load(open(base_f, encoding="utf-8")) if base_f.exists() else {}
        gradual = {**base.get("gradual", {}), **cfg.get("gradual", {})}  # empty: no pruning (baseline)
        n_epochs = 100
        val, prunes, mtime, step = read_log(r["name"])
        ref_val, _, _, _ = read_log(r["ref"]) if (RUNS / r["ref"]).exists() else ({}, None, None, 0)
        done = (RUNS / r["name"] / "DONE").exists()
        if not val:
            age = f"{(now - mtime) / 60:.0f} min" if mtime else "no log yet"
            out.append(f"- {r['label']} [{r['name']}]: started, step {step}, first epoch not finished "
                       f"(log updated {age} ago)")
            continue
        ep = max(val)
        d = val[ep]
        times = [val[e].get("time/epoch", 0) / 60 for e in sorted(val)[-3:]]
        per_ep = sum(times) / len(times)
        left_h = per_ep * (n_epochs - 1 - ep) / 60
        kept = prunes[-1] if prunes else None
        prune_txt = (f"{kept['prune/level']:.0%} of the cost removed, {kept['prune/kept']}/857 channels kept"
                     if kept else "no pruning yet" if gradual else "no pruning (baseline)")
        ref = ref_val.get(ep)
        ref_txt = (f"ref {r['ref']} at ep. {ep}: F {1000 * ref['val/force_mae']:.1f}, "
                   f"E {1000 * ref['val/energy_mae_per_atom']:.1f}" if ref else f"ref {r['ref']}: no ep. {ep}")
        out.append(
            f"- {r['label']} [{r['name']}]: {'DONE' if done else f'epoch {ep}/{n_epochs - 1} done'}, "
            f"{'finished' if done else phase_of(ep + 1, gradual) if gradual else 'training'}; {prune_txt}\n"
            f"    val F {1000 * d['val/force_mae']:.1f} meV/Å, E {1000 * d['val/energy_mae_per_atom']:.1f} meV/atom "
            f"({ref_txt}); {per_ep:.0f} min/epoch, ~{left_h:.0f} h left; log updated {(now - mtime) / 60:.0f} min ago")
    print(f"{plan.get('title', 'Gradual-pruning runs')}, {time.strftime('%a %d %b %H:%M')}")
    print("\n".join(out))

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # forces (left) and energy (right), validation error per epoch, log scale
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        colors = ["#be123c", "#4b5563", "#059669", "#7c3aed"]
        ref_style = {}   # dense reference run -> color of its dashed line
        logs = {r["name"]: read_log(r["name"]) for r in plan["runs"]}
        for r in plan["runs"]:
            if r["ref"] not in ref_style and (RUNS / r["ref"]).exists():
                ref_style[r["ref"]] = ["#9ca3af", "#c4b5fd"][len(ref_style) % 2]
        ref_logs = {ref: read_log(ref)[0] for ref in ref_style}
        for ax, key, ylab, ticks in [
                (axes[0], "val/force_mae", "Validation force MAE (meV/Å)", [30, 40, 50, 60, 70, 80, 100, 150]),
                (axes[1], "val/energy_mae_per_atom", "Validation energy MAE (meV/atom)", [10, 12, 15, 20, 25, 30, 40, 50])]:
            for r, c in zip(plan["runs"], colors):
                val, prunes, _, _ = logs[r["name"]]
                if val:
                    ax.plot(sorted(val), [1000 * val[e][key] for e in sorted(val)], color=c, lw=2, label=r["label"])
                for p in prunes:
                    ax.axvline(p["epoch_start"], color=c, lw=0.6, ls=":", alpha=0.6)
            for ref, rc in ref_style.items():
                rv = ref_logs[ref]
                wd = ref.rsplit("wd", 1)[-1].replace("1e3", "0.001").replace("1e2", "0.01")
                ax.plot(sorted(rv), [1000 * rv[e][key] for e in sorted(rv)], color=rc, lw=1.4, ls="--",
                        label=f"dense BMuon, wd {wd} (no pruning)")
            ax.set_yscale("log")
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.3, which="both")
        axes[0].set_ylim(30, 150)
        axes[1].set_ylim(10, 50)
        axes[0].legend(fontsize=8)
        fig.suptitle("Gradual pruning on full MPtrj (dotted lines: pruning events, 20/40/60/70/80/90 % of the cost removed)",
                     fontsize=11)
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"plot: {args.plot}")


if __name__ == "__main__":
    main()
