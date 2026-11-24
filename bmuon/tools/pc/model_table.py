"""Print the markdown table of bmuon/models/ (validation errors read from each run's log.jsonl)."""
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])  # .../bmuon


def run_of(exp, f):
    if exp.startswith("2026-10-13"):
        m = {"bmuon-budget30_wd1e-2": "mp1-bmuon-wd1e2", "bmuon-budget30_wd1e-3": "mp1-bmuon-wd1e3",
             "bmuon-budget30_selective-wd_": "mp1-bmuon-selwd", "bmuon-budget30_selective-wd-then": "mp1-selwd-late-wd0p1",
             "uniform-small-cost35_wd1e-2": "mp1-uniform-shrink-wd1e2", "uniform-small-cost35_wd1e-3": "mp1-uniform-shrink-wd1e3"}
        return next(v for k, v in m.items() if f.startswith(k))
    if exp.startswith("2026-11-03"):
        if f.startswith("gradual-"):
            return "fig2-" + f.split("_")[0]
        return {"bmuon": "fig2-bmuon", "muon": "fig2-muon", "muon-group-lasso-2": "fig2-muon-gl-2",
                "numuon": "fig2-numuon"}[f.split("_")[0]]
    if exp.startswith("2026-11-06"):
        b, w = f.split("_")[:2]
        return f"mp1-{b}-{w.replace('wd1e-', 'wd1e')}"
    return "mp1-uniform-c20-wd1e3"


for exp in sorted(p for p in (root / "models").iterdir() if p.is_dir()):
    print(f"\n### {exp.name}\n")
    print("| Model | Training run | Epoch | Val force MAE (meV/Å) | Val energy MAE (meV/atom) |")
    print("|---|---|---|---|---|")
    for f in sorted(exp.glob("*.nqx")):
        run = run_of(exp.name, f.name)
        ep = int(re.search(r"epoch(\d{3})", f.name).group(1))
        log = next((root / "runs").glob(f"*/{run}/log.jsonl"))
        val = {}
        for line in open(log):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "val/force_mae" in d and "epoch" in d:
                val[int(d["epoch"])] = d
        d = val.get(ep)
        fe = f"{1000 * d['val/force_mae']:.1f} | {1000 * d['val/energy_mae_per_atom']:.1f}" if d else "– | –"
        alloc = " (+ `_alloc.npz`)" if f.with_name(f.stem + "_alloc.npz").exists() else ""
        print(f"| `{f.name}`{alloc} | `{run}` | {ep} | {fe} |")
