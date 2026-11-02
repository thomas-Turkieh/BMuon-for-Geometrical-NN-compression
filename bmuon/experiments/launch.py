"""Spread training runs over the lab machines by SSH.

    python -m bmuon.experiments.launch plan   bmuon/experiments/runs.yml bmuon/experiments/machines.txt
    python -m bmuon.experiments.launch start  bmuon/experiments/runs.yml bmuon/experiments/machines.txt
    python -m bmuon.experiments.launch status bmuon/experiments/runs.yml

machines.txt: one host per line, followed by a GPU tag, e.g. ``ablette 3090``
(lines starting with # are ignored). runs.yml:

    repo: ~/nequix                  # same path on every machine (shared home)
    data_src: ~/data/bmuon          # optional, copied once to /tmp on each machine
    python: ~/venvs/nequix/bin/python
    runs:
      - name: mp1-muon-s0
        config: bmuon/configs/nequix-mp-1-muon.yml
        set: ["seed=0"]
        gpu: [3090, a5000]          # allowed GPU tags, in order of preference
      - grid:                       # expands to one run per combination
          name: mp1-bmuon-b{budget}-s{seed}
          config: bmuon/configs/nequix-mp-1-bmuon.yml
          set: ["seed={seed}", "bmuon.budget={budget}"]
          gpu: [3090]
          values: {budget: [0.2, 0.3, 0.5, 0.7], seed: [0, 1, 2]}

Each run gets one machine; jobs live in bmuon/runs/<name> and resume on their own
(bmuon/experiments/run_forever.sh). Assignments are stored in bmuon/runs/assignments.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shlex
import subprocess
from pathlib import Path

import yaml

ASSIGN = Path("bmuon/runs/assignments.json")


def expand_runs(spec: dict) -> list:
    runs = []
    for r in spec["runs"]:
        if "grid" not in r:
            runs.append(r)
            continue
        g = r["grid"]
        keys = list(g["values"])
        for combo in itertools.product(*(g["values"][k] for k in keys)):
            v = dict(zip(keys, combo))
            runs.append({
                "name": g["name"].format(**v),
                "config": g["config"].format(**v),
                "set": [s.format(**v) for s in g.get("set", [])],
                "gpu": g.get("gpu", []),
            })
    names = [r["name"] for r in runs]
    if len(names) != len(set(names)):
        raise ValueError("duplicate run names")
    return runs


def read_machines(path: str) -> list:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            host, *tag = line.split()
            out.append((host, tag[0].lower() if tag else "any"))
    return out


def plan(runs: list, machines: list, taken: dict) -> dict:
    busy = set(taken.values())
    assignment = {}
    for r in runs:
        if r["name"] in taken:
            continue
        prefs = [str(g).lower() for g in r.get("gpu", [])] or ["any"]
        for pref in prefs:
            host = next((h for h, tag in machines if h not in busy and (pref in (tag, "any"))), None)
            if host:
                assignment[r["name"]] = host
                busy.add(host)
                break
        else:
            print(f"no free machine for {r['name']} ({prefs})")
    return assignment


def remote_command(spec: dict, run: dict) -> str:
    repo = spec.get("repo", "~/nequix")
    env = []
    if spec.get("data_src"):
        env.append(f"BMUON_DATA_SRC={spec['data_src']}")
    if spec.get("python"):
        env.append(f"BMUON_PYTHON={spec['python']}")
    args = " ".join(shlex.quote(s) for s in run.get("set", []))
    out = f"bmuon/runs/{run['name']}"
    job = (f"{' '.join(env)} nohup bash bmuon/experiments/run_forever.sh {out} "
           f"{shlex.quote(run['config'])} {args} > /dev/null 2>&1 &")
    return f"cd {repo} && mkdir -p {out} && {job}"


def ssh(host: str, command: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command],
        capture_output=True, text=True, timeout=timeout,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["plan", "start", "status"])
    p.add_argument("runs")
    p.add_argument("machines", nargs="?")
    p.add_argument("--only", nargs="*", help="restrict to these run names")
    args = p.parse_args()

    spec = yaml.safe_load(Path(args.runs).read_text())
    runs = expand_runs(spec)
    if args.only:
        runs = [r for r in runs if r["name"] in set(args.only)]
    taken = json.loads(ASSIGN.read_text()) if ASSIGN.exists() else {}

    if args.action == "status":
        for r in runs:
            out = Path("runs") / r["name"]
            state = "done" if (out / "DONE").exists() else ("running" if out.exists() else "pending")
            last = ""
            log = out / "log.jsonl"
            if log.exists():
                lines = log.read_text().splitlines()
                if lines:
                    d = json.loads(lines[-1])
                    last = (f"step {d.get('step')} epoch {d.get('epoch', '')} "
                            f"val_force {d.get('val/force_mae', '')}")
            print(f"{r['name']:40s} {taken.get(r['name'], '-'):12s} {state:8s} {last}")
        return

    machines = read_machines(args.machines)
    assignment = plan(runs, machines, taken)
    for name, host in assignment.items():
        print(f"{name:40s} -> {host}")
    if args.action == "plan":
        return

    by_name = {r["name"]: r for r in runs}
    for name, host in assignment.items():
        res = ssh(host, remote_command(spec, by_name[name]))
        if res.returncode == 0:
            taken[name] = host
        else:
            print(f"failed on {host}: {res.stderr.strip()[:200]}")
    ASSIGN.parent.mkdir(parents=True, exist_ok=True)
    ASSIGN.write_text(json.dumps(taken, indent=2))


if __name__ == "__main__":
    main()
