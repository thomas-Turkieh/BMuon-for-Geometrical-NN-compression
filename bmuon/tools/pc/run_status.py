"""Cluster-side status of runs, run from the PC (Git Bash) through the gateway:

    ssh -o BatchMode=yes -l thomas.turkieh aerides.polytechnique.fr python3 - RUN1 RUN2 ... \
        < bmuon/tools/pc/run_status.py

For each run: last validated epoch, force / energy MAE (meV), minutes since log.jsonl was last
written (a few minutes = training; hours = stopped), DONE flag.
"""
import json
import os
import sys
import time

os.chdir(os.path.expanduser("~/nequix/bmuon/runs"))
print("now", time.strftime("%a %H:%M"))
for r in sys.argv[1:]:
    log = os.path.join(r, "log.jsonl")
    val = [json.loads(x) for x in open(log) if "val/force_mae" in x]
    d = val[-1] if val else {}
    age = (time.time() - os.path.getmtime(log)) / 60
    print("%-36s epoch=%-3s F=%.2f E=%.2f last_write=%.0f min ago done=%s" % (
        r, d.get("epoch"), 1000 * d.get("val/force_mae", float("nan")),
        1000 * d.get("val/energy_mae_per_atom", float("nan")), age, os.path.exists(os.path.join(r, "DONE"))))
