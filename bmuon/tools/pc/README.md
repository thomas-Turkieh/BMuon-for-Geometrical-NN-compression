# PC-side tools

Scripts run from the Windows PC that drives the cluster (Git Bash or PowerShell). The cluster
account is `thomas.turkieh`, the gateway `aerides.polytechnique.fr`; the training code runs from
`~/nequix` on the cluster. Paths below are relative to the repository root.

| Script | What it does |
|---|---|
| `run_status.py` | Epoch, latest validation errors and time since the last log write of the given runs (cluster side, through one ssh). |
| `pull_logs.sh` | Tar of the `config.yml` and validation lines of `log.jsonl` of the given runs, to plot them on the PC. |
| `hosts_check.sh` | Per-host health of runs (processes, GPU memory, out-of-memory errors). One ssh per host: use sparingly. |
| `start_guard.ps1` / `stop_guard.ps1` | Start the night guard detached from any terminal / stop it. |
| `keep_awake_until.ps1` | Keeps the PC awake until a date (the guard needs the PC awake and plugged in). |
| `cluster_export_for_repo.sh` | Tar of finished-run files and final models for the repo (edit the lists first). |
| `model_table.py` | Markdown table of `bmuon/models/` with the validation errors of each model. |

## Common commands (Git Bash)

```bash
cd ~/OneDrive/Documents/EA/code
RUNS="mp1-wide-x1.25-gradual-bmuon-mu mp1-wide-x3-gradual-bmuon-mu mp1-arch-x1.5-pruned-scratch"
GW="ssh -o BatchMode=yes -l thomas.turkieh aerides.polytechnique.fr"

# status of the runs
$GW python3 - $RUNS < bmuon/tools/pc/run_status.py
# guard activity
tail -20 bmuon/results/wide_arch_guard.out

# pull the logs of all runs needed by the figures into a local cache (not versioned)
mkdir -p ~/runs_cache && (cd ~/runs_cache && $GW bash -s -- mp1-bmuon-selwd mp1-bmuon-wd1e3 \
  mp1-wide-x1.25-gradual-bmuon-mu mp1-wide-x1.5-gradual-bmuon-mu mp1-wide-x2-gradual-bmuon-mu \
  mp1-wide-x2.5-gradual-bmuon-mu mp1-wide-x3-gradual-bmuon-mu mp1-arch-x1.5-pruned-scratch \
  < "$OLDPWD/bmuon/tools/pc/pull_logs.sh" | tar -xmf -)

# figures (see bmuon/HANDOFF.md for the list)
python -m bmuon.experiments.wide_fig --runs ~/runs_cache
python -m bmuon.experiments.scratch_vs_dense_fig --runs ~/runs_cache
python -m bmuon.experiments.preread_figs --runs ~/runs_cache \
  --bench bmuon/results/2026-11-10_full-mptrj_wide-models-pruned-to-nequix-size/equal_size_bench
```

## Night guard (PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File bmuon\tools\pc\start_guard.ps1   # refuses if one is already running
powershell -ExecutionPolicy Bypass -File bmuon\tools\pc\stop_guard.ps1
```

Never run two guards on the same plan (for example on two PCs), and never start a run by hand
while a guard is running: stop the guard, start the run, wait for its first log write, restart
the guard.
