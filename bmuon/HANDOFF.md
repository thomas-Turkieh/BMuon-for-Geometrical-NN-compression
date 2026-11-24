# Handoff: driving the BMuon × Nequix experiments from a new PC

Written on 24 November 2026 for a Claude Code session that takes over this project on a new
computer, a **Mac** (the old one was a Windows PC: Git Bash, PowerShell, WMI; ignore those parts). Read it entirely before acting. `bmuon/CLAUDE.md` describes a collaborator's setup
(another cluster account, other commit rules): it does **not** apply here, this file does.

## 1. Working rules (from the user, mandatory)

- Answer the user in **French**. Code, figures and documents in **English**, labelled professionally.
- **Anonymize**: no person's name in figures, documents or commits' content. Never send the user's email anywhere.
- **Ask before any `git commit` and before any `git push`.** If unsure which GitHub repo to push to, ask, change nothing.
- All non-Nequix code goes in `bmuon/`.
- **No SSH key added to the cluster's `authorized_keys`, no cron job on the cluster.**
- Runs only inside the IT department's window: weekdays from 18:30, Friday from 18:00, everything stopped before 07:25.
- Hosts to avoid: mayenne, ardennes, indre, manche, baudroie (100 Mb/s network), morbihan (GPU errors), atlas, nissan (other users), saone, aerides (gateway only), raie (lent to someone else, never use).
- Never `night.sh stop` on hosts shared with other runs. Keep the SSH volume low (one ssh to the gateway rather than one per host).
- **Never launch a new experiment without the user's approval.** Maintenance of approved runs (restarts, host swaps) is fine.
- Never start a run by hand while the guard is running (it sees missing ranks and kills it): stop the guard, start the run, wait for its first log write, restart the guard.
- Commits to this repo end with a `Co-Authored-By` line for Claude (as in the history).

## 2. Where things are

| What | Where |
|---|---|
| This repository (results, figures, models, configs, PC tools) | Mac: `~/EA/code` (old PC: `%USERPROFILE%\OneDrive\Documents\EA\code`), GitHub `thomas-Turkieh/BMuon-for-Geometrical-NN-compression` (public) |
| Code that actually trains | cluster `~/nequix` (home on NFS, same on every machine): a separate clone of the collaborator's repo with local fixes. Not this GitHub repo. |
| Cluster access | user `thomas.turkieh`, gateway `aerides.polytechnique.fr`; the PC's key in `~/.ssh/id_ed25519` works on every host (shared home) |
| Training data and environment on each GPU host | `/tmp/$USER` (wiped after 10 days; `night.sh provision` reinstalls it from `~/bmuon-env.tar.zst`) |
| Layout convention | `bmuon/{configs,runs,models,results}/<start date>_<data>_<what is tested>/<run id>/`, one README per top folder |
| PC helper scripts | `bmuon/tools/pc/` (see its README) |

## 3. Setting up the new PC

macOS:
Step-by-step setup for Claude Code on the Mac: `bmuon/MAC_SETUP.md`. In short:
1. Homebrew, then `brew install git bash coreutils python@3.11`; MacTeX for the LaTeX pre-read
   (`/Library/TeX/texbin/pdflatex`). The guard needs bash ≥ 4 and GNU `timeout`: macOS ships bash 3.2 and no
   `timeout`, `start_guard.sh` puts the Homebrew versions first in `PATH`.
2. Python environment: `python3.11 -m venv ~/.venvs/ea && source ~/.venvs/ea/bin/activate &&
   pip install numpy matplotlib scipy pyyaml` (old PC: matplotlib 3.10, scipy 1.15, numpy 2.2).
   Activate it before any `python -m bmuon...` command.
3. Repository: `mkdir -p ~/EA && git clone https://github.com/thomas-Turkieh/BMuon-for-Geometrical-NN-compression ~/EA/code`.
   Open `~/EA` as the working folder.
4. SSH: copy `id_ed25519`, `id_ed25519.pub`, `config` and `known_hosts` from the old PC's `.ssh`
   folder into `~/.ssh/` by hand (USB or similar, **never through git**), then
   `chmod 700 ~/.ssh && chmod 600 ~/.ssh/id_ed25519 ~/.ssh/config`. Check with
   `ssh -o BatchMode=yes -l thomas.turkieh aerides.polytechnique.fr hostname`.
5. Claude Code session and memory: they live in `~/.claude/projects/<encoded working folder>/` (session
   `.jsonl` files and the `memory/` folder), copied by hand from the old PC. The memory notes are not
   in this repo (they contain personal information). Some notes describe Windows-only procedures
   (WMI, PowerShell, Git Bash): on the Mac use the `.sh` tools of `bmuon/tools/pc/` instead.
6. Move the guard (section 5): stop it on the old PC, then start it on the Mac with
   `bash bmuon/tools/pc/start_guard.sh`. Keep the Mac **plugged in with the lid open**: `caffeinate`
   (started by the script) prevents idle sleep, not lid-close sleep. A sleeping computer means no
   restarts at night.

## 4. Experiments and their state (24 November 2026, morning)

All runs: MPtrj, BMuon optimizer, learning rate 0.01, weight decay 1e-3, global batch 256, 100 epochs,
seed 0, validation split 5 % (79,020 structures), errors from EMA weights.

**Gradual pruning of wider models** (plan `bmuon/experiments/wide_arch_runs.yml`, configs
`bmuon/configs/2026-11-10_full-mptrj_wide-models-pruned-to-nequix-size/`). Nequix-MP-1 with tensor
products k× wider (`tp_expand: k`), pruned during training with BM-μ (channels ranked by the BMuon
allocation μ, budget in tensor-product paths) in 6 steps after epochs 29, 38, 47, 56, 65, 74, down
to the size of Nequix-MP-1 (3,236 tensor-product paths), then 25 epochs of recovery. Pruned channels
are masked during training (not removed), so each run trains at full width.
Baseline: dense Nequix-MP-1 with the same recipe, run `mp1-bmuon-selwd` (final 34.35 meV/Å,
10.61 meV/atom as logged; exact re-evaluation 34.29 / 10.61).

| Run | State | Force / energy MAE (meV/Å, meV/atom) |
|---|---|---|
| x1.25 | epoch 98 on 23 Nov, should finish the night of 23→24 Nov | 35.58 / 10.96 |
| x1.5 | done | 33.80 / 10.10 (exact: −1.4 % / −4.7 % vs dense, paired bootstrap 95 % CI [−1.7, −1.1] / [−5.3, −4.1]) |
| x2 | done | 34.17 / 10.25 |
| x2.5 | done | 34.31 / 10.25 |
| x3 | epoch 88 on 23 Nov, about two more nights | 35.12 / 10.39 |
| x4, x6 | stopped early (12 Nov), not resumed | |

Final x1.5, x2, x2.5 models (pruned channels physically removed, 3,232 paths) are in
`bmuon/models/2026-11-10_full-mptrj_wide-models-pruned-to-nequix-size/`; load with
`bmuon.model.load_pruned`. Exporter: `bmuon/experiments/export_pruned.py` (run on the cluster with
`JAX_PLATFORMS=cpu`). Export x1.25 and x3 the same way when they finish (ask before pushing).

**Architecture found by pruning the x1.5 model, trained from scratch** (`mp1-arch-x1.5-pruned-scratch`,
same plan, config in `bmuon/configs/2026-11-22_full-mptrj_x1.5-pruned-architecture-from-scratch/`):
epoch 26 of 99 on 23 Nov, about 1 % behind the dense model at the same epoch, about 47 min per
epoch on 8 GPUs, so several more nights.

Earlier experiments (finished, documented in the READMEs of `bmuon/runs` and `bmuon/results`):
pruning-method comparison (20 % of MPtrj), post-training pruning, gradual pruning to 90 % of the cost
removed, small uniform models from scratch, learned vs uniform architecture at 30 % of the cost.
Three gradual-pruning runs stopped at epochs 91 to 96 and the small models c80/c60/c14 stopped early;
no guard restarts them, finish them only if the user asks.

## 5. Running and watching the trainings

- Launcher on the cluster: `bmuon/experiments/night.sh {free|provision|check|start|status|stop|deadline}`
  (run from the PC in Git Bash, it ssh-es to the hosts). `night_rank.sh` restarts a rank locally and
  stops before the end of the window; it refuses to start on a GPU with ≥ 3000 MiB already used.
- **Night guard**: `bmuon/experiments/night_guard.sh <plan>` runs on the PC, checks every 10 min,
  restarts runs that lost a rank (resume from `state.pkl`), restarts a run whose log has not moved
  for 40 min, replaces a host unreachable for 15 min by an idle host of the same GPU type, pauses
  outside the window and exits when every run is DONE. Start/stop on the Mac with
  `bash bmuon/tools/pc/start_guard.sh` / `stop_guard.sh` (background `nohup` process, survives closing
  Claude Code; `.ps1` versions for Windows). Log: `bmuon/results/wide_arch_guard.out`. Only one guard
  per plan, on one computer.
- Status: see `bmuon/tools/pc/README.md` (`run_status.py` through one ssh; `last_write` of a few
  minutes means training; outside the window the runs are stopped on purpose).
- Allow about one hour after 18:30 before concluding that everything runs (host swaps).

Known incidents and what to do:
- Every night at about 00:33 and 06:33 an external SIGTERM kills our processes on part of the hosts:
  the guard restarts them; without a guard they stay dead.
- Continuous "gateway unreachable" in the guard log while a manual ssh works (seen on Windows: Git
  Bash could no longer fork after about 30 h): stop and restart the guard.
- Computer asleep, lid closed, or rebooted for updates: the guard is frozen or dead. After any reboot
  or wake-up, check that the guard process exists (`pgrep -fl night_guard`) and still writes its log.
- Another user takes our GPUs (runs crash out of memory): the guard swaps busy hosts; look for
  `RESOURCE_EXHAUSTED` in `/tmp/$USER/night-<run>-rank*.log` before blaming the network.
- "start refused" because no idle host of the same GPU type exists: edit `runs/<run>/hosts.txt` on
  the cluster with idle hosts of another type that fits in memory, the guard retries.
- Hosts that come back after a network loss keep ghost ranks ("different incarnation" errors):
  kill our processes on those hosts.
- Hosts provisioned more than 10 days ago lose `/tmp`: the guard runs `night.sh provision` before
  each restart.
- After any edit of `night_guard.sh`: `bash -n` it and test the edited line.

## 6. Figures

Style: matplotlib, paper look (serif font, closed axes, inward ticks, viridis for widths, black
dashed line for the dense baseline), English, factual titles (no questions), no person's name,
no jargon abbreviations in labels (write "model size (number of tensor-product paths)", not "TP cost").
Call the method "gradual pruning of wider models", never "wide-then-prune".

| Script | Output | Data |
|---|---|---|
| `bmuon/experiments/preread_figs.py` | the 4 pre-read figures, `summary.json` (all numbers quoted in the pre-read) | logs (`--runs`), `--bench` = `results/2026-11-10_.../equal_size_bench` |
| `bmuon/experiments/wide_fig.py` | training curves of all wide runs, `val_errors_per_epoch.csv` | logs |
| `bmuon/experiments/scratch_vs_dense_fig.py` | from-scratch run vs dense | logs |
| `bmuon/experiments/equal_size_fig.py` | x1.5 vs dense comparison (older, 4 panels) | bench folder |
| `bmuon/experiments/equal_size_bench.py` | measures errors and timings of x1.5 vs dense (GPU, run on the cluster) | snapshots on the cluster |
| `bmuon/experiments/gradual_fig.py`, `fig2.py` | earlier experiments | see their docstrings |

Logs for plotting are pulled into a local cache with `bmuon/tools/pc/pull_logs.sh` (README there).

## 7. Meeting pre-read

A weekly pre-read is sent to the supervisor and the team before the meeting, as one PDF
(the email only says "pre-read attached"). Latest: `bmuon/results/2026-11-24_full-mptrj_meeting-pre-read/`
(`pre-read.tex`, compiled twice with `pdflatex` (MacTeX on the Mac), output renamed `pre-read_<date>.pdf`). Start the next one
in a new dated folder. The figures use Times New Roman (present on macOS and Windows).
To update: pull the logs, run `preread_figs.py`, update the numbers in `pre-read.tex` from
`summary.json`, compile twice, check the page count.
The user's feedback on its form:
- 1 to 2 pages; setup in 2 bullets, then 3 to 4 key results in plain English, then the figures.
- Only results since the last meeting; concise; no takeaways under figures; no bold in key results;
  **no colons anywhere** (text, captions, figures).
- State exactly what is compared. "Same size" means the same number of tensor-product paths, not the
  same training cost (training of the wider model costs more).
- Every number must be checked against the logs or `summary.json`; nothing approximate or invented.

## 8. Open questions for the user

- Longer recovery between pruning steps for the wider models (likely better, but training even heavier).
- Remove pruned channels during training (cheaper training) and prune earlier.
- More seeds for x1.5 and dense; why x1.25 is worse than dense even before pruning.
- Whether to finish the stopped gradual and small-model runs; push the x1.25 and x3 models when done.
