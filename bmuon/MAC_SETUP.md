# Setting up the Mac: instructions for Claude Code

You are Claude Code running on the user's new Mac. The user is moving a research project from an
old Windows PC to this Mac and wants you to do the whole setup. **Talk to the user in French.**
Do every step yourself with your tools; ask the user only for what you cannot do (marked USER).
Report progress briefly after each step. Never print the private SSH key, never put it in git or
send it anywhere, never commit or push without asking the user first.

Inputs:
- `~/Downloads/transfert_mac.zip` (the user copied it from the old PC). It contains `ssh/` (SSH key
  and config for the cluster), `claude_project/` (the previous Claude Code conversations and memory),
  `GUARD_STOPPED.txt` (time the night guard was stopped on the old PC) and a copy of this file.
  If it is elsewhere, ask the user where.
- The project repository on GitHub: `https://github.com/thomas-Turkieh/BMuon-for-Geometrical-NN-compression` (public).

## Step 1. Tools (no Homebrew: the user prefers uv)

1. Apple Command Line Tools (git, a C compiler): `xcode-select -p`. If it fails, run
   `xcode-select --install`, then USER: ask the user to click "Install" in the window that opens
   and to tell you when it is finished (a few minutes). Check again with `xcode-select -p && git --version`.
2. uv: `command -v uv || ls ~/.local/bin/uv`. If missing:
   `curl -LsSf https://astral.sh/uv/install.sh | sh` (installs into `~/.local/bin`, no password).
   Use `~/.local/bin/uv` by full path if `uv` is not on your PATH.
3. Python environment (Python 3.11 managed by uv):
   `uv venv --python 3.11 ~/.venvs/ea && uv pip install --python ~/.venvs/ea/bin/python numpy matplotlib scipy pyyaml`
4. LaTeX: the user installs MacTeX themselves. Check `ls /Library/TeX/texbin/pdflatex`; if missing,
   USER: ask the user whether the MacTeX installation is finished. Use `/Library/TeX/texbin/pdflatex`
   by full path if it is not on your PATH.
5. bash ≥ 4 for the night guard (macOS ships bash 3.2), after Step 2 (the script is in the repo):
   `bash ~/EA/code/bmuon/tools/pc/mac_install_bash.sh` builds GNU bash 5 into `~/.local/bin` (about
   2 min, uses the Command Line Tools). If the build fails, read the log it names and fix it (for
   example try another version: `bash .../mac_install_bash.sh 5.3`). The guard's other need, GNU
   `timeout`, is covered by the stand-in `bmuon/tools/pc/macbin/timeout`; `start_guard.sh` sets all
   of this up by itself.

## Step 2. Repository

`mkdir -p ~/EA && git clone https://github.com/thomas-Turkieh/BMuon-for-Geometrical-NN-compression ~/EA/code`
(if `~/EA/code` already exists, `git -C ~/EA/code pull` instead). Then read `~/EA/code/bmuon/HANDOFF.md`
entirely: it holds the project rules (section 1), the state of the experiments, how the trainings
are run and watched, the figures and the meeting pre-read. Follow it from now on.
Then do Step 1.5 (bash 5).

## Step 3. Unpack the transfer archive

`mkdir -p ~/transfert_mac && ditto -x -k ~/Downloads/transfert_mac.zip ~/transfert_mac`

## Step 4. SSH access to the cluster

1. `mkdir -p ~/.ssh && chmod 700 ~/.ssh`. If `~/.ssh/id_ed25519` or `~/.ssh/config` already exist
   and differ, ask the user before replacing them (back them up with a `.bak` suffix).
2. Copy `~/transfert_mac/ssh/{id_ed25519,id_ed25519.pub,config,known_hosts}` into `~/.ssh/`, then
   `chmod 600 ~/.ssh/id_ed25519 ~/.ssh/config && chmod 644 ~/.ssh/id_ed25519.pub ~/.ssh/known_hosts`.
   If the files contain Windows line endings (`\r`), strip them (`sed -i '' 's/\r$//' FILE`).
3. Test: `ssh -o BatchMode=yes -o ConnectTimeout=15 -l thomas.turkieh aerides.polytechnique.fr hostname`
   must print `aerides`. If it fails with a network error, the Mac may need the school network or VPN:
   tell the user. Never add a key to the cluster's `authorized_keys`.

## Step 5. Previous conversations and memory

Claude Code stores the conversations of a working folder in `~/.claude/projects/<folder path with
every non-alphanumeric character replaced by "-">/`. The working folder on the Mac is `~/EA`:

```bash
P=~/.claude/projects/$(echo "$HOME/EA" | sed 's/[^a-zA-Z0-9]/-/g')
mkdir -p "$P" && cp -R ~/transfert_mac/claude_project/. "$P/"
ls "$P" "$P/memory"
```

This brings the previous conversations (`*.jsonl` and their folders) and the memory (`memory/`,
with `MEMORY.md` as index). Some memory notes describe Windows-only procedures (WMI, PowerShell,
Git Bash, the scratchpad folder): on the Mac they are replaced by the tools of `bmuon/tools/pc/`.

## Step 6. Check that everything works (changes nothing on the cluster)

1. Status of the runs, through one ssh:
   `cd ~/EA/code && ssh -o BatchMode=yes -l thomas.turkieh aerides.polytechnique.fr python3 - mp1-wide-x1.25-gradual-bmuon-mu mp1-wide-x3-gradual-bmuon-mu mp1-arch-x1.5-pruned-scratch < bmuon/tools/pc/run_status.py`
2. Pull the logs into `~/runs_cache` and redraw the pre-read figures in a temporary folder (do not
   overwrite the repository's results):
   ```bash
   mkdir -p ~/runs_cache && cd ~/runs_cache && ssh -o BatchMode=yes -l thomas.turkieh aerides.polytechnique.fr \
     bash -s -- mp1-bmuon-selwd mp1-bmuon-wd1e3 mp1-wide-x1.25-gradual-bmuon-mu mp1-wide-x1.5-gradual-bmuon-mu \
     mp1-wide-x2-gradual-bmuon-mu mp1-wide-x2.5-gradual-bmuon-mu mp1-wide-x3-gradual-bmuon-mu \
     mp1-arch-x1.5-pruned-scratch < ~/EA/code/bmuon/tools/pc/pull_logs.sh | tar -xmf -
   cd ~/EA/code && ~/.venvs/ea/bin/python -m bmuon.experiments.preread_figs --runs ~/runs_cache \
     --bench bmuon/results/2026-11-10_full-mptrj_wide-models-pruned-to-nequix-size/equal_size_bench --out /tmp/preread_test
   cp bmuon/results/2026-11-24_full-mptrj_meeting-pre-read/pre-read.tex /tmp/preread_test/ && cd /tmp/preread_test && \
     /Library/TeX/texbin/pdflatex -interaction=nonstopmode pre-read.tex > /dev/null && \
     /Library/TeX/texbin/pdflatex -interaction=nonstopmode pre-read.tex | grep "Output written"
   ```
3. Report the run status to the user (epoch, errors, minutes since last log write; outside 18:30–07:25
   on weekdays the runs are stopped on purpose).

## Step 7. Move the night guard to the Mac

The guard restarts the trainings every evening and after crashes. Exactly one guard may run.
1. Read `~/transfert_mac/GUARD_STOPPED.txt`. If it is missing, USER: ask the user to confirm the
   guard is stopped on the old PC before going on.
2. `cd ~/EA/code && bash bmuon/tools/pc/start_guard.sh` and check it printed "guard started" with
   a bash 5 version and the python of `~/.venvs/ea` (`pgrep -fl night_guard` shows it). It writes to
   `bmuon/results/wide_arch_guard.out`; its first lines must say "guard started, plan ..." and,
   inside the window, never "every run of the plan is DONE" while runs are unfinished (that would
   mean the plan could not be read: stop the guard and fix it).
3. Tell the user: keep the Mac plugged in, lid open, every night (the script prevents idle sleep,
   not lid-close sleep). After a reboot, the guard must be started again (same command).
4. If started before 18:30, the guard waits for the window; around 19:30, check the end of
   `bmuon/results/wide_arch_guard.out` and the run status again.

## Step 8. Clean up and hand over

1. Delete the archive and its unpacked copy, which contain the private key:
   `rm -rf ~/Downloads/transfert_mac.zip ~/transfert_mac`. Remind the user to delete the copy on
   the USB stick.
2. Optional, USER: to let Claude push to GitHub later, install GitHub CLI without Homebrew (macOS
   zip of the latest release from https://github.com/cli/cli/releases, its `bin/gh` copied into
   `~/.local/bin`), then the user runs `gh auth login` in the Terminal app.
3. Tell the user, in French: "Ferme VS Code, rouvre-le sur le dossier EA (File > Open Folder > ~/EA),
   ouvre Claude Code, va dans l'historique des conversations et reprends la conversation sur le
   projet BMuon. Premier message : « On est sur le Mac, relis bmuon/HANDOFF.md et bmuon/MAC_SETUP.md »."
   If that conversation does not appear or does not open, a new conversation in `~/EA` should start
   with: "Lis code/bmuon/HANDOFF.md en entier et la mémoire, respecte les règles de la section 1,
   puis fais-moi un point sur les runs et la garde sans rien lancer ni modifier."
4. Give the user a short summary: what was installed, ssh test result, run status, guard status,
   and anything left for them to do.
