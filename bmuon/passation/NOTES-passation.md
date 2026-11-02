# Passation — BMuon × Nequix (13 oct. 2026)

Pour le Claude Code qui reprend. Le dépôt sur GitHub est à jour jusqu'au commit `bb79332` (brouillon du papier dans `bmuon/paper/`).

## État des machines
- Environnement GPU déjà installé sur `ablette` dans `/tmp/$USER/nequix-venv`. Il contient jax 0.5.3, e3nn-jax 0.20.8, equinox 0.13.1, optax 0.2.6, lmdb 1.7.5 et les kernels OpenEquivariance (`openequivariance_extjax` 0.7.0), compilés avec une racine CUDA 12 minimale dans `/tmp/$USER/cuda12`.
- Données MPtrj prétraitées dans `/tmp/$USER/bmuon-data/mptrj-aselmdb` (16 fichiers, marqueur `.complete`).
- Toute autre machine s'installe avec `bash bmuon/experiments/setup_machine.sh` (~10 min, idempotent). Il ne faut pas faire un `uv sync` à la main : sans l'étape de la racine CUDA, les kernels ne se compilent pas, parce que le nvcc 13 du système est incomplet.
- Le home est partagé (NFS, quota 30 Go, ~10 Go libres). L'environnement et les données vont dans `/tmp/$USER` sur chaque machine. Le code et `bmuon/runs/` restent sur le home.

## Mesures sur ablette (RTX 3090, modèle complet nequix-mp-1, BMuon budget 0.3)
- Lots de 16 × accumulation 16 : 80 s par époque sur 0,3 % des données, GPU à 72 %, 2,5 Go.
- Lots de 64 × accumulation 4 : 72 s par époque, GPU à 100 %, 8,7 Go. Le GPU est limité par le calcul.
- Run complet : environ 6,7 h par époque, soit environ 670 h de GPU pour 100 époques (environ 28 jours sur une machine).
- La reprise depuis `out_dir/state.pkl` fonctionne.

## Patch `multi-machines.patch` (non commité, s'applique sur `bb79332`)
Pour l'appliquer : `git apply multi-machines.patch`. Il contient :
- `bmuon/train.py` :
  - options `--coordinator hôte:port --num-processes N --process-id i` (appel à `jax.distributed.initialize`) ;
  - données d'entraînement et de validation découpées par process (`shard`) ;
  - même nombre de micro-pas par époque dans tous les process (`(n_train // N) // batch_size`, via `prefetch_n`). Sinon le batching dynamique donne des nombres de lots différents et les collectives bloquent. Conséquence : environ 8 % de lots en moins par époque qu'en mono-machine ;
  - perte et métriques moyennées entre process (`pmean`) ;
  - validation répartie puis sommée (`evaluate_sums` + `all_sum`) ;
  - seul le process 0 écrit les logs, les checkpoints (écriture atomique de `state.pkl`) et `DONE`.
  - En mono-machine, le comportement est inchangé.
- `bmuon/experiments/launch_dist.sh` : `check` / `start` / `status` / `stop` par SSH. Le premier hôte est le process 0. Les logs vont dans `/tmp/$USER/dist-NAME-rank{i}.log`.
- `bmuon/configs/nequix-mp-1-muon.yml` : 64 × 4 au lieu de 16 × 16 (lot global de 256 inchangé).
- `bmuon/README.md` : section sur les commandes multi-machines.

Le patch a été testé avec 2 process sur CPU (gloo, config `toy-bmuon.yml`). Entraînement, validation, reprise et sortie propre fonctionnent, et les 14 tests passent. Deux bugs trouvés pendant ce test sont corrigés dans le patch :
1. Les statistiques du jeu de données (shift, scale, énergies atomiques) étaient calculées sur la part de chaque process, ce qui donnait des modèles différents. Elles sont maintenant calculées avant le découpage. nequix-mp-1 n'était pas touché, car sa config les fournit déjà.
2. Les process ne s'arrêtaient pas à la fin. `jax.distributed` installe un gestionnaire SIGTERM dont héritent les workers du DataLoader (fork), donc le `join` de multiprocessing attendait indéfiniment. Correction : un `atexit` qui fait `kill()` sur les enfants, avec `multiprocessing.util` importé tôt pour que ce hook passe en premier.

Pas encore testé : GPU (NCCL) et réseau réel. L'essai prévu est sur 2 machines :

    bash bmuon/experiments/launch_dist.sh check "ablette anchois"
    bash bmuon/experiments/launch_dist.sh start essai2m bmuon/configs/nequix-mp-1-bmuon.yml "ablette anchois" \
      --set train_subset_frac=0.003 valid_subset_frac=0.01 n_epochs=2 log_every=1 batch_size=64 grad_accum_steps=2
    bash bmuon/experiments/launch_dist.sh status essai2m "ablette anchois"
    grep -o "'time/epoch': [0-9.]*" bmuon/runs/essai2m/log.jsonl     # attendu ~36 s pour la 2e époque (72 s sur 1 machine)

Pour garder le lot global à 256 : 4 machines en 64 × 1, 8 en 32 × 1, 16 en 16 × 1. Au-delà, il faudrait réajuster le taux d'apprentissage.

## En suspens
- **`.gitignore`** : le `.gitignore` racine de Nequix ignore `*.log` et `logs/`, donc les logs dans `bmuon/results` ne partent pas. Il faut les ré-inclure dans `bmuon/.gitignore`, tout en continuant d'ignorer `stdout.log` et `setup.log`.
- **Code Taylor (local, non commité, gardé hors du dépôt à la demande d'Octave)** : `taylor_scores` dans `prune.py`, et l'option `--score taylor` dans `fragility.py`.
- **Papier** : les axes de coût de la Figure 1 ont été calculés avec le modèle de coût « cg ». Il faut les refaire avec « paths » (modèle par défaut du papier).
- **Plan expérimental proposé** :
  - 1 ou 2 runs BMuon complets (100 époques, multi-machines, budgets 30 % et 15 %), comparés au modèle Muon publié ;
  - des balayages sur un calendrier court (~10 époques), Muon et BMuon dans les mêmes conditions, une machine par run.
- **Règles d'Octave** :
  - demander avant tout commit ou push ;
  - commits à son nom (`Octave Koenig <OctaveKNG@users.noreply.github.com>`), sans ligne Co-Authored-By Claude ;
  - tout le code non-Nequix va dans `bmuon/`.
