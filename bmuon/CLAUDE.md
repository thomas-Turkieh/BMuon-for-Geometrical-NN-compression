# Contexte pour Claude Code — cluster Polytechnique & projet BMuon×Nequix

Ce PC (Windows) sert à piloter, **par SSH**, un entraînement distribué sur le parc GPU de
l'École Polytechnique. Le vrai projet n'est pas sur ce PC : il est sur le cluster, dans
`~/nequix`. Lis ce fichier en entier avant d'agir.

## 1. Accès SSH (déjà configuré, sans mot de passe)

- Utilisateur cluster : `octave.koenig`. Adresse d'une machine : `<nom>.polytechnique.fr`
  (ex. `ssh ablette.polytechnique.fr`).
- **PC → cluster** : sans mot de passe. `~/.ssh/config` (Windows) a un bloc
  `Host *.polytechnique.fr` → `IdentityFile ~/.ssh/id_polytechnique`, `IdentitiesOnly yes`,
  `StrictHostKeyChecking accept-new`. La clé publique est déjà dans l'`authorized_keys` du cluster.
- **Machine ↔ machine** : sans mot de passe (maillage complet). Une clé `~/.ssh/id_cluster`
  (sans passphrase) vit dans le home, et le home est **partagé par NFS** → la même clé vaut sur
  toutes les machines. La config SSH du cluster a `Host * !github.com` → `id_cluster`.
- **Home NFS** : `/users/eleves-b/2024/octave.koenig`, monté identique sur toutes les machines.
  Quota **~30 Go, ~10 Go libres** → voir §3.
- Parc : salles 30 à 36, ~169 machines, GPU variés. Salle 34 (noms de poissons : ablette,
  anchois, saumon, thon…) = **RTX 3090 24 Go**. Inventaire complet : `bmuon/experiments/machines.txt`.
- Le message SSH « post-quantum key exchange » est **inoffensif**, l'ignorer.

## 2. Le projet

- Dépôt GitHub `OctaveKNG/bmuon-nequix`, cloné sur le cluster dans **`~/nequix`** (home NFS, donc
  visible depuis toutes les machines). GitHub en SSH via `~/.ssh/id_ed25519_github` (ssh.github.com:443).
- Sujet : **BMuon × Nequix** — étendre Budget Muon (budget de calcul par canal + élagage
  structuré équivariant) aux couches à produit tensoriel du potentiel interatomique **Nequix**
  (en **JAX** : jax, e3nn-jax, equinox, optax, jraph).
- Tout le code du projet est dans **`bmuon/`** ; le reste du dépôt est du Nequix amont non modifié.

## 3. Environnement & données (NE PAS installer à la main)

- Le quota NFS est trop petit → **l'environnement et les données vont dans `/tmp/$USER`** (disque
  local NVMe de chaque machine, ~200 Go libres), **pas** dans le home.
- Env GPU : **`/tmp/$USER/nequix-venv`** (jax 0.5.3, e3nn-jax 0.20.8, equinox 0.13.1, optax 0.2.6,
  lmdb 1.7.5, kernels OpenEquivariance compilés contre une racine CUDA 12 dans `/tmp/$USER/cuda12`).
- Données MPtrj prétraitées : **`/tmp/$USER/bmuon-data/mptrj-aselmdb`** (marqueur `.complete`).
- **Provisionner une machine neuve** : `bash bmuon/experiments/setup_machine.sh` (~10 min,
  idempotent). **Ne jamais faire `uv sync` à la main** : sans l'étape racine CUDA, les kernels
  OEQ ne compilent pas (le nvcc 13 du système est incomplet).
- Activer l'env : `source /tmp/$USER/nequix-venv/bin/activate`.
- Pins critiques : lmdb 3.x casse les workers de données (fork) ; jax ≥ 0.10 supprime
  `jax.device_put_replicated`, utilisé par la boucle d'entraînement.

## 4. Règles d'Octave (impératives)

- **Demander avant tout `git commit` et avant tout `git push`.**
- Commits à son nom : **`Octave Koenig <OctaveKNG@users.noreply.github.com>`**, **sans** ligne
  `Co-Authored-By`. La config git du cluster est vide → passer
  `-c user.name="Octave Koenig" -c user.email="OctaveKNG@users.noreply.github.com"` à chaque commit.
- Tout code non-Nequix va dans **`bmuon/`** ; garder le Nequix amont intact
  (`git diff da0fb24 -- . ':!bmuon'` doit rester vide).
- Les checkpoints (`*.nqx`) et états d'optimiseur (`state.pkl`) **ne sont pas versionnés** ;
  seuls config, `log.jsonl`, `hosts.log` et les `alloc_*.npz` de `bmuon/runs/` le sont.

## 5. Entraînement multi-machines

- Lanceur : `bash bmuon/experiments/launch_dist.sh {check|start|status|stop} ...`. Le **premier
  hôte = process 0** (coordinateur, écrit logs/checkpoints dans `bmuon/runs/NAME` sur le home).
- Données découpées par process, gradients/métriques/validation moyennés (`pmean`), écritures
  réservées au process 0. En mono-machine le comportement est inchangé.
- Garder le lot global à 256 : 2 machines = `batch_size=64 grad_accum_steps=2` ; 4 = 64×1 ;
  8 = 32×1 ; 16 = 16×1. Au-delà, réajuster le learning rate.
- Repères de temps (RTX 3090, nequix-mp-1, 0.3 % des données) : ~72 s/époque sur 1 machine,
  ~36 s attendues sur 2 machines.

## 6. Spécificités Windows / outils

- **PowerShell 5.1 massacre les guillemets doubles** passés à `ssh.exe` → bash distant voit des
  guillemets non fermés. Pour une commande distante complexe : écrire un script en **LF** et le
  piper — `Get-Content -Raw f.sh | ssh hôte bash`. Via l'outil **Bash** (Git Bash) le quoting est normal.
- `gh` (GitHub CLI) n'est pas installé ; utiliser `git`. Le connecteur GitHub de claude.ai
  n'est **pas** accessible depuis Claude Code.

## 7. Pour aller plus loin (à lire sur le cluster)

- `bmuon/passation/NOTES-passation.md` — état détaillé, mesures, essais, points en suspens.
- `bmuon/README.md` — architecture du dossier `bmuon/`, install, tests, commandes types.
- `bmuon/paper/draft.md` — brouillon du papier (méthode, protocoles d'expériences §6).

## 8. État au 2026-10-13

- Accès SSH PC→cluster et maillage machine↔machine : **opérationnels**.
- Patch multi-machines **appliqué et commité** : `24bd6d5` sur `main` (train.py, launch_dist.sh,
  nequix-mp-1-muon.yml, README). **Non poussé.**
- `anchois` en cours de provisionnement ; essai 2 machines `essai2m` à lancer pour valider
  (~36 s à la 2ᵉ époque). Commande de l'essai dans `bmuon/passation/NOTES-passation.md`.
