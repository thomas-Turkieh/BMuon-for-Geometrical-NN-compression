# Registre des runs — BMuon × Nequix

> **Convention de nommage :** `mp1` = Nequix-MP-1 (le modèle de référence de la section 5 du papier).
> Ajouter une entrée dès qu'un run démarre ; mettre à jour `status` et `résultat` quand il se termine.

---

## Runs actifs (oct 2026)

### `mp1-bmuon-wd1e2` — BMuon référence, WD uniforme 0.01
- **Config :** `bmuon/configs/nequix-mp-1-bmuon.yml`
- **Modèle :** Nequix-MP-1 plein — `128x0e + 64x1o + 32x2e + 32x3o`
- **Optimizer :** BMuon · budget=0.3 · anneal_fraction=0.25 · cost_model="paths"
- **weight_decay :** 0.01 uniforme sur tous les canaux budgétés
- **Machines :** 16 × RTX 3090 24 Go — poissons (ablette, anchois, …, rouget) · salle 34
- **Coordinateur (rank 0) :** ablette · logs NFS : `bmuon/runs/mp1-bmuon-wd1e2/`
- **Lancé :** nuit 13 oct 2026
- **But :** run principal de la thèse — mesurer l'allocation apprise par BMuon et comparer
  la performance au modèle shrink uniforme de même coût total (B=0.3 paths).
- **Ce qui le distingue :** seul run avec BMuon *et* WD=0.01. Hypothèse à tester : ce WD élevé
  crée-t-il un "plafond de verre" qui empêche les canaux actifs d'atteindre leur plein potentiel ?
- **Statut :** en cours — 10 époques terminées au 14/10/2026

---

### `mp1-uniform-shrink-wd1e2` — Baseline uniforme, WD 0.01
- **Config :** `bmuon/configs/nequix-mp-1-uniform.yml`
- **Modèle :** shrink uniforme à 30 % du coût paths — `40x0e + 16x1o + 8x2e + 8x3o`
  (même coût total que BMuon budget=0.3 → comparaison équitable)
- **Optimizer :** Muon standard (pas de budget, pas d'allocation)
- **weight_decay :** 0.01 — identique à mp1-bmuon-wd1e2
- **Machines :** 16 × oiseaux (albatros, autruche, …, ombrette)
- **Coordinateur :** albatros
- **Lancé :** nuit 13 oct 2026
- **But :** baseline principale — à iso-coût de calcul, est-ce que BMuon fait mieux qu'une
  réduction uniforme naïve ? Même WD que le run BMuon pour isoler l'effet du budget.
- **Statut :** en cours

---

### `mp1-uniform-shrink-wd1e3` — Baseline uniforme, WD 0.001 (défaut Nequix)
- **Config :** `bmuon/configs/nequix-mp-1-uniform-wd1e3.yml`
- **Modèle :** `40x0e + 16x1o + 8x2e + 8x3o` — identique à uniform-wd1e2
- **Optimizer :** Muon standard
- **weight_decay :** 0.001 — valeur par défaut de Nequix (×10 moins fort que wd1e2)
- **Machines :** 16 × anatomy (acromion, apophyse, …, perone) · RTX 4000 Ada 20 Go
- **Coordinateur :** acromion
- **Lancé :** nuit 14 oct 2026 (relancé après crash de 3 machines mal provisionnées)
- **But :** vérifier si WD=0.01 pénalise les runs wd1e2 (uniform et BMuon) par rapport au
  défaut Nequix. Si uniform-wd1e3 > uniform-wd1e2, alors WD=0.01 est bien un handicap.
- **Ce qui le distingue :** seul WD à differer entre wd1e2 et wd1e3 — tout le reste est identique.
- **Statut :** en cours — époque 1 en cours au 14/10/2026

---

### `mp1-bmuon-selwd` — BMuon avec weight decay sélectif
- **Config :** `bmuon/configs/nequix-mp-1-bmuon-selwd.yml`
- **Modèle :** Nequix-MP-1 plein — `128x0e + 64x1o + 32x2e + 32x3o`
- **Optimizer :** BMuon · budget=0.3 · anneal_fraction=0.25
- **weight_decay :** 0.001 (wd_base) + WD sélectif : λ(μ_c) = 0.001 + 0.049 · max(0, 1−μ_c/μ̄)²
  → canaux actifs (μ_c ≈ 1) : WD ≈ 0.001 ; canaux mourants (μ_c → 0) : WD → 0.050
- **Machines :** 16 × voitures (bentley, bugatti, …, pontiac) · RTX 4000 Ada 20 Go
- **Coordinateur :** bentley
- **Lancé :** nuit 14 oct 2026
- **But :** supprimer le "plafond de verre" hypothétique de wd1e2 — les canaux actifs ne sont
  plus pénalisés au-delà du défaut Nequix, mais les mourants sont poussés plus fort.
  Implémenté dans `bmuon/optimizer.py` (`scale_by_bmuon`, paramètre `selective_wd`).
- **Ce qui le distingue :** WD n'est pas uniforme — dépend de l'allocation μ_c courante.
  Nouveau mécanisme, pas encore dans la littérature. À comparer à mp1-bmuon-wd1e2.
- **Statut :** en cours — époque 1 en cours au 14/10/2026

---

### `mp1-bmuon-wd1e3` — BMuon, WD uniforme 0.001 (défaut Nequix)
- **Config :** `bmuon/configs/nequix-mp-1-bmuon.yml`, `--set weight_decay=0.001` (tout le reste identique à `mp1-bmuon-wd1e2`)
- **Modèle :** Nequix-MP-1 plein — `128x0e + 64x1o + 32x2e + 32x3o`
- **Optimizer :** BMuon · budget=0.3 · anneal_fraction=0.25 · cost_model="paths"
- **weight_decay :** 0.001 uniforme (valeur par défaut de Nequix)
- **Machines :** 16 × RTX 4000 Ada 20 Go (manche, marne, tibia, morbihan, …, temporal) · salles 33/35 ;
  `mayenne` (carte réseau à 100 Mb/s) remplacée par `tibia` à la relance du 16/10 à 18h49
- **Coordinateur :** manche · logs NFS : `bmuon/runs/mp1-bmuon-wd1e3/`
- **Lancé :** vendredi 16 oct 2026, 18h06, départ à zéro (week-end DSI)
- **But :** compléter le carré 2×2 {BMuon, Muon uniforme} × {wd 1e-2, wd 1e-3} : même BMuon que
  `mp1-bmuon-wd1e2`, avec le WD par défaut de Nequix. Permet de séparer l'effet de l'optimiseur de celui du WD.
- **Ce qui le distingue :** seul WD à différer de `mp1-bmuon-wd1e2` ; seul optimiseur à différer de `mp1-uniform-shrink-wd1e3`.
- **Statut :** en cours — démarré le 16/10/2026 (~55 min/époque mesurés au démarrage)

---

## Runs terminés / archivés

### `essai`, `essai1m`, `essai2m`, `essai16m`, `essai64` — Essais techniques
- Tests de fonctionnement du launcher distribué et de la scalabilité (1, 2, 16 machines).
- `essai2m` : validation ~36 s/époque sur 2 × RTX 3090 (conforme aux repères CLAUDE.md).
- Pas de résultats scientifiques — infrastructure uniquement.

---

## Matrice de comparaison

| Run | Modèle | Optimizer | WD | WD type | GPU |
|-----|--------|-----------|-----|---------|-----|
| mp1-bmuon-wd1e2 | plein (B=1.0) | BMuon B=0.3 | 0.01 | uniforme | 3090 (ép. 0-17) puis 4000 Ada |
| mp1-bmuon-wd1e3 | plein (B=1.0) | BMuon B=0.3 | 0.001 | uniforme | 4000 Ada |
| mp1-uniform-shrink-wd1e2 | shrunk (B≈0.3) | Muon | 0.01 | uniforme | A5000 |
| mp1-uniform-shrink-wd1e3 | shrunk (B≈0.3) | Muon | 0.001 | uniforme | 4000 Ada |
| mp1-bmuon-selwd | plein (B=1.0) | BMuon B=0.3 | 0.001+0.049 | **sélectif** | 4000 Ada |
| mp1-selwd-late-wd0p1 | plein (B=1.0) | BMuon B=0.3 | 0.1 (dès ép. 81) | sélectif + 0.1 | 4000 Ada (8) |

> **Comparaisons clés :**
> - **Carré 2×2** {BMuon, Muon uniforme} × {wd 1e-2, wd 1e-3} : `mp1-bmuon-wd1e2`, `mp1-bmuon-wd1e3`,
>   `mp1-uniform-shrink-wd1e2`, `mp1-uniform-shrink-wd1e3` (+ `mp1-bmuon-selwd` en plus)
> - BMuon vs baseline iso-coût : `mp1-bmuon-wd1e2` vs `mp1-uniform-shrink-wd1e2`
> - Effet du WD sur baseline : `mp1-uniform-shrink-wd1e2` vs `mp1-uniform-shrink-wd1e3`
> - WD sélectif vs uniforme : `mp1-bmuon-selwd` vs `mp1-bmuon-wd1e2`
>   (NB : GPU différents — normaliser par temps/époque ou par coût FLOPs)

---

## Week-end du 16-19 oct 2026 (fenêtre DSI : ven 18h00 (accord DSI du 16 oct.) -> lun 7h30)

Carré 2×2 `{BMuon, Muon uniforme} × {wd 1e-2, wd 1e-3}` + le run `selwd`. Arrêt automatique de chaque
process à 7h10 (`timeout`, voir `bmuon/experiments/window.sh` et `launch_dist.sh`) ; filets de sécurité :
`fleet.sh stop` à 7h20 (cron ablette + bentley), comptage à 7h35 dans `~/fleet_check.log`.

### `mp1-bmuon-wd1e3`
Nouveau run du week-end : voir son entrée dans « Runs actifs » ci-dessus.

### Changement de matériel de `mp1-bmuon-wd1e2` (à signaler dans le papier)
- Les ép. 0-17 ont tourné sur 16 × RTX 3090 (poissons, GPU partagés, ~65 min/époque).
- Reprise le 16 oct. sur 16 × RTX 4000 Ada (coordinateur `ain`), depuis `state.pkl` : même lot global (256),
  même code, seuls changent le GPU et la vitesse (~26 min/époque estimé). Les résultats ne dépendent pas de la machine.

### Arrêts du 15 oct. 2026
- Tous les runs ont été arrêtés le 15 oct. entre 08h05 (wd1e2) et 09h39 (les trois autres) ; dernières époques :
  wd1e2 ép. 17, uniform-wd1e2 ép. 17, uniform-wd1e3 ép. 25, selwd ép. 21.

### Incidents du 16-17 oct. 2026
- **Cartes réseau à 100 Mb/s** sur `ardennes`, `indre` (groupe `bmuon-wd1e2`) et `mayenne` (groupe `bmuon-wd1e3`) au lieu de 1000 :
  ces deux runs tournaient à ~60 min/époque au lieu de ~30. Remplacées par `xiphoide` / `porsche` (wd1e2, relancé à 19h09 juste
  après un checkpoint) et `tibia` (wd1e3, 18h49). Gain mesuré : ~2,3x. Sur 90 machines contrôlées, ce sont les seules à 100 Mb/s.
- **Arrêt de `bmuon-wd1e3` le 16/10 à 18h43** : 7 machines d'un autre sous-réseau ont cessé d'envoyer leurs heartbeats JAX, pendant une
  copie rsync de l'environnement entre salles (cause probable, non prouvée). Relancé depuis son state, rien perdu hors ~8 min.
- **Arrêt des 5 runs le 17/10 de 7h20 à 12h38** (5h19 perdues) : un cron de sécurité `fleet.sh stop` prévu pour lundi s'exécutait
  tous les jours ; les relances de la surveillance en session ne se sont pas déclenchées (PC probablement en veille). Crons restreints
  au lundi ; un cron `fleet.sh repair` toutes les 15 min tourne maintenant sur ablette. States intacts, reprise propre.

### `mp1-selwd-late-wd0p1` — branche de `selwd` avec weight decay fort tardif
- **Idée :** entraîner `selwd` jusqu'à l'époque 80, puis augmenter beaucoup le weight decay pour vider les canaux faibles avant d'élaguer.
- **Config :** `nequix-mp-1-bmuon-selwd.yml`, `--set weight_decay=0.1` (terme sélectif `selective_weight_decay_prune` 0.049 inchangé).
- **Départ :** copie de `state.pkl` de `selwd` juste après l'époque 80 (copie dans `~/selwd-fork80`, le run `selwd` n'a pas été touché), reprise à l'époque 81.
- **Machines :** 8 × RTX 4000 Ada (acromion, astragale, axis, cote, frontal, malleole, metacarpe, parietal), lot 32 par machine SANS accumulation
  (l'état d'optimiseur de `selwd` n'a pas la structure MultiSteps), remplissage doublé (max_n_nodes=2048, max_n_edges=104000) ; lot global 256.
- **Lancé :** dim. 18/10/2026 03h32 ; terminé (DONE) le même jour, ~10,9 h de calcul.
- **Résultat (ép. 99) :** E 18.92 meV/atome, F 51.00 meV/Å, stress 0.000250, contre `selwd` : E 10.61, F 34.35. Précision nettement dégradée (F +48 %) :
  la valeur 0.1 est trop forte. Pic d'erreur à l'ép. 88 (F 54.2), puis récupération partielle.
- **Allocation :** presque binaire dès l'ép. 90 : canaux au plancher 166 (ép. 81) -> 542 (ép. 90) -> 548 (ép. 99) sur 857, canaux intermédiaires 502 -> 8.

### Suite du week-end (17-18 oct. 2026)
- **Remplacement de 7 machines de `bmuon-wd1e3`** (marne, morbihan, moselle, somme, rotule, sacrum, sternum -> apophyse, atlas, coccyx, cubitus, cuboide, femur, humerus),
  dim. 01h37, à cause d'une charge CPU d'un autre compte (~15 cœurs par machine) ; l'autre compte a rattrapé 2 des 7 machines en moins d'une demi-heure.
- **Réparation automatique de `bmuon-wd1e2`** dim. 05h00 (log figé 17 min, 16/16 processus vivants), reprise depuis le state de 04h35.
- **Runs terminés :** `uniform-wd1e3` (sam. 19h25, ép. 99 : E 18.53 / F 52.61), `uniform-wd1e2` (dim. 05h09 : E 19.04 / F 52.95),
  `selwd` (dim. 12h05 : E 10.61 / F 34.35).
