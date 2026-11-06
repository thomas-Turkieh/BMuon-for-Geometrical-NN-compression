# Plan : la Figure 2 du papier BMuon, refaite pour Nequix

*Rien n'est lancé tant que ce plan n'est pas validé.*

## 1. Ce que montre la Figure 2 du papier

**Figure 2 du papier BMuon (CIFAR-100, ViT).** On entraîne un modèle avec un optimiseur donné. Ensuite, **sans aucun réentraînement**, on supprime une fraction croissante des canaux cachés des MLP (0 → 95 %) et on mesure la précision. Une méthode est bonne si sa courbe reste plate le plus longtemps possible.

| Courbe | Entraînement | Comment on compresse après l'entraînement |
|---|---|---|
| **Muon** | Muon standard | On enlève les canaux de plus petite magnitude couplée s_j = ‖W1[j,:]‖² + ‖W2[:,j]‖²)^½, classés globalement sur tous les blocs |
| **Muon+GL** | Muon + pas proximal **Group-Lasso** exact sur chaque canal (λ = 0,05, choisi parmi 6 valeurs) | Magnitude, comme Muon |
| **NuMuon** | Muon dont la mise à jour est de **rang réduit** (rang relatif 1 → 0,25 en cosinus) | **SVD tronquée** de chaque matrice, avec un rang tel que le nombre de paramètres égale celui du modèle élagué par canaux au même niveau |
| **BM-mag** | **BMuon** (α = ½, budget réduit linéairement de 100 % à 20 % des canaux pendant le premier tiers) | Magnitude |
| **BM-μ** | Le même modèle BMuon | Classement par l'**allocation apprise μ** |

**La Figure 2 ne contient pas d'élagage graduel.** Tout y est élagué *après* l'entraînement, en une fois. L'élagage graduel est traité à part, en §5.2 (Table 1) : on élague pendant l'entraînement à intervalles réguliers, on reprend l'entraînement à partir des poids survivants, et on finit par une phase de récupération. Les trois variantes sont Muon-mag, BMuon-mag et BMuon-μ. **Comme le demande le tuteur, on ajoute ces trois variantes sur notre figure**, sous forme d'étoiles à leur niveau final d'élagage, prolongées par l'élagage post-entraînement au-delà.

## 2. Ce que deviennent ces notions pour Nequix

| Papier (ViT) | Nous (Nequix-MP-1, code d'Octave) |
|---|---|
| canal caché d'un MLP | **canal de type I** : une copie d'irrep à l'entrée du produit tensoriel. Il possède une colonne de `linear_1`, ses lignes de `linear_2` et ses colonnes de la dernière couche du MLP radial. **857 canaux** (89 dans le bloc 0, 256 dans chacun des blocs 1 à 3) |
| W1[j,:], W2[:,j] | toutes les « tranches » du canal (rôles A, B, C de `bmuon/channels.py`) |
| précision top-1 (plus haut = mieux) | **MAE des forces (meV/Å) et des énergies (meV/atome)**, plus bas = mieux, en échelle log |
| jeu de test | les **mêmes structures tenues à l'écart** (split de validation 5 % de MPtrj, identique pour tous les runs, y compris ceux d'Octave) |
| « au moins 4 canaux par bloc » | au moins 1 canal par (bloc, irrep), comme `fragility.py` d'Octave |

Une colonne de compression de 70 % signifie qu'il reste 30 % des 857 canaux. L'export physique (`bmuon.prune.prune`) donne exactement le même modèle que la mise à zéro de ces canaux, ce qui est testé. L'évaluation se contente donc de les mettre à zéro.

## 3. Protocole commun : 20 % des données, 25 époques

Nequix-MP-1 a été entraîné 100 époques sur tout MPtrj. Octave mesurait ~26 min par époque complète sur 16 GPU, soit ~43 h. La nuit en offre ~7. Le choix :

- **`train_subset_frac: 0.2`** : 20 % de l'ensemble d'entraînement, soit ~300 000 structures, tirées une fois pour toutes. C'est le même sous-ensemble pour tous les runs.
- **`n_epochs: 25`** : ~5 passages équivalents sur MPtrj complet et ~29 000 pas d'optimisation de 256 structures. Le calendrier complet (warmup, cosinus, annealing du budget) se déroule sur ces 25 époques. Chaque modèle est donc un vrai modèle « fin d'entraînement », pas une coupe au milieu d'un long run.
- **Pourquoi pas moins de données et plus d'époques ?** MPtrj est très divers. Garder plus de structures différentes rapproche la généralisation de celle d'un entraînement complet, plus que de répéter 100 fois 5 % des données. 25 époques suffisent pour que l'allocation BMuon se stabilise : le budget est atteint à l'époque ~6, puis il reste 19 époques à budget fixe. Elles suffisent aussi à loger 5 élagages graduels suivis de 5 époques de récupération.
- Tout le reste est commun à toutes les méthodes : modèle Nequix-MP-1 complet (`128x0e + 64x1o + 32x2e + 32x3o`, 4 couches), lr 0,01 en warmup puis cosinus, weight decay 0,001 (défaut de Nequix), lot global 256, seed 0.

## 4. Les 8 entraînements

8 GPU par run, soit 64 GPU, plus 1 GPU pour la figure en direct.

| # | Run (`bmuon/runs/…`) | Config (`bmuon/configs/fig2/`) | Courbe(s) | Détail |
|---|---|---|---|---|
| 1 | `fig2-bmuon` | `bmuon.yml` | **BM-mag** et **BM-μ** | BMuon, budget 100 % → 30 % des canaux sur les 25 premiers % de l'entraînement, α = ½, pas MM tous les 5 pas |
| 2 | `fig2-muon` | `muon.yml` | **Muon** | l'optimiseur de Nequix (`optax.contrib.muon`) |
| 3 | `fig2-gradual-bmuon-mu` | `gradual-bmuon-mu.yml` | ★ **BM-μ graduel** | voir §5 |
| 4 | `fig2-numuon` | `numuon.yml` | **NuMuon** | rang relatif 1 pendant 10 %, cosinus jusqu'à 0,25 à 80 %, puis 0,25 |
| 5 | `fig2-gradual-muon-mag` | `gradual-muon-mag.yml` | ★ **Muon-mag graduel** | |
| 6 | `fig2-muon-gl-2` | `muon-gl-2.yml` | **Muon+GL (λ = 2)** | |
| 7 | `fig2-gradual-bmuon-mag` | `gradual-bmuon-mag.yml` | ★ **BM-mag graduel** | |
| 8 | `fig2-muon-gl-0p5` | `muon-gl-0p5.yml` | **Muon+GL (λ = 0,5)** | |

Les runs sont listés par priorité. S'il n'y a pas assez de GPU libres, le lanceur démarre les premiers et signale ceux qui manquent.

**Courbes de référence, sans calcul d'entraînement.** On les obtient en évaluant des modèles existants entraînés sur **tout** MPtrj :
- « Muon (Nequix-MP-1 publié, 100 époques) » : `models/nequix-mp-1.nqx`, élagué par magnitude ;
- « BM-mag / BM-μ (Octave selwd, époque 90) » : le meilleur run BMuon d'Octave (budget 30 %, weight decay sélectif).

Ces deux courbes montrent ce qui change entre notre protocole réduit et un entraînement complet.

## 5. Élagage graduel (§5.2 et annexe E.4 du papier)

- Élagages après les époques 4, 8, 12, 16 et 19, avec des niveaux cumulés de **20, 40, 55, 65 et 70 %** des canaux (686, 514, 386, 300 puis 257 canaux restants), puis **5 époques de récupération** (20 à 24).
- À chaque élagage, on classe globalement les canaux survivants : par magnitude couplée pour Muon-mag et BM-mag, par l'allocation μ pour BM-μ. Les plus faibles sont supprimés **définitivement** : leurs poids, leur EMA et leur moment sont mis à zéro et le restent, ce qui est testé. L'entraînement reprend sur les poids survivants.
- Pour BMuon : dans chaque phase, le budget va linéairement des survivants actuels au niveau suivant pendant la première moitié de la phase, puis il est maintenu. Après chaque élagage, l'allocation des survivants repasse à 1. C'est exactement le protocole du papier.
- Le niveau final de 70 % correspond au budget de 30 % du run BMuon (n° 1). On compare donc élagage graduel et post-entraînement au même niveau.
- Le masque des canaux supprimés est stocké dans l'état de l'optimiseur (`state.pkl`), et un fichier `prune_epochNNN.npz` est écrit à chaque élagage : une reprise après crash ne refait ni ne perd aucun élagage.

## 6. Écarts au papier, à connaître pour la réunion

1. **Une seule seed**, contre 3 dans le papier : pas de barres d'erreur. Deux autres seeds coûteraient deux nuits de plus.
2. **Budget BMuon de 30 % sur 25 % de l'entraînement**, contre 20 % sur un tiers pour la Figure 2 du papier. C'est le réglage ImageNet du papier et celui de tous les runs d'Octave. Il est mesuré en **nombre de canaux** (`cost_model: channel`, comme le papier), pas en coût « paths » comme chez Octave.
3. **λ de Group-Lasso remis à l'échelle.** Les canaux de Nequix ont une norme de ~17 (mesuré sur Nequix-MP-1), contre ~1 dans un ViT. Le λ = 0,05 du papier serait donc ~20× trop faible en relatif. Faute de pouvoir balayer 6 valeurs, on en prend deux, **0,5 et 2**. Le papier garde la valeur la plus forte dont la précision dense reste à moins d'un point de BMuon ; on fera de même au vu des courbes.
4. **NuMuon appliqué seulement aux matrices des canaux** (celles qu'on compresse) ; les autres matrices gardent Muon. Le papier l'appliquait à toutes les matrices éligibles à Muon. Notre direction de rang réduit est **exacte**, calculée par décomposition propre de la matrice de Gram, là où NuMuon utilise une approximation Krylov aléatoire. Un rang minimal de 4 est imposé à la compression, comme dans le papier.
5. **Classement par magnitude brute**, comme le papier. Octave avait remarqué que les normes brutes ne sont pas comparables d'un bloc à l'autre : il utilisait une magnitude normalisée par groupe. Cette variante n'est pas dans la figure, mais on peut l'ajouter à l'évaluation sans réentraîner.

## 7. Durée et sécurité

- **GPU disponibles (essai à blanc à 23h20)** : il y a de quoi lancer **7 runs sur 8**. Les 5 premiers tournent sur des GPU rapides (4000 Ada, A5000, 3090). `fig2-muon-gl-2` et `fig2-gradual-bmuon-mag` tournent sur des A4000, ~1,4× plus lentes, avec une fin estimée entre ~4 et 6 h. **`fig2-muon-gl-0p5` reste sans GPU** et partira une autre nuit. On peut aussi le lancer tout de suite sur des RTX 2000 Ada (`TYPES="4000ada+a5000+3090 a4000 2000ada"`), mais sans garantie qu'il finisse. Un run ne mélange jamais des GPU de vitesses très différentes.
- Durée estimée par run : **2 h 45 à 4 h 30** sur 8 GPU rapides. La fourchette va de ma mesure sur A5000 aux chiffres plus lents d'Octave. NuMuon coûte ~+13 %. Lancé avant minuit, tout finit avec plusieurs heures de marge avant **6h57**.
- **Si c'est plus lent que prévu**, la colonne « fin estimée dans » de `STATUS.md` le montre dès l'époque 1-2. On peut alors relâcher la nuit suivante : la reprise est automatique, avec le même nom de run.
- Tous les process **s'arrêtent seuls à 6h57** (SIGKILL au plus tard à 6h58). Aucun redémarrage n'est tenté à moins de 5 min de l'heure limite.
- **Crash ou coupure réseau** : chaque machine relance son process seule. La reprise se fait depuis `state.pkl`, écrit toutes les 300 étapes (~4 fois par époque), au milieu de l'époque si besoin. Un run bloqué 25 min sans log est relancé. Tout cela a été testé le 2 novembre au soir.
- **Non couvert** : redémarrage ou coupure de courant d'une machine. Le run concerné s'arrête alors pour la nuit, mais rien n'est perdu jusqu'au dernier `state.pkl`.

## 8. Traçabilité : suivre la nuit et ne rien perdre

**Un checkpoint par époque.** Chaque run écrit `snap_epochNNN.nqx` (le modèle EMA, 2,8 Mo) et `alloc_epochNNN.npz` (l'allocation μ) à **chaque époque**, sur le home partagé. Même si un run meurt, on peut tracer sa courbe de Figure 2 à partir de sa dernière époque.

**Figure en direct.** Une machine dédiée (`fig2_watch.sh`) redessine la figure toutes les ~10 min avec le dernier snapshot de chaque run, sur 500 structures. Seuls les nouveaux snapshots sont évalués, les autres sont en cache. La légende indique l'époque de chaque courbe.
- `~/nequix/bmuon/results/fig2/fig2.png` : la figure (forces à gauche, énergies à droite) ;
- `~/nequix/bmuon/results/fig2/fig2_points.csv` : tous les points ;
- `~/nequix/bmuon/results/fig2/STATUS.md` : pour chaque run, époque atteinte, minutes par époque, **fin estimée**, MAE de validation, canaux restants, âge du dernier log.

Quand tous les runs ont fini, le même script fait une **passe finale sur 2000 structures**, dans `bmuon/results/fig2_final/`. **C'est la figure pour la réunion.**

**Commandes pour suivre** (Git Bash sur le PC, depuis `EA/code`) :

```bash
bash bmuon/experiments/night.sh status                         # machines vivantes + dernière époque de chaque run
ssh ablette.polytechnique.fr cat nequix/bmuon/results/fig2/STATUS.md
scp ablette.polytechnique.fr:nequix/bmuon/results/fig2/fig2.png .   # la figure du moment
```

**Les logs.** `bmuon/runs/<run>/log.jsonl` contient toutes les métriques, toutes les 50 étapes et à chaque époque. `hosts.log` contient chaque (re)démarrage. Les logs process sont dans `/tmp/$USER/night-<run>-rank<i>.log` sur chaque machine.

## 9. Lancement, à faire après validation

```bash
bash bmuon/experiments/fig2_launch.sh        # à blanc : choisit 65 GPU libres et affiche le plan
bash bmuon/experiments/fig2_launch.sh --go   # installe les machines qui en ont besoin, lance les 8 runs + la figure en direct
```

## 10. Ce qui a été testé (sans aucun entraînement sur les vraies données)

- Tests unitaires (`bmuon/tests/test_fig2_methods.py`, 7 tests), plus les 14 tests existants d'Octave :
  - le prox Group-Lasso est exact ;
  - la direction NuMuon a le bon rang et vaut le facteur polaire à rang plein ;
  - les canaux élagués restent exactement nuls après de nombreuses mises à jour, et les autres bougent ;
  - les calendriers de budget et de rang sont corrects ;
  - la compression SVD est exacte à rang plein et respecte le budget de paramètres ;
  - **le budget plein reste exactement Muon** (test d'Octave).
- Boucle d'entraînement complète sur le jeu jouet synthétique d'Octave, en CPU, quelques secondes par méthode, pour Muon+GL, NuMuon, BM-μ graduel et Muon-mag graduel. Les élagages tombent aux bonnes époques (58 → 46 → 34 → 29 canaux) et la figure se génère.
- Évaluation de la figure sur les vrais modèles de référence (5 structures, en CPU), pour valider les formes de matrices.

## 11. À valider

1. Le protocole « 20 % des données × 25 époques ».
2. Les deux valeurs de λ pour Muon+GL (0,5 et 2).
3. Le budget BMuon à 30 % (choix d'Octave et du papier sur ImageNet), plutôt que 20 % (choix du papier sur CIFAR).
4. Le niveau final d'élagage graduel à 70 %.
5. Le nombre de GPU : 57 ce soir (7 runs × 8, plus 1 pour la figure), soit environ un tiers du parc.
6. Lancer λ = 0,5 sur des RTX 2000 Ada (risque de ne pas finir) ou le garder pour une autre nuit.
