# Budgeted Muon for Equivariant Interatomic Potentials — brouillon

Oct 12, 2026 · @Octave

> **Note de lecture.** Brouillon du papier au 12 octobre 2026. Le texte du papier est en anglais ; les blocs **TODO** (en français) disent quoi faire pour remplir chaque partie. Les sections 3 et 4 (contexte et méthode) sont rédigées ; les sections 5 à 7 sont des énoncés ou des protocoles à valider. Tous les chiffres entre crochets \[X\] sont à remplacer par des mesures.
>
> **TODO — cadrage.** Choisir la venue cible (workshop AI4Mat / ML4PS ou conférence principale) : elle fixe la longueur, le poids de la théorie et l'ampleur des expériences. Le titre est provisoire.

## Abstract

E(3)-equivariant interatomic potentials built on tensor products are accurate, but their per-edge tensor products dominate inference cost. Recent compact foundation models such as Nequix are trained with the Muon optimizer. Budget Muon (BMuon) reshapes Muon's spectral geometry so that hidden channels compete for a global budget, which makes Transformers robust to structured pruning; it is, however, defined only for MLP hidden channels.

We extend BMuon to the tensor-product layers of equivariant networks. A channel is one copy of an irreducible representation entering the tensor product; it couples slices of the preceding linear layer, of the radial network and of the following linear layer. Allocations are shared across the 2l+1 components of each copy, so training and pruning preserve equivariance. We further introduce a cost-weighted budget, under which channels compete for compute rather than for count. The BMuon oracle remains normalized steepest descent under an induced norm, its allocation problem stays concave, and its minorize–maximize update keeps a closed form.

On Nequix trained on MPtrj, budgeted training yields \[X\]× faster inference at matched \[force MAE / Matbench-Discovery CPS\], outperforming magnitude pruning of Muon-trained models and post-hoc (k,l)-block pruning. The learned allocations reveal \[non-uniform irrep multiplicities / a per-block effective l\_max\] that \[outperform uniform architectures when retrained from scratch\].

> **TODO — Abstract.** À réécrire une fois les expériences 6.1 à 6.5 faites. Remplacer chaque crochet par le chiffre mesuré ; supprimer la dernière phrase si la découverte d'architecture (6.5) n'est pas confirmée. Garder ≤ 200 mots.

## 1 Introduction

Machine-learned interatomic potentials (MLIPs) based on E(3)-equivariant message passing, such as NequIP and its descendants, reach near-DFT accuracy with high data efficiency. Their features are direct sums of irreducible representations of O(3), and each message is a tensor product between a neighbour's features and the spherical harmonics of the bond direction. This tensor product is evaluated on every edge at every step of a molecular-dynamics run, so it dominates inference cost. Reducing it without breaking equivariance is a direct lever on simulation throughput.

Structured pruning is the natural tool, because removing whole units shrinks dense tensors and translates into speed. In an equivariant network the smallest unit that can be removed without breaking equivariance is one copy of an irrep, with all its 2l+1 components. Existing work prunes equivariant models after training \[Wang et al.\] or targets group-convolutional networks \[Alnemari & Al-Omair\]. Neither shapes training so that compressibility emerges.

Compact equivariant foundation models are now trained with the Muon optimizer. Nequix \[Koker et al., 2025\] pairs a simplified NequIP with Muon and equivariant RMSNorm and is trained on MPtrj for 100 A100-hours. BMuon \[anonymous, 2026\] showed that Muon-trained Transformers collapse under structured MLP pruning, and that reweighting Muon's spectral geometry under a global channel budget restores compressibility at almost no cost in accuracy. \[Our Figure 1 shows the same fragility for Nequix.\]

BMuon is defined for MLP hidden channels: a row of the up-projection and a column of the down-projection around a pointwise nonlinearity. Tensor-product layers differ in two ways. First, a channel enters a product with the spherical harmonics that is linear in the features, weighted per path by a radial network; it is therefore coupled to slices of three families of matrices. Second, channels have very different costs: a copy of degree l = 3 takes part in more paths, each with larger Clebsch–Gordan contractions, than a scalar copy.

**Contributions.**

1. We extend BMuon to tensor-product layers by defining a channel as a coupled group of matrix slices with shared, possibly repeated, allocations. Allocations are shared across the 2l+1 components, so every update and every pruned model remains E(3)-equivariant.
2. We introduce a cost-weighted budget, under which channels compete for compute rather than count. We give the induced norm, show that it recovers Muon, that the allocation problem is concave for α ≤ 1/2, and that the minorize–maximize update keeps a closed form.
3. On Nequix trained on MPtrj, we obtain \[X\]× faster inference at matched accuracy on Matbench-Discovery and the MDR phonon benchmark, outperforming magnitude pruning and post-hoc (k,l)-block pruning.
4. The learned allocations reveal \[non-uniform irrep multiplicities across layers / a per-block effective l\_max\] that \[outperform uniform architectures when retrained from scratch\].

> **TODO — Introduction.** (a) Ajouter les citations entre crochets (voir section 2). (b) Produire la Figure 1 = expérience 6.1 (fragilité de Nequix entraîné avec Muon) : si Nequix n'est PAS fragile, réécrire le 3e paragraphe autour du gain de vitesse plutôt que de la robustesse. (c) Contributions 3 et 4 : remplir après 6.3–6.5, supprimer la 4 si non confirmée. (d) Vérifier l'affirmation « le produit tensoriel domine le coût » avec le benchmark 6.2 et la chiffrer (part du temps par arête).

## 2 Related work

> **TODO — Related work (à rédiger, \~1 page).** Un paragraphe par thème, chacun terminé par une phrase qui dit en quoi notre travail diffère. Lire chaque papier avant de le citer ; ne rien citer de mémoire.

| Thème | À couvrir | Phrase de positionnement à écrire |
| --- | --- | --- |
| MLIPs équivariants | NequIP, Allegro, MACE, SevenNet (Park et al., 2024), EquiformerV2, eSEN, Nequix | ils fixent les multiplicités à la main ; nous les apprenons sous un budget de calcul |
| Efficacité des produits tensoriels | kernels (OpenEquivariance, cuEquivariance, FlashTP), Xie et al. 2025 sur expressivité vs coût | travaux orthogonaux : nous réduisons le nombre de contractions, les kernels accélèrent chacune ; on les combine |
| Compression de réseaux équivariants | Wang et al. (arXiv 2605.08885, élagage post-hoc par blocs (k,l)), Alnemari & Al-Omair (Symmetry 2026, G-CNN), distillation de MLIPs | eux élaguent après coup ; chez nous l'optimiseur rend le modèle compressible pendant l'entraînement |
| Muon et variantes | Muon (Jordan et al., 2024), analyses en norme spectrale, NuMuon, BMuon, Stupariu & Manolache (arXiv 2605.27662, Muon vs Adam sur réseaux équivariants) | première extension de BMuon hors des MLP, et première à un réseau équivariant |
| Élagage sous contrainte de ressources | MorphNet (Gordon et al., 2018, régularisation pondérée par les FLOPs), group lasso, normes k-support structurées | l'idée d'un budget pondéré par le coût existe ; la nouveauté est de l'intégrer à la géométrie de BMuon (norme induite, oracle, mise à jour MM) |

Points à vérifier en lisant : (1) la granularité exacte des blocs (k,l) de Wang et al. et si elle coïncide avec nos copies d'irrep ; (2) ce que Stupariu & Manolache mesurent sur la robustesse à l'élagage ; (3) s'il existe déjà un élagage de MLIPs équivariants pendant l'entraînement.

## 3 Background

### 3.1 Features and interaction blocks in NequIP and Nequix

Node features are direct sums of irreducible representations (irreps) of O(3), written in e3nn notation, e.g. 128x0e + 64x1o + 32x2e + 32x3o. An irrep (l, p) has dimension 2l+1 and parity p; one copy of it is a vector of 2l+1 real coefficients that transforms as a whole under rotations. Edges carry the spherical harmonics Y^(l)(r̂\_ij) for l ≤ l\_max, of parity (−1)^l, and a radial embedding φ(|r\_ij|) made of Bessel functions times a polynomial cutoff.

An interaction block maps node features h to new features h' in five steps.

```latex
\tilde h_j^{(l)} = A^{(l)} h_j^{(l)} \qquad \text{(linear\_1, per irrep)}
```

```latex
m_{ij}^{(p,k)} = R_{p,k}\big(|r_{ij}|\big)\,\Big(\tilde h^{(l_1)}_{j,k} \otimes Y^{(l_2)}(\hat r_{ij})\Big)^{(l_3)}, \qquad p = (l_1, l_2 \to l_3)
```

```latex
x_i^{(l_3)} = B^{(l_3)} \Big[\tfrac{1}{\sqrt{\bar n}} \textstyle\sum_{j \in \mathcal N(i)} m_{ij}^{(p,\cdot)}\Big]_{p:\, l_3(p)=l_3} + S\, h_i, \qquad h_i' = \mathrm{Gate}\big(\mathrm{RMSNorm}(x_i)\big)
```

Here the tensor product runs in the uvu mode: copy k of the input irrep produces copy k of every path p, so it mixes no channels. The per-edge weights R\_{p,k} come from a radial MLP whose last layer is a matrix C. Linear layers act only within one irrep and with the same weights on all 2l+1 components (Schur's lemma); the learnable parameters are therefore the multiplicity matrices A^(l), B^(l\_3), S^(l), and the identity factor on the components is implicit. The gate multiplies each copy of degree l > 0 by σ(g\_k), where g\_k is an extra scalar output of the block.

Nequix \[Koker et al., 2025\] simplifies NequIP in three ways: the species-dependent self-connection becomes a single linear map S, the last block outputs only scalars, and an equivariant RMSNorm precedes the gate. Its released model nequix-mp-1 uses the configuration below.

| Hyper-parameter | Value |
| --- | --- |
| Hidden irreps | 128x0e + 64x1o + 32x2e + 32x3o |
| l\_max (edges) | 3 |
| Interaction blocks | 4 (last one scalar-only) |
| Radial cutoff | 6 Å |
| Radial basis / MLP | 8 Bessel functions, 2 layers of 64 |
| Species | 89 |
| Parameters | 708K |
| Optimizer | Muon, lr 0.01, weight decay 1e-3 |

Sources: [Nequix paper](https://arxiv.org/abs/2508.16067), [configs/nequix-mp-1.yml](https://github.com/atomicarchitects/nequix).

### 3.2 Muon

For a weight matrix W with momentum M, Muon updates W ← W − η Polar(M), where Polar(M) = UV^T for the SVD M = UΣV^T. By spectral–nuclear duality this is normalized steepest descent under the operator norm:

```latex
\mathrm{Polar}(M) \in \arg\max_{\|\Delta\|_{\mathrm{op}} \le 1} \langle M, \Delta \rangle, \qquad \max_{\|\Delta\|_{\mathrm{op}} \le 1} \langle M, \Delta \rangle = \|M\|_*
```

In practice Polar is approximated by five Newton–Schulz iterations and rescaled by sqrt(max(1, fan\_out/fan\_in)). Nequix uses optax's Muon, which routes every 2D parameter to Muon and every other parameter to AdamW. In nequix-mp-1 this leaves only the RMSNorm scales (about 1.3K parameters) to AdamW; every multiplicity matrix, the radial MLP and the readout are trained with Muon. Each multiplicity matrix is orthogonalized separately. e3nn-jax stores one matrix per irrep for linear\_1 and, for linear\_2, one matrix per output irrep l\_3 whose input rows are the copies of all paths ending in l\_3.

### 3.3 BMuon

BMuon \[anonymous, 2026\] considers MLP blocks x ↦ W\_2 φ(W\_1 x) and assigns each hidden channel j an allocation μ\_j ∈ \[0, 1\] under a global budget Σ\_j μ\_j = K shared by all blocks. The admissible updates scale row j of Δ\_1 and column j of Δ\_2 by μ\_j^α; the oracle chooses the allocation and the direction jointly to maximize the first-order decrease of the loss. The allocation problem is concave for α ≤ 1/2, the oracle is steepest descent under an induced norm that recovers Muon at K = N, and a minorize–maximize (MM) step gives a closed-form allocation update. With weight decay, channels whose allocation vanishes see their weights driven to zero, which makes the network robust to removing them.

> **TODO — Section 3.** (a) Recompter les paramètres et la part Muon/AdamW avec un script sur le modèle publié (mon décompte : 708K, dont \~1.3K sous AdamW). (b) Vérifier dans e3nn-jax que `linear_2` stocke bien une matrice par chemin (affirmé en 3.2). (c) Ajouter la figure du bloc d'interaction (ci-dessous, à placer en Figure 2).

&#91;embedded content: Figure 2 · interaction block and the slices of one type-I channel\]

Figure 2. Interaction block of Nequix. The shaded matrices hold the slices of one type-I channel: row k of A, one row of C and one column of B per path fed by copy k, all scaled by the same allocation.

## 4 Method: budgeted Muon on tensor-product channels

All matrices are written for y = Wx. (e3nn-jax stores weights as (in, out) and computes x @ W, so rows and columns are swapped in code.)

### 4.1 Tensor-product channels

For block t, let I\_t be the irreps entering the tensor product, with multiplicities m\_{t,l}, and P\_t the set of paths p = (l\_1, l\_2 → l\_3). A **channel** is a triple c = (t, l, k): copy k of irrep l at the output of linear\_1 in block t. The set of channels is C, with N = |C| = Σ\_t Σ\_l m\_{t,l}.

Channel c reaches the output of the block only through three families of slices:

| Matrix | Role | Shape | Slices of channel (t, l, k) |
| --- | --- | --- | --- |
| A\_t^(l) | linear\_1, irrep l | m\_{t,l} × m^in\_{t,l} | row k |
| B\_t^(l\_3) | linear\_2, output irrep l\_3 | n\_{t,l\_3} × Σ\_{p: l\_3(p)=l\_3} m\_{t,l\_1(p)} | column (p, k), for every p with l\_1(p) = l (in the matrix of l\_3(p)) |
| C\_t | last layer of the radial MLP | (Σ\_p m\_{t,l\_1(p)}) × H | row (p, k), for every p with l\_1(p) = l |

Setting all slices of c to zero removes c exactly from the computation. A copy of degree l takes part in n\_{t,l} = |{p : l\_1(p) = l}| paths, so it owns 1 + 2 n\_{t,l} slices. In block 2 of nequix-mp-1, for instance, a 1o copy feeds six paths and owns thirteen slices. A channel is the analogue of a BMuon hidden channel, with two differences: the operator between its slices is a tensor product with the spherical harmonics, linear in the features, rather than a pointwise nonlinearity; and its allocation is repeated across the rows of C\_t.

All slices act on the multiplicity index only. An allocation therefore scales the 2l+1 components of a copy together, which is what equivariance requires (Proposition 1).

### 4.2 Cost-weighted budget

Each channel receives an allocation μ\_c ∈ \[0, 1\]. BMuon constrains their sum; we constrain their cost:

```latex
\mathcal K_{K,w} = \Big\{ \mu \in [0,1]^N : \textstyle\sum_{c \in \mathcal C} w_c\, \mu_c = K \Big\}, \qquad 0 < K \le W := \textstyle\sum_c w_c
```

The weight w\_c is the inference cost of channel c, dominated by its per-edge contractions: w\_{(t,l,k)} ∝ Σ\_{p : l\_1(p) = l} cost(p). Timings on an RTX 3090 (Section 6.2) show that cost(p) = 1, i.e. w\_c proportional to the number of paths the copy feeds, fits the measured savings best (correlation 0.89, against 0.84 for Clebsch–Gordan nonzeros); we use it throughout. Uniform weights w\_c = 1 recover the channel budget of BMuon. With measured weights, K becomes a compute budget, and channels of high degree must earn their cost.

### 4.3 The oracle

From μ we form one diagonal matrix per budgeted matrix, acting on the slices of its channels; in D\_C the allocation of a channel is repeated once per path it feeds:

```latex
D_{A_t^{(l)}} = \mathrm{diag}\big(\mu_{(t,l,k)}\big)_{k}, \qquad D_{B_t^{(l_3)}} = \mathrm{diag}\big(\mu_{(t,l_1(p),k)}\big)_{(p,k):\, l_3(p) = l_3}, \qquad D_{C_t} = \mathrm{diag}\big(\mu_{(t,l_1(p),k)}\big)_{(p,k)}
```

D\_A acts on the rows of A, D\_B on the columns of B and D\_C on the rows of C. For α ∈ (0, 1\], the admissible updates of these matrices are

```latex
\mathcal A_\alpha(\mu) = \Big\{ \Delta :\; \Delta_A = D_A^\alpha P_A,\;\; \Delta_B = P_B D_B^\alpha,\;\; \Delta_C = D_C^\alpha P_C,\;\; \|P\|_{\mathrm{op}} \le 1 \text{ for every matrix} \Big\}
```

All other matrices of the network keep the Muon constraint ‖Δ‖\_op ≤ 1. A channel with μ\_c = 1 is updated as by Muon; a channel with small μ\_c moves little in all its slices; a channel with μ\_c = 0 does not move. The budget makes channels compete across blocks and irreps. The oracle chooses allocation and direction jointly:

```latex
\max_{\mu \in \mathcal K_{K,w}}\; \max_{\Delta \in \mathcal A_\alpha(\mu)}\; \langle M, \Delta \rangle
```

where M collects the momenta. Since the constraints on the P separate across matrices, spectral–nuclear duality reduces it to the allocation problem

```latex
\max_{\mu \in \mathcal K_{K,w}} \Phi_M(\mu), \qquad \Phi_M(\mu) = \sum_{t=1}^{T} \Big[ \sum_{l \in \mathcal I_t} \big\| D_{A_t^{(l)}}^\alpha M_{A_t^{(l)}} \big\|_* + \sum_{l_3} \big\| M_{B_t^{(l_3)}} D_{B_t^{(l_3)}}^\alpha \big\|_* + \big\| D_{C_t}^\alpha M_{C_t} \big\|_* \Big]
```

Given a maximizer μ\*, the optimal directions are

```latex
\Delta_A^\star = D^\alpha \mathrm{Polar}\big(D^\alpha M_A\big), \qquad \Delta_B^\star = \mathrm{Polar}\big(M_B D^\alpha\big) D^\alpha, \qquad \Delta_C^\star = D^\alpha \mathrm{Polar}\big(D^\alpha M_C\big)
```

and each matrix is updated as W ← (1 − ηλ) W − η Δ\*, with weight decay λ.

### 4.4 Practical allocation update

As in BMuon, we replace the exact allocation problem by one MM step every ΔT\_μ optimizer steps. From the current allocation μ̄, we compute the polar factors P̄\_A = Polar(D\_μ̄^α M\_A), P̄\_B = Polar(M\_B D\_μ̄^α), P̄\_C = Polar(D\_μ̄^α M\_C) and a score per channel that sums over all its slices:

```latex
a_c = \big\langle M_{A_t^{(l)}}[k,:],\, \bar P_{A_t^{(l)}}[k,:] \big\rangle + \sum_{p:\, l_1(p)=l} \Big[ \big\langle M_{B_t^{(l_3(p))}}[:,(p,k)],\, \bar P_{B_t^{(l_3(p))}}[:,(p,k)] \big\rangle + \big\langle M_{C_t}[(p,k),:],\, \bar P_{C_t}[(p,k),:] \big\rangle \Big]
```

Then Φ\_M(μ) ≥ Σ\_c a\_c μ\_c^α, with equality at μ̄. Maximizing this separable surrogate over {μ ∈ \[ε, 1\]^N : Σ\_c w\_c μ\_c = K} gives, by the KKT conditions,

```latex
\mu_c^+ = \mathrm{clip}_{[\varepsilon, 1]}\Big[ \Big( \frac{\alpha\, a_c}{\lambda_\mu\, w_c} \Big)^{1/(1-\alpha)} \Big]
```

where λ\_μ > 0 is found by bisection so that Σ\_c w\_c μ\_c^+ = K. The only change from BMuon is the factor w\_c. The full update is:

1. Update the momenta of all matrices as in Muon.
2. Every ΔT\_μ steps: compute P̄ and the scores a\_c, then μ^+ by the formula above.
3. For A, B, C: apply Δ\* with the current allocation. For every other 2D matrix: plain Muon. RMSNorm scales: AdamW.
4. Anneal K from W (pure Muon) to the target budget over the first part of training.

We use α = 1/2, ΔT\_μ = 5 and ε = 1e-6, as in BMuon.

### 4.5 Pruning and export

After training, channels are ranked globally, either by μ\_c or by the coupled magnitude s\_c = (‖A\[k,:\]‖² + Σ\_p ‖B^(p)\[:,k\]‖² + Σ\_p ‖C\[(p,k),:\]‖²)^{1/2}, and the lowest-ranked are removed: rows of A, columns of B and rows of C. The result is a dense Nequix whose linear\_1 outputs smaller multiplicities than it receives; every block keeps a minimum number of copies per irrep. Removing all copies of a degree l in a block lowers that block's effective l\_max.

> **TODO — Section 4.** (a) **Modèle de coût w\_c** : fait — w\_c = nombre de chemins alimentés par la copie, le meilleur accord avec les temps mesurés sur RTX 3090 (corrélation 0,89, section 6.2) ; à recontrôler sur une autre carte. Les coûts mesurés copie par copie sont trop bruités pour servir directement. (b) **Pondération par 2l+1** : décider si les scores a\_c portent sur W ou sur W ⊗ I\_{2l+1} (facteur 2l+1 sur la norme nucléaire) ; tester les deux en ablation. (c) **Normalisation e3nn** : les chemins sont multipliés par des constantes 1/sqrt(fan\_in) dans le passage avant ; vérifier qu'elles ne biaisent pas les scores entre irreps (le même problème fausse l'élagage par magnitude brute, voir 6.1). (d) **Arrondi** : 6.2 ne montre pas de paliers nets avec OpenEquivariance (le temps baisse régulièrement avec les copies gardées) ; l'arrondi à un multiple de 8 reste une option, à tester. (e) **Code** : fait (`PrunableConvolution` dans `bmuon/model.py`, testé sur GPU avec les kernels). (f) Ajouter le tableau complet des tranches du bloc 2 en annexe A.

## 5 Theoretical properties

The results below transfer those of BMuon to the tensor-product setting with a cost-weighted budget. Their statements are fixed; their proofs remain to be written (TODO under each).

**Proposition 1 (equivariance).** For every allocation μ and every momentum M, the updated network is E(3)-equivariant. For every subset of channels, the network obtained by removing them is an E(3)-equivariant network of the same architecture with smaller multiplicities.

> **TODO — preuve.** Les paramètres sont les matrices de multiplicité ; toute valeur donne une couche équivariante (Schur), donc toute mise à jour aussi. Retirer la copie k retire un sous-espace invariant de dimension 2l+1 (et toutes les contributions des chemins uvu qui en partent) : le reste est un réseau Nequix valide. Une demi-page.

**Proposition 2 (induced geometry).** Let B\_{α,K,w} be the convex hull of the union of the sets A\_α(μ) over μ in K\_{K,w}, and Ω\_{α,K,w} its Minkowski functional. Then Ω is a norm, the oracle is normalized steepest descent under it, and

```latex
\|\Delta\|_{\mathrm{Muon}} \le \Omega_{\alpha,K,w}(\Delta) \le \Big(\frac{W}{K}\Big)^{\alpha} \|\Delta\|_{\mathrm{Muon}}, \qquad W = \textstyle\sum_c w_c
```

In particular, K = W forces μ = 1 and recovers Muon.

> **TODO — preuve.** Reprendre le Lemme 3 de BMuon. Seule modification : l'allocation uniforme devient μ = (K/W)·1, qui est bien dans K\_{K,w}, d'où le facteur (W/K)^α. Compacité, convexité et symétrie inchangées.

**Proposition 3 (concavity).** For 0 < α ≤ 1/2, Φ\_M is concave on K\_{K,w}; the allocation problem is a convex program.

> **TODO — preuve.** Le Théorème 1 de BMuon donne la concavité de μ ↦ norme nucléaire de D(μ)^α M (et du côté droit). Pour C, la diagonale est D(Sμ), où S est l'application linéaire positive qui duplique chaque μ\_c une fois par chemin ; une fonction concave composée avec une application linéaire reste concave. Φ\_M est une somme de fonctions concaves, et la contrainte reste linéaire, donc convexe, même pondérée.

**Proposition 4 (MM update).** Assume every channel is nontrivial (at least one of its slices has nonzero momentum). Then every score a\_c is positive, the surrogate has a unique maximizer given by the clipped closed form of Section 4.4, and for α ≤ 1/2 repeated MM steps converge in objective value to the global maximum of Φ\_M over the floored set, with every accumulation point globally optimal.

> **TODO — preuve.** Minorisation : écrire Φ\_M ≥ Σ ⟨D^α M, P̄⟩ et regrouper les tranches par canal (la répétition dans C se regroupe dans a\_c). Positivité : Lemme 8 et éq. (22) de BMuon, tranche par tranche. KKT : refaire l'annexe D.2 avec la contrainte pondérée, ce qui donne le facteur w\_c. Convergence : reprendre le Théorème 9 (Danskin + concavité) et la Proposition 10 (effet du plancher ε) avec W à la place de N.

**Proposition 5 (allocation controls channel norms).** With decoupled weight decay λ and step sizes η\_t, let r\_{c,t} be the Euclidean norm of all slices of channel c at step t and n\_c its number of paths. Then

```latex
r_{c,t+1} \le (1 - \eta_t \lambda)\, r_{c,t} + \sqrt{1 + 2 n_c}\; \eta_t\, \mu_{c,t}^{\alpha}
```

so a channel whose allocation stays below a level μ̄ ends with norm at most sqrt(1 + 2 n\_c) μ̄^α / λ, and channels whose allocation vanishes are driven to zero.

> **TODO — preuve.** Chaque tranche de Δ\* a une norme ≤ μ\_c^α (ligne ou colonne d'une matrice de norme opérateur ≤ 1, multipliée par μ\_c^α) ; le canal a 1 + 2n\_c tranches, d'où le facteur sqrt(1 + 2 n\_c) à la place du sqrt(2) de BMuon. Itérer comme pour le Théorème 6 et le Corollaire 7.

**Open question (not claimed).** BMuon's Theorem 2 shows that, under a scarce budget, minimum-cost linear factorizations W\_2 W\_1 are width-minimal. In our setting a block is linear in its features and factorizes as B T A, where T (tensor product with the spherical harmonics, radial weights, neighbour sum) is block-diagonal over channels in uvu mode. Each channel then contributes a term of rank up to n\_c rather than 1. A width-recovery result would require a structured notion of rank; we leave it to future work.

> **TODO — Section 5.** Écrire les cinq preuves en annexe B (citer BMuon pour tout ce qui est repris à l'identique). Ne tenter l'analogue du Théorème 2 que si les expériences sont terminées.

## 6 Experiments

> **TODO — Section 6 entière.** Les sous-sections ci-dessous sont des protocoles, à exécuter dans l'ordre : 6.1 et 6.2 ne demandent aucun entraînement et décident si le projet tient. Remplacer chaque protocole par le texte de résultats (anglais) une fois la mesure faite.

### 6.0 Setup

> **TODO.** Données : MPtrj avec le découpage de Nequix (5 % en validation) ; évaluation sur Matbench-Discovery (F1, RMSD, κSRME, CPS) et sur le benchmark de phonons MDR. Modèle : configuration nequix-mp-1 ; développement sur la version réduite 128x0e + 64x1o et sur un sous-ensemble de MPtrj. Budget de calcul : l'entraînement complet a coûté 100 h A100 ; estimer le temps sur les RTX 3090 disponibles et décider combien de runs complets sont faisables (3 graines × méthodes). Fixer α = 1/2, ΔT\_μ = 5, ε = 1e-6 et le calendrier de K.

### 6.1 Motivation: Muon-trained Nequix under channel pruning (Figure 1)

Magnitude pruning of the Muon-trained nequix-mp-1 collapses at the first cuts. Removing 10 % of the type-I channel cost multiplies the energy error by 12 to 23 (9.9 → 116–229 meV/atom) and the force error by 7 to 9 (33 → 233–287 meV/Å), without fine-tuning, whether channels are ranked globally or the same fraction is removed in every (block, irrep).

&#91;embedded content: Figure 1 · bmuon/results/fragility\_mp1\_global.csv · nequix-mp-1, RTX 3090, 13 Oct 2026\]

The unpruned model reproduces the published accuracy on the same validation split (9.9 meV/atom, 33.0 meV/Å against 10.05 and 32.79 reported), so the split matches the one Nequix held out. Channels are ranked globally by their coupled magnitude divided by the median of their (block, irrep) group; block 0, which reads the one-hot species, is never pruned. The cuts are spread across blocks and irreps: at 10 % of the cost, every group keeps 77–97 % of its copies.

**Why the normalization matters.** Ranking raw slice norms instead empties the last block first (one copy per irrep left at 10 % of the cost) and gives a 1.35 eV/atom energy error; norms are not comparable across blocks and irreps. On a small sample of synthetic crystals, the normalized magnitude ranks channels no better than a random order, in line with Muon flattening weight spectra. This is the failure BMuon is designed to remove.

> **TODO — 6.1.** (a) Variante « par groupe » : faite jusqu'à 20 % (résultats dans bmuon/results/fragility\_mp1\_pergroup\_log.txt) ; suffisant pour la figure. (b) Les fractions de coût de ce run utilisent le modèle de coût CG (valeur par défaut de fragility.py) ; refaire les axes avec le modèle « paths » pour être cohérent avec 4.2. (c) Préciser dans les limites qu'un critère fondé sur les gradients (Taylor) fait sans doute mieux ; la comparaison du papier porte sur Muon vs BMuon à critère égal. (d) Superposer la même courbe pour un modèle BMuon dès que le pilote est fini — c'est la figure qui porte le message.

### 6.2 Cost model and speed (Table 1)

Inference time falls almost in proportion to the copies entering the tensor product. Keeping one copy in eight in every block makes energy-plus-forces evaluation 3.2× faster on 512 atoms and 3.7× faster on 216 atoms.

&#91;embedded content: Figure 4 · bmuon/results/speed\_3090\_v2.csv · RTX 3090, OpenEquivariance kernels, Si supercells, 13 Oct 2026\]

On 512 atoms the measured time is close to 0.25 + 0.75 f, where f is the fraction of copies kept: about three quarters of the inference time scales with the type-I channels, which bounds the attainable speed-up near 4×. On 64 atoms the gain saturates near 1.7×, because fixed per-call costs dominate small systems.

**Protocol.** Nequix-MP-1 with random weights of the reduced shape, jitted energy and forces on rattled Si supercells (64, 216, 512 atoms). Each variant is timed interleaved with the full model (base, variant, base, …; 5 rounds of 10 calls) and reported as the median ratio to the surrounding base times. A first, non-interleaved run drifted by up to +20 % over the benchmark, more than the effects measured; interleaving removes that drift.

**Cost of one copy.** Removing half of the copies of a single (block, irrep) changes the time by 1–11 %, close to the noise for groups of 16 copies, so per-copy costs cannot be read off individually. Their pattern, however, singles out a cost model:

| Cost model for a copy | Correlation with measured savings |
| --- | --- |
| number of tensor-product paths it feeds | 0.89 |
| nonzero Clebsch–Gordan coefficients of those paths | 0.84 |
| product of irrep dimensions of those paths | 0.80 |

Scalar copies are therefore not cheap in practice, and a copy of the last block, which feeds only the paths l ⊗ l → 0e, costs four to five times less than a copy of the middle blocks. We use the path count as the cost weight w\_c (Section 4.2).

> **TODO — 6.2.** (a) Répéter sur une RTX 4000 Ada et une A5000 pour montrer que le gain ne dépend pas de la carte. (b) Ajouter des structures MPtrj réelles (pas seulement du Si) et des systèmes plus grands (1 000–5 000 atomes), où le gain compte en dynamique moléculaire. (c) Mesurer en mode « inférence MD » (pas de temps successifs sur la même structure) en plus de l'appel isolé. (d) Remplacer les poids aléatoires par un vrai modèle élagué une fois disponible (le temps ne dépend pas des valeurs des poids, mais c'est plus convaincant).

### 6.3 Post-training compression (Figure 2, Table 2)

> **TODO — protocole.** Entraîner avec la même recette que Nequix : (i) Muon ; (ii) BMuon budget uniforme ; (iii) BMuon budget pondéré par le coût. Puis élagage global sans récupération, classé par magnitude couplée et, pour BMuon, par μ. Baselines : Muon + magnitude ; élagage post-hoc par blocs (k,l) de Wang et al. sur le modèle Muon. **Figure 2** : MAE forces en fonction du temps d'inférence mesuré (pas du nombre de canaux), une courbe par méthode. **Table 2** : pour 2–3 niveaux de vitesse, MAE énergie / forces / stress, CPS Matbench-Discovery et MAE phonons. 3 graines si le calcul le permet.

### 6.4 Gradual pruning with recovery (Table 3)

> **TODO — protocole.** Alterner entraînement et élagage jusqu'à des cibles de coût (par ex. 50 % et 25 % du coût initial), avec phases de récupération, comme la section 5.2 de BMuon. Comparer Muon-magnitude, BMuon-magnitude et BMuon-μ. Optionnel si le calcul manque : 6.3 suffit pour un workshop.

### 6.5 Learned architectures (Figure 3, Table 4)

> **TODO — protocole.** Relever les multiplicités finales par bloc et par irrep issues de BMuon-μ. **Figure 3** : carte bloc × irrep des multiplicités retenues (montrer si des degrés élevés disparaissent dans certains blocs, c.-à-d. un l\_max effectif par bloc). **Table 4** : réentraîner depuis zéro, avec Muon, l'architecture découverte et une architecture uniforme de même coût mesuré ; comparer MAE et CPS. N'écrire la contribution 4 que si l'architecture découverte gagne.

### 6.6 Ablations (Table 5)

> **TODO — protocole.** Sur la version réduite : budget uniforme vs pondéré (à vitesse égale) ; scores sur W vs sur W ⊗ I\_{2l+1} ; α ∈ {1/4, 1/2} ; budget cible K ; fréquence ΔT\_μ ; plancher ε.

### 6.7 Molecular-dynamics sanity check

> **TODO — protocole.** Pour le meilleur modèle élagué et le modèle de référence : dynamique NVE sur 2–3 systèmes, mesurer la dérive d'énergie et le nombre de pas par seconde. Montre que le gain de vitesse est réel en simulation et que la stabilité est conservée.

## 7 Limitations

- **One family of channels.** We budget the channels entering the tensor product. Budgeting the feature channels between blocks as well would make linear\_1, linear\_2 and the self-connection budgeted on both sides; the allocation objective then contains products of allocations, and neither the concavity result nor the closed-form MM update applies.
- **Theory in the linear-factorization sense is open.** We do not establish a width-recovery result for blocks of the form B T A.
- **Kernel-dependent cost.** The weights w\_c are measured for one implementation and hardware; another kernel can change their ratios.
- **One model and dataset.** Results are for Nequix on MPtrj. \[Extension to MACE or Allegro, or to OMat24, is left out.\]

## 8 Conclusion

We extended Budget Muon from MLP channels to the tensor-product layers of equivariant networks, with allocations shared across the components of each irrep copy and a budget that counts compute rather than channels. \[On Nequix, this yields X× faster inference at matched accuracy, and allocations that reveal Y.\] The construction applies to any network whose units are coupled groups of matrix slices with heterogeneous costs.

> **TODO — Sections 7–8.** Ajuster les limites aux expériences réellement faites (supprimer la ligne « one model » si une seconde architecture est testée). Remplir la phrase entre crochets de la conclusion avec le résultat principal ; ne pas dépasser 5 phrases.

## Appendices (à écrire)

| Annexe | Contenu | Comment la remplir |
| --- | --- | --- |
| A. Slice map | tableau complet des tranches de chaque canal, bloc par bloc, chemin par chemin | le générer par script à partir des instructions du produit tensoriel de Nequix (indices dans A, B, C), le vérifier en mettant un canal à zéro et en contrôlant que la sortie ne dépend plus de lui |
| B. Proofs | preuves des Propositions 1–5 | suivre les TODO de la section 5 ; citer BMuon pour les étapes reprises à l'identique |
| C. Implementation | GradientTransformation optax qui remplace Muon sur A, B, C ; convention (in, out) d'e3nn-jax ; export du modèle élagué | décrire le code une fois écrit ; donner le surcoût mesuré par rapport à Muon (comme l'annexe D.6 de BMuon) |
| D. Hyper-parameters | tous les réglages par expérience | tableau par expérience, calqué sur la Table A.1 de Nequix |
| E. Extra results | résultats complets, toutes graines, courbes d'entraînement | générer depuis les logs |

## Global checklist

- [x] 6.1 : courbe de fragilité de nequix-mp-1 sous élagage par magnitude (décide le cadrage)
- [x] 6.2 : benchmark de vitesse et modèle de coût w\_c (décide si les canaux de type I valent le coup)
- [ ] Choisir la venue cible
- [ ] Lire et résumer Wang et al., Stupariu & Manolache, Alnemari & Al-Omair, MorphNet
- [ ] Script de la carte des tranches (annexe A) et test d'élimination d'un canal
- [ ] Modifier Nequix pour que linear\_1 accepte des irreps de sortie différentes de l'entrée
- [ ] Implémenter la transformation BMuon (optax) et la tester sur la version réduite
- [ ] Trancher : pondération par 2l+1, normalisation e3nn, arrondi des multiplicités
- [ ] 6.3 puis 6.5 sur le modèle complet ; 6.4 et 6.6 si le calcul le permet
- [ ] 6.7 : dynamique moléculaire de contrôle
- [ ] Écrire les preuves (annexe B)
- [ ] Réécrire abstract, introduction, conclusion avec les chiffres

## References (provisional)

- Batzner et al. (2022). E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. Nature Communications 13, 2453.
- Koker, Kotak, Smidt (2025). [Training a foundation model for materials on a budget](https://arxiv.org/abs/2508.16067) (Nequix). [Code](https://github.com/atomicarchitects/nequix).
- Anonymous (2026). BMuon: Budgeting Muon's spectral geometry for structured compressibility. Under review at ICLR 2027.
- Jordan et al. (2024). [Muon: an optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/).
- Park et al. (2024). Scalable parallel algorithm for graph neural network interatomic potentials in molecular dynamics simulations. JCTC 20(11).
- Liao et al. (2023). EquiformerV2: improved equivariant transformer for scaling to higher-degree representations.
- Bharadwaj et al. (2025). An efficient sparse kernel generator for O(3)-equivariant deep networks (OpenEquivariance).
- Gordon et al. (2018). MorphNet: fast and simple resource-constrained structure learning of deep networks. CVPR.
- Wang et al. (2026). arXiv 2605.08885. \[TODO : titre exact après lecture\]
- Stupariu & Manolache (2026). arXiv 2605.27662. \[TODO : titre exact après lecture\]
- Alnemari & Al-Omair (2026). Symmetry. \[TODO : titre et référence complète\]
