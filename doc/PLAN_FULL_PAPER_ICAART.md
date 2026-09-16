# Plan d'exécution — Transformer le papier B1 en Full Paper ICAART 2027

> **Destinataire : Claude Code.** Ce document est un plan d'exécution. Suis les phases dans l'ordre.
> Chaque phase a une **Definition of Done (DoD)** vérifiable. Ne passe pas à la phase suivante
> tant que la DoD n'est pas satisfaite. Les phases marquées ⛔ sont bloquantes.
>
> **Contexte.** Le papier existe (`Example.tex`, template SCITEPRESS) et rapporte une étude
> scope × mécanisme (fine-tuning vs LoRA) pour la reconnaissance faciale forensique sur SCface.
> Il est actuellement calibré pour un Short Paper. L'objectif est de le faire classer **Full Paper**
> (12 pages) **sans refaire l'expérimentation principale** : on exploite les checkpoints et les
> scores déjà produits.
>
> **Cible de soumission.** ICAART 2027, deuxième tour, **22 octobre 2026**, catégorie
> **Regular Paper** (surtout pas Position Paper : celui-ci est plafonné à Short d'office).

---

## Table des phases

| Phase | Objet | Durée | Bloquant |
|---|---|---|---|
| −1 | Reconnaissance du dépôt + gate de faisabilité | 0,5 j | ⛔ |
| 0 | Réparations de compilation | 1 j | ⛔ |
| 1 | Analyse spectrale des checkpoints | 4 j | cœur |
| 2 | Exploitation des scores existants | 3 j | cœur |
| 3 | Deux expériences de contrôle | 4 j | |
| 4 | Restructuration et rédaction | 12 j | |
| 5 | Finalisation et conformité | 5 j | ⛔ |

---

# PHASE −1 — Reconnaissance et gate de faisabilité ⛔

**Ne produis aucun code d'analyse avant d'avoir terminé cette phase.**

## −1.1 Inventaire

Explore le dépôt et produis `AUDIT.md` répondant précisément à :

1. **Checkpoints.** Où sont les poids sauvegardés ? Liste tous les fichiers `.pt` / `.pth` / `.ckpt`
   avec leur chemin, leur taille et la configuration à laquelle ils correspondent
   (mécanisme × scope × seed). Attendu : FT layer4, FT layer3+4, FT full, LoRA layer4,
   LoRA layer3+4 (r=8), LoRA layer3+4 (r=16), LoRA full, Hybrid — × 3 seeds.
2. **Checkpoint pré-entraîné.** Où est le IResNet-50 ArcFace/MS1MV3 d'origine (`W_0`) ?
3. **Scores d'évaluation.** Les matrices de similarité probe×gallery ou les listes de
   prédictions par terrain ont-elles été sauvegardées ? Sous quel format ?
4. **Code d'entraînement.** Chemin du script d'entraînement, du builder de modèle,
   de l'implémentation LoRA (comment les adaptateurs sont injectés dans les conv 3×3).
5. **Données.** Chemin de SCface, partition train/test, nombre exact d'identités
   d'entraînement et de test, méthode d'alignement, résolution d'entrée, taille de batch.
6. **Environnement.** Version PyTorch, GPU disponible, temps mesuré d'un run complet.

## −1.2 Gate de décision

Écris en tête de `AUDIT.md` un verdict explicite :

- **GO COMPLET** — checkpoints FT layer3+4, FT full, LoRA layer3+4 et le pré-entraîné existent.
  → Exécute tout le plan.
- **GO PARTIEL** — les checkpoints manquent mais le code d'entraînement est fonctionnel et un
  run coûte < 2 h. → Relance uniquement les 3 configs nécessaires (FT l3+4, Full FT, LoRA l3+4,
  1 seed chacune suffit pour l'analyse spectrale), puis exécute tout le plan.
- **NO-GO** — ni checkpoints ni code réutilisable. → **Arrête-toi et signale-le.** Bascule sur le
  plan de repli (voir « Repli » en fin de document) : Phases 0, 2, 4 seulement, cible Short Paper.

## −1.3 Espace de travail

Crée l'arborescence :

```
analysis/
  scripts/          # scripts d'analyse
  outputs/          # CSV, JSON de résultats
  figures/          # PDF des figures pour LaTeX
paper/              # copie de travail du .tex
```

Ne modifie jamais les sources originales en place : travaille sur une copie versionnée (branche git dédiée `full-paper`).

**DoD −1 :** `AUDIT.md` existe, contient le verdict, et liste des chemins réels vérifiés (pas supposés).

---

# PHASE 0 — Réparations de compilation ⛔

Le papier **ne compile pas actuellement**. Rien d'autre ne compte tant que ce n'est pas réglé.

## 0.1 Corrections du préambule

Dans `Example.tex` :

| Action | Détail |
|---|---|
| Ajouter | `\usepackage{siunitx}` **avant** `\usepackage{SCITEPRESS}` |
| Supprimer | `\usepackage{lipsum}` (ligne ~21) et son commentaire français |
| Supprimer | le `\usepackage{subcaption}` en double (il apparaît lignes ~4 et ~26 ; n'en garder qu'un) |

## 0.2 Suppression du faux texte ⛔ CRITIQUE

Section 3.1, autour de `\input{fig_pipeline.tex}` :

```latex
% À SUPPRIMER intégralement :
\lipsum[1-2]
\lipsum[3-6]
```

Vérifie ensuite par `grep -n lipsum Example.tex` → doit retourner 0 résultat.

## 0.3 Unité invalide

Remplace `\SI{43.6}{\mega\nothing}` (caption de la table des configurations) par `43.6\,M`.
`\nothing` n'existe plus en siunitx v3.

## 0.4 Insérer la figure LoRA orpheline

`fig_lora_block.tex` existe mais n'est jamais inclus. C'est actuellement la **seule** description
de la manière dont LoRA est injecté dans les convolutions 3×3 — sans elle, le mécanisme central
du papier n'est décrit nulle part.

Insère `\input{fig_lora_block.tex}` dans la Section 3.2, juste après le paragraphe
« Low-rank adaptation », et ajoute dans le corps du texte 3 à 4 phrases décrivant la
paramétrisation : `A` réduit les canaux via une conv 1×1 au rang `r`, `B` ré-étend via le
noyau spatial 3×3 d'origine, sortie mise à l'échelle par `α/r`, `B` initialisé à zéro.
Référence la figure avec `Fig.~\ref{fig:lora_block}`.

## 0.5 Erreur factuelle interne ⛔

Section 5.1, la phrase affirmant que LoRA layer4 a « a similar footprint on that stage » est
**fausse** : 0,50 M contre 26,02 M, soit un facteur 52, contredit par la table des configurations
du papier lui-même.

Remplace par une formulation exacte, du type :
> even though this stage carries the largest share of the backbone's parameters, the rank-8
> adapters can only reach 0.50M trainable weights against 26.02M for direct fine-tuning.

## 0.6 Contradiction figure/texte ⛔

Le texte affirme que LoRA layer4 « does not adapt at all (−1.2, no gain over the un-adapted
model) ». Or la figure du terrain `ir 4.20 m` place LoRA l4 à **49,7 %** contre une baseline de
**35,6 %**, soit +14,1 points sur la condition la plus dure.

LoRA layer4 n'est pas inerte : il **redistribue** (il gagne sur l'IR long et perd ailleurs), et
ΣΔ ≈ 0 masque cette redistribution.

Reformule le passage pour dire exactement cela, et ajoute une phrase d'interprétation : sous
contrainte de capacité, l'adaptateur ne peut satisfaire qu'une partie des terrains et sacrifie
les conditions saturées. C'est un **résultat intéressant**, pas une erreur à cacher.

## 0.7 Abstract

Il fait actuellement **247 mots**, la limite SCITEPRESS est de **200**. Réduis-le en coupant
la redondance entre la 2ᵉ et la 3ᵉ « finding ». Vérifie par script.

## 0.8 Nettoyage

- Supprime tous les commentaires en français dans `Example.tex`, `fig_pipeline.tex`,
  `fig_lora_block.tex`, `references.bib` (dont l'en-tête « Papier B1 CoopIS 2026 » —
  il ne doit subsister aucune trace d'un autre lieu de soumission).
- Supprime les grands blocs de commentaires résiduels du template SCITEPRESS (lignes ~850–1080).

**DoD 0 :**
- [ ] `pdflatex` + `bibtex` + `pdflatex` ×2 s'exécutent avec **0 erreur**
- [ ] `grep -c lipsum Example.tex` → `0`
- [ ] Abstract ≤ 200 mots (script de comptage)
- [ ] Aucune référence non résolue (`??`) dans le PDF
- [ ] Le PDF contient les deux figures TikZ (pipeline **et** bloc LoRA)

---

# PHASE 1 — Analyse spectrale des checkpoints ⭐ LE CŒUR

**C'est cette phase qui fait basculer le papier de Short à Full.** Aucun entraînement :
uniquement du chargement de poids et de l'algèbre linéaire.

**Justification.** Le papier affirme actuellement que LoRA échoue sur l'infrarouge « parce que
l'adaptation requise est de rang élevé ». C'est une **conjecture non mesurée**. Les poids
nécessaires pour la mesurer sont déjà sur disque.

## 1.1 Rang effectif de ΔW

Crée `analysis/scripts/spectral_analysis.py`.

Pour chaque couche convolutive et linéaire adaptée, pour chaque checkpoint fine-tuné :

```python
import numpy as np, torch

def delta_spectrum(W0: torch.Tensor, Wft: torch.Tensor):
    """W: (C_out, C_in, kh, kw) pour une conv, (d_out, d_in) pour fc."""
    dW = (Wft - W0).reshape(W0.shape[0], -1).float().cpu().numpy()
    s = np.linalg.svd(dW, compute_uv=False)
    s = s[s > 0]
    energy = s**2
    p = energy / energy.sum()
    erank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))   # Roy & Vetterli
    cum = np.cumsum(energy) / energy.sum()
    r90 = int(np.searchsorted(cum, 0.90) + 1)
    r99 = int(np.searchsorted(cum, 0.99) + 1)
    return dict(
        max_rank=int(min(dW.shape)),
        erank=erank, r90=r90, r99=r99,
        rel_norm=float(np.linalg.norm(dW) / np.linalg.norm(
            W0.reshape(W0.shape[0], -1).float().cpu().numpy())),
        spectrum=s.tolist(),
    )
```

**Checkpoints à traiter :** `FT layer3+4` (prioritaire), `Full FT`, et pour comparaison
`LoRA layer3+4` (reconstruis son ΔW effectif par `(α/r)·B·A` avant de le comparer).

**Sorties :**
- `analysis/outputs/spectral_per_layer.csv` — colonnes :
  `checkpoint, seed, layer_name, stage, max_rank, erank, r90, r99, rel_norm`
- `analysis/outputs/spectra/` — un `.npy` de valeurs singulières par couche

**Agrégation demandée :** `erank` médian par étage (layer3, layer4, fc), moyenné sur les seeds.

## 1.2 Figure « spectres de valeurs singulières »

`analysis/figures/fig_spectrum.pdf` — valeurs singulières normalisées (s_i / s_1) en échelle log,
axe x = indice, une courbe par étage (layer3, layer4, fc) pour le FT layer3+4.

**Trace une ligne verticale à r = 8 et r = 16** (les rangs LoRA testés) pour rendre visible
d'un coup d'œil la fraction d'énergie que l'adaptateur ne peut pas capturer.

Annote sur la figure : « LoRA r=8 captures only X % of the fine-tuning update energy ».
Calcule X réellement, ne l'invente pas.

## 1.3 Profil de déplacement par étage — valide H1

Sur le checkpoint **Full FT** uniquement, calcule pour chaque étage
(stem, layer1, layer2, layer3, layer4, fc) :

```
rel_displacement(stage) = ||W_ft - W_0||_F / ||W_0||_F   # agrégé sur les paramètres de l'étage
```

**Sortie :** `analysis/figures/fig_stage_displacement.pdf`, diagramme en barres.

**Lecture attendue :** si le fine-tuning complet, laissé libre d'adapter tout le réseau,
concentre spontanément son déplacement sur layer3+4, alors H1 cesse d'être une observation
empirique et devient une **explication structurelle**. C'est un résultat fort pour ~20 lignes
de code. Si le profil est plat, dis-le honnêtement : c'est aussi une information.

## 1.4 Contrôle BatchNorm ⛔ OBLIGATOIRE

Compare les buffers `running_mean` et `running_var` des checkpoints **LoRA** à ceux du
pré-entraîné.

```python
drift = (bn_lora.running_mean - bn_pre.running_mean).norm() / bn_pre.running_mean.norm()
```

**Enjeu.** Si les statistiques BN se mettent à jour pendant l'entraînement LoRA, le backbone
n'est pas réellement « gelé », la comparaison est confondue, et — l'infrarouge ayant des
statistiques d'image radicalement différentes — cela pourrait constituer une **explication
alternative à tout l'effet H3**. Un relecteur compétent posera la question.

**Deux issues :**
- Dérive négligeable (< 1 %) → ajoute en Section 3.1 : *« the BN running statistics are kept
  frozen in all LoRA configurations »*. Le confondant disparaît en une phrase.
- Dérive significative → déclare-le, et déclenche l'expérience de contrôle 3.2.

**Sortie :** `analysis/outputs/bn_drift.csv`.

## 1.5 Directions intruses — BONUS, seulement si 1.1–1.4 sont terminés

Réplique l'analyse de Shuttleworth et al. en convolutionnel : pour les top-k vecteurs singuliers
gauches de `W_adapted`, calcule le cosinus maximal avec le sous-espace de tête de `W_0`.
Une direction « intruse » est une direction de tête du modèle adapté qui n'a aucun correspondant
dans le pré-entraîné.

Compare FT vs LoRA. **Ne dépasse pas 1 journée dessus.** Si ça résiste, abandonne : les points
1.1 à 1.4 suffisent.

**DoD 1 :**
- [ ] `spectral_per_layer.csv` rempli pour au moins FT l3+4 et Full FT
- [ ] `erank` médian de layer3 et layer4 connu et comparé numériquement à r=8 et r=16
- [ ] `fig_spectrum.pdf` et `fig_stage_displacement.pdf` générés
- [ ] `bn_drift.csv` produit et interprété
- [ ] Un fichier `analysis/outputs/FINDINGS_PHASE1.md` résumant en 10 lignes ce que les chiffres disent réellement — **y compris s'ils contredisent l'hypothèse du papier**

> ⚠️ **Honnêteté scientifique.** Si le rang effectif mesuré s'avère *faible* (proche de 8),
> l'explication actuelle du papier est fausse et il faut la réécrire, pas forcer les chiffres.
> Un résultat qui infirme une hypothèse posée a priori se rapporte tel quel — c'est
> précisément ce que la structure H1/H2/H3 du papier permet de faire proprement.

---

# PHASE 2 — Exploitation des scores existants

Toujours aucun entraînement : inférence seule, ou réutilisation des matrices de scores.

## 2.1 Courbes CMC rang-1 → rang-10

Depuis les matrices de similarité, trace la CMC sur le terrain `ir 4.20 m` pour les 5 méthodes
(Base, FT l3+4, Full FT, LoRA l3+4, Hybrid).

**Sortie :** `analysis/figures/fig_cmc_ir420.pdf`.

**Intérêt :** si LoRA rattrape au rang 5 mais pas au rang 1, cela signifie qu'il préserve
l'information d'identité mais dégrade le classement — nuance qualitative absente du papier.

## 2.2 Écart de domaine dans l'espace d'embedding ⭐

Pour chaque identité de test, calcule le cosinus entre l'embedding de son mugshot (galerie) et
l'embedding de ses images de surveillance, décomposé par modalité et distance, pour :
pré-entraîné / FT l3+4 / LoRA l3+4.

**Sortie :** `analysis/figures/fig_embedding_gap.pdf` — boxplots ou violons,
axe x = les 6 terrains, une couleur par méthode.

**Résultat attendu et très démonstratif :** LoRA referme l'écart sur le visible mais **pas** sur
l'infrarouge. Tu montres alors la défaillance **dans la représentation elle-même**, et pas
seulement dans la métrique finale. C'est ce qui répond à la question de review
« Improve critical discussion? ».

## 2.3 Statistiques robustes — répare la faiblesse n=3 ⛔

Abandonne les p-values calculées sur 3 seeds (le papier en rapporte trois : 0,54 / 0,05 / 0,03,
sans nommer le test, sans correction de comparaisons multiples). Remplace par un **bootstrap au
niveau identité**, qui exploite les 30 identités de test plutôt que les 3 seeds :

```python
# Pour chaque méthode : vecteur binaire correct/incorrect par identité et par terrain
# 10 000 rééchantillonnages avec remise des 30 identités
# -> IC 95 % sur le rank-1 de chaque méthode
# -> IC 95 % sur la DIFFÉRENCE APPARIÉE FT - LoRA (rééchantillonner les mêmes identités
#    pour les deux méthodes : c'est ce qui rend le test puissant malgré n=30)
```

Rapporte aussi une taille d'effet (Cohen's d apparié).

**Point important de rédaction :** supprime l'usage actuel de « p = 0.05 » pour argumenter
l'**équivalence** en H2. C'est une inversion logique (p = 0,05 est conventionnellement
*significatif*, et l'absence de significativité ne démontre jamais l'équivalence). Remplace par
l'IC de la différence, en disant explicitement ce qu'il exclut ou n'exclut pas.

**Sortie :** `analysis/outputs/bootstrap_ci.csv`.

## 2.4 Métriques complémentaires

Depuis les mêmes matrices : rang-5 et TPIR@FAR = 1 %. Ajoute-les en colonnes de la table
par terrain.

**DoD 2 :**
- [ ] 3 figures générées (`fig_cmc_ir420`, `fig_embedding_gap`, + une visualisation des IC)
- [ ] `bootstrap_ci.csv` produit
- [ ] Toutes les p-values sur 3 seeds retirées du `.tex` et remplacées par des IC

---

# PHASE 3 — Deux expériences de contrôle

Le strict minimum de calcul neuf. Budget : **6 runs, ~6 heures machine**.

## 3.1 Baseline `fc-only` — 3 seeds

Fine-tuning de la seule projection finale `fc` + tête ArcFace, backbone entièrement gelé.
Même protocole unifié que tout le reste (AdamW, lr 1e-4, wd 0.1, 20 époques, batch 50/50).

**Pourquoi c'est indispensable.** Le papier adapte `fc` dans **toutes** ses configurations,
y compris LoRA. Un relecteur objectera donc légitimement : *« et si tout le gain venait de fc ? »*.
Sans cette baseline, l'objection est imparable. Avec elle, tu prouves que le gain provient des
étages convolutifs.

## 3.2 Seconde baseline PEFT — 3 seeds

**Choix conditionné par le résultat de la Phase 1.4 :**

- **Si dérive BN détectée** → relance `LoRA layer3+4` avec BN explicitement gelées
  (`bn.eval()` + `requires_grad=False` sur les affines). Élimine le confondant.
- **Si pas de dérive BN** → implémente **BitFit** (entraînement des seuls termes de biais +
  fc + tête), scope layer3+4. Coût : quelques dizaines de milliers de paramètres.

**Apport :** tu ne compares plus seulement deux mécanismes mais **cinq**, sur un axe de coût
croissant : `fc-only` → `BitFit` → `LoRA` → `FT sélectif` → `FT complet`. C'est exactement ce
que vise la question de review « Needs comparative evaluation? ».

## 3.3 Intégration

Ajoute ces configurations à la table des configurations (avec leur budget de paramètres mesuré)
et à la grille de résultats, et fais-les apparaître dans la figure du terrain `ir 4.20 m`.

## 3.4 Contrôle optionnel — FT sans ancrage (voir 4.3.2.C)

Un 7ᵉ run optionnel, **si le budget le permet** : `FT layer3+4` sans le mélange mugshot/dégradé
(100 % dégradé), 1 seed. Objectif : tester si l'ancrage est bien ce qui empêche l'oubli
catastrophique et l'instabilité observés chez PETALface en son absence — argument central de la
Section 4.3/6.x. Coût ≈ 1h. Si le temps manque, ne bloque pas la Phase 3 pour ça : reporte-toi à
la mesure passive de 4.3.2.B et assume l'hypothèse non démontrée dans le texte.

**DoD 3 :**
- [ ] 6 runs terminés, résultats par terrain enregistrés
- [ ] Les 2 nouvelles configurations figurent dans les tables et figures
- [ ] Leur ΣΔ est calculé avec la même convention que les autres
- [ ] (Optionnel) Run de contrôle sans ancrage terminé et rapporté dans `FINDINGS_PHASE1.md`
      ou un fichier dédié, qu'il confirme ou infirme l'hypothèse

---

# PHASE 4 — Restructuration et rédaction

**Cible : ~42 000 caractères hors espaces** (le papier est actuellement à ~27 500 ; la fourchette
Regular Paper est 10 000–50 000). Un papier à 27 500 caractères *ressemble* à un Short Paper de
8 pages, et sera classé comme tel.

Script de comptage à créer, à lancer après chaque session de rédaction :

```bash
# depuis le PDF compilé, hors espaces
pdftotext paper.pdf - | tr -d '[:space:]' | wc -c
```

## 4.1 Réhabiliter l'hybride — gratuit, et corrige un défaut grave ⛔

**Constat actuel :** le papier écarte en une seule phrase son **meilleur résultat**.
L'hybride obtient ΣΔ = +92,3 (le plus élevé du papier), 84,2 % sur `ir 4.20 m` (le plus élevé,
+4,5 points au-dessus de la méthode recommandée), pour **14,2 M paramètres contre 42,3 M** pour
le fine-tuning sélectif recommandé — soit **3× moins**.

Pire, la Discussion le disqualifie en le comparant à LoRA (« trains far more parameters than
LoRA ») au lieu de le comparer à la méthode que le papier recommande, face à laquelle il est
meilleur **sur les deux axes**. En l'état, cela ressemble à du cherry-picking et sera relevé.

**Actions :**
1. Intègre l'hybride comme **ligne pleine** de la table de grille (pas seulement une colonne
   de la table par terrain), et vérifie que sa valeur ΣΔ = +92,3 apparaît bien dans un tableau —
   elle n'existe actuellement que dans une phrase du texte.
2. Fais-en une **quatrième contribution** dans l'introduction :
   > *combining low-rank adapters on the low-level stages with direct fine-tuning of the
   > high-level stages reaches the best infrared accuracy of the whole grid at one third of the
   > trainable parameters of selective fine-tuning.*
3. Discute-le honnêtement : la variance sur `ir 4.20 m` (±2,1) et le coût de complexité
   justifient-ils ou non d'en faire la recommandation ? Tranche, et assume.

Tu passes de trois à quatre findings **sans une seule expérience nouvelle**.

## 4.2 Requalifier la contribution n°2

Le fine-tuning `layer3+4` représente **96,9 % du backbone**. L'argument actuel — « on garde le
reste du réseau gelé » — n'économise que 3 % des paramètres par rapport au fine-tuning complet,
et l'affirmation que le full backbone est « wasteful » est une surinterprétation.

**Reformule :** l'intérêt n'est pas l'économie de paramètres mais le fait que les étages bas
n'ont **rien à apprendre** du domaine forensique — ce que la Phase 1.3 démontre désormais
quantitativement. Renvoie explicitement à la figure de déplacement par étage.

## 4.3 Traiter la contradiction avec PETALface ⛔ LE POINT LE PLUS IMPORTANT

**PETALface (Narayan, Nair, Xu, Chellappa, Patel — WACV 2025, arXiv:2412.07771), votre voisin
le plus proche, conclut l'inverse de vous** : sur TinyFace et BRIAR, le PEFT **surpasse** le
fine-tuning complet, y compris **sur le domaine cible basse résolution lui-même** (pas
seulement en préservation haute résolution). Le papier le cite pourtant comme un travail qu'il
prolonge (« We follow this line of work »). Un relecteur informé attaquera cela en premier.

### 4.3.1 Ce que le protocole de PETALface révèle réellement (lu dans le papier, pas supposé)

> ⚠️ Ces éléments proviennent d'une extraction automatisée du PDF/HTML du papier (deux passes
> ont donné des chiffres légèrement différents : 71.11 vs 71.32 sur TinyFace). La tendance et
> l'ordre de grandeur sont cohérents entre les deux extractions, mais **vérifie manuellement
> sur le PDF source avant toute citation dans le texte final** — une citation inexacte d'un
> papier concurrent est pire qu'une absence de citation.

| Élément | PETALface | Ce papier |
|---|---|---|
| Ancrage anti-oubli (mélange HQ/dégradé en entraînement) | **Absent** sur tous les baselines — full FT et LoRA naïf entraînés directement sur le train set basse résolution, sans image haute qualité | **Présent** — 50/50 mugshot/dégradé à chaque batch, sur toutes les configurations y compris full FT |
| Comportement du full FT sur le domaine cible | **Dégrade** la performance cible elle-même (TinyFace : pré-entraîné ~72.7–73.3 → full FT ~71.1–71.3 ; BRIAR : 55.3 → 44.8) | **Améliore** fortement la performance cible, partout |
| Explication donnée par les auteurs (leur Annexe C) | Gradients initiaux du full FT énormes, « even after clipping » → convergence instable | non mesuré chez nous (à faire, voir 4.3.2.C) |
| Architecture | Transformer (Swin-B), LoRA sur attention qkv + MLP | CNN (IResNet-50), LoRA sur convs 3×3 |
| Méthode PEFT comparée | LoRA **double**, pondéré dynamiquement par un score de qualité d'image — pas un LoRA simple | LoRA statique standard |
| Échelle des données cible | Très rare par identité (TinyFace : ~3 images/identité) | Rare également (SCface), mais compensée par l'ancrage |
| Modalité couverte | Résolution seule (aucun décalage spectral) | Résolution **et** infrarouge (décalage spectral) |

**Conclusion : la contradiction est réelle, mais les deux protocoles ne sont pas
superposables.** Trois mécanismes distincts l'expliquent, pas un seul — et les deux premiers
sont désormais vérifiables dans le texte même de PETALface, donc citables, plutôt qu'une
hypothèse interne non étayée :

1. **Absence d'ancrage anti-oubli chez PETALface.** Leur full FT n'a jamais vu d'image haute
   qualité pendant l'entraînement basse résolution → il oublie. Le nôtre en voit à chaque
   batch → il n'oublie pas. C'est l'explication du plan initial, maintenant étayée par le
   protocole documenté de PETALface plutôt qu'affirmée sans preuve externe.
2. **Instabilité d'optimisation du full FT sans contrainte, sur données cible rares.** Chez
   PETALface, le full FT dégrade la performance sur le domaine cible *lui-même*, pas seulement
   en HQ — un phénomène d'optimisation (gradients initiaux énormes, leur Annexe C), distinct de
   l'oubli catastrophique. Chez nous, le full FT n'échoue jamais sur le domaine cible. C'est
   *peut-être* aussi un effet régularisateur de l'ancrage (pas seulement mémoriel), mais **cela
   reste une hypothèse non démontrée** tant qu'un contrôle sans ancrage n'a pas été fait
   (voir 4.3.2.C).
3. **Architecture et méthode PEFT différentes.** Transformer/attention vs CNN/convolution ;
   LoRA simple statique (nous) vs LoRA double pondéré par qualité (PETALface — une méthode plus
   riche que le LoRA nu que nous étudions). Cet axe prolonge notre propre cadrage théorique
   (Biderman/Shuttleworth viennent aussi du monde Transformer/langage) : nous testons
   précisément si leurs constats survivent en convolutionnel. Le nommer explicitement en
   comparaison à PETALface renforce cette originalité au lieu de l'affaiblir.

### 4.3.2 Actions concrètes pour rendre la comparaison solide

**A. Rédaction (obligatoire, aucune expérience nouvelle) :**
- Rédige le paragraphe de Section 6 (Discussion) autour des **trois mécanismes** ci-dessus,
  appuyé par le tableau de comparaison de protocoles — la transparence sur la
  non-comparabilité renforce la crédibilité, elle ne l'affaiblit pas.
- Cite le fait que le **LoRA naïf seul** de PETALface (pas seulement leur méthode complète)
  bat déjà leur full FT (75.64 vs ~71.1–71.3 sur TinyFace, à vérifier précisément) : cela montre
  que la divergence vient du protocole d'entraînement du full FT chez eux, pas d'un baseline
  volontairement affaibli — argument plus honnête et donc plus solide face à un relecteur.
- Mentionne que notre étude couvre une modalité (infrarouge, décalage spectral) absente de
  PETALface (TinyFace/BRIAR = dégradation de résolution uniquement) — différenciation légitime,
  pas seulement défense.
- **Vérifie manuellement chaque chiffre/citation PETALface sur le PDF source avant intégration**
  (voir avertissement 4.3.1).

**B. Mesure passive (prévue dans la version initiale du plan, coût ≈ 0, inférence seule) :**
- Performance des modèles FT sur les mugshots haute qualité tenus à l'écart après adaptation →
  preuve directe de l'absence d'oubli (mécanisme 1).

**C. Contrôle actif — NOUVEAU, optionnel, à rattacher à la Phase 3 (1 run, ~1h) :**
- **FT layer3+4 (ou full) SANS ancrage** (mugshots retirés du batch, 100 % dégradé) — 1 seed
  suffit pour un signal qualitatif. Si la performance cible se dégrade (comme chez PETALface)
  ou que la performance mugshot chute fortement, cela **confirme empiriquement** le mécanisme 1
  (et possiblement 2) au lieu de rester une hypothèse. Si rien ne change, c'est aussi une
  information honnête à rapporter : cela affaiblirait l'explication par l'ancrage et orienterait
  vers l'architecture (mécanisme 3) comme explication principale.
- **Ne dépasse pas une demi-journée dessus.** Si ça ne tient pas dans le budget de la Phase 3
  (6h déjà allouées à 3.1/3.2), reste sur la mesure passive (B) et assume l'hypothèse non
  démontrée en toutes lettres dans le texte plutôt que de la sur-affirmer.

**DoD 4.3 :**
- [ ] Tableau de comparaison de protocoles (PETALface vs nous) présent en Section 6
- [ ] Paragraphe citant les trois mécanismes, pas un seul
- [ ] Chiffres/citations PETALface vérifiés manuellement sur le PDF source avant intégration
- [ ] Mesure mugshot post-adaptation rapportée (4.3.2.B)
- [ ] (Optionnel) Résultat du contrôle sans ancrage rapporté honnêtement, qu'il confirme ou
      infirme l'hypothèse du mécanisme 1/2

## 4.4 Recadrage pour ICAART

ICAART est une conférence agents & IA, pas vision. **Relevance** est le premier critère noté et
c'est votre note la plus fragile — elle ne se corrige que par le cadrage.

1. **Titre** — déplace le centre de gravité vers la question générale. Exemple :
   *« When Does Low-Rank Adaptation Fail? A Systematic Scope × Mechanism Study on Degraded
   Infrared Surveillance »*. Le forensique devient le cas d'étude, pas le sujet.
2. **Introduction** — le premier paragraphe doit poser la question générale (quand l'adaptation
   à faible rang est-elle suffisante, et peut-on le prédire avant d'entraîner ?), le SCface
   n'arrivant qu'au troisième.
3. **Ajoute un paragraphe « Contribution and applicability »** en fin d'introduction. Les
   guidelines l'exigent noir sur blanc : *« Each paper should clearly indicate the nature of its
   technical/scientific contribution, and the problems, domains or environments to which it is
   applicable. »* Assume explicitement le type de contribution — étude empirique systématique
   avec analyse mécaniste, pas nouvelle méthode. L'assumer est plus solide que le maquiller.
4. **Topics à la soumission** : *Vision and Perception* en principal,
   *Privacy, Safety, Security, and Ethical Issues* en secondaire (rareté et sensibilité des
   données forensiques).

## 4.5 Bibliographie : 11 → ~28 références ⛔

**Le point le plus rentable du plan.** 11 références, c'est visuellement un Short Paper ;
un Full Paper en porte 25 à 35.

### 4.5.1 Corriger les entrées fausses

| Clé | Problème | Correction |
|---|---|---|
| `ye2024lgaf` | **Titre ET auteurs inventés** | Vrai titre : *Local and Global Feature Attention Fusion Network for Face Recognition*. Auteurs : **Wang Yu, Wei Wei**. arXiv:2411.16169 |
| `narayan2025petalface` | Auteurs faux | **Narayan, Kartik; Nair, Nithin Gopalakrishnan; Xu, Jennifer; Chellappa, Rama; Patel, Vishal M.** — WACV 2025 |
| `papantoniou2024arc2face` | « Kotsia, Bernhard » conflate deux noms | Vérifier sur la page ECCV/arXiv officielle et corriger |
| `shekhar2019scface` | Attribution et chiffres (73,3 / 93,5 / 98,0) non vérifiés | Vérifier la source ; si introuvable, retirer la ligne du tableau |
| `ding2024lorac`, `butt2024heterogeneous` | `and others` | Développer les listes d'auteurs complètes |

**Règle absolue : aucune entrée ne doit rester avec `and others`.** Le rendu apalike produit un
« et al. » inacceptable en camera-ready, et une référence inventée détectée fait basculer un
relecteur en mode hostile sur tout le reste du papier.

### 4.5.2 Ajouter ~17 références

- **PEFT** : AdaLoRA, DoRA, (IA)³, Adapters (Houlsby et al.), prefix-tuning, LoRA+
- **Cross-spectral / NIR-VIS** : CASIA NIR-VIS 2.0, LAMP-HQ, une méthode NIR-VIS récente
- **Benchmarks basse qualité** : TinyFace, IJB-S
- **Losses de reconnaissance** : MagFace, CurricularFace
- **Fondements théoriques de l'analyse Phase 1** : Roy & Vetterli (effective rank),
  Aghajanyan et al. (intrinsic dimensionality), Li et al. (intrinsic dimension of objective landscapes)

Ces trois dernières sont **nécessaires** : elles fondent la métrique que tu utilises en Phase 1.

## 4.6 Nouvelles sections à rédiger

| Section | Contenu | Caractères |
|---|---|---|
| **6.1 Spectral analysis of the learned updates** (nouvelle) | Phase 1 : rang effectif, spectres, déplacement par étage. **C'est la section qui justifie le statut Full Paper.** | +5 000 |
| **5.x Embedding-space analysis** (nouvelle) | Phase 2.2 | +2 500 |
| Baselines `fc-only` / BitFit dans la grille | Phase 3 | +1 200 |
| Discussion PETALface | 4.3 | +1 800 |
| Hybride réhabilité | 4.1 | +1 500 |
| Recadrage intro + contribution/applicability | 4.4 | +1 200 |
| Bibliographie étendue | 4.5 | +2 500 |
| **Total** | | **+15 700** |

27 500 + 15 700 ≈ **43 200 caractères**. Cible atteinte.

## 4.7 Limitations affinées

Les nouvelles mesures en **retirent** deux :
- le BatchNorm n'est plus un confondant (mesuré en 1.4)
- le « n = 3 seeds » est remplacé par des IC bootstrap au niveau identité

**Conserve honnêtement** : un seul backbone (IResNet-50), un seul benchmark (SCface),
30 identités de test — protocole non standard, les protocoles SCface usuels utilisant 43/44 ou
50 identités, ce qui limite la comparabilité avec la littérature —, et le fait que train et test
partagent les mêmes familles de caméras (conclusions valables pour de nouvelles identités,
pas de nouveaux capteurs).

**DoD 4 :**
- [ ] Comptage ≥ 40 000 caractères hors espaces
- [ ] PDF formaté ≤ 12 pages
- [ ] ≥ 25 références, **0** entrée avec `and others`, **0** référence non vérifiée
- [ ] 4 findings annoncés en introduction et repris en conclusion
- [ ] Paragraphe « Contribution and applicability » présent
- [ ] Paragraphe PETALface présent

---

# PHASE 5 — Finalisation et conformité ⛔

## 5.1 Vérification préalable — À FAIRE AVANT TOUT LE RESTE ⛔

Le fichier `references.bib` portait l'en-tête « Papier B1 **CoopIS 2026** » et le projet contient
`coopis26_2.pdf`.

**Si ce papier est actuellement en review à CoopIS ou ailleurs, la soumission à ICAART constitue
une soumission simultanée** : rejet automatique sans review et signalement éthique INSTICC.
Tous les papiers sont passés à l'analyse anti-plagiat *avant* review.

**Signale ce point aux auteurs et exige une réponse explicite avant de finaliser.**
Si la version CoopIS a été publiée ou déposée publiquement, l'outil la détectera.

## 5.2 Anonymisation double-blind ⛔

Supprime de `Example.tex` :
- les noms d'auteurs, les blocs `\author{}` et `\affiliation{}`
- toute mention de « HANALAB », « ENSI », « Ecole Nationale des Sciences de l'Informatique »,
  « Manouba », « Tunisia »
- les adresses e-mail
- la section remerciements
- toute formulation auto-identifiante (« in our previous work [ref] we showed… » → citer à la
  troisième personne)

**Test final :** relis le PDF en te demandant « est-ce que ce texte trahit l'équipe ? ».

**Contrainte associée :** ne déposer le papier ni sur arXiv, ni sur un site personnel, ni sur un
dépôt institutionnel entre la soumission et la notification.

## 5.3 Déclaration outils d'IA

Les guidelines l'exigent dans les remerciements — mais les remerciements doivent être supprimés
pour le double-blind. **Prépare le texte dès maintenant dans un fichier séparé
`CAMERA_READY_NOTES.md`, à réinsérer au camera-ready uniquement.**

## 5.4 Vérifications finales

- [ ] Caractères hors espaces sur le PDF final : entre 10 000 et 50 000 (viser 40–45 000)
- [ ] PDF formaté ≤ 12 pages
- [ ] Compilation propre, aucune référence `??`, aucun `Overfull \hbox` grave
- [ ] Toutes les figures lisibles en niveaux de gris
- [ ] Les captions de tables sont **au-dessus**, celles de figures **en dessous** (règle SCITEPRESS)
- [ ] Anglais relu intégralement
- [ ] ≤ 9 auteurs

## 5.5 Soumission

- [ ] Plateforme **PRIMORIS**
- [ ] Catégorie : **Regular Paper** ⛔ — Position Paper est plafonné à Short Paper d'office.
      **C'est le seul choix irréversible de toute la procédure.**
- [ ] Deadline : **22 octobre 2026** (confirmer auprès du secrétariat que les Regular Papers
      sont bien acceptés à cette date : le libellé du site, « Position Papers / Regular Papers »,
      est ambigu)
- [ ] Topics : *Vision and Perception* + *Privacy, Safety, Security, and Ethical Issues*

## 5.6 Préparer le rebuttal dès maintenant

Crée `REBUTTAL_PREP.md` listant les objections anticipées et la réponse factuelle à chacune :

| Objection anticipée | Réponse préparée |
|---|---|
| « Un seul backbone » | Limitation assumée ; l'analyse spectrale est architecture-agnostique et reproductible |
| « 30 identités de test » | IC bootstrap au niveau identité ; les écarts principaux restent séparables |
| « PETALface conclut l'inverse » | Section 6.x : condition d'inversion identifiée (ancrage mugshot vs oubli catastrophique) |
| « Pourquoi pas l'hybride ? » | Section 4.1 : discuté et tranché explicitement |
| « Pas de méthode nouvelle » | Contribution revendiquée comme empirique + mécaniste, annoncée dès l'introduction |

Un relecteur qui coche « needs more experiments » se répond avec **des résultats**, pas des
arguments. Garde donc en réserve, non publiées, une ou deux analyses supplémentaires prêtes
à être ajoutées au rebuttal.

---

# Repli — si la Phase 1 échoue

Si au **jour 6** l'analyse spectrale n'a rien donné (checkpoints perdus, code non fonctionnel,
résultat plat sans signal) : **arrête les frais.**

Exécute Phases 0, 2, 4.1 à 4.5, 5. Tu obtiens un Short Paper propre, indexé Scopus, sans
erreur, avec une bibliographie correcte et une discussion honnête. Ce n'est pas un échec :
c'est une publication indexée qui laisse davantage de marge de nouveauté pour l'extension journal
prévue ensuite.

**Ne soumets jamais un Full Paper bancal plutôt qu'un Short Paper solide.**

---

# Justification — pourquoi ce plan satisfait les critères de review

## Les neuf questions du formulaire ICAART

| Question posée aux relecteurs | Avant | Après | Ce qui le prouve |
|---|---|---|---|
| Needs more experimental results? | **Oui** | Non | Analyse spectrale, déplacement par étage, CMC, embeddings, bootstrap, 2 baselines : de ~8 à ~15 expériences rapportées |
| Needs comparative evaluation? | **Oui** | Non | 5 mécanismes sur un axe de coût croissant au lieu de 2 |
| Improve critical discussion? | Oui | Non | Section mécaniste, PETALface résolu, hybride argumenté |
| Figures are Adequate? | Non | Oui | Lipsum retiré, figure LoRA insérée, contradiction corrigée, +4 figures analytiques |
| References up-to-date and appropriate? | Non | Oui | 11 → 28, 4 entrées fausses corrigées |
| Abstract and Introduction adequate? | Non (247 mots) | Oui | 200 mots, contribution explicitée |
| Paper formatting needs adjustment? | **Ne compile pas** | Oui | Phase 0 |
| Conclusions/Future Work convincing? | Moyen | Oui | 4 findings mesurés, future work concret |
| Improve English? | Non | Non | Déjà un point fort du papier |

## Les cinq critères notés

**Technical Quality** — le gain principal. Un papier qui *mesure* le rang effectif des mises à
jour apprises et le compare à la contrainte imposée par l'adaptateur n'est plus une ablation :
c'est une analyse mécaniste. Les IC bootstrap remplacent des p-values sur 3 seeds. Le confondant
BatchNorm est levé explicitement.

**Significance** — le papier ne dit plus seulement « LoRA échoue sur l'IR », mais « voici la
quantité mesurable qui prédit cet échec ». Un praticien peut calculer le rang effectif sur son
propre domaine et décider **avant** d'entraîner. Le résultat devient actionnable au-delà de SCface.

**Originality** — restera la note la plus faible : pas de méthode nouvelle, et il faut l'assumer
plutôt que le maquiller. Mais la mesure du rang effectif des updates en convolutionnel sur un
écart spectral visible→NIR n'existe pas dans la littérature. C'est une originalité empirique
réelle, pas revendiquée à tort.

**Presentation** — déjà correcte, devient forte : compile, structure hypothèses → mesure →
explication, figures cohérentes, bibliographie propre.

**Relevance** — point faible structurel, ICAART n'étant pas une conférence vision. Seul le
recadrage de 4.4 agit dessus. C'est le risque résiduel non éliminable.

## Le raisonnement de fond

Un **Short Paper**, c'est un résultat correct sans analyse du *pourquoi*.
Un **Full Paper**, c'est un résultat **expliqué et mesuré**.

Le papier actuel constate une défaillance et propose une explication non testée. Après ce plan,
il constate, **mesure la cause**, la relie à un résultat établi du domaine langage, identifie la
condition qui inverse la conclusion du travail concurrent le plus proche, et en tire une règle
utilisable ailleurs. C'est la définition d'un Full Paper — et cela ne demande **aucune
expérimentation nouvelle au-delà de six runs d'une heure**.

## Estimation honnête

Ce plan fait passer Full Paper d'improbable à **sérieusement jouable — de l'ordre d'une chance
sur deux**. Ce qui l'empêche d'aller au-delà : un seul backbone, un seul benchmark, 30 identités
de test, aucune méthode nouvelle. Ces limites ne sont pas franchissables en six semaines et il ne
faut pas chercher à les masquer : les déclarer clairement dans les Limitations vaut mieux que de
laisser un relecteur les découvrir.

**Le facteur le plus risqué n'est pas technique mais séquentiel : la Phase 1.**
Exécute-la dès le jour 2, avant toute rédaction.
