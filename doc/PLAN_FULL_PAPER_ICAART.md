# Plan d'exécution — Transformer le papier B1 en Full Paper ICAART 2027

> **Version 2 — 19/09/2026.** Révision complète après l'exécution des Phases −1, 1 et de la
> correction BatchNorm. La version 1 (début septembre) reposait sur quatre hypothèses que les
> mesures ont renversées ; ce document les remplace. Voir « Journal des révisions » ci-dessous.
>
> **Destinataire : Claude Code.** Ce document est un plan d'exécution. Suis les phases dans l'ordre.
> Chaque phase a une **Definition of Done (DoD)** vérifiable. Ne passe pas à la phase suivante
> tant que la DoD n'est pas satisfaite. Les phases marquées ⛔ sont bloquantes.
>
> **Contexte.** Le papier (`paper/Example.tex`, template SCITEPRESS) rapporte une étude
> scope × mécanisme (fine-tuning vs LoRA) pour la reconnaissance faciale forensique sur SCface.
> Les sections Experimental Setup et Results ont été réécrites dans `paper/exp_results.tex`
> sur la base de la grille corrigée (BN gelées) — **ce fichier n'est pas encore branché dans
> `Example.tex`**. Le code d'expérimentation est modularisé dans `src/forensic_fr/` et piloté
> par `notebooks/colab_runner.ipynb` ; les résultats consolidés sont dans
> `analysis/outputs/resultats_exp.md` (source unique, gitignorée).
>
> **Cible de soumission.** ICAART 2027, deuxième tour, **22 octobre 2026**, catégorie
> **Regular Paper** (surtout pas Position Paper : celui-ci est plafonné à Short d'office).
> Ambition : Full Paper solide, conférence classe B au minimum.

---

## Journal des révisions (v1 → v2)

| Hypothèse de la v1 | Ce que les mesures ont montré | Conséquence sur le plan |
|---|---|---|
| LoRA échoue sur l'infrarouge parce que l'adaptation requise est de rang élevé | **Faux.** LoRA échouait parce que les statistiques BatchNorm dérivaient pendant l'entraînement (backbone gelé en paramètres, pas en fonction). Une fois BN gelées, LoRA égale FT dès r=8 et le dépasse à r=32 sur tous les terrains | Le récit central du papier est inversé : ce n'est plus « pourquoi LoRA échoue » mais « à quelle condition et à quel rang LoRA égale le fine-tuning, et pourquoi » |
| L'hybride (FT l3+4 + LoRA l1+2) est le meilleur résultat, à réhabiliter comme 4ᵉ contribution (§4.1) | **Caduc.** `LoRA layer3+4 r=32` bat FT et l'hybride sur les six terrains avec 4.14 M paramètres. Aucune combinaison ne fait mieux | §4.1 inversée : « un seul mécanisme, un seul scope, un seul hyperparamètre suffit » |
| Contradiction avec PETALface (ils trouvent PEFT > FT, nous FT > PEFT) à expliquer (§4.3) | **Dissoute.** Nos résultats corrigés vont dans le même sens que PETALface. La « contradiction » était un artefact BN | §4.3 devient une *convergence*, avec un argument inédit : une partie des désaccords de la littérature PEFT-vs-FT sur backbones convolutifs pourrait avoir la même origine |
| Phase 2.2 : « LoRA referme l'écart d'embedding sur le visible mais pas sur l'IR » | **Faux à r=32.** | Phase 2.2 conservée, mais avec une question différente : *où* le gain se produit dans l'espace d'embedding, et ce que la dérive BN y fait |
| Titre proposé : « When Does Low-Rank Adaptation Fail? » | Elle n'échoue pas | Titre à reformuler (§4.4) |
| Aucun checkpoint disponible → repli possible | Checkpoints régénérés pour toute la grille, 3 seeds, BN gelées | Section « Repli » réécrite : la Phase 1 ne peut plus échouer, elle est faite |

**Ce qui n'a pas changé** : la structure hypothèses → mesure → explication, la décomposition
terrains conquis / challenge, le protocole unifié (AdamW, 20 époques, ancrage 50/50, 3 seeds),
l'analyse spectrale comme cœur du statut Full Paper, et l'ensemble des Phases 0 et 5.

---

## Table des phases

| Phase | Objet | État au 19/09 | Bloquant |
|---|---|---|---|
| −1 | Reconnaissance + gate | **faite** | — |
| 0 | Réparations de compilation + branchement de `exp_results.tex` | **à faire** | ⛔ |
| 1 | Analyse spectrale | **faite** (1.1–1.4) | — |
| 1bis | Correction BN, grille relancée, ablation de rang, runs de complétude | **faite** | — |
| 2 | Exploitation des checkpoints : bootstrap, embeddings, CMC | à faire | cœur |
| 3 | Contrôles : `fc-only` ✅, mesure mugshot, sans-ancrage (optionnel) | 3.1 faite | |
| 4 | Restructuration et rédaction | Setup + Results faits ; le reste à faire | |
| 5 | Finalisation et conformité | à faire | ⛔ |

Ordre d'exécution recommandé : **0 → 2.3 → 2.2/2.1 → 3.1 → 4 → 5**. La Phase 0 d'abord parce
que tant que le papier ne compile pas, ni le nombre de pages ni le comptage de caractères ne
sont connus — et ce sont les deux critères de la Phase 4.

---

# PHASE −1 — Reconnaissance et gate ✅ FAITE

Verdict initial : **GO PARTIEL** — aucun checkpoint n'avait jamais été sauvegardé
(`piepline_version_stable.ipynb` créait `checkpoints/` mais n'y écrivait rien), mais le code
était fonctionnel et un run coûte ~20 min. Conséquences :

- `src/forensic_fr/` : package modulaire (config, data, models, training, evaluation),
  `training/checkpoint.py` sauvegarde les poids à la fin de chaque run.
- `training/grid.py` : grille unique (`GRID`), `build_plan(only, seeds, anchor, freeze_bn)`.
- `notebooks/colab_runner.ipynb` : point d'entrée Colab (git pull, paramètres `#@param`, grille,
  analyses).
- Historique : un JSON par run (`exp1_{mode}_r{r}_seed{s}_{anchor}[_bnfrozen].json`) — l'ancien
  fichier partagé par mode subissait une race condition Drive.
- Dépôt public `MarouaneAyech/foresynch_b2` ; `paper/`, `old code/`, `analysis/outputs/`
  gitignorés (manuscrit non anonymisé, résultats non publiés).

---

# PHASE 0 — Réparations de compilation ⛔ À FAIRE

Le papier **ne compile pas** et les sections révisées ne sont pas branchées. Rien d'autre ne
compte tant que ce n'est pas réglé.

## 0.1 Préambule (`Example.tex`)

| Action | Détail |
|---|---|
| Ajouter | `\usepackage{siunitx}` **avant** `\usepackage{SCITEPRESS}` — `\SI` est utilisé partout dans `exp_results.tex` |
| Supprimer | `\usepackage{lipsum}` et son commentaire français |
| Supprimer | le `\usepackage{subcaption}` en double |

## 0.2 Faux texte ⛔ CRITIQUE

Supprimer `\lipsum[1-2]` et `\lipsum[3-6]` (Section 3.1). Vérifier : `grep -n lipsum Example.tex` → 0.

## 0.3 Brancher `exp_results.tex` ⛔ NOUVEAU

- Remplacer les anciennes sections 4 (Experimental Setup) et 5 (Results) de `Example.tex` par
  `\input{exp_results.tex}`.
- Supprimer les anciens tableaux/figures devenus redondants (l'ancienne table de grille,
  l'ancienne figure `ir 4.20 m` avec l'hybride, les anciens résultats LoRA dérive-BN).
- Vérifier que les labels référencés ailleurs (`tab:configs`, `sec:results`, …) existent encore.
- `paper/figures/` contient `fig_full_lora_bn.pdf` et `fig_full_ft_vs_full_lora.pdf`
  (générées par `analysis/scripts/fig_headline.py`) — les chemins dans `exp_results.tex` sont
  relatifs à `paper/`.

## 0.4 Unité invalide

`\SI{43.6}{\mega\nothing}` → `43.6\,M` (si encore présent dans les parties non réécrites).

## 0.5 Figure LoRA orpheline

`fig_lora_block.tex` n'est jamais inclus. L'insérer dans la Section 3.2 après le paragraphe
« Low-rank adaptation », avec 3–4 phrases : `A` = conv 1×1 (in→r), `B` = conv 3×3 (r→out),
échelle α/r, `B` initialisé à zéro. **Ajouter la remarque** (résultat de la Phase 1.1) : le ΔW
aplati d'un adaptateur conv n'est pas plafonné à r mais à ~r·k² = 9r, parce que `B` porte le
noyau spatial ; seul l'adaptateur `fc` est plafonné à r exactement.

## 0.6 Sections 5.1 et 6.1 de l'ancienne version — ⚠️ REMPLACÉES, PAS CORRIGÉES

Les points 0.5/0.6 de la v1 (« similar footprint », « does not adapt at all ») corrigeaient des
phrases qui décrivaient des résultats **faux** (dérive BN). Ne pas les corriger : ces passages
disparaissent avec le branchement de `exp_results.tex` (0.3). La Discussion (§6) devra être
réécrite en Phase 4, pas rafistolée.

## 0.7 Abstract

247 mots → ≤ 200. À réécrire entièrement de toute façon (Phase 4) : le résumé actuel affirme
que LoRA échoue.

## 0.8 Nettoyage

Commentaires français, en-tête « Papier B1 CoopIS 2026 » dans `references.bib`, blocs
résiduels du template SCITEPRESS.

**DoD 0 :**
- [ ] `pdflatex` + `bibtex` + `pdflatex` ×2 → **0 erreur**
- [ ] `grep -c lipsum Example.tex` → `0`
- [ ] `exp_results.tex` inclus, anciennes sections 4–5 retirées, 0 référence `??`
- [ ] Les deux figures TikZ (pipeline, bloc LoRA) et les deux figures PDF de la §4.1 apparaissent
- [ ] Nombre de pages et comptage de caractères hors espaces notés dans `resultats_exp.md`

---

# PHASE 1 — Analyse spectrale ✅ FAITE (1.1–1.4)

Scripts : `analysis/scripts/spectral_analysis.py` (1.1), `spectrum_figure.py` (1.2),
`stage_displacement.py` (1.3), `bn_drift_check.py` (1.4). Résultats détaillés dans
`resultats_exp.md`. Ce que les chiffres disent :

**1.1 Rang effectif** (médian par étage, checkpoints BN gelées) :

| Config | layer3 | layer4 | fc | ratio réel/nominal (layer4) |
|---|---|---|---|---|
| FT layer3+4 | 181.6 | 368.4 | 117.4 | — |
| LoRA l3+4 r=8 | 34.6 | 32.4 | 7.3 | ×4.05 |
| LoRA l3+4 r=16 | 60.2 | 59.3 | 13.5 | ×3.71 |
| LoRA l3+4 r=32 | 91.6 | 97.7 | 20.2 | ×3.05 |
| LoRA l3+4 r=64 | 120.3 | 159.4 | 25.2 | ×2.49 |

Le rendement marginal du rang s'effondre (×4.05 → ×2.49) : explication mécaniste du plateau
puis du recul à r=64. Et LoRA **dépasse FT dès r=32 avec un rang effectif ~4× plus faible**
(97.7 vs 368.4 sur layer4) : il n'a pas besoin d'égaler le rang de FT pour égaler sa performance.

**1.2 Énergie capturée** (FT l3+4, layer4) : r=8 → 8.6 %, r=16 → 14.3 %, r=32 → 23.5 %,
r=64 → 37.5 %. Même à r=64, plus de 60 % de l'énergie de la mise à jour FT reste hors de
portée — et pourtant la performance est déjà atteinte à r=32.

**1.3 Déplacement par étage** (Full FT) : stem 1.3 % → layer1 6.8 % → layer2 8.8 % →
layer3 14.6 % → layer4 20.5 % → fc 22.9 %. Gradient croissant régulier, pas un escalier ;
layer3+4 = 3.1× les étages bas. **H1 confirmée structurellement.** Note : `fc` est l'étage qui
bouge le plus en proportion — c'est ce qui rend la baseline `fc-only` (3.1) indispensable.

**1.4 Contrôle BN** : dérive significative détectée sur tous les checkpoints LoRA d'origine →
a déclenché la Phase 1bis. C'est **le** résultat qui a changé le papier.

**1.5 Directions intruses** : abandonné. Les points 1.1–1.4 suffisent ; le temps est mieux
investi en Phase 2.

---

# PHASE 1bis — Correction BN et grille finale ✅ FAITE

Non prévue en v1. Déclenchée par 1.4.

**Le piège.** `requires_grad=False` gèle les paramètres affines γ, β d'une BatchNorm mais pas
ses buffers `running_mean`/`running_var`, mis à jour par moyenne mobile à chaque forward en
mode `.train()`. Un backbone « gelé » au sens des paramètres continue donc de changer au sens
de la fonction. Correctif : `freeze_all_batchnorm()` (`.eval()` sur chaque BN après chaque
`model.train()`), option `freeze_bn` de `RunConfig`, suffixe `_bnfrozen` sur les fichiers.

**Grille finale** (3 seeds mesurés, ± par terrain partout, écart-type de population) :

| Config | Params | ir 4.20 m | ΣΔ |
|---|---|---|---|
| FT layer4 | 26.0 M | 52.5±4.5 | +39.2±6.9 |
| FT layer3+4 | 42.3 M | 79.7±2.4 | +90.2±5.9 |
| Full FT | 43.6 M | 80.2±6.5 | +87.8±8.6 |
| LoRA layer4 r=8 | 0.50 M | 50.9±6.8 | +46.0±12.0 (dérive BN : −1.2) |
| LoRA l3+4 r=8 | 1.07 M | 72.9±5.1 | +82.9±0.9 (dérive BN : +58.7) |
| LoRA l3+4 r=16 | 2.09 M | 80.8±0.8 | +93.9±1.8 |
| **LoRA l3+4 r=32** | **4.14 M** | **87.6±5.5** | **+99.9±3.9** |
| LoRA l3+4 r=64 | 8.22 M | 81.9±3.5 | +93.2±3.5 |
| Full LoRA r=8 | 1.18 M | 75.7±3.9 | +90.0±6.0 (dérive BN : +53.9±10.1, par terrain mesuré) |

Décisions prises et documentées dans `resultats_exp.md` :
- r=64 fixé **à l'avance** comme dernier point de l'ablation (pas après coup).
- Hybride abandonné : plus aucun terrain où FT garde un avantage mesurable.
- Question BN tranchée une fois, à pleine portée, 3 seeds × 2 bras par terrain ; corroborée en
  ΣΔ aux deux autres portées. `lora_34 r=8` dérive-BN volontairement non relancé.
- Reproductibilité : `full_lora` dérive-BN, `full_ft` et `ft_34` relancés reproduisent le
  brouillon **à la décimale** (pipeline déterministe par seed).

---

# PHASE 2 — Exploitation des checkpoints ⭐ À FAIRE

Aucun entraînement. Tous les checkpoints de la grille existent (BN gelées pour LoRA).

## 2.0 Prérequis : `analysis/scripts/eval_checkpoint.py`

Le trainer ne sauvegarde ni embeddings ni matrices de similarité — seulement le rank-1 par
terrain. Écrire un script qui, pour un checkpoint donné (réinjection LoRA via
`load_checkpoint_model` de `spectral_analysis.py`), produit dans `analysis/outputs/eval/` :

- `embeddings_{run}.npz` : embeddings L2-normalisés galerie + probes, avec identité et terrain ;
- `scores_{run}.npz` : matrice cosinus probe × galerie par terrain ;
- `correct_{run}.csv` : vecteur correct/incorrect par probe (identité, terrain, rang de la
  bonne identité).

Vérification : le rank-1 recalculé doit être **identique** au `test` de la dernière époque du
JSON d'historique (même checkpoint = époque 20). Le script sert aussi de secours si un JSON
est perdu.

Checkpoints à traiter (1 seed chacun suffit pour 2.1/2.2, les 3 seeds pour 2.3) :
pré-entraîné, FT l3+4, Full FT, LoRA l3+4 r=8, **LoRA l3+4 r=32**, Full LoRA r=8, et
Full LoRA r=8 dérive-BN (pour montrer l'effet BN dans l'espace d'embedding).

## 2.3 Statistiques robustes — PRIORITÉ 1 de la phase ⛔

`exp_results.tex` contient encore des affirmations « Welch, p<0.01 », « not significant at the
5 % level » **non documentées** (aucun script, aucun test nommé précisément). Sur 3 seeds,
ces tests n'ont de toute façon pas de puissance. Remplacer par un **bootstrap au niveau
identité** (30 identités de test, 10 000 rééchantillonnages) :

- IC 95 % sur le rank-1 de chaque config par terrain ;
- IC 95 % sur la **différence appariée** LoRA r=32 − FT l3+4, LoRA r=32 − Full FT,
  Full LoRA − Full FT, BN gelées − BN dérivante (mêmes identités rééchantillonnées des deux côtés) ;
- taille d'effet (Cohen's d apparié).

**Rédaction :** ne jamais argumenter une *équivalence* par l'absence de significativité. Dire ce
que l'IC de la différence exclut ou n'exclut pas. Retirer toutes les p-values de `exp_results.tex`.

**Sortie :** `analysis/outputs/bootstrap_ci.csv` + une figure des IC (forest plot) sur
`ir 4.20 m`.

## 2.2 Écart de domaine dans l'espace d'embedding ⭐

Pour chaque identité de test : cosinus entre l'embedding mugshot (galerie) et ses probes,
par terrain, pour pré-entraîné / FT l3+4 / LoRA r=32 / Full LoRA dérive-BN.

**Question révisée** (la v1 prédisait un échec sur l'IR qui n'existe plus) : montrer *où* le
gain se produit — l'écart intra-identité se referme-t-il uniformément ou surtout sur l'IR long ?
— et ce que la dérive BN fait : on s'attend à ce qu'elle éloigne les probes des mugshots
**y compris sur les terrains conquis** (cohérent avec la régression sous baseline mesurée).
C'est la version « dans la représentation » du résultat BN, plus démonstrative qu'un rank-1.

**Sortie :** `analysis/figures/fig_embedding_gap.pdf`, boxplots, 6 terrains, une couleur par
config (palette de `fig_headline.py`).

## 2.1 Courbes CMC rang-1 → rang-10

Terrain `ir 4.20 m`, configs : Base, FT l3+4, Full FT, LoRA r=8, LoRA r=32, Full LoRA.
Intérêt : si FT rattrape LoRA r=32 au rang 5, la différence est de classement, pas
d'information — nuance utile pour la Discussion.

**Sortie :** `analysis/figures/fig_cmc_ir420.pdf`.

## 2.4 Métriques complémentaires (optionnel)

Rang-5 et TPIR@FAR=1 % depuis les mêmes matrices. Seulement si la place le permet.

**DoD 2 :**
- [ ] `eval_checkpoint.py` reproduit le rank-1 des JSON à l'identique
- [ ] `bootstrap_ci.csv` produit ; **0 p-value** dans `exp_results.tex`, remplacées par des IC
- [ ] `fig_embedding_gap.pdf`, `fig_cmc_ir420.pdf` générées, même style que `fig_headline.py`
- [ ] Constats consignés dans `resultats_exp.md`

---

# PHASE 3 — Contrôles À FAIRE (réduits)

## 3.1 Baseline `fc-only` — 3 seeds ✅ FAITE (19–20/09/2026)

**Résultat : +55.5±2.7** (100.0 | 100.0 | 87.8 | 97.8 | 90.4 | 58.8). ≈ LoRA layer4 terrain par
terrain avec 26× plus de paramètres ; la projection seule fait la moitié du gain, l'autre
moitié (ir 4.20 m) exige les convolutions et layer3. Intégré dans `exp_results.tex`
(`tab:configs`, ligne de contrôle de `tab:scope`, paragraphe dédié). Détails dans
`resultats_exp.md`.


Fine-tuning de la seule projection `fc` + tête ArcFace, backbone gelé, **BN gelées**
(même lecture que LoRA). Même protocole. Scénario à ajouter dans `training/scenarios.py`
(`configure_fc_only`) et à `GRID`.

**Pourquoi c'est plus nécessaire qu'en v1 :** toutes les configs adaptent `fc`, et la Phase 1.3
montre que `fc` est l'étage au plus grand déplacement relatif (22.9 %). L'objection « tout le
gain vient de fc » est donc *renforcée* par nos propres mesures. Sans cette baseline, elle est
imparable. Coût : 3 runs, ~1 h.

## 3.2 Seconde baseline PEFT — ✅ FAITE (branche BN)

La v1 conditionnait 3.2 au résultat de 1.4 : dérive détectée → LoRA avec BN gelées. C'est
devenu toute la grille. **BitFit reste optionnel** (scope layer3+4, biais + fc + tête) — à
faire seulement si la Phase 4 révèle un manque sur l'axe « coût croissant »
(`fc-only` → BitFit → LoRA → FT sélectif → FT complet).

## 3.3 Mesure passive mugshot (ex-4.3.2.B) — ❌ NON RÉALISABLE avec le cache actuel

Le cache d'évaluation ne contient qu'un mugshot par identité de test — la galerie elle-même.
Aucune probe HQ tenue à l'écart n'existe ; la mesure demanderait de reconstruire le cache
depuis SCface (autres mugshots par identité). Abandonnée. L'argument anti-oubli repose sur
les terrains conquis, préservés à 100 / 100 / 98.3 par toutes les configs BN gelées — une
mesure directe, déjà dans les tables.

## 3.4 Contrôle sans ancrage — optionnel, 1 run

`FT layer3+4` avec `ANCHOR=False` (100 % dégradé), 1 seed. Teste si l'ancrage est bien ce qui
évite l'oubli et l'instabilité observés chez PETALface. Le code le supporte déjà
(`build_train_loader(anchor=False)`). Ne pas y passer plus d'une demi-journée ; sinon, assumer
l'hypothèse non démontrée dans le texte.

**DoD 3 :**
- [x] ~~`fc-only` × 3 seeds dans la grille, la table des configs et la table de portée~~
- [x] ~~Tableau mugshot post-adaptation~~ — non réalisable, voir 3.3
- [ ] (Optionnel) contrôle sans ancrage rapporté, qu'il confirme ou infirme

---

# PHASE 4 — Restructuration et rédaction

**Cible : ~42 000 caractères hors espaces**, ≤ 12 pages. Script de comptage :
`pdftotext paper.pdf - | tr -d '[:space:]' | wc -c`.

## 4.0 Le nouveau récit (à tenir d'un bout à l'autre)

1. Un backbone convolutif « gelé » admet deux lectures ; la mauvaise fait échouer LoRA de 36
   points et dégrade même les terrains déjà maîtrisés. **Contribution méthodologique.**
2. Avec la bonne lecture, LoRA égale le fine-tuning complet dès r=8 (2.6 % des paramètres) et
   le dépasse à r=32 sur les six terrains (8.7 %), y compris sur le plus dur.
3. Le scope `layer3+4` n'est pas un choix : c'est là que le fine-tuning libre déplace ses poids
   (Phase 1.3). Un seul mécanisme, un seul scope, un seul hyperparamètre suffisent — aucune
   hybridation nécessaire.
4. L'analyse spectrale explique le rang : le rendement du rang nominal s'effondre, LoRA n'a
   besoin que d'un quart du rang effectif de FT, et l'ablation est en cloche, pas monotone.
5. Convergence avec PETALface, et l'hypothèse que des désaccords de la littérature sur
   PEFT-vs-FT en convolutionnel sont en partie des artefacts BN.

## 4.1 Un seul mécanisme suffit (inverse de la v1) 

L'hybride est retiré des tableaux (jamais publié : aucune justification de retrait nécessaire).
Une phrase en Discussion : *la combinaison FT+LoRA n'est pas motivée dès lors que LoRA
layer3+4 r=32 domine le fine-tuning sur les six terrains.* Limite à déclarer : « LoRA r=32 sur
layer3+4 + petit LoRA sur layer1+2 » n'a pas été testé directement ; le faisceau d'indices
(`full_lora` r=8 < `lora_34` r=32 ; Phase 1.3) le rend improbable, pas exclu.

## 4.2 Requalifier le scope `layer3+4`

Inchangé, renforcé : l'intérêt n'est pas l'économie de paramètres (96.9 % du backbone en FT)
mais le fait que les étages bas n'ont rien à apprendre du domaine — mesuré en 1.3. Ajouter :
FT l3+4 est aussi **plus stable** que Full FT sur ir 4.20 m (±2.4 vs ±6.5).

## 4.3 PETALface : de la contradiction à la convergence

Le tableau de protocoles de la v1 (ancrage absent chez eux, full FT qui dégrade le domaine
cible, Transformer vs CNN, LoRA double pondéré vs LoRA simple, résolution seule vs résolution
+ IR) reste **exact et utile** — il est reproduit ci-dessous. Ce qui change, c'est l'usage :

- **Convergence** : PEFT ≥ FT chez eux et chez nous, sur des architectures et des modalités
  différentes. C'est un résultat de généralisation, pas une défense.
- **Différence honnête** : chez eux le full FT *dégrade* le domaine cible ; chez nous il
  l'améliore fortement. L'ancrage 50/50 est l'explication candidate (3.3/3.4 la testent).
- **Argument inédit** : notre brouillon initial *contredisait* PETALface, et la contradiction
  était un artefact BN. Suggérer, prudemment, que des résultats « FT > PEFT sur CNN » de la
  littérature méritent la même vérification. Ne pas accuser un papier en particulier.
- Vérifier chaque chiffre PETALface sur le PDF source avant citation.

| Élément | PETALface | Ce papier |
|---|---|---|
| Ancrage HQ/dégradé | Absent | 50/50 à chaque batch |
| Full FT sur le domaine cible | Dégrade (TinyFace ~72.7→~71.1 ; BRIAR 55.3→44.8) | Améliore fortement |
| Architecture / cible LoRA | Swin-B, qkv + MLP | IResNet-50, convs 3×3 + fc |
| Méthode PEFT | LoRA double pondéré par qualité | LoRA simple statique |
| Modalité | Résolution | Résolution + infrarouge |
| Lecture BN | sans objet (LayerNorm) | **variable contrôlée** |

## 4.4 Recadrage pour ICAART

*Relevance* est le critère le plus fragile. Le titre de la v1 est faux. Candidats :

- *When Low-Rank Adaptation Matches Fine-Tuning: Scope, Rank and a BatchNorm Pitfall in
  Convolutional Face Recognition under Degradation*
- *A Frozen Backbone Is Not a Frozen Function: Making LoRA Match Fine-Tuning on Degraded
  Infrared Surveillance*

Introduction : question générale d'abord (à quelle condition l'adaptation à faible rang
suffit-elle, et peut-on le mesurer ?), SCface au troisième paragraphe. Paragraphe
« Contribution and applicability » obligatoire : étude empirique systématique + analyse
mécaniste + un piège d'implémentation documenté, pas une méthode nouvelle. Topics :
*Vision and Perception* + *Privacy, Safety, Security, and Ethical Issues*.

## 4.5 Bibliographie : 11 → ~28 références ⛔

Corrections de la v1 inchangées (`ye2024lgaf` inventé, auteurs PETALface, `and others`, etc.).
Ajouts : PEFT (AdaLoRA, DoRA, (IA)³, Adapters, LoRA+), NIR-VIS (CASIA NIR-VIS 2.0, LAMP-HQ),
TinyFace, IJB-S, MagFace, CurricularFace, Roy & Vetterli, Aghajanyan et al., Li et al.
**Nouveaux, pour la sous-section BN** : Ioffe & Szegedy 2015 ; Li et al. 2016 (AdaBN) ;
au moins une référence PEFT vision qui gèle explicitement les BN, et une sur la
normalisation en transfert (l'affirmation « both readings are found in practice » dans
`exp_results.tex` a besoin d'une citation ou d'un adoucissement).

## 4.6 Sections à rédiger

| Section | Contenu | Source | Caractères |
|---|---|---|---|
| **3.x A frozen backbone is not a frozen function** (nouvelle) | paramètres vs buffers, deux lectures, mesure 1.4, régression sous baseline, variance ×1.6 | 1.4, 1bis | +3 000 |
| **6.1 Spectral analysis of the learned updates** (nouvelle) | rang effectif, énergie par rang, déplacement par étage, rendement du rang | 1.1–1.3 | +5 000 |
| 5.x Embedding-space analysis | 2.2 | 2 | +2 000 |
| Intervalles bootstrap | 2.3 | 2 | +800 |
| `fc-only` dans la grille | 3.1 | 3 | +1 000 |
| PETALface en convergence | 4.3 | — | +1 800 |
| Un mécanisme suffit | 4.1 | — | +600 |
| Intro + contribution/applicability | 4.4 | — | +1 200 |
| Bibliographie | 4.5 | — | +2 500 |
| **Total** | | | **≈ +18 000** |

## 4.7 Limitations

Retirer : « BN est un confondant possible » → devient « BN était un confondant, mesuré et
contrôlé ». Retirer « n=3 seeds » → IC bootstrap par identité. **Conserver** : un backbone, un
benchmark, 30 identités de test (protocole non standard), mêmes familles de caméras en train
et test, hyperparamètres non retunés par rang (explication candidate du recul à r=64),
« LoRA r=32 + LoRA l1+2 » non testé.

## 4.8 Points ouverts dans `exp_results.tex` (à régler avant 5)

- [ ] p-values Welch non documentées (§4.2, §4.3) → Phase 2.3
- [ ] Les deux chiffres dérive-BN cités sans table (layer4 −1.2, layer3+4 +58.7) → note de
      bas de page « same protocol, three seeds, not tabulated » ou petite table annexe
- [ ] « both readings are found in practice » → citation ou adoucissement
- [ ] Nombre d'identités train/test (100 / 30) absent de `tab:setup-data`
- [ ] Seed 42 de `ft_34` : valeur reconstituée, à confirmer avec le JSON du Drive

**DoD 4 :**
- [ ] ≥ 40 000 caractères hors espaces, ≤ 12 pages
- [ ] ≥ 25 références, 0 `and others`, 0 référence non vérifiée
- [ ] Les 5 points du récit 4.0 annoncés en introduction et repris en conclusion
- [ ] Paragraphe « Contribution and applicability » présent
- [ ] Sous-section BN et §6.1 spectrale présentes
- [ ] Tous les points 4.8 fermés

---

# PHASE 5 — Finalisation et conformité ⛔

## 5.1 Soumission simultanée — ✅ RÉGLÉ

La soumission CoopIS est **abandonnée**. Aucun risque de soumission simultanée. Retirer
néanmoins toute trace de CoopIS des sources (0.8).

## 5.2 Anonymisation double-blind ⛔

Inchangé : auteurs, affiliations (HANALAB, ENSI, Manouba, Tunisia), e-mails, remerciements,
formulations auto-identifiantes. Ne rien déposer publiquement entre soumission et
notification — **y compris ne pas lever le `.gitignore` de `paper/` sur le dépôt public.**

## 5.3 Déclaration outils d'IA

Texte préparé dans `CAMERA_READY_NOTES.md`, réinséré au camera-ready uniquement.

## 5.4 Vérifications finales

Inchangé : caractères 40–45 k, ≤ 12 pages, 0 `??`, figures lisibles en niveaux de gris (les
palettes de `fig_headline.py` sont validées daltonisme et N&B), captions tables au-dessus /
figures en dessous, anglais relu, ≤ 9 auteurs.

## 5.5 Soumission

PRIMORIS, **Regular Paper** ⛔ (seul choix irréversible), 22 octobre 2026, topics ci-dessus.

## 5.6 Rebuttal — objections anticipées (mises à jour)

| Objection | Réponse préparée |
|---|---|
| « Le résultat BN est un bug, pas une contribution » | Les deux lectures sont défendables et présentes dans la pratique ; l'effet est mesuré (36 pts, 6 terrains, 3 portées, variance ×1.6) et explique un désaccord avec la littérature. C'est un résultat reproductible, pas une anecdote |
| « LoRA > FT est déjà connu (PETALface) » | Sur Transformer, sans ancrage, résolution seule. Nous : CNN, ancrage, infrarouge, et *à quelle condition* — plus le rang effectif qui explique *pourquoi* r=32 |
| « Un seul backbone / 30 identités » | Limitation assumée ; IC bootstrap par identité ; analyse spectrale architecture-agnostique |
| « Pourquoi pas l'hybride ? » | Testé, dominé sur les six terrains par un mécanisme seul |
| « r=64 recule : sur-apprentissage ? » | Hyperparamètres non retunés par rang, déclaré ; rendement du rang mesuré (×2.49) |
| « Tout le gain vient de fc » | Baseline `fc-only` (3.1) |
| « Pas de méthode nouvelle » | Contribution empirique + mécaniste + méthodologique, annoncée dès l'introduction |

Garder en réserve, non publiées : CMC, TPIR@FAR, BitFit si fait.

---

# Repli — réécrit

La Phase 1 ne peut plus échouer : elle est faite, et la grille corrigée porte seule le statut
Full Paper. Si la Phase 2 (bootstrap, embeddings) ne tient pas dans le temps : soumettre avec
la grille + Phase 1 + sous-section BN, en retirant **toutes** les affirmations statistiques non
documentées plutôt qu'en les laissant. Si la Phase 3.1 ne tient pas : déclarer l'objection
`fc` en limitation, en toutes lettres. **Ne jamais soumettre une affirmation que le dépôt ne
peut pas reproduire.**

---

# Justification — pourquoi ce plan satisfait les critères de review

## Les neuf questions du formulaire ICAART

| Question | v1 (brouillon) | Après ce plan | Ce qui le prouve |
|---|---|---|---|
| Needs more experimental results? | Oui | Non | Grille 9 configs × 3 seeds ± partout, ablation de rang 4 points, contrôle BN 2 bras, spectrale, embeddings, bootstrap, `fc-only` |
| Needs comparative evaluation? | Oui | Non | `fc-only` → LoRA (4 rangs) → FT sélectif → FT complet ; deux lectures BN |
| Improve critical discussion? | Oui | Non | Sous-section BN, §6.1 mécaniste, PETALface en convergence, hybride tranché |
| Figures are Adequate? | Non | Oui | 2 figures par terrain (style unifié), spectres, déplacement, embeddings, IC |
| References? | Non | Oui | 11 → 28, entrées fausses corrigées |
| Abstract and Introduction? | Non | Oui | ≤ 200 mots, récit 4.0, contribution explicitée |
| Formatting? | Ne compile pas | Oui | Phase 0 |
| Conclusions convincing? | Moyen | Oui | 5 constats mesurés, une recommandation unique, limites déclarées |
| English? | Bon | Bon | — |

## Les cinq critères notés

**Technical Quality** — un confondant identifié, mesuré, contrôlé et expliqué ; une grille
complète avec dispersion partout ; le rang effectif des mises à jour comparé à la contrainte de
l'adaptateur ; des IC par identité à la place de p-values sur 3 seeds.

**Significance** — le papier ne dit plus « LoRA échoue sur l'IR » mais « LoRA égale ou dépasse
le fine-tuning à condition de geler les statistiques BN, à r=32 sur layer3+4, et voici la
quantité mesurable qui explique ce rang ». Le piège BN est actionnable pour quiconque adapte
un backbone convolutif avec des adaptateurs — bien au-delà de SCface.

**Originality** — pas de méthode nouvelle, à assumer. Mais : la mesure du rang effectif des
mises à jour en convolutionnel sur un décalage visible→NIR n'existe pas dans la littérature ;
l'ablation en cloche expliquée par le rendement du rang non plus ; et le piège BN comme
explication d'un désaccord de littérature est un angle inédit.

**Presentation** — compile, structure question → mesure → explication, figures homogènes,
bibliographie propre.

**Relevance** — point faible structurel (ICAART n'est pas une conférence vision). Le
recadrage 4.4 et le caractère général du piège BN (tout backbone convolutif) sont les deux
leviers.

## Le raisonnement de fond

Un Short Paper constate ; un Full Paper **mesure la cause**. Le brouillon constatait un échec de
LoRA et proposait une explication non testée. Le papier révisé montre que l'échec était un
artefact, mesure ce que l'artefact fait (par terrain, en variance, dans la représentation),
établit le résultat corrigé sur une grille complète, en explique le rang par l'analyse
spectrale, et le situe dans la littérature. C'est un papier plus solide que celui que la v1
visait — parce que le résultat est plus intéressant que l'hypothèse de départ.

## Estimation honnête

Full Paper : **jouable, mieux qu'une chance sur deux** si les Phases 0, 2.3 et 3.1 sont faites.
Ce qui reste hors de portée en six semaines : un second backbone, un second benchmark, plus
d'identités. À déclarer, pas à masquer. Le risque résiduel principal est *Relevance* ; le
second est le temps de rédaction (≈ +18 000 caractères de texte neuf à écrire proprement).

**Le facteur critique est maintenant séquentiel : la Phase 0.** Tant que le papier ne compile
pas avec `exp_results.tex` branché, personne ne sait combien de pages restent pour le reste.
