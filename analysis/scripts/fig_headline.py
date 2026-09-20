#!/usr/bin/env python
"""Figures de la sous-section « Full Fine-tuning versus Full LoRA » du papier.

Deux figures independantes, chacune dans son propre fichier (PDF vectoriel + PNG de
controle), en largeur de colonne :
  fig_full_lora_bn.pdf         -- Full LoRA sous les deux lectures de BatchNorm :
                                  statistiques courantes laissees deriver (violet)
                                  vs gelees (orange).
  fig_full_ft_vs_full_lora.pdf -- Full FT (bleu) vs Full LoRA BN gelees (orange),
                                  les deux mecanismes compares a reglage fixe.
Meme style pour les deux (barres pleines, deux teintes). LoRA BN gelees garde
l'orange d'une figure a l'autre : c'est la meme entite. Paires validees CVD :
bleu/orange delta E 24.7, violet/orange delta E 29.5, contraste >= 3:1 partout.

Dans les deux : la baseline non adaptee est un repere horizontal par terrain (une
reference, pas un concurrent -- l'ecart barre/repere est le gain) ; barres d'erreur =
ecart-type (population) sur 3 seeds, sur TOUTES les series qui en ont une ; terrains
conquis a gauche, challenge a droite, comme dans le papier ; valeurs directes
uniquement sur les terrains de challenge.

Les chiffres sont lus dans analysis/outputs/headline_results.csv (dossier exclu du
depot public) -- aucun resultat n'est code en dur ici.

Usage :
    python analysis/scripts/fig_headline.py            # les deux figures
    python analysis/scripts/fig_headline.py --only bn  # ou : --only ft
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "analysis" / "outputs" / "headline_results.csv"

# Ordre du papier : conquis puis challenge.
TERRAINS = [
    ("vis_1.00m", "Visible\n1.00 m"),
    ("vis_2.60m", "Visible\n2.60 m"),
    ("ir_1.00m",  "IR\n1.00 m"),
    ("vis_4.20m", "Visible\n4.20 m"),
    ("ir_2.60m",  "IR\n2.60 m"),
    ("ir_4.20m",  "IR\n4.20 m"),
]
N_CONQ = 3

C_FT = "#2a78d6"
C_LORA = "#eb6834"
C_DRIFT = "#4a3aa7"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

W = 0.30      # largeur d'une barre
GAP = 0.03    # ecart surface entre les deux barres d'un terrain


def load(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """-> {config: {"mean": {terrain: v}, "std": {terrain: v}}} ; std absent = pas de ±."""
    out: dict = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            vals = {k: float(v) for k, v in r.items() if k not in ("config", "stat")}
            out.setdefault(r["config"], {})[r["stat"]] = vals
    return out


def panel(ax, base, left, right, left_style, right_style):
    """Un panneau : deux barres par terrain + repere baseline + habillage."""
    keys = [k for k, _ in TERRAINS]
    x = list(range(len(keys)))
    xs_l = [i - W / 2 - GAP / 2 for i in x]
    xs_r = [i + W / 2 + GAP / 2 for i in x]

    err = {"ecolor": INK_2, "elinewidth": 0.8, "capsize": 1.5, "capthick": 0.8}
    ax.bar(xs_l, [left["mean"][k] for k in keys], width=W, linewidth=0, zorder=3,
           yerr=[left["std"][k] for k in keys] if "std" in left else None,
           error_kw=err, **left_style)
    ax.bar(xs_r, [right["mean"][k] for k in keys], width=W, linewidth=0, zorder=3,
           yerr=[right["std"][k] for k in keys] if "std" in right else None,
           error_kw=err, **right_style)

    half = W + GAP / 2 + 0.06
    for i, k in zip(x, keys):
        ax.plot([i - half, i + half], [base["mean"][k]] * 2, color=INK_2, linewidth=1.2,
                solid_capstyle="butt", zorder=4)

    for i, k in zip(x, keys):
        if i < N_CONQ:
            continue
        # Etiquette posee au-dessus de la barre d'erreur, pas de la barre.
        for xpos, side, dx in ((xs_l[i], left, -0.07), (xs_r[i], right, +0.07)):
            val = side["mean"][k]
            top = val + (side["std"][k] if "std" in side else 0)
            ax.text(xpos + dx, top + 1.2, f"{val:.1f}", ha="center", va="bottom",
                    fontsize=6, color=INK, zorder=5)

    ax.plot([N_CONQ - 0.5] * 2, [0, 104], color=AXIS, linewidth=0.8, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in TERRAINS], color=INK_2)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Rank-1 accuracy (%)", color=INK_2)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="both", length=0, colors=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.8)


def make_figure(base, left, right, left_style, right_style, handles, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.3 / 2.54, 5.9 / 2.54))
    fig.patch.set_facecolor("white")
    panel(ax, base, left, right, left_style, right_style)

    # Etiquettes de groupe sous les noms de terrain.
    for xc, lab in (((N_CONQ - 1) / 2, "conquered terrains"),
                    ((N_CONQ + len(TERRAINS) - 1) / 2, "challenge terrains")):
        ax.text(xc, -27, lab, ha="center", va="top", fontsize=7, color=INK_2,
                style="italic", clip_on=False)

    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               frameon=False, ncol=3, handlelength=1.0, handleheight=0.8,
               columnspacing=1.2, handletextpad=0.5, labelcolor=INK_2)

    fig.tight_layout(pad=0.3, rect=(0, 0.10, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Figure ecrite : {out} (+ .png de controle)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "paper" / "figures")
    ap.add_argument("--only", choices=["bn", "ft"], default=None)
    args = ap.parse_args()

    d = load(args.data)
    base = d["Baseline"]
    ft = d["Full FT"]
    lora = d["Full LoRA (BN frozen)"]
    lora_drift = d["Full LoRA (BN drifting)"]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
    })

    drift_style = {"color": C_DRIFT}
    lora_style = {"color": C_LORA}
    ft_style = {"color": C_FT}
    h_base = Line2D([0], [0], color=INK_2, linewidth=1.2, label="Baseline")

    if args.only in (None, "bn"):
        make_figure(base, lora_drift, lora, drift_style, lora_style, [
            Patch(facecolor=C_DRIFT, label="BN drifting"),
            Patch(facecolor=C_LORA, label="BN frozen"),
            h_base,
        ], args.out_dir / "fig_full_lora_bn.pdf")

    if args.only in (None, "ft"):
        make_figure(base, ft, lora, ft_style, lora_style, [
            Patch(facecolor=C_FT, label="Full FT"),
            Patch(facecolor=C_LORA, label="Full LoRA (BN frozen)"),
            h_base,
        ], args.out_dir / "fig_full_ft_vs_full_lora.pdf")


if __name__ == "__main__":
    main()
