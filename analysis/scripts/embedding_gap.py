#!/usr/bin/env python
"""Phase 2.2 du plan -- ecart de domaine dans l'espace d'embedding.

Pour chaque probe, deux quantites lues dans les matrices de scores d'eval_checkpoint.py :
  genuine  = cos(probe, mugshot de SA propre identite)      -> l'ecart de domaine
  margin   = genuine - max cos(probe, mugshot d'une AUTRE identite)
             (> 0 <=> rank-1 correct : c'est la marge de decision)

Quatre figures (deux jeux x deux mesures), memes conventions que fig_headline.py (terrains conquis a gauche,
challenge a droite, une couleur par configuration, legende sous l'axe) :
  fig_embedding_gap.pdf      `genuine`, jeu "gain" : pre-entraine / FT l3+4 / LoRA r=32
  fig_margin.pdf             `margin` (frontiere de decision a 0), meme jeu
  fig_embedding_gap_bn.pdf   `genuine`, jeu "BN" : pre-entraine / Full LoRA gelee / derivante
  fig_margin_bn.pdf          `margin`, meme jeu

Configurations tracees (seeds regroupes) : pre-entraine (reference, gris), FT layer3+4,
LoRA layer3+4 r=32 (config recommandee), Full LoRA r=8 avec BN derivante (l'effet BN
vu dans la representation). Modifiable via --configs.

Question du plan v2 : *ou* le gain se produit-il dans la representation, et que fait la
derive BN ? Attente : la derive eloigne les probes de leur mugshot y compris sur les
terrains conquis (coherent avec la regression sous baseline mesuree en rank-1).

Usage :
    python analysis/scripts/embedding_gap.py
    python analysis/scripts/embedding_gap.py --eval-dir <dossier> --configs pretrained ft_34_anchor ...
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "analysis" / "outputs" / "eval"
FIG_DIR = ROOT / "paper" / "figures"

TERRAINS = [("visible_1.00m", "Visible\n1.00 m"), ("visible_2.60m", "Visible\n2.60 m"),
            ("ir_1.00m", "IR\n1.00 m"), ("visible_4.20m", "Visible\n4.20 m"),
            ("ir_2.60m", "IR\n2.60 m"), ("ir_4.20m", "IR\n4.20 m")]
N_CONQ = 3

# Jeu "gain" : ou le gain se produit dans la representation (config recommandee vs FT).
MAIN_CONFIGS = [
    ("pretrained", "Pre-trained", "#898781"),
    ("ft_34_anchor", "FT layer3+4", "#2a78d6"),
    ("lora_34_r32_anchor_bnfrozen", "LoRA layer3+4 r=32", "#eb6834"),
]
# Jeu "BN" : la paire equitable, meme config, seules les statistiques BN changent.
BN_CONFIGS = [
    ("pretrained", "Pre-trained", "#898781"),
    ("full_lora_r8_anchor_bnfrozen", "Full LoRA r=8, BN frozen", "#eb6834"),
    ("full_lora_r8_anchor", "Full LoRA r=8, BN drifting", "#4a3aa7"),
]
DEFAULT_CONFIGS = MAIN_CONFIGS
INK, INK_2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def config_of(run: str) -> str:
    return re.sub(r"_seed\d+", "", run)


def load_scores(eval_dir: Path) -> dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]]:
    """-> data[config][terrain] = liste (par seed) de (genuine, margin) par probe."""
    data: dict = defaultdict(lambda: defaultdict(list))
    for f in sorted(eval_dir.glob("eval_*.npz")):
        run = f.stem[len("eval_"):]
        z = np.load(f, allow_pickle=False)
        gallery_ids = list(z["gallery_ids"])
        col = {iid: k for k, iid in enumerate(gallery_ids)}
        for t, _ in TERRAINS:
            if f"scores__{t}" not in z:
                continue
            S = z[f"scores__{t}"]
            ids = z[f"probes__{t}__ids"]
            true_col = np.array([col[i] for i in ids])
            genuine = S[np.arange(len(ids)), true_col]
            S_imp = S.copy()
            S_imp[np.arange(len(ids)), true_col] = -np.inf
            margin = genuine - S_imp.max(axis=1)
            data[config_of(run)][t].append((genuine, margin))
    return data


def boxes(ax, data, configs, which: int, title_zero: bool):
    n = len(configs)
    width = 0.8 / n
    for ci, (cfg, label, color) in enumerate(configs):
        vals, pos = [], []
        for ti, (t, _) in enumerate(TERRAINS):
            if t not in data[cfg]:
                continue
            v = np.concatenate([pair[which] for pair in data[cfg][t]])
            vals.append(v)
            pos.append(ti - 0.4 + width * (ci + 0.5))
        bp = ax.boxplot(vals, positions=pos, widths=width * 0.85, patch_artist=True,
                        showfliers=False, whis=(5, 95), zorder=3,
                        medianprops={"color": "white", "linewidth": 1.2},
                        whiskerprops={"color": color, "linewidth": 0.8},
                        capprops={"color": color, "linewidth": 0.8},
                        boxprops={"facecolor": color, "edgecolor": "none"})
    if title_zero:
        ax.axhline(0, color=INK_2, linewidth=0.8, zorder=2)
    ax.plot([N_CONQ - 0.5] * 2, ax.get_ylim(), color=AXIS, linewidth=0.8, zorder=1)
    ax.set_xticks(range(len(TERRAINS)))
    ax.set_xticklabels([lab for _, lab in TERRAINS], color=INK_2)
    ax.set_xlim(-0.6, len(TERRAINS) - 0.4)
    ax.tick_params(axis="both", length=0, colors=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    for xc, lab in (((N_CONQ - 1) / 2, "conquered terrains"),
                    ((N_CONQ + len(TERRAINS) - 1) / 2, "challenge terrains")):
        ax.text(xc, -0.24, lab, ha="center", va="top", fontsize=7, color=INK_2,
                style="italic", clip_on=False, transform=ax.get_xaxis_transform())


def figure(data, configs, which, ylabel, out: Path, zero: bool):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8.3 / 2.54, 6.2 / 2.54))
    fig.patch.set_facecolor("white")
    boxes(ax, data, configs, which, zero)
    ax.set_ylabel(ylabel, color=INK_2)
    handles = [Patch(facecolor=c, label=l) for _, l, c in configs]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), frameon=False,
               ncol=2, handlelength=1.0, handleheight=0.8, columnspacing=1.2,
               handletextpad=0.5, labelcolor=INK_2)
    fig.tight_layout(pad=0.3, rect=(0, 0.16, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Figure ecrite : {out}")


def summary(data, configs):
    print("\nMediane par terrain (genuine | margin), seeds regroupes :")
    print(f"{'config':<28s}" + "".join(f"{t[:11]:>16s}" for t, _ in TERRAINS))
    for cfg, label, _ in configs:
        cells = []
        for t, _ in TERRAINS:
            if t not in data[cfg]:
                cells.append(f"{'--':>16s}"); continue
            g = np.concatenate([p[0] for p in data[cfg][t]])
            m = np.concatenate([p[1] for p in data[cfg][t]])
            cells.append(f"{np.median(g):7.3f}|{np.median(m):+7.3f}")
        print(f"{label:<28s}" + "".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    ap.add_argument("--configs", nargs="*", default=None, help="labels de config (sans seed)")
    args = ap.parse_args()

    data = load_scores(args.eval_dir)
    if not data:
        raise SystemExit(f"aucun eval_*.npz dans {args.eval_dir}")
    configs = DEFAULT_CONFIGS
    if args.configs:
        palette = [c for _, _, c in DEFAULT_CONFIGS]
        configs = [(c, c, palette[i % len(palette)]) for i, c in enumerate(args.configs)]
    missing = [c for c, _, _ in configs if c not in data]
    if missing:
        print(f"configs absentes, ignorees : {missing}")
        configs = [c for c in configs if c[0] in data]

    if args.configs:
        figure(data, configs, 0, "cos(probe, own mugshot)", FIG_DIR / "fig_embedding_gap_custom.pdf", zero=False)
        figure(data, configs, 1, "margin to best impostor", FIG_DIR / "fig_margin_custom.pdf", zero=True)
        summary(data, configs)
        return
    for tag, cfgs in (("", MAIN_CONFIGS), ("_bn", BN_CONFIGS)):
        cfgs = [c for c in cfgs if c[0] in data]
        figure(data, cfgs, 0, "cos(probe, own mugshot)", FIG_DIR / f"fig_embedding_gap{tag}.pdf", zero=False)
        figure(data, cfgs, 1, "margin to best impostor", FIG_DIR / f"fig_margin{tag}.pdf", zero=True)
        summary(data, cfgs)


if __name__ == "__main__":
    main()
