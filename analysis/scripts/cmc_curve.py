#!/usr/bin/env python
"""Phase 2.1 du plan -- courbes CMC (rang 1 a 10) sur un terrain.

CMC(k) = fraction des probes dont la bonne identite est classee dans les k premieres.
Lue directement dans la colonne `rank` des correct_{run}.csv d'eval_checkpoint.py,
seeds regroupes par configuration.

Interet : deux modeles au meme rank-1 peuvent differer au rang 5 -- l'un met la bonne
personne 2e quand il rate, l'autre la met 15e. Pour un usage forensique (liste courte
de candidats), cette difference compte. Et si FT rattrape LoRA r=32 au rang 5, l'ecart
au rang 1 est un ecart de classement, pas d'information.

Usage :
    python analysis/scripts/cmc_curve.py                       # terrain ir_4.20m
    python analysis/scripts/cmc_curve.py --terrain visible_4.20m
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "analysis" / "outputs" / "eval"
FIG_DIR = ROOT / "paper" / "figures"

# (label de config, legende, couleur, style). Le pre-entraine est une reference : gris, tirets.
DEFAULT_CONFIGS = [
    ("pretrained", "Pre-trained", "#898781", "--"),
    ("ft_34_anchor", "FT layer3+4", "#2a78d6", "-"),
    ("full_ft_anchor", "Full FT", "#4a3aa7", "-"),
    ("lora_34_r8_anchor_bnfrozen", "LoRA layer3+4 r=8", "#1baf7a", "-"),
    ("lora_34_r32_anchor_bnfrozen", "LoRA layer3+4 r=32", "#eb6834", "-"),
]
INK, INK_2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
K_MAX = 10


def config_of(run: str) -> str:
    return re.sub(r"_seed\d+", "", run)


def load_ranks(eval_dir: Path, terrain: str) -> dict[str, list[int]]:
    ranks: dict[str, list[int]] = defaultdict(list)
    for f in sorted(eval_dir.glob("correct_*.csv")):
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                if row["terrain"] == terrain:
                    ranks[config_of(row["run"])].append(int(row["rank"]))
    if not ranks:
        raise SystemExit(f"aucune ligne pour le terrain {terrain!r} dans {eval_dir}")
    return ranks


def cmc(r: np.ndarray, k_max: int = K_MAX) -> np.ndarray:
    return np.array([100.0 * (r <= k).mean() for k in range(1, k_max + 1)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    ap.add_argument("--terrain", default="ir_4.20m")
    args = ap.parse_args()

    ranks = load_ranks(args.eval_dir, args.terrain)
    configs = [c for c in DEFAULT_CONFIGS if c[0] in ranks]
    missing = [c[0] for c in DEFAULT_CONFIGS if c[0] not in ranks]
    if missing:
        print(f"configs absentes, ignorees : {missing}")

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8.3 / 2.54, 5.8 / 2.54))
    fig.patch.set_facecolor("white")
    ks = np.arange(1, K_MAX + 1)

    print(f"CMC sur {args.terrain} (seeds regroupes) :")
    print(f"{'config':<22s}" + "".join(f"  k={k:<3d}" for k in (1, 2, 3, 5, 10)))
    for cfg, label, color, ls in configs:
        r = np.array(ranks[cfg])
        c = cmc(r)
        ax.plot(ks, c, color=color, linewidth=2, linestyle=ls, zorder=3,
                marker="o", markersize=4, markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=1.0)
        ax.text(K_MAX + 0.25, c[-1], f"{c[-1]:.1f}", va="center", ha="left", fontsize=6.5, color=INK_2)
        print(f"{label:<22s}" + "".join(f"  {c[k-1]:5.1f}" for k in (1, 2, 3, 5, 10)))

    ax.set_xticks(ks)
    ax.set_xlim(0.7, K_MAX + 1.6)
    ax.set_xlabel("Rank k", color=INK_2)
    ax.set_ylabel(f"CMC (%) -- {args.terrain.replace('_', ' ')}", color=INK_2)
    ax.tick_params(axis="both", length=0, colors=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.legend([plt.Line2D([0], [0], color=c, linestyle=ls, linewidth=2) for _, _, c, ls in configs],
              [l for _, l, _, _ in configs], loc="lower right", frameon=False, labelcolor=INK_2,
              handlelength=1.6)

    fig.tight_layout(pad=0.3)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"fig_cmc_{args.terrain.replace('.', '')}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"Figure ecrite : {out}")


if __name__ == "__main__":
    main()
