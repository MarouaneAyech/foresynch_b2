#!/usr/bin/env python
"""Phase 1.2 du plan — figure des spectres de valeurs singulieres.

Charge un checkpoint FT layer3+4 (ft_34), calcule le spectre normalise (s_i/s_1)
de chaque couche adaptee, agrege par etage (layer3, layer4, fc) via la mediane, et
trace les 3 courbes en echelle log-log. Des lignes verticales marquent les rangs
LoRA testes dans l'ablation (8/16/32/64) pour rendre visible d'un coup d'oeil la
fraction d'energie de la mise a jour FT que chaque rang ne capture pas.

Ne necessite aucun checkpoint LoRA : les lignes verticales ne font que marquer une
position x=r sur le spectre FT, elles ne comparent pas a un LoRA reellement entraine
(cette comparaison-la est deja faite par l'ablation de rang elle-meme).

Usage :
    python analysis/scripts/spectrum_figure.py
    python analysis/scripts/spectrum_figure.py --checkpoint ft_34_seed42_anchor.pt
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forensic_fr import config as cfg  # noqa: E402
from spectral_analysis import (  # noqa: E402
    delta_spectrum, energy_at_rank, iter_target_layers, load_checkpoint_model,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
RANKS_TO_MARK = [8, 16, 32, 64]
STAGE_COLORS = {"layer3": "tab:blue", "layer4": "tab:orange", "fc": "tab:green"}


def collect_stage_spectra(model, pretrained_model) -> dict[str, list[np.ndarray]]:
    pretrained_modules = dict(pretrained_model.named_modules())
    stage_spectra: dict[str, list[np.ndarray]] = {"layer3": [], "layer4": [], "fc": []}
    for layer_name, module, kind in iter_target_layers(model):
        if kind != "ft":
            continue  # cette figure ne concerne que le spectre FT
        stage = "fc" if layer_name == "fc" else layer_name.split(".")[0]
        if stage not in stage_spectra:
            continue
        W0 = pretrained_modules[layer_name].weight.detach()
        Wft = module.weight.detach()
        s = delta_spectrum(W0, Wft)["spectrum"]
        if len(s) == 0:
            continue
        stage_spectra[stage].append(s / s[0])  # normalise s_i/s_1
    return stage_spectra


def median_curve(curves: list[np.ndarray]) -> np.ndarray:
    max_len = max(len(c) for c in curves)
    padded = np.full((len(curves), max_len), np.nan)
    for i, c in enumerate(curves):
        padded[i, : len(c)] = c
    return np.nanmedian(padded, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-id", default="E5_LoRA")
    parser.add_argument("--checkpoint", default=None,
                         help="nom exact du checkpoint FT (defaut: premier ft_34_*.pt trouve)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = cfg.load_config()
    run_dir = cfg.run_dir(args.experiment_id)
    ckpt_dir = run_dir / "checkpoints"

    if args.checkpoint:
        ckpt_path = ckpt_dir / args.checkpoint
    else:
        candidates = sorted(ckpt_dir.glob("ft_34_*.pt"))
        if not candidates:
            raise SystemExit(f"aucun checkpoint ft_34_*.pt trouve dans {ckpt_dir}")
        ckpt_path = candidates[0]
    print(f"Checkpoint utilise : {ckpt_path.name}")

    model, _ckpt = load_checkpoint_model(ckpt_path, config["pretrained_path"],
                                          config["embedding_dim"], args.device)
    from spectral_analysis import load_pretrained  # noqa: E402
    pretrained_model = load_pretrained(config["pretrained_path"], config["embedding_dim"],
                                        args.device)

    stage_spectra = collect_stage_spectra(model, pretrained_model)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    energy_report: dict[str, dict[int, float]] = {}
    for stage, curves in stage_spectra.items():
        if not curves:
            continue
        curve = median_curve(curves)
        ax.plot(np.arange(1, len(curve) + 1), curve, label=stage,
                 color=STAGE_COLORS[stage], linewidth=1.6)
        energy_report[stage] = {
            r: float(np.mean([energy_at_rank(c, r) for c in curves])) for r in RANKS_TO_MARK
        }

    ymin = min(np.nanmin(c) for curves in stage_spectra.values() for c in curves if len(c))
    for r in RANKS_TO_MARK:
        ax.axvline(r, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(r, ymin, f"r={r}", rotation=90, fontsize=7, va="bottom", ha="right",
                 color="gray")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("indice de la valeur singuliere")
    ax.set_ylabel(r"$s_i / s_1$ (normalise)")
    ax.set_title("Spectre de Delta W (FT layer3+4), median par etage")
    ax.legend()
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "fig_spectrum.pdf"
    fig.savefig(out_path)
    print(f"\nFigure ecrite : {out_path}")

    print("\nEnergie de la mise a jour FT capturee par rang (moyenne des couches du stage) :")
    print("-> repond a \"LoRA r=X captures only Y% of the fine-tuning update energy\"")
    for stage, energies in energy_report.items():
        for r, e in energies.items():
            print(f"  {stage:<8s} r={r:<3d} -> {e*100:5.1f}%")


if __name__ == "__main__":
    main()
