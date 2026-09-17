#!/usr/bin/env python
"""Phase 1.3 du plan — profil de deplacement par etage (valide H1).

Sur le checkpoint Full FT uniquement (le seul scenario ou TOUT le backbone est
libre de bouger), calcule pour chaque etage du reseau :

    rel_displacement(stage) = ||W_ft - W_0||_F / ||W_0||_F

agrege sur TOUS les parametres entrainables de l'etage (pas seulement les convs
3x3 comme en Phase 1.1 -- ici on veut le tableau complet : stem, layer1, layer2,
layer3, layer4, fc). Si le fine-tuning, laisse libre d'adapter tout le reseau,
concentre spontanement son deplacement sur layer3+4, H1 cesse d'etre une
observation empirique du seul grid de resultats et devient une explication
structurelle.

`bn2` (la BN finale apres layer4) est regroupe avec `layer4`, et `features` (la
BN1d finale apres fc, toujours gelee par construction) est regroupe avec `fc` --
ce sont les composants immediatement adjacents dans l'architecture (voir
fig_pipeline.tex, qui ne les distingue pas non plus comme noeuds separes).

Usage :
    python analysis/scripts/stage_displacement.py
    python analysis/scripts/stage_displacement.py --checkpoint full_ft_seed42_anchor.pt
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forensic_fr import config as cfg  # noqa: E402
from spectral_analysis import load_checkpoint_model, load_pretrained  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
STAGES = ["stem", "layer1", "layer2", "layer3", "layer4", "fc"]


def stage_of_param(name: str) -> str | None:
    top = name.split(".")[0]
    if top in ("conv1", "bn1", "prelu"):
        return "stem"
    if top in ("layer1", "layer2", "layer3"):
        return top
    if top in ("layer4", "bn2"):
        return "layer4"
    if top in ("fc", "features"):
        return "fc"
    return None


def stage_displacement(model_ft: nn.Module, model0: nn.Module) -> dict[str, float]:
    sq_diff: dict[str, float] = defaultdict(float)
    sq_w0: dict[str, float] = defaultdict(float)
    params_ft = dict(model_ft.named_parameters())
    for name, p0 in model0.named_parameters():
        stage = stage_of_param(name)
        if stage is None:
            continue
        pft = params_ft[name]
        sq_diff[stage] += torch.sum((pft - p0) ** 2).item()
        sq_w0[stage] += torch.sum(p0 ** 2).item()
    return {s: (sq_diff[s] / sq_w0[s]) ** 0.5 for s in sq_diff}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-id", default="E5_LoRA")
    parser.add_argument("--checkpoint", default=None,
                         help="nom exact du checkpoint full_ft (defaut: premier full_ft_*.pt trouve)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = cfg.load_config()
    run_dir = cfg.run_dir(args.experiment_id)
    ckpt_dir = run_dir / "checkpoints"

    if args.checkpoint:
        ckpt_path = ckpt_dir / args.checkpoint
    else:
        candidates = sorted(ckpt_dir.glob("full_ft_*.pt"))
        if not candidates:
            raise SystemExit(f"aucun checkpoint full_ft_*.pt trouve dans {ckpt_dir}")
        ckpt_path = candidates[0]
    print(f"Checkpoint utilise : {ckpt_path.name}")

    model_ft, _ckpt = load_checkpoint_model(ckpt_path, config["pretrained_path"],
                                             config["embedding_dim"], args.device)
    model0 = load_pretrained(config["pretrained_path"], config["embedding_dim"], args.device)

    disp = stage_displacement(model_ft, model0)

    print("\nDeplacement relatif par etage (||W_ft - W_0||_F / ||W_0||_F) :")
    for s in STAGES:
        if s in disp:
            print(f"  {s:<8s} {disp[s]*100:6.2f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "stage_displacement.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "rel_displacement"])
        for s in STAGES:
            if s in disp:
                writer.writerow([s, disp[s]])
    print(f"\nCSV ecrit : {csv_path}")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    values = [disp[s] * 100 for s in STAGES if s in disp]
    labels = [s for s in STAGES if s in disp]
    ax.bar(labels, values, color="tab:blue")
    ax.set_ylabel("Déplacement relatif (%)")
    ax.set_title("Full FT — déplacement par étage vs. pré-entraîné")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / "fig_stage_displacement.pdf"
    fig.savefig(fig_path)
    print(f"Figure ecrite : {fig_path}")

    early = [disp[s] for s in ("stem", "layer1", "layer2") if s in disp]
    late = [disp[s] for s in ("layer3", "layer4") if s in disp]
    if early and late:
        print(f"\nMoyenne stem/layer1/layer2 : {sum(early)/len(early)*100:.2f}%")
        print(f"Moyenne layer3/layer4      : {sum(late)/len(late)*100:.2f}%")
        print("-> Si layer3/layer4 >> stem/layer1/layer2, le fine-tuning libre confirme "
              "quantitativement H1 (le scope layer3+4 est structurellement suffisant).")


if __name__ == "__main__":
    main()
