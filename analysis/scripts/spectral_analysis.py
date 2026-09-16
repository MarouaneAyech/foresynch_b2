#!/usr/bin/env python
"""Phase 1.1 du plan — rang effectif de Delta W sur les checkpoints réels.

Charge les checkpoints produits par scripts/run_grid.py (FT layer3+4, Full FT,
LoRA layer3+4 r=8/r=16), calcule Delta W = W_ft - W_0 pour chaque couche adaptée
(convs 3x3 de layer3/layer4 + fc), et en extrait le rang effectif via SVD.

Pour les checkpoints FT : Delta W = poids fine-tunes - poids pre-entraines, directement.
Pour les checkpoints LoRA : le backbone gele est identique au pre-entraine (jamais
modifie) ; le "Delta W effectif" est reconstruit en composant les deux matrices de
l'adaptateur, (alpha/r)*B*A, comme demande par le plan (Phase 1.1).

Usage :
    python analysis/scripts/spectral_analysis.py
    python analysis/scripts/spectral_analysis.py --experiment-id E5_LoRA --only ft_34 full_ft
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from forensic_fr import config as cfg  # noqa: E402
from forensic_fr.models import LoRAConv2d, LoRALinear, iresnet50  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
SPECTRA_DIR = OUT_DIR / "spectra"


def delta_spectrum(W0: torch.Tensor, Wft: torch.Tensor) -> dict:
    """W: (C_out, C_in, kh, kw) pour une conv, (d_out, d_in) pour fc."""
    dW = (Wft - W0).reshape(W0.shape[0], -1).float().cpu().numpy()
    s = np.linalg.svd(dW, compute_uv=False)
    s = s[s > 0]
    energy = s**2
    p = energy / energy.sum()
    erank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))  # Roy & Vetterli
    cum = np.cumsum(energy) / energy.sum()
    r90 = int(np.searchsorted(cum, 0.90) + 1)
    r99 = int(np.searchsorted(cum, 0.99) + 1)
    return dict(
        max_rank=int(min(dW.shape)),
        erank=erank, r90=r90, r99=r99,
        rel_norm=float(np.linalg.norm(dW) / np.linalg.norm(
            W0.reshape(W0.shape[0], -1).float().cpu().numpy())),
        spectrum=s,
    )


def energy_at_rank(spectrum: np.ndarray, r: int) -> float:
    """Fraction de l'energie de la mise a jour capturee par les r premieres
    directions -- repond a "LoRA r=8 captures only X% of the update energy"."""
    energy = spectrum**2
    total = energy.sum()
    if total == 0 or len(spectrum) == 0:
        return 0.0
    k = min(r, len(spectrum))
    return float(energy[:k].sum() / total)


def lora_effective_delta_conv(m: LoRAConv2d) -> torch.Tensor:
    """(alpha/r)*B*A pour un LoRAConv2d : A est 1x1 (melange de canaux), B porte
    le noyau spatial d'origine -- leur composition equivaut a une seule conv kxk."""
    A = m.lora_A.weight.detach().squeeze(-1).squeeze(-1)  # (r, in_ch)
    B = m.lora_B.weight.detach()                          # (out_ch, r, k, k)
    return torch.einsum("orkl,ri->oikl", B, A) * m.scaling


def lora_effective_delta_linear(m: LoRALinear) -> torch.Tensor:
    A = m.lora_A.weight.detach()  # (r, in_f)
    B = m.lora_B.weight.detach()  # (out_f, r)
    return (B @ A) * m.scaling


def stage_of(layer_name: str) -> str:
    if layer_name == "fc":
        return "fc"
    for p in ("layer3", "layer4"):
        if layer_name.startswith(p):
            return p
    return "other"


def iter_target_layers(model: nn.Module):
    """(layer_name, module, kind) pour les convs 3x3 (ou LoRAConv2d) de layer3/
    layer4 et le fc (ou LoRALinear) -- la portee "layer3+4" adaptee dans le papier.
    Les sous-modules internes d'un wrapper LoRA (.conv/.linear/.lora_A/.lora_B)
    sont ignores : c'est le wrapper qui les represente."""
    for name, module in model.named_modules():
        if not (name.startswith("layer3") or name.startswith("layer4") or name == "fc"):
            continue
        if name.endswith((".conv", ".linear", ".lora_A", ".lora_B")):
            continue
        if isinstance(module, LoRAConv2d):
            yield name, module, "lora"
        elif isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3):
            yield name, module, "ft"
        elif isinstance(module, LoRALinear) and name == "fc":
            yield name, module, "lora"
        elif isinstance(module, nn.Linear) and name == "fc":
            yield name, module, "ft"


def load_pretrained(pretrained_path: str, embedding_dim: int, device: str) -> nn.Module:
    model = iresnet50(num_features=embedding_dim, fp16=False)
    model.load_state_dict(torch.load(pretrained_path, map_location="cpu"), strict=False)
    return model.to(device).eval()


def load_checkpoint_model(ckpt_path: Path, pretrained_path: str, embedding_dim: int,
                           device: str):
    from forensic_fr.training import SCENARIOS  # import tardif, evite un cycle a l'init

    ckpt = torch.load(ckpt_path, map_location=device)
    mode = ckpt["mode"]
    lora_r = ckpt.get("lora_r")
    lora_alpha = (ckpt.get("run_config") or {}).get("lora_alpha", 16)

    model = iresnet50(num_features=embedding_dim, fp16=False)
    model.load_state_dict(torch.load(pretrained_path, map_location="cpu"), strict=False)
    SCENARIOS[mode](model, lora_r=lora_r or 8, lora_alpha=lora_alpha)
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model.to(device).eval(), ckpt


def checkpoint_label(ckpt: dict) -> str:
    mode = ckpt["mode"]
    lora_r = ckpt.get("lora_r")
    return f"{mode}_r{lora_r}" if lora_r else mode


def analyze_checkpoint(ckpt_path: Path, pretrained_model: nn.Module, pretrained_path: str,
                        embedding_dim: int, device: str) -> list[dict]:
    model, ckpt = load_checkpoint_model(ckpt_path, pretrained_path, embedding_dim, device)
    label = checkpoint_label(ckpt)
    seed = ckpt.get("seed")
    pretrained_modules = dict(pretrained_model.named_modules())

    rows = []
    for layer_name, module, kind in iter_target_layers(model):
        if kind == "ft":
            W0 = pretrained_modules[layer_name].weight.detach()
            Wft = module.weight.detach()
        else:  # kind == "lora"
            if isinstance(module, LoRAConv2d):
                W0 = module.conv.weight.detach()
                dW = lora_effective_delta_conv(module)
            else:
                W0 = module.linear.weight.detach()
                dW = lora_effective_delta_linear(module)
            Wft = W0 + dW

        stats = delta_spectrum(W0, Wft)
        spectrum = stats.pop("spectrum")

        SPECTRA_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = layer_name.replace(".", "_")
        np.save(SPECTRA_DIR / f"{label}__{safe_name}.npy", spectrum)

        row = {"checkpoint": label, "seed": seed, "layer_name": layer_name,
               "stage": stage_of(layer_name), **stats}
        rows.append(row)

        e8 = energy_at_rank(spectrum, 8)
        e16 = energy_at_rank(spectrum, 16)
        print(f"  {label:<14s} {layer_name:<20s} erank={stats['erank']:6.1f}  "
              f"r90={stats['r90']:4d}  r99={stats['r99']:4d}  "
              f"energy@8={e8*100:5.1f}%  energy@16={e16*100:5.1f}%")

    return rows


def summarize(rows: list[dict]) -> None:
    print("\n== Rang effectif median par etage (agrege sur les checkpoints/seeds) ==")
    by_ckpt_stage: dict[tuple, list] = {}
    for r in rows:
        by_ckpt_stage.setdefault((r["checkpoint"], r["stage"]), []).append(r["erank"])
    for (ckpt, stage), eranks in sorted(by_ckpt_stage.items()):
        print(f"  {ckpt:<14s} {stage:<8s} erank_median={float(np.median(eranks)):6.1f} "
              f"(n={len(eranks)} couches)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-id", default="E5_LoRA")
    parser.add_argument("--only", nargs="+", default=None,
                         help="sous-ensemble de checkpoints a traiter, par mode "
                              "(ex: ft_34 full_ft lora_34)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = cfg.load_config()
    run_dir = cfg.run_dir(args.experiment_id)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_paths = sorted(ckpt_dir.glob("*.pt"))
    if args.only:
        ckpt_paths = [p for p in ckpt_paths if any(p.name.startswith(m) for m in args.only)]
    if not ckpt_paths:
        raise SystemExit(f"aucun checkpoint trouve dans {ckpt_dir} (filtre --only={args.only})")

    print(f"{len(ckpt_paths)} checkpoint(s) a analyser dans {ckpt_dir}")
    pretrained_model = load_pretrained(config["pretrained_path"], config["embedding_dim"],
                                        args.device)

    all_rows: list[dict] = []
    for p in ckpt_paths:
        print(f"\n=== {p.name} ===")
        all_rows.extend(analyze_checkpoint(p, pretrained_model, config["pretrained_path"],
                                            config["embedding_dim"], args.device))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "spectral_per_layer.csv"
    fieldnames = ["checkpoint", "seed", "layer_name", "stage", "max_rank", "erank",
                  "r90", "r99", "rel_norm"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV ecrit : {csv_path}")
    print(f"Spectres (.npy) ecrits dans : {SPECTRA_DIR}")

    summarize(all_rows)


if __name__ == "__main__":
    main()
