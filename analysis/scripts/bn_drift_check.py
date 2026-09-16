#!/usr/bin/env python
"""Phase 1.4 du plan -- controle du confondant BatchNorm.

Compare les buffers `running_mean`/`running_var` de chaque BatchNorm des
checkpoints LoRA a ceux du modele pre-entraine. Enjeu : `model.train()` est
actif pendant tout l'entrainement (voir training/trainer.py) ; un BatchNorm
continue de mettre a jour ses statistiques courantes des qu'il voit des
donnees en mode train, meme si ses parametres affines (weight/bias) sont geles
via requires_grad=False -- ces deux choses sont independantes dans PyTorch.
Si les stats BN dérivent notablement sous LoRA, le backbone n'est pas
reellement "fige", ce qui pourrait etre une explication alternative a une
partie de l'ecart LoRA/FT en infrarouge (les stats visible/IR etant tres
differentes).

Usage :
    python analysis/scripts/bn_drift_check.py
    python analysis/scripts/bn_drift_check.py --experiment-id E5_LoRA --only lora_34
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from forensic_fr import config as cfg  # noqa: E402

from spectral_analysis import checkpoint_label, load_checkpoint_model, load_pretrained  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def bn_modules(model: nn.Module) -> dict:
    return {name: m for name, m in model.named_modules()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))}


def relative_drift(a: torch.Tensor, b_ref: torch.Tensor) -> float:
    denom = b_ref.norm().item()
    if denom == 0:
        return 0.0
    return float((a - b_ref).norm().item() / denom)


def analyze_checkpoint(ckpt_path: Path, pretrained_model: nn.Module, pretrained_path: str,
                        embedding_dim: int, device: str) -> list[dict]:
    model, ckpt = load_checkpoint_model(ckpt_path, pretrained_path, embedding_dim, device)
    label = checkpoint_label(ckpt)
    seed = ckpt.get("seed")

    pre_bn = bn_modules(pretrained_model)
    rows = []
    for name, bn in bn_modules(model).items():
        pre = pre_bn[name]
        drift_mean = relative_drift(bn.running_mean, pre.running_mean)
        drift_var = relative_drift(bn.running_var, pre.running_var)
        rows.append({"checkpoint": label, "seed": seed, "layer_name": name,
                      "drift_mean": drift_mean, "drift_var": drift_var})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-id", default="E5_LoRA")
    parser.add_argument("--only", nargs="+", default=["lora"],
                         help="filtre les checkpoints par prefixe de mode "
                              "(defaut: uniquement les configs LoRA)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = cfg.load_config()
    run_dir = cfg.run_dir(args.experiment_id)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_paths = sorted(ckpt_dir.glob("*.pt"))
    ckpt_paths = [p for p in ckpt_paths if any(m in p.name for m in args.only)]
    if not ckpt_paths:
        raise SystemExit(f"aucun checkpoint LoRA trouve dans {ckpt_dir} (filtre --only={args.only})")

    print(f"{len(ckpt_paths)} checkpoint(s) LoRA a controler dans {ckpt_dir}")
    pretrained_model = load_pretrained(config["pretrained_path"], config["embedding_dim"],
                                        args.device)

    all_rows: list[dict] = []
    for p in ckpt_paths:
        print(f"\n=== {p.name} ===")
        rows = analyze_checkpoint(p, pretrained_model, config["pretrained_path"],
                                   config["embedding_dim"], args.device)
        all_rows.extend(rows)
        for r in rows:
            flag = "!!" if max(r["drift_mean"], r["drift_var"]) > 0.01 else "  "
            print(f"  {flag} {r['layer_name']:<24s} drift_mean={r['drift_mean']*100:6.2f}%  "
                  f"drift_var={r['drift_var']*100:6.2f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "bn_drift.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "seed", "layer_name",
                                                 "drift_mean", "drift_var"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV ecrit : {csv_path}")

    max_drift = max((max(r["drift_mean"], r["drift_var"]) for r in all_rows), default=0.0)
    print(f"\nDerive maximale observee : {max_drift*100:.2f}%")
    if max_drift < 0.01:
        print("-> Negligeable (<1%). Le backbone LoRA peut etre decrit comme gele en "
              "Section 3.1 : 'BN running statistics are kept frozen in all LoRA "
              "configurations'.")
    else:
        print("-> NON negligeable. A declarer explicitement dans le papier, et "
              "declenche le controle de la Phase 3.2 (LoRA avec BN explicitement gelees).")


if __name__ == "__main__":
    main()
