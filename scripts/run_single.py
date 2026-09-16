#!/usr/bin/env python
"""Lance UN run de la grille scope x mécanisme.

Exemples :
    # cellule "FT layer3+4", seed 42, avec ancrage (comportement du papier)
    python scripts/run_single.py --mode ft_34 --seed 42

    # contrôle Phase 4.3.2.C : même config, SANS ancrage
    python scripts/run_single.py --mode ft_34 --seed 42 --no-anchor

    # ablation de rang LoRA
    python scripts/run_single.py --mode lora_34 --seed 7 --lora-r 16

    # contrôle Phase 3.2 : LoRA avec les BatchNorm explicitement gelées
    python scripts/run_single.py --mode lora_34 --seed 42 --freeze-bn
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forensic_fr.training import SCENARIOS, RunConfig, run_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=sorted(SCENARIOS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--no-anchor", action="store_true",
                         help="désactive le mélange 50/50 mugshot/surveillance")
    parser.add_argument("--freeze-bn", action="store_true",
                         help="gèle aussi les statistiques courantes BatchNorm "
                              "(controle Phase 3.2, pas seulement les poids/affines)")
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--experiment-id", default="E5_LoRA")
    parser.add_argument("--save-every", type=int, default=0)
    args = parser.parse_args()

    rc = RunConfig(
        mode=args.mode, seed=args.seed, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        anchor=not args.no_anchor, freeze_bn=args.freeze_bn,
        n_epochs=args.n_epochs, warmup=args.warmup,
        base_lr=args.base_lr, weight_decay=args.weight_decay,
        experiment_id=args.experiment_id, save_every=args.save_every,
    )
    result = run_training(rc)
    print(f"\nCheckpoint : {result['checkpoint_path']}")
    print(f"Historique : {result['history_path']}")


if __name__ == "__main__":
    main()
