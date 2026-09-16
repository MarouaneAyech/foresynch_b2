#!/usr/bin/env python
"""Lance tout ou partie de la grille scope x mécanisme du papier, automatiquement.

Remplace le changement manuel de MODE/SEED/LORA_R en tête du notebook d'origine :
la grille complète (8 configs x 3 seeds = 24 runs) est une donnée déclarée une seule
fois dans forensic_fr.training.grid, pas du code à éditer à la main.

Exemples :
    # tout le papier (24 runs)
    python scripts/run_grid.py

    # seulement les 3 configs prioritaires de la Phase 1 (analyse spectrale), 1 seed
    python scripts/run_grid.py --only ft_34 full_ft lora_34 --seeds 42

    # le contrôle FT-sans-ancrage de la Phase 3.4 / 4.3.2.C, 1 seed
    python scripts/run_grid.py --only ft_34 --seeds 42 --no-anchor --experiment-id E5_control

    # voir le plan sans rien lancer
    python scripts/run_grid.py --dry-run
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forensic_fr.training import DEFAULT_SEEDS, build_plan, run_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", default=None,
                         help="sous-ensemble de modes à lancer (ex: ft_34 full_ft lora_34)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--no-anchor", action="store_true")
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--experiment-id", default="E5_LoRA")
    parser.add_argument("--dry-run", action="store_true", help="affiche le plan sans entraîner")
    args = parser.parse_args()

    plan = build_plan(only=args.only, seeds=args.seeds, anchor=not args.no_anchor,
                       n_epochs=args.n_epochs, experiment_id=args.experiment_id)

    print(f"{len(plan)} run(s) planifiés :")
    for rc in plan:
        r_tag = f" r={rc.lora_r}" if "lora" in rc.mode or rc.mode == "hybrid" else ""
        anchor_tag = "avec ancrage" if rc.anchor else "SANS ancrage"
        print(f"  - {rc.mode}{r_tag}  seed={rc.seed}  {anchor_tag}  ({rc.n_epochs} epochs)")

    if args.dry_run:
        return

    for i, rc in enumerate(plan, 1):
        print(f"\n[{i}/{len(plan)}] === {rc.mode} seed={rc.seed} ===")
        run_training(rc)


if __name__ == "__main__":
    main()
