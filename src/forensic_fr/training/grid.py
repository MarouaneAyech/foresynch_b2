"""Définition de la grille scope x mécanisme du papier — source unique, partagée par
`scripts/run_grid.py` (CLI) et `notebooks/colab_runner.ipynb` (Colab).
"""
from .trainer import RunConfig

GRID: list[dict] = [
    {"mode": "ft_4"},
    {"mode": "ft_34"},
    {"mode": "full_ft"},
    {"mode": "lora_4", "lora_r": 8, "lora_alpha": 16},
    {"mode": "lora_34", "lora_r": 8, "lora_alpha": 16},
    # ablation de rang (Sec. 4.3 du papier / Table hyperparams) : alpha=32, PAS 16 —
    # le scaling alpha/r doit rester 2.0 comme pour r=8 (16/8=2.0), pas retomber a 1.0.
    {"mode": "lora_34", "lora_r": 16, "lora_alpha": 32},
    # 3e point de l'ablation de rang, ajoute apres correction BN gelees (r=16 rapprochait
    # deja LoRA du fine-tuning sur ir 4.20m) -- meme ratio alpha/r=2.0 que les deux autres.
    {"mode": "lora_34", "lora_r": 32, "lora_alpha": 64},
    # 4e et DERNIER point de l'ablation de rang (decide a l'avance, pas apres coup selon
    # le resultat) -- complete la serie geometrique 8/16/32/64 pour une courbe
    # log(rang)->performance a comparer au rang effectif mesure en Phase 1.1.
    {"mode": "lora_34", "lora_r": 64, "lora_alpha": 128},
    {"mode": "full_lora", "lora_r": 8, "lora_alpha": 16},
    # Baseline de controle (Phase 3.1) : fc + tete seulement, backbone gele. A lancer
    # avec freeze_bn=True comme LoRA -- sinon les stats BN derivent et la baseline
    # mesure autre chose que "fc seule".
    {"mode": "fc_only"},
    {"mode": "hybrid", "lora_r": 8, "lora_alpha": 16},
]
DEFAULT_SEEDS = [7, 42, 123]


def cell_id(cell: dict) -> str:
    """Identifiant precis d'une cellule, ex. 'lora_34_r16' -- distingue les deux
    cellules qui partagent le meme mode (ablation de rang lora_34 r=8 vs r=16)."""
    lora_r = cell.get("lora_r")
    return f"{cell['mode']}_r{lora_r}" if lora_r is not None else cell["mode"]


def build_plan(
    only: list[str] | None = None,
    seeds: list[int] | None = None,
    anchor: bool = True,
    freeze_bn: bool = False,
    n_epochs: int = 20,
    experiment_id: str = "E5_LoRA",
) -> list[RunConfig]:
    """`only` matche soit le mode (ex. 'lora_34' -> ses deux cellules r=8 et r=16),
    soit l'id precis d'une cellule (ex. 'lora_34_r16' -> seulement celle-la)."""
    seeds = seeds if seeds is not None else DEFAULT_SEEDS
    cells = [c for c in GRID
             if only is None or c["mode"] in only or cell_id(c) in only]
    if not cells:
        raise ValueError(f"aucune config ne correspond a only={only!r}")
    return [
        RunConfig(
            mode=cell["mode"], seed=seed, lora_r=cell.get("lora_r", 8),
            lora_alpha=cell.get("lora_alpha", 16),
            anchor=anchor, freeze_bn=freeze_bn,
            n_epochs=n_epochs, experiment_id=experiment_id,
        )
        for cell in cells
        for seed in seeds
    ]
