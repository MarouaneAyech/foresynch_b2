"""Définition de la grille scope x mécanisme du papier — source unique, partagée par
`scripts/run_grid.py` (CLI) et `notebooks/colab_runner.ipynb` (Colab).
"""
from .trainer import RunConfig

GRID: list[dict] = [
    {"mode": "ft_4"},
    {"mode": "ft_34"},
    {"mode": "full_ft"},
    {"mode": "lora_4", "lora_r": 8},
    {"mode": "lora_34", "lora_r": 8},
    {"mode": "lora_34", "lora_r": 16},   # ablation de rang (Sec. 4.3 du papier)
    {"mode": "full_lora", "lora_r": 8},
    {"mode": "hybrid", "lora_r": 8},
]
DEFAULT_SEEDS = [7, 42, 123]


def build_plan(
    only: list[str] | None = None,
    seeds: list[int] | None = None,
    anchor: bool = True,
    n_epochs: int = 20,
    experiment_id: str = "E5_LoRA",
) -> list[RunConfig]:
    seeds = seeds if seeds is not None else DEFAULT_SEEDS
    cells = [c for c in GRID if only is None or c["mode"] in only]
    if not cells:
        raise ValueError(f"aucune config ne correspond a only={only!r}")
    return [
        RunConfig(
            mode=cell["mode"], seed=seed, lora_r=cell.get("lora_r", 8),
            anchor=anchor, n_epochs=n_epochs, experiment_id=experiment_id,
        )
        for cell in cells
        for seed in seeds
    ]
