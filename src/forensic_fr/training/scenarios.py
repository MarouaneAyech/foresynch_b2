"""Registre des 7 scénarios de la grille scope x mécanisme.

Remplace le bloc `if MODE == ... elif MODE == ...` du notebook d'origine, où le
scénario était changé en commentant/décommentant une ligne à la main. Ici, chaque
scénario est une fonction enregistrée dans SCENARIOS ; changer de scénario revient à
passer un nom différent en paramètre (voir training.trainer.run_training), ce qui
permet de boucler dessus automatiquement (scripts/run_grid.py).
"""
from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from ..models import inject_lora

FROZEN_STEM = ("conv1", "bn1", "prelu", "layer1", "layer2")


@dataclass
class ScenarioResult:
    n_lora_convs: int = 0
    n_lora_fc: int = 0


def _freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def _freeze_prefixes(model: nn.Module, prefixes: tuple[str, ...]) -> None:
    for name, module in model.named_modules():
        if name.split(".")[0] in prefixes:
            for p in module.parameters():
                p.requires_grad = False


def configure_full_ft(model: nn.Module, **_) -> ScenarioResult:
    _unfreeze_all(model)
    model.features.weight.requires_grad = False
    return ScenarioResult()


def configure_ft_34(model: nn.Module, **_) -> ScenarioResult:
    _unfreeze_all(model)
    _freeze_prefixes(model, FROZEN_STEM)
    model.features.weight.requires_grad = False
    return ScenarioResult()


def configure_ft_4(model: nn.Module, **_) -> ScenarioResult:
    _unfreeze_all(model)
    _freeze_prefixes(model, FROZEN_STEM + ("layer3",))
    model.features.weight.requires_grad = False
    return ScenarioResult()


def configure_full_lora(model: nn.Module, lora_r: int = 8, lora_alpha: int = 16, **_) -> ScenarioResult:
    _freeze_all(model)
    n_conv, n_fc = inject_lora(
        model, r=lora_r, alpha=lora_alpha,
        target_layers=("layer1", "layer2", "layer3", "layer4"), include_fc=True,
    )
    return ScenarioResult(n_conv, n_fc)


def configure_lora_34(model: nn.Module, lora_r: int = 8, lora_alpha: int = 16, **_) -> ScenarioResult:
    _freeze_all(model)
    n_conv, n_fc = inject_lora(
        model, r=lora_r, alpha=lora_alpha,
        target_layers=("layer3", "layer4"), include_fc=True,
    )
    return ScenarioResult(n_conv, n_fc)


def configure_lora_4(model: nn.Module, lora_r: int = 8, lora_alpha: int = 16, **_) -> ScenarioResult:
    _freeze_all(model)
    n_conv, n_fc = inject_lora(
        model, r=lora_r, alpha=lora_alpha,
        target_layers=("layer4",), include_fc=False,
    )
    return ScenarioResult(n_conv, n_fc)


def configure_hybrid(model: nn.Module, lora_r: int = 8, lora_alpha: int = 16, **_) -> ScenarioResult:
    _unfreeze_all(model)
    _freeze_prefixes(model, FROZEN_STEM)
    model.features.weight.requires_grad = False
    n_conv, n_fc = inject_lora(
        model, r=lora_r, alpha=lora_alpha,
        target_layers=("layer1", "layer2"), include_fc=False,
    )
    return ScenarioResult(n_conv, n_fc)


SCENARIOS: dict[str, Callable[..., ScenarioResult]] = {
    "full_ft": configure_full_ft,
    "ft_34": configure_ft_34,
    "ft_4": configure_ft_4,
    "full_lora": configure_full_lora,
    "lora_34": configure_lora_34,
    "lora_4": configure_lora_4,
    "hybrid": configure_hybrid,
}
