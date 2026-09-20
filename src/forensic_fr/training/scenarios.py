"""Registre des 8 scénarios de la grille scope x mécanisme.

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


def configure_fc_only(model: nn.Module, **_) -> ScenarioResult:
    """Baseline de controle (Phase 3.1 du plan) : seule la projection finale `fc`
    (features conv aplaties 25088 -> embedding 512, 12.85M parametres) est
    entrainee, avec la tete ArcFace. Tout le tronc convolutif reste gele, et la
    BN1d finale `features` reste gelee comme dans tous les autres scenarios.

    Repond a l'objection "et si tout le gain venait de fc ?" : toutes les
    configurations de la grille adaptent fc (directement en FT, par adaptateur en
    LoRA), et fc est l'etage au plus grand deplacement relatif en fine-tuning libre
    (Phase 1.3, 22.9%). A lancer avec freeze_bn=True (meme lecture que LoRA : le
    backbone gele doit l'etre aussi au sens de la fonction).

    Attention : fc n'est PAS la tete ArcFace. La tete (512 x n_identites) est jetee
    a l'inference ; fc produit l'embedding et reste dans le chemin d'inference.
    """
    _freeze_all(model)
    for p in model.fc.parameters():
        p.requires_grad = True
    return ScenarioResult()


def configure_hybrid(model: nn.Module, lora_r: int = 8, lora_alpha: int = 16, **_) -> ScenarioResult:
    _unfreeze_all(model)
    _freeze_prefixes(model, FROZEN_STEM)
    model.features.weight.requires_grad = False
    n_conv, n_fc = inject_lora(
        model, r=lora_r, alpha=lora_alpha,
        target_layers=("layer1", "layer2"), include_fc=False,
    )
    return ScenarioResult(n_conv, n_fc)


def freeze_all_batchnorm(model: nn.Module) -> int:
    """Met tous les BatchNorm du modele en mode eval (statistiques courantes
    figees), meme si le reste du modele reste en mode train pour le calcul du
    gradient -- controle Phase 3.2 du plan (BN explicitement gelees). A rappeler
    a chaque debut d'epoque : `full_model.train()` remet tout en mode train,
    y compris les BN qu'on veut garder figees.

    requires_grad=False sur les parametres affines (weight/bias) d'un BatchNorm
    n'empeche PAS la mise a jour de running_mean/running_var : ces deux choses
    sont independantes dans PyTorch. Seul .eval() bloque cette mise a jour.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
            n += 1
    return n


SCENARIOS: dict[str, Callable[..., ScenarioResult]] = {
    "full_ft": configure_full_ft,
    "ft_34": configure_ft_34,
    "ft_4": configure_ft_4,
    "full_lora": configure_full_lora,
    "lora_34": configure_lora_34,
    "lora_4": configure_lora_4,
    "fc_only": configure_fc_only,
    "hybrid": configure_hybrid,
}
