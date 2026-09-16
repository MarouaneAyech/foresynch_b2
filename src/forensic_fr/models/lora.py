"""Adaptateurs LoRA pour Conv2d et Linear.

Réf. Hu et al. 2022 (LoRA, couches linéaires) ; Ding et al. 2024 (LoRA-C, extension
aux couches convolutives). Repris tel quel du notebook d'origine.
"""
import math

import torch.nn as nn


class LoRAConv2d(nn.Module):
    """W_effectif = W_figé + (B @ A) * (alpha/r).

    A : réduction 1x1 (in_ch -> r), sans composante spatiale.
    B : expansion avec le kernel spatial original (r -> out_ch), initialisée à zéro
    (l'adaptation démarre exactement au modèle pré-entraîné).
    """

    def __init__(self, conv_layer: nn.Conv2d, r: int = 8, alpha: int = 16):
        super().__init__()
        self.conv = conv_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        for p in self.conv.parameters():
            p.requires_grad = False

        in_ch = conv_layer.in_channels
        out_ch = conv_layer.out_channels
        k = conv_layer.kernel_size[0]

        self.lora_A = nn.Conv2d(in_ch, r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(
            r, out_ch, kernel_size=k,
            stride=conv_layer.stride, padding=conv_layer.padding, bias=False,
        )

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.conv(x) + self.lora_B(self.lora_A(x)) * self.scaling


class LoRALinear(nn.Module):
    """Même principe que LoRAConv2d, pour la couche fc finale."""

    def __init__(self, linear_layer: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.linear = linear_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        for p in self.linear.parameters():
            p.requires_grad = False

        in_f = linear_layer.in_features
        out_f = linear_layer.out_features

        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x)) * self.scaling


def inject_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    target_layers: tuple[str, ...] = ("layer3", "layer4"),
    include_fc: bool = False,
) -> tuple[int, int]:
    """Remplace les Conv2d 3x3 des `target_layers` par des LoRAConv2d, et
    optionnellement le `fc` final par un LoRALinear.

    Collecte d'abord toutes les cibles puis remplace (ne modifie pas l'arbre de
    modules pendant qu'on le parcourt).

    Retourne (n_convs_injectées, n_fc_injecté [0 ou 1]).
    """
    conv_targets = []
    for name, module in model.named_modules():
        if not any(name.startswith(t) for t in target_layers):
            continue
        for child_name, child in module.named_children():
            if isinstance(child, nn.Conv2d) and child.kernel_size == (3, 3):
                conv_targets.append((module, child_name, child))

    n_injected = 0
    for parent_module, child_name, conv in conv_targets:
        setattr(parent_module, child_name, LoRAConv2d(conv, r=r, alpha=alpha))
        n_injected += 1

    n_fc = 0
    if include_fc and hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = LoRALinear(model.fc, r=r, alpha=alpha)
        n_fc = 1

    return n_injected, n_fc
