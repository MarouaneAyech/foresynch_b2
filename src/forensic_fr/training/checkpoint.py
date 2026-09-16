"""Sauvegarde/chargement des poids de modèle.

Le notebook d'origine créait le dossier `checkpoints/` mais n'y écrivait jamais rien
(seul un JSON d'historique de métriques par époque était sauvegardé) — voir AUDIT.
Ce module comble ce manque : c'est lui qui fournit les poids nécessaires à la Phase 1
du plan (analyse spectrale de Delta W).
"""
from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(model: nn.Module, path: Path, **meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), **meta}, path)


def load_checkpoint(model: nn.Module, path: Path, map_location: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"], strict=False)
    return ckpt


def checkpoint_name(mode: str, seed: int, lora_r: int | None, anchor: bool,
                     freeze_bn: bool = False) -> str:
    r_tag = f"_r{lora_r}" if lora_r is not None else ""
    anchor_tag = "anchor" if anchor else "noanchor"
    bn_tag = "_bnfrozen" if freeze_bn else ""
    return f"{mode}{r_tag}_seed{seed}_{anchor_tag}{bn_tag}.pt"
