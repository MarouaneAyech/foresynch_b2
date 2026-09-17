"""Orchestrateur d'un run unique de la grille scope x mécanisme.

Équivalent du contenu manuel des cellules 9 à 36 du notebook `piepline_version_stable.ipynb`,
mais piloté par une RunConfig au lieu de variables MODE/SEED/LORA_R codées en dur à changer
à la main entre deux exécutions.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from .. import config as cfg_module
from ..data import build_train_loader, load_eval_cache
from ..evaluation import quick_eval_all_terrains
from ..models import ArcFaceHead, FullModel, LoRAConv2d, LoRALinear, iresnet50
from ..seeding import set_seed
from .checkpoint import checkpoint_name, save_checkpoint
from .diagnostics import diagnose_trainable
from .scenarios import SCENARIOS, freeze_all_batchnorm
from .scheduler import cosine_lr


@dataclass
class RunConfig:
    mode: str                      # une clé de training.scenarios.SCENARIOS
    seed: int
    lora_r: int = 8
    lora_alpha: int = 16
    anchor: bool = True            # False -> contrôle Phase 4.3.2.C (sans ancrage HQ)
    freeze_bn: bool = False        # True -> contrôle Phase 3.2 (BN explicitement gelées)
    n_epochs: int = 20
    warmup: int = 1
    base_lr: float = 1e-4
    weight_decay: float = 0.1
    experiment_id: str = "E5_LoRA"
    save_every: int = 0            # 0 = uniquement le checkpoint final
    device: str | None = None


def run_training(rc: RunConfig, verbose: bool = True) -> dict:
    if rc.mode not in SCENARIOS:
        raise ValueError(f"mode inconnu : {rc.mode!r} (attendu un de {list(SCENARIOS)})")

    set_seed(rc.seed)
    device = rc.device or ("cuda" if torch.cuda.is_available() else "cpu")

    config = cfg_module.load_config()
    run_dir = cfg_module.run_dir(rc.experiment_id)

    # 1) backbone pré-entraîné
    model = iresnet50(num_features=config["embedding_dim"], fp16=False)
    model.load_state_dict(torch.load(config["pretrained_path"], map_location="cpu"), strict=False)

    # 2) freeze/injection LoRA selon le scénario
    lora_r = rc.lora_r if "lora" in rc.mode or rc.mode == "hybrid" else None
    scenario_result = SCENARIOS[rc.mode](model, lora_r=rc.lora_r, lora_alpha=rc.lora_alpha)
    model = model.to(device)

    # 3) assemblage backbone + tête ArcFace
    head = ArcFaceHead(config["embedding_dim"], config["num_classes"],
                        config["arcface_margin"], config["arcface_scale"])
    full_model = FullModel(model, head).to(device)

    if verbose:
        diagnose_trainable(model, head=head, lora_class=(LoRAConv2d, LoRALinear))

    # 4) optimiseur — AdamW, identique pour tous les modes (contrainte du papier)
    trainable_params = [p for p in full_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=rc.base_lr, weight_decay=rc.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # 5) données
    loader = build_train_loader(
        cfg_module.CACHE_DIR, config["batch_size"], config["horizontal_flip_prob"],
        anchor=rc.anchor,
    )
    eval_cache = load_eval_cache(cfg_module.CACHE_DIR / "eval_aligned_cache.npz")
    baseline = quick_eval_all_terrains(full_model.backbone, eval_cache, device)

    # 6) boucle d'entraînement
    history = []
    for epoch in range(rc.n_epochs):
        lr = cosine_lr(epoch, rc.warmup, rc.n_epochs, rc.base_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        full_model.train()
        if rc.freeze_bn:
            freeze_all_batchnorm(full_model.backbone)
        correct, total = 0, 0
        for images, labels in tqdm(loader, desc=f"{rc.mode} seed{rc.seed} epoch{epoch+1}",
                                    disable=not verbose):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = full_model(images, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                emb = F.normalize(full_model.backbone(images), p=2, dim=1)
                w = F.normalize(full_model.head.weight, p=2, dim=1)
                correct += (F.linear(emb, w).argmax(1) == labels).sum().item()
                total += labels.size(0)

        train_acc = 100 * correct / total
        test_res = quick_eval_all_terrains(full_model.backbone, eval_cache, device)
        deltas = {t: test_res.get(t, 0) - baseline[t] for t in baseline}
        history.append({
            "epoch": epoch + 1, "train_acc": train_acc,
            "test": test_res, "deltas": deltas, "sum_delta": sum(deltas.values()),
        })
        if verbose:
            print(f"epoch {epoch+1}: train_acc={train_acc:.1f}%  sum_delta={sum(deltas.values()):+.1f}")

        if rc.save_every and (epoch + 1) % rc.save_every == 0:
            _save(model, run_dir, rc, lora_r, epoch=epoch + 1)

    ckpt_path = _save(model, run_dir, rc, lora_r, epoch=rc.n_epochs)

    # Un fichier par run (nom derive de celui du checkpoint), pas un fichier partage
    # par mode. L'ancien schema ("exp1_{mode}.json", lu-modifie-reecrit a chaque run)
    # subit une race condition des que deux sessions Colab (ou un restart de runtime
    # avec un cache Drive perime) ecrivent a peu pres en meme temps : la deuxieme
    # ecriture, basee sur une lecture anterieure a la premiere, ecrase les entrees
    # que l'autre venait d'ajouter. Un fichier unique par run rend cette race
    # impossible : aucune lecture-modification-ecriture partagee entre runs.
    run_label = checkpoint_name(rc.mode, rc.seed, lora_r, rc.anchor, rc.freeze_bn)[:-3]
    history_path = run_dir / f"exp1_{run_label}.json"
    history_path.write_text(json.dumps([{
        "mode": rc.mode, "optimizer": "AdamW", "base_lr": rc.base_lr,
        "weight_decay": rc.weight_decay, "lora_r": lora_r, "lora_alpha": rc.lora_alpha,
        "n_lora_convs": scenario_result.n_lora_convs, "anchor": rc.anchor,
        "freeze_bn": rc.freeze_bn,
        "n_epochs": rc.n_epochs, "warmup": rc.warmup, "seed": rc.seed,
        "checkpoint_path": str(ckpt_path), "results_b_epochs": history,
    }], indent=2))

    return {"history": history, "baseline": baseline, "checkpoint_path": str(ckpt_path),
            "history_path": str(history_path)}


def _save(model: nn.Module, run_dir: Path, rc: RunConfig, lora_r: int | None, epoch: int) -> Path:
    name = checkpoint_name(rc.mode, rc.seed, lora_r, rc.anchor, rc.freeze_bn)
    path = run_dir / "checkpoints" / name
    save_checkpoint(model, path, mode=rc.mode, seed=rc.seed, lora_r=lora_r,
                     anchor=rc.anchor, freeze_bn=rc.freeze_bn, epoch=epoch,
                     run_config=asdict(rc))
    return path
