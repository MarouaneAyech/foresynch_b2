from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from .cache_dataset import CachedDataset


def build_train_loader(
    cache_dir: Path,
    batch_size: int,
    hflip_prob: float,
    anchor: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    """Construit le DataLoader d'entraînement.

    anchor=True  (comportement du papier) : mélange ~50/50 mugshots HQ (ancres) et
        surveillance dégradée réelle des identités d'entraînement, via un sampler pondéré.
    anchor=False (contrôle Phase 4.3.2.C du plan) : surveillance dégradée seule, sans
        aucune image haute qualité — sert à tester si l'ancrage est bien ce qui empêche
        l'oubli catastrophique/l'instabilité observés chez PETALface en son absence.
    """
    surv_imgs = np.load(cache_dir / "surv_train_images.npy")
    surv_lbls = np.load(cache_dir / "surv_train_labels.npy")

    if not anchor:
        dataset = CachedDataset(surv_imgs, surv_lbls, hflip_prob)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )

    real_imgs = np.load(cache_dir / "real_images.npy")
    real_lbls = np.load(cache_dir / "real_labels.npy")

    all_imgs = np.concatenate([real_imgs, surv_imgs])
    all_lbls = np.concatenate([real_lbls, surv_lbls])
    is_real = np.concatenate([np.ones(len(real_imgs)), np.zeros(len(surv_imgs))])

    w_real = len(surv_imgs) / len(real_imgs)
    weights = np.where(is_real == 1, w_real, 1.0)
    sampler = WeightedRandomSampler(weights, num_samples=len(surv_imgs) * 2, replacement=True)

    dataset = CachedDataset(all_imgs, all_lbls, hflip_prob)
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
