import random

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedDataset(Dataset):
    """Dataset depuis un cache d'images déjà alignées (pas de ré-alignement ici)."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, hflip_prob: float = 0.5):
        self.images = images
        self.labels = labels
        self.hflip_prob = hflip_prob

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        aligned = self.images[idx].copy()
        label = int(self.labels[idx])
        if random.random() < self.hflip_prob:
            aligned = aligned[:, ::-1, :].copy()
        rgb = aligned[:, :, ::-1].copy()
        x = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
        x = (x - 127.5) / 127.5
        return torch.from_numpy(x), label
