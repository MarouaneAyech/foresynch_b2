import json
from pathlib import Path

import numpy as np
import torch


def load_eval_cache(path: Path) -> dict:
    """Recharge le cache d'évaluation (gallery + probes des 6 terrains) depuis un .npz."""
    data = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(data["__meta__"]).decode("utf-8"))

    cache = {"gallery": {}, "probes": {}}
    for iid in meta["gallery_ids"]:
        cache["gallery"][iid] = torch.from_numpy(data[f"gallery__{iid}"])

    for tname, iids in meta["terrains"].items():
        stacked = data[f"probes__{tname}"]
        cache["probes"][tname] = [
            {"iid": iids[k], "tensor": torch.from_numpy(stacked[k])}
            for k in range(stacked.shape[0])
        ]
    return cache
