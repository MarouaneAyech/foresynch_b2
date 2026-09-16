import numpy as np
import torch
import torch.nn as nn


def quick_eval_all_terrains(
    backbone_model: nn.Module,
    cache: dict,
    device: str,
    batch_size: int = 64,
) -> dict[str, float]:
    """Rank-1 identification accuracy (%) par terrain (cosinus embeddings L2-normalisés)."""
    backbone_model.eval()

    def embed_tensors(tensor_list):
        embs = []
        with torch.no_grad():
            for i in range(0, len(tensor_list), batch_size):
                batch = torch.stack(tensor_list[i : i + batch_size]).to(device)
                out = backbone_model(batch).cpu().numpy().astype(np.float32)
                out = out / np.linalg.norm(out, axis=1, keepdims=True)
                embs.append(out)
        return np.concatenate(embs) if embs else np.empty((0,))

    gallery_ids = list(cache["gallery"].keys())
    gmat = embed_tensors([cache["gallery"][i] for i in gallery_ids])

    res = {}
    for tname, probes in cache["probes"].items():
        if not probes:
            continue
        pmat = embed_tensors([pr["tensor"] for pr in probes])
        preds = gmat @ pmat.T
        pred_ids = [gallery_ids[np.argmax(preds[:, j])] for j in range(pmat.shape[0])]
        correct = sum(1 for j, pr in enumerate(probes) if pred_ids[j] == pr["iid"])
        res[tname] = round(100 * correct / len(probes), 2)

    backbone_model.train()
    return res
