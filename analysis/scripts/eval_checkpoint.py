#!/usr/bin/env python
"""Phase 2.0 du plan -- evaluation complete d'un checkpoint, avec sauvegarde de tout
ce que l'entrainement ne sauvegarde pas : embeddings, matrices de scores, et le vecteur
correct/incorrect par probe (avec le rang de la bonne identite).

C'est le prerequis de toute la Phase 2 :
  2.3 bootstrap au niveau identite  -> lit  correct_{run}.csv
  2.2 ecart de domaine en embedding -> lit  eval_{run}.npz (embeddings)
  2.1 courbes CMC                   -> lit  eval_{run}.npz (scores) ou correct_{run}.csv (rank)

Le rank-1 recalcule par terrain doit etre IDENTIQUE au champ `test` de la derniere
epoque du JSON d'historique du run (meme checkpoint = epoque 20) : le script le
verifie et le signale. Sert donc aussi de secours si un JSON a ete perdu.

Sorties (dans <run_dir>/eval/) :
  eval_{run}.npz     gallery_ids, gallery_emb (n_g x 512), et par terrain T :
                     probes__T__ids, probes__T__emb (n_p x 512), scores__T (n_p x n_g)
  correct_{run}.csv  run, seed, terrain, probe_idx, iid, top1_iid, rank, correct

Usage (Colab, apres la section 2 du colab_runner) :
    python analysis/scripts/eval_checkpoint.py --pretrained            # baseline
    python analysis/scripts/eval_checkpoint.py --only lora_34_r32     # 3 seeds r=32
    python analysis/scripts/eval_checkpoint.py --all                   # tout
    python analysis/scripts/eval_checkpoint.py --checkpoint full_ft_seed42_anchor.pt
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forensic_fr import config as cfg  # noqa: E402
from forensic_fr.data import load_eval_cache  # noqa: E402
from forensic_fr.evaluation import quick_eval_all_terrains  # noqa: E402
from spectral_analysis import load_checkpoint_model, load_pretrained  # noqa: E402


@torch.no_grad()
def embed(model: nn.Module, tensors: list[torch.Tensor], device: str, batch_size: int = 64) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(tensors), batch_size):
        x = torch.stack(tensors[i:i + batch_size]).to(device)
        e = model(x).float().cpu().numpy()
        out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
    return np.concatenate(out) if out else np.empty((0, 512), dtype=np.float32)


def evaluate(model: nn.Module, cache: dict, device: str) -> dict:
    gallery_ids = list(cache["gallery"].keys())
    g_emb = embed(model, [cache["gallery"][i] for i in gallery_ids], device)
    gid_index = {iid: k for k, iid in enumerate(gallery_ids)}

    result = {"gallery_ids": gallery_ids, "gallery_emb": g_emb, "terrains": {}}
    for tname, probes in cache["probes"].items():
        if not probes:
            continue
        p_ids = [p["iid"] for p in probes]
        p_emb = embed(model, [p["tensor"] for p in probes], device)
        scores = p_emb @ g_emb.T                       # (n_probes, n_gallery), cosinus
        top1 = scores.argmax(axis=1)
        rows = []
        for j, iid in enumerate(p_ids):
            true_col = gid_index[iid]
            # rang de la bonne identite = 1 + nb d'identites de la galerie mieux notees
            rank = 1 + int((scores[j] > scores[j, true_col]).sum())
            rows.append({"probe_idx": j, "iid": iid, "top1_iid": gallery_ids[top1[j]],
                         "rank": rank, "correct": int(rank == 1)})
        rank1 = round(100.0 * sum(r["correct"] for r in rows) / len(rows), 2)
        result["terrains"][tname] = {"ids": p_ids, "emb": p_emb, "scores": scores,
                                     "rows": rows, "rank1": rank1}
    return result


def save(result: dict, run_label: str, seed, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"gallery_ids": np.array(result["gallery_ids"]),
              "gallery_emb": result["gallery_emb"].astype(np.float32)}
    for t, r in result["terrains"].items():
        arrays[f"probes__{t}__ids"] = np.array(r["ids"])
        arrays[f"probes__{t}__emb"] = r["emb"].astype(np.float32)
        arrays[f"scores__{t}"] = r["scores"].astype(np.float32)
    npz_path = out_dir / f"eval_{run_label}.npz"
    np.savez_compressed(npz_path, **arrays)

    csv_path = out_dir / f"correct_{run_label}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "seed", "terrain", "probe_idx", "iid", "top1_iid", "rank", "correct"])
        for t, r in result["terrains"].items():
            for row in r["rows"]:
                w.writerow([run_label, seed, t, row["probe_idx"], row["iid"],
                            row["top1_iid"], row["rank"], row["correct"]])
    return npz_path, csv_path


def check_against_history(run_label: str, run_dir: Path, rank1: dict) -> None:
    hist = run_dir / f"exp1_{run_label}.json"
    if not hist.exists():
        print(f"  (pas de JSON d'historique {hist.name} -- pas de verification possible)")
        return
    last = json.loads(hist.read_text())[0]["results_b_epochs"][-1]["test"]
    diffs = {t: (rank1.get(t), last.get(t)) for t in last if abs(rank1.get(t, -1) - last[t]) > 1e-6}
    if diffs:
        print(f"  !! ECART avec le JSON (recalcule vs historique) : {diffs}")
    else:
        print(f"  OK : rank-1 identique au JSON d'historique sur les {len(last)} terrains")


def run_one(model: nn.Module, run_label: str, seed, cache: dict, device: str,
            run_dir: Path, out_dir: Path, check: bool = True) -> None:
    result = evaluate(model, cache, device)
    rank1 = {t: r["rank1"] for t, r in result["terrains"].items()}
    # controle : la fonction d'evaluation de l'entrainement donne le meme rank-1
    ref = quick_eval_all_terrains(model, cache, device)
    assert all(abs(ref[t] - rank1[t]) < 1e-6 for t in ref), (ref, rank1)
    print(f"  rank-1 : " + "  ".join(f"{t}={v:.2f}" for t, v in rank1.items()))
    if check:
        check_against_history(run_label, run_dir, rank1)
    npz_path, csv_path = save(result, run_label, seed, out_dir)
    print(f"  -> {npz_path.name}, {csv_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-id", default="E5_LoRA")
    ap.add_argument("--pretrained", action="store_true", help="evalue le backbone pre-entraine (baseline)")
    ap.add_argument("--checkpoint", nargs="+", default=None, help="nom(s) exact(s) de checkpoint")
    ap.add_argument("--only", nargs="+", default=None,
                    help="prefixe(s) de nom de checkpoint, ex: lora_34_r32 full_ft fc_only")
    ap.add_argument("--all", action="store_true", help="tous les checkpoints + le pre-entraine")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    config = cfg.load_config()
    run_dir = cfg.run_dir(args.experiment_id)
    ckpt_dir = run_dir / "checkpoints"
    out_dir = run_dir / "eval"
    cache = load_eval_cache(cfg.CACHE_DIR / "eval_aligned_cache.npz")
    print(f"Cache d'evaluation : {len(cache['gallery'])} identites en galerie, "
          + ", ".join(f"{t}={len(p)}" for t, p in cache["probes"].items()))

    if args.pretrained or args.all:
        print("\n=== pretrained (baseline) ===")
        model = load_pretrained(config["pretrained_path"], config["embedding_dim"], args.device)
        run_one(model, "pretrained", None, cache, args.device, run_dir, out_dir, check=False)
        del model

    paths: list[Path] = []
    if args.checkpoint:
        paths = [ckpt_dir / c for c in args.checkpoint]
    elif args.only:
        paths = [p for p in sorted(ckpt_dir.glob("*.pt")) if any(p.name.startswith(o) for o in args.only)]
    elif args.all:
        paths = sorted(ckpt_dir.glob("*.pt"))
    if not paths and not args.pretrained:
        raise SystemExit("rien a evaluer : utilisez --pretrained, --checkpoint, --only ou --all")

    for p in paths:
        print(f"\n=== {p.name} ===")
        model, ckpt = load_checkpoint_model(p, config["pretrained_path"], config["embedding_dim"], args.device)
        run_one(model, p.stem, ckpt.get("seed"), cache, args.device, run_dir, out_dir)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"\nSorties dans : {out_dir}")


if __name__ == "__main__":
    main()
