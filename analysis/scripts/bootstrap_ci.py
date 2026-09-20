#!/usr/bin/env python
"""Phase 2.3 du plan -- intervalles de confiance par bootstrap au niveau IDENTITE.

Remplace les tests sur 3 seeds (aucune puissance) par un reechantillonnage des 30
identites de test : la question "ce resultat tient-il sur d'autres personnes ?" est
celle qui compte, et 30 identites x 3 seeds x ~5 probes donnent de la matiere.

Entree : les correct_{run}.csv produits par eval_checkpoint.py (une ligne par probe :
run, seed, terrain, iid, rank, correct). Les seeds d'une meme configuration sont
regroupes : pour chaque identite et chaque terrain, la precision est la moyenne des
probes de cette identite sur tous les seeds. L'unite reechantillonnee est l'identite.

Deux sorties :
  bootstrap_ci.csv        par config x terrain : rank-1 (point) et IC 95 %
  bootstrap_pairs.csv     par paire (A - B) x terrain (+ SUM = somme des 6 terrains) :
                          difference, IC 95 % APPARIE (memes identites tirees pour A et
                          B -- c'est ce qui donne de la puissance malgre n=30), Cohen's d
                          apparie, et la fraction des tirages ou A > B.

Regle de redaction : ne jamais argumenter une equivalence par l'absence de
significativite. Dire ce que l'IC de la difference exclut ou n'exclut pas.

Usage :
    python analysis/scripts/bootstrap_ci.py                       # lit analysis/outputs/eval/
    python analysis/scripts/bootstrap_ci.py --eval-dir <dossier> --n-boot 10000
    python analysis/scripts/bootstrap_ci.py --pairs lora_34_r32_anchor_bnfrozen:ft_34_anchor
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "analysis" / "outputs" / "eval"
OUT_DIR = ROOT / "analysis" / "outputs"
TERRAINS = ["visible_1.00m", "visible_2.60m", "visible_4.20m", "ir_1.00m", "ir_2.60m", "ir_4.20m"]

# Paires d'interet par defaut (labels = nom de checkpoint sans seed ni extension).
DEFAULT_PAIRS = [
    ("lora_34_r32_anchor_bnfrozen", "ft_34_anchor"),
    ("lora_34_r32_anchor_bnfrozen", "full_ft_anchor"),
    ("full_lora_r8_anchor_bnfrozen", "full_ft_anchor"),
    ("full_lora_r8_anchor_bnfrozen", "full_lora_r8_anchor"),        # BN gelees - BN derivante
    ("lora_34_r8_anchor_bnfrozen", "fc_only_anchor_bnfrozen"),
    ("lora_34_r32_anchor_bnfrozen", "lora_34_r8_anchor_bnfrozen"),
    ("lora_34_r32_anchor_bnfrozen", "lora_34_r16_anchor_bnfrozen"),
    ("lora_34_r32_anchor_bnfrozen", "lora_34_r64_anchor_bnfrozen"),
    ("lora_34_r16_anchor_bnfrozen", "lora_34_r8_anchor_bnfrozen"),
    ("lora_34_r64_anchor_bnfrozen", "lora_34_r16_anchor_bnfrozen"),
    ("ft_34_anchor", "lora_34_r8_anchor_bnfrozen"),                 # le deficit de r=8 face a FT
    ("full_lora_r8_anchor_bnfrozen", "lora_34_r8_anchor_bnfrozen"),  # effet du scope, LoRA
    ("full_ft_anchor", "ft_34_anchor"),                              # effet du scope, FT
    ("lora_34_r8_anchor_bnfrozen", "lora_4_r8_anchor_bnfrozen"),     # layer3, LoRA
]


def config_of(run: str) -> str:
    """'lora_34_r32_seed42_anchor_bnfrozen' -> 'lora_34_r32_anchor_bnfrozen' ; 'pretrained' inchange."""
    return re.sub(r"_seed\d+", "", run)


def load(eval_dir: Path) -> dict[str, dict[str, dict[str, list[int]]]]:
    """-> data[config][terrain][iid] = liste des 0/1 (tous seeds, toutes probes)."""
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    files = sorted(eval_dir.glob("correct_*.csv"))
    if not files:
        raise SystemExit(f"aucun correct_*.csv dans {eval_dir}")
    for f in files:
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                data[config_of(row["run"])][row["terrain"]][row["iid"]].append(int(row["correct"]))
    return data


def per_identity(data_cfg: dict, terrain: str, ids: list[str]) -> np.ndarray:
    """Precision par identite (moyenne des probes x seeds), dans l'ordre `ids`."""
    return np.array([np.mean(data_cfg[terrain][i]) for i in ids])


def n_probes(data_cfg: dict, terrain: str, ids: list[str]) -> np.ndarray:
    return np.array([len(data_cfg[terrain][i]) for i in ids], dtype=float)


def boot_mean(acc: np.ndarray, w: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Precision ponderee par le nombre de probes, sur des tirages d'identites (n_boot x n_id)."""
    num = (acc[idx] * w[idx]).sum(axis=1)
    den = w[idx].sum(axis=1)
    return 100.0 * num / den


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", nargs="*", default=None, help="paires A:B (labels sans seed)")
    args = ap.parse_args()

    data = load(args.eval_dir)
    configs = sorted(data)
    print(f"{len(configs)} configuration(s) : {configs}")

    # Identites communes a toutes les configs (elles doivent l'etre : meme split de test).
    ids = sorted(set.intersection(*[set(data[c][TERRAINS[0]].keys()) for c in configs]))
    n_id = len(ids)
    print(f"{n_id} identites de test, {args.n_boot} tirages")
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n_id, size=(args.n_boot, n_id))    # memes tirages pour tout le monde

    # --- IC par config x terrain ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "bootstrap_ci.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "terrain", "rank1", "ci_lo", "ci_hi", "n_identities", "n_probes_total"])
        for c in configs:
            for t in TERRAINS:
                if t not in data[c]:
                    continue
                acc, wt = per_identity(data[c], t, ids), n_probes(data[c], t, ids)
                point = 100.0 * (acc * wt).sum() / wt.sum()
                b = boot_mean(acc, wt, idx)
                lo, hi = np.percentile(b, [2.5, 97.5])
                w.writerow([c, t, f"{point:.2f}", f"{lo:.2f}", f"{hi:.2f}", n_id, int(wt.sum())])
    print(f"-> {OUT_DIR / 'bootstrap_ci.csv'}")

    # --- Differences appariees ---
    if args.pairs:
        pairs = [tuple(p.split(":")) for p in args.pairs]
    else:
        pairs = [(a, b) for a, b in DEFAULT_PAIRS if a in data and b in data]
        missing = [(a, b) for a, b in DEFAULT_PAIRS if a not in data or b not in data]
        if missing:
            print(f"paires ignorees (config absente) : {missing}")

    with open(OUT_DIR / "bootstrap_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["A", "B", "terrain", "diff", "ci_lo", "ci_hi", "cohen_d_paired", "p_A_gt_B"])
        for a, b in pairs:
            total = np.zeros(args.n_boot)
            total_point = 0.0
            for t in TERRAINS:
                acc_a, acc_b = per_identity(data[a], t, ids), per_identity(data[b], t, ids)
                wt = n_probes(data[a], t, ids)           # meme nombre de probes par identite
                d_id = acc_a - acc_b                      # difference par identite (appariee)
                point = 100.0 * (d_id * wt).sum() / wt.sum()
                bd = boot_mean(acc_a, wt, idx) - boot_mean(acc_b, wt, idx)
                lo, hi = np.percentile(bd, [2.5, 97.5])
                sd = d_id.std(ddof=1)
                cohen = float(d_id.mean() / sd) if sd > 0 else float("nan")
                w.writerow([a, b, t, f"{point:+.2f}", f"{lo:+.2f}", f"{hi:+.2f}",
                            f"{cohen:.2f}", f"{(bd > 0).mean():.3f}"])
                total += bd
                total_point += point
            lo, hi = np.percentile(total, [2.5, 97.5])
            w.writerow([a, b, "SUM", f"{total_point:+.2f}", f"{lo:+.2f}", f"{hi:+.2f}", "",
                        f"{(total > 0).mean():.3f}"])
            print(f"  {a} - {b} : SUM {total_point:+.1f}  IC95 [{lo:+.1f}, {hi:+.1f}]"
                  f"  P(A>B)={(total > 0).mean():.3f}")
    print(f"-> {OUT_DIR / 'bootstrap_pairs.csv'}")


if __name__ == "__main__":
    main()
