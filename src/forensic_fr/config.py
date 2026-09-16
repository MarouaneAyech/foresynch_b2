"""Résolution des chemins et chargement des fichiers de configuration.

Tous les chemins étaient codés en dur sur `/content/drive/MyDrive/research/forensic_fr`
dans le notebook d'origine (Colab uniquement). Ici la racine est configurable via la
variable d'environnement FORENSIC_FR_ROOT, pour pouvoir exécuter le même code en local
ou sur toute autre machine, tout en gardant le chemin Colab comme valeur par défaut.
"""
import json
import os
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("FORENSIC_FR_ROOT")
    if env:
        return Path(env)
    return Path("/content/drive/MyDrive/research/forensic_fr")


PROJECT_ROOT = project_root()
OUT_ROOT = PROJECT_ROOT / "outputs"
PHASE1_DIR = OUT_ROOT / "phase1_finetune"
CACHE_DIR = PHASE1_DIR / "aligned_cache"


def load_config(path: Path | None = None) -> dict:
    path = path or (PHASE1_DIR / "config_phase1.json")
    with open(path) as f:
        return json.load(f)


def load_split(path: Path | None = None) -> dict:
    path = path or (PHASE1_DIR / "dataset_split.json")
    with open(path) as f:
        return json.load(f)


def run_dir(experiment_id: str) -> Path:
    d = PHASE1_DIR / "runs" / experiment_id
    (d / "checkpoints").mkdir(parents=True, exist_ok=True)
    return d


def bootstrap_configs(repo_dir: Path) -> None:
    """Copie configs/phase1_finetune/*.json du dépôt (source de vérité, versionnée)
    vers PHASE1_DIR sur Drive, uniquement si absent — Drive sert de cache d'exécution
    pour le notebook Colab, le dépôt git reste la référence pour la reproductibilité.
    N'écrase jamais un fichier déjà présent sur Drive (ex: si modifié manuellement).
    """
    src_dir = Path(repo_dir) / "configs" / "phase1_finetune"
    PHASE1_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("config_phase1.json", "dataset_split.json"):
        dst = PHASE1_DIR / name
        if dst.exists():
            continue
        src = src_dir / name
        if src.exists():
            dst.write_bytes(src.read_bytes())
            print(f"bootstrap: {src} -> {dst}")
