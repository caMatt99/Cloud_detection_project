"""
Reproducibility, environment and config helpers shared by every experiment.
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

ENVS = ("kaggle", "local")
KAGGLE_INPUT = Path("/kaggle/input")


# ─────────────────────────────────────────────
# Seeding
# ─────────────────────────────────────────────

def seed_everything(seed: int, deterministic: bool = True) -> None:
    """
    Seeds Python, NumPy and PyTorch (CPU + CUDA).

    deterministic=True also disables cuDNN autotuning and asks PyTorch for
    deterministic kernels (warn_only, so ops without a deterministic
    implementation emit a warning instead of crashing). On a T4 this costs a
    few percent of throughput and removes most of the run-to-run noise that
    is NOT due to the seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker_init_fn: derives Python/NumPy seeds from the per-worker
    torch seed, so augmentations that use `random` (Cutout) or NumPy are
    reproducible too. torch's own RNG (RandAugment) is already seeded per
    worker by the DataLoader generator.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────

def get_env() -> str:
    """'kaggle' inside a Kaggle notebook, 'local' otherwise."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or KAGGLE_INPUT.is_dir():
        return "kaggle"
    return "local"


def print_env() -> None:
    """One-screen summary of the runtime; warns if the PyTorch build cannot use the GPU."""
    import timm
    import torchvision

    print(f"env={get_env()}  torch={torch.__version__}  torchvision={torchvision.__version__}  "
          f"timm={timm.__version__}  cpus={os.cpu_count()}")
    if not torch.cuda.is_available():
        print("GPU: none (CPU only)")
        return
    archs = set(torch.cuda.get_arch_list())
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        supported = f"sm_{major}{minor}" in archs
        print(f"GPU{i}: {name}  {mem:.1f} GB  sm_{major}{minor}{'' if supported else '  <- NOT supported by this torch build'}")
        if not supported:
            print("  Switch the accelerator to 'GPU T4 x2' (Settings -> Accelerator).")


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

def _is_dataset_root(path: Path) -> bool:
    return all((path / split / cls).is_dir() for split in ("train", "val", "test") for cls in ("true", "false"))


def find_dataset_root(base: Path = KAGGLE_INPUT, max_depth: int = 4) -> str:
    """
    Locates the directory containing train/ val/ test/ (each with true/ false/)
    under `base`. Kaggle's mount point depends on the dataset slug and on how
    the zip was nested, so it is discovered rather than hardcoded.
    macOS zip artefacts (__MACOSX) and hidden folders are ignored.
    """
    base = Path(base)
    matches: List[Path] = []
    for root, dirs, _ in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d != "__MACOSX" and not d.startswith("."))
        root_path = Path(root)
        if _is_dataset_root(root_path):
            matches.append(root_path)
            dirs[:] = []
        elif len(root_path.relative_to(base).parts) >= max_depth:
            dirs[:] = []

    if len(matches) == 1:
        return str(matches[0])
    if not matches:
        raise FileNotFoundError(
            f"No train/val/test dataset found under {base}. Attach the dataset to the notebook "
            "(Add Input) or set dataset.path.kaggle explicitly in the config."
        )
    raise RuntimeError(
        "Several candidate datasets found, set dataset.path.kaggle explicitly:\n"
        + "\n".join(f"  {m}" for m in matches)
    )


def load_config(path: str, env: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads a YAML experiment config and resolves every `{kaggle: ..., local: ...}`
    entry to the current environment, so downstream code reads plain strings
    (cfg["dataset"]["path"], cfg["paths"]["results"], ...).
    dataset.path = 'auto' is resolved with find_dataset_root().
    """
    env = env or get_env()
    with open(path) as f:
        cfg = yaml.safe_load(f)

    def resolve(node, where=""):
        if isinstance(node, dict):
            if node and set(node) <= set(ENVS):
                if env not in node:
                    raise KeyError(f"{path}: '{where}' has no entry for env '{env}'")
                return node[env]
            return {k: resolve(v, f"{where}.{k}".lstrip(".")) for k, v in node.items()}
        return node

    cfg = resolve(cfg)
    if cfg.get("dataset", {}).get("path") == "auto":
        cfg["dataset"]["path"] = find_dataset_root()
    cfg["_env"] = env
    cfg["_config_path"] = os.path.abspath(path)
    return cfg
