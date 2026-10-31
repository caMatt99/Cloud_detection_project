"""
Reproducibility and config helpers shared by every experiment.
"""

import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml


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


def get_env() -> str:
    """'colab' on Google Colab, 'local' otherwise."""
    return "colab" if os.path.exists("/content") else "local"


def load_config(path: str, env: str = None) -> Dict[str, Any]:
    """
    Loads a YAML experiment config and resolves every `{colab: ..., local: ...}`
    path entry to the current environment, so downstream code reads plain
    strings (cfg["dataset"]["path"], cfg["paths"]["checkpoints"], ...).
    """
    env = env or get_env()
    with open(path) as f:
        cfg = yaml.safe_load(f)

    def resolve(node):
        if isinstance(node, dict):
            if set(node) == {"colab", "local"}:
                return node[env]
            return {k: resolve(v) for k, v in node.items()}
        return node

    cfg = resolve(cfg)
    cfg["_env"] = env
    cfg["_config_path"] = path
    return cfg


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """{'a': {'b': 1}} -> {'a.b': 1}. Used to log the whole config to W&B."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, f"{key}."))
        else:
            out[key] = v
    return out
