from __future__ import annotations

from pathlib import Path

import yaml

from .hp import Hyperparameters

HPARAMS_FILENAME = "hparams.yaml"


def save_hparams(args: Hyperparameters, log_dir: Path) -> Path:
    """Dump the full hyperparameter config to ``log_dir/hparams.yaml``."""
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / HPARAMS_FILENAME
    data = args.model_dump(mode="json")
    out.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")
    return out


def load_hparams(log_dir: Path) -> Hyperparameters:
    """Reconstruct a Hyperparameters instance from a saved ``hparams.yaml``."""
    path = log_dir / HPARAMS_FILENAME
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Hyperparameters(**data)
