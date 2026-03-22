from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, computed_field


class Hyperparameters(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Data paths.
    data_path: str = Field(default="./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = Field(default="./data/tokenizers/fineweb_1024_bpe.model")

    # Run identity.
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    seed: int = Field(default=1337)

    # Validation cadence and batch size.
    val_batch_size: int = Field(default=524_288)
    val_loss_every: int = Field(default=1000)
    train_log_every: int = Field(default=200)

    # Training length.
    iterations: int = Field(default=20000)
    warmdown_iters: int = Field(default=1200)
    warmup_steps: int = Field(default=20)
    train_batch_tokens: int = Field(default=524_288)
    train_seq_len: int = Field(default=1024)
    max_wallclock_seconds: float = Field(default=600.0)
    qk_gain_init: float = Field(default=1.5)

    # Model shape.
    vocab_size: int = Field(default=1024)
    num_layers: int = Field(default=9)
    num_kv_heads: int = Field(default=4)
    model_dim: int = Field(default=512)
    num_heads: int = Field(default=8)
    mlp_mult: int = Field(default=2)
    tie_embeddings: bool = Field(default=True)
    rope_base: float = Field(default=10000.0)
    logit_softcap: float = Field(default=30.0)
    use_bitlinear: bool = Field(default=False)
    bitlinear_layers: list[int] | None = Field(default=None)
    attention_bitlinear: bool = Field(default=True)
    mlp_bitlinear: bool = Field(default=True)

    # Optimizer hyperparameters.
    embed_lr: float = Field(default=0.6)
    head_lr: float = Field(default=0.008)
    tied_embed_lr: float = Field(default=0.05)
    tied_embed_init_std: float = Field(default=0.005)
    matrix_lr: float = Field(default=0.04)
    scalar_lr: float = Field(default=0.04)
    muon_momentum: float = Field(default=0.95)
    muon_backend_steps: int = Field(default=5)
    muon_momentum_warmup_start: float = Field(default=0.85)
    muon_momentum_warmup_steps: int = Field(default=500)
    beta1: float = Field(default=0.9)
    beta2: float = Field(default=0.95)
    adam_eps: float = Field(default=1e-8)
    grad_clip_norm: float = Field(default=0.0)

    # Runtime environment (auto-detected, not user-configurable).
    num_gpus: int = Field(default_factory=lambda: int(os.environ.get("WORLD_SIZE", "1")))
    gpu_name: str = Field(default_factory=lambda: (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    ))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def train_files(self) -> str:
        return os.path.join(self.data_path, "fineweb_train_*.bin")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def val_files(self) -> str:
        return os.path.join(self.data_path, "fineweb_val_*.bin")

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> Hyperparameters:
        """Load config from a YAML file, with env var overrides on top.

        Priority (highest wins): env vars > YAML file > field defaults.
        """
        file_values: dict[str, Any] = {}
        if path is not None:
            file_values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        env_overrides: dict[str, Any] = {}
        for name, field_info in cls.model_fields.items():
            env_val = os.environ.get(name.upper())
            if env_val is not None:
                target_type = field_info.annotation
                if target_type is bool:
                    env_overrides[name] = bool(int(env_val))
                else:
                    env_overrides[name] = env_val

        return cls(**{**file_values, **env_overrides})
