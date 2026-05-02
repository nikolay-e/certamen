from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from certamen.domain.errors import ConfigurationError


class SlimModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider: str
    model_name: str
    display_name: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    reasoning_effort: str | None = None


class SlimSecretProvider(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_var: str
    op_path: str | None = None


class SlimSecrets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    providers: dict[str, SlimSecretProvider] = Field(default_factory=dict)


class SlimLogging(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"
    file_logging: bool = True
    console_logging: bool = True


class SlimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    models: dict[str, SlimModel]
    workflow: str
    overrides: dict[str, Any] = Field(default_factory=dict)
    secrets: SlimSecrets | None = None
    outputs_dir: str | None = None
    logging: SlimLogging | None = None


_LEGACY_KEYS = (
    "tournament",
    "knowledge_bank",
    "features",
    "prompts",
    "retry",
    "reasoning_perspectives",
)


def is_slim_config(raw: dict[str, Any]) -> bool:
    return "workflow" in raw and not any(key in raw for key in _LEGACY_KEYS)


def load_slim_config(path: str | Path) -> SlimConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(
            f"Slim config not found: {config_path.resolve()}"
        )

    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"Slim config must be a YAML mapping, got {type(raw).__name__}"
        )

    legacy_keys_present = [k for k in _LEGACY_KEYS if k in raw]
    if legacy_keys_present:
        raise ConfigurationError(
            "Slim config rejects legacy keys "
            f"{legacy_keys_present}. Move them into the workflow YAML "
            "or apply via 'overrides:'."
        )

    try:
        return SlimConfig(**raw)
    except ValidationError as exc:
        raise ConfigurationError(
            f"Invalid slim config at {config_path}: {exc}"
        ) from exc
