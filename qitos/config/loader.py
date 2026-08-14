"""YAML agent configuration loader."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints


@dataclass(slots=True)
class ModelConfig:
    """Model configuration from YAML."""

    provider: str = "openai"
    model: str = ""
    model_name: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    context_window: Optional[int] = None
    api_mode: str = ""

    @classmethod
    def from_env(cls) -> Optional[ModelConfig]:
        """Load the existing QitOS provider environment into typed config."""

        provider = (
            (os.getenv("QITOS_MODEL_PROVIDER") or os.getenv("MODEL_PROVIDER") or "")
            .strip()
            .lower()
        )
        if not provider:
            if os.getenv("OPENAI_API_KEY") or os.getenv("QITOS_API_KEY"):
                provider = "openai"
            elif os.getenv("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
                provider = "gemini"
            elif os.getenv("LITELLM_MODEL"):
                provider = "litellm"
            elif os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL"):
                provider = "ollama"
            elif os.getenv("LM_STUDIO_BASE_URL"):
                provider = "lmstudio"
            else:
                return None

        model: str = os.getenv("QITOS_MODEL", "")
        api_key: str = ""
        base_url: str = ""
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("QITOS_API_KEY") or ""
            base_url = os.getenv("OPENAI_BASE_URL", "")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            base_url = os.getenv("ANTHROPIC_BASE_URL", "")
        elif provider in {"gemini", "google"}:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
            base_url = os.getenv("GEMINI_BASE_URL", "")
        elif provider == "litellm":
            model = model or os.getenv("LITELLM_MODEL", "")
            api_key = os.getenv("LITELLM_API_KEY", "")
            base_url = os.getenv("LITELLM_API_BASE", "")
        elif provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or ""
        elif provider == "lmstudio":
            base_url = os.getenv("LM_STUDIO_BASE_URL", "")
        elif provider == "vllm":
            base_url = os.getenv("VLLM_BASE_URL", "")

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "model_name": self.model_name,
            "api_key": "***REDACTED***" if self.api_key else "",
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "context_window": self.context_window,
            "api_mode": self.api_mode,
        }


@dataclass(slots=True)
class DatasetItem:
    """A single task in the dataset."""

    task: str = ""
    expected: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentConfig:
    """Agent configuration loaded from YAML."""

    name: str = "agent"
    max_steps: int = 10
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: List[DatasetItem] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    protocol: Optional[str] = None
    parser: Optional[str] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "max_steps": self.max_steps,
            "model": self.model.to_dict(),
            "dataset": [
                {"task": d.task, "expected": d.expected, "metadata": d.metadata}
                for d in self.dataset
            ],
            "tools": self.tools,
            "protocol": self.protocol,
            "parser": self.parser,
            "environment": self.environment,
            "seed": self.seed,
            "metadata": self.metadata,
        }


_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")
_NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class _ModelConfigInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    provider: _NonEmptyString = "openai"
    model: str = ""
    model_name: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.7
    max_tokens: int = Field(default=2048, gt=0)
    context_window: int | None = Field(default=None, gt=0)
    api_mode: str = ""


class _DatasetItemInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    task: str = ""
    expected: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class _AgentConfigInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: _NonEmptyString = "agent"
    max_steps: int = Field(default=10, gt=0)
    model: _ModelConfigInput = Field(default_factory=_ModelConfigInput)
    dataset: list[_DatasetItemInput | str] = Field(default_factory=list)
    tools: list[_NonEmptyString] = Field(default_factory=list)
    protocol: str | None = None
    parser: str | None = None
    environment: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def resolve_env_vars(value: Any) -> Any:
    """Replace ${VAR} patterns with environment variable values."""
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(lambda m: os.getenv(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_env_vars(item) for item in value]
    return value


def load_agent_config(path: str | Path) -> AgentConfig:
    """Load a YAML config file and return an AgentConfig.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed AgentConfig with environment variables resolved.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping, got {type(raw).__name__}"
        )

    raw = resolve_env_vars(raw)
    validated = _AgentConfigInput.model_validate(raw)
    return _to_agent_config(validated)


def _to_agent_config(config: _AgentConfigInput) -> AgentConfig:
    model = config.model
    dataset = [
        (
            DatasetItem(task=item)
            if isinstance(item, str)
            else DatasetItem(
                task=item.task,
                expected=item.expected,
                metadata=dict(item.metadata),
            )
        )
        for item in config.dataset
    ]
    return AgentConfig(
        name=config.name,
        max_steps=config.max_steps,
        model=ModelConfig(
            provider=model.provider,
            model=model.model,
            model_name=model.model_name,
            api_key=model.api_key,
            base_url=model.base_url,
            temperature=model.temperature,
            max_tokens=model.max_tokens,
            context_window=model.context_window,
            api_mode=model.api_mode,
        ),
        dataset=dataset,
        tools=list(config.tools),
        protocol=config.protocol,
        parser=config.parser,
        environment=dict(config.environment),
        seed=config.seed,
        metadata=dict(config.metadata),
    )


__all__ = [
    "AgentConfig",
    "ModelConfig",
    "DatasetItem",
    "load_agent_config",
    "resolve_env_vars",
]
